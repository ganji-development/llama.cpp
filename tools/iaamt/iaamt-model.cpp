// GGUF loading for instrument-agnostic-amt models.

#include "iaamt.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace {

std::string fmt(const char * f, ...) {
    va_list ap;
    va_start(ap, f);
    char buf[512];
    vsnprintf(buf, sizeof(buf), f, ap);
    va_end(ap);
    return std::string(buf);
}

struct kv_reader {
    gguf_context * ctx;

    bool has(const std::string & key) const {
        return gguf_find_key(ctx, key.c_str()) >= 0;
    }

    int32_t get_i32(const std::string & key, int32_t def) const {
        const int id = gguf_find_key(ctx, key.c_str());
        if (id < 0) {
            return def;
        }
        switch (gguf_get_kv_type(ctx, id)) {
            case GGUF_TYPE_INT32:  return gguf_get_val_i32(ctx, id);
            case GGUF_TYPE_UINT32: return (int32_t) gguf_get_val_u32(ctx, id);
            case GGUF_TYPE_INT64:  return (int32_t) gguf_get_val_i64(ctx, id);
            default:               return def;
        }
    }

    float get_f32(const std::string & key, float def) const {
        const int id = gguf_find_key(ctx, key.c_str());
        if (id < 0 || gguf_get_kv_type(ctx, id) != GGUF_TYPE_FLOAT32) {
            return def;
        }
        return gguf_get_val_f32(ctx, id);
    }

    bool get_bool(const std::string & key, bool def) const {
        const int id = gguf_find_key(ctx, key.c_str());
        if (id < 0 || gguf_get_kv_type(ctx, id) != GGUF_TYPE_BOOL) {
            return def;
        }
        return gguf_get_val_bool(ctx, id);
    }

    std::string get_str(const std::string & key, const std::string & def) const {
        const int id = gguf_find_key(ctx, key.c_str());
        if (id < 0 || gguf_get_kv_type(ctx, id) != GGUF_TYPE_STRING) {
            return def;
        }
        return gguf_get_val_str(ctx, id);
    }

    std::vector<float> get_f32_array(const std::string & key) const {
        std::vector<float> out;
        const int id = gguf_find_key(ctx, key.c_str());
        if (id < 0 || gguf_get_kv_type(ctx, id) != GGUF_TYPE_ARRAY) {
            return out;
        }
        const int n = (int) gguf_get_arr_n(ctx, id);
        const gguf_type t = gguf_get_arr_type(ctx, id);
        const void * data = gguf_get_arr_data(ctx, id);
        out.resize(n);
        for (int i = 0; i < n; ++i) {
            if (t == GGUF_TYPE_FLOAT32) {
                out[i] = ((const float *) data)[i];
            } else if (t == GGUF_TYPE_INT32) {
                out[i] = (float) ((const int32_t *) data)[i];
            }
        }
        return out;
    }

    std::vector<int> get_i32_array(const std::string & key) const {
        std::vector<int> out;
        const int id = gguf_find_key(ctx, key.c_str());
        if (id < 0 || gguf_get_kv_type(ctx, id) != GGUF_TYPE_ARRAY) {
            return out;
        }
        const int n = (int) gguf_get_arr_n(ctx, id);
        const gguf_type t = gguf_get_arr_type(ctx, id);
        const void * data = gguf_get_arr_data(ctx, id);
        out.resize(n);
        for (int i = 0; i < n; ++i) {
            if (t == GGUF_TYPE_INT32) {
                out[i] = ((const int32_t *) data)[i];
            } else if (t == GGUF_TYPE_UINT32) {
                out[i] = (int) ((const uint32_t *) data)[i];
            }
        }
        return out;
    }
};

} // namespace

iaamt_model::~iaamt_model() {
    if (buf_weights) {
        ggml_backend_buffer_free(buf_weights);
    }
    if (ctx_data) {
        ggml_free(ctx_data);
    }
    if (backend && backend != backend_cpu) {
        ggml_backend_free(backend);
    }
    if (backend_cpu) {
        ggml_backend_free(backend_cpu);
    }
}

bool iaamt_model_load(iaamt_model & model, const std::string & fname, bool use_gpu, std::string & err) {
    ggml_context * ctx_meta = nullptr;
    gguf_init_params gparams = {
        /* no_alloc = */ true,
        /* ctx      = */ &ctx_meta,
    };

    gguf_context * gguf = gguf_init_from_file(fname.c_str(), gparams);
    if (!gguf) {
        err = "failed to open " + fname;
        return false;
    }

    bool ok = false;
    do {
        kv_reader kv{gguf};

        const std::string arch = kv.get_str("general.architecture", "");
        if (arch != "iaamt") {
            err = "not an iaamt GGUF (general.architecture = '" + arch + "')";
            break;
        }

        iaamt_hparams & hp = model.hparams;
        hp.model_type = kv.get_str("iaamt.model_type", "");
        const bool is_velocity   = (hp.model_type == IAAMT_MODEL_TYPE_VELOCITY);
        const bool is_beat_chord = (hp.model_type == IAAMT_MODEL_TYPE_BEAT_CHORD);
        if (hp.model_type != IAAMT_MODEL_TYPE_TRANSCRIPTION && !is_velocity && !is_beat_chord) {
            err = "unsupported iaamt.model_type '" + hp.model_type + "'";
            break;
        }

        hp.sample_rate          = kv.get_i32("iaamt.sample_rate", 22050);
        hp.hop_length           = kv.get_i32("iaamt.hop_length", 512);
        hp.input_audio_channels = kv.get_i32("iaamt.input_audio_channels", 2);
        hp.num_audio_channels   = kv.get_i32("iaamt.num_audio_channels", 10);
        hp.cqt_fmin             = kv.get_f32("iaamt.cqt_fmin", 27.5f);
        hp.cqt_fmin_large       = kv.get_f32("iaamt.cqt.fmin_large", hp.cqt_fmin);
        hp.cqt_n_bins           = kv.get_i32("iaamt.cqt_n_bins", 312);
        hp.cqt_actual_bins      = kv.get_i32("iaamt.cqt.actual_bins", hp.cqt_n_bins);
        hp.cqt_n_bins_large     = kv.get_i32("iaamt.cqt.n_bins_large", hp.cqt_n_bins);
        hp.cqt_bins_per_octave  = kv.get_i32("iaamt.cqt_bins_per_octave", 36);
        hp.cqt_filter_scale     = kv.get_f32("iaamt.cqt_filter_scale", 0.475f);
        hp.cqt_log_scale        = kv.get_bool("iaamt.cqt_log_scale", false);
        hp.harmonics            = kv.get_f32_array("iaamt.harmonics");
        hp.harmonic_shifts      = kv.get_f32_array("iaamt.cqt.harmonic_shifts");
        hp.cqt_kernels_embedded = kv.get_bool("iaamt.cqt.kernels_embedded", true);

        hp.hidden_size          = kv.get_i32("iaamt.hidden_size", 384);
        hp.base_ch              = kv.get_i32("iaamt.base_ch", 64);
        // beat_chord names these plainly; the audio models prefix them
        hp.n_layer              = kv.get_i32("iaamt.encoder_num_layers",
                                             kv.get_i32("iaamt.num_layers", 6));
        hp.n_head               = kv.get_i32("iaamt.encoder_num_heads",
                                             kv.get_i32("iaamt.num_heads", 12));
        hp.num_input_channels   = kv.get_i32("iaamt.num_input_channels", 72);
        hp.num_global_tokens    = kv.get_i32("iaamt.num_global_tokens", 4);
        hp.num_meter_classes    = kv.get_i32("iaamt.num_meter_classes", 60);
        hp.num_root_chord_classes = kv.get_i32("iaamt.num_root_chord_classes", 745);
        hp.time_downsample      = kv.get_i32("iaamt.time_downsample_factor", 8);
        hp.ffn_multiplier       = kv.get_i32("iaamt.ffn_multiplier", 4);
        {
            const std::vector<int> r = kv.get_i32_array("iaamt.inter_refine_layers");
            hp.inter_refine_layers = r;
        }
        hp.pitch_min            = kv.get_i32("iaamt.pitch_min", 21);
        hp.pitch_max            = kv.get_i32("iaamt.pitch_max", 108);
        // velocity checkpoints express the pitch range instead of a query count
        hp.pitch_query_count    = kv.get_i32("iaamt.pitch_query_count",
                                             hp.pitch_max - hp.pitch_min + 1);
        hp.num_stem_classes     = kv.get_i32("iaamt.num_stem_classes", 7);
        hp.note_hidden_size     = kv.get_i32("iaamt.note_hidden_size", 256);
        hp.absolute_velocity_energy =
            kv.get_bool("iaamt.use_absolute_velocity_energy", false);
        hp.local_frame_offsets  = kv.get_f32_array("iaamt.local_frame_offsets");
        if (is_velocity && hp.local_frame_offsets.empty()) {
            err = "velocity model is missing iaamt.local_frame_offsets";
            break;
        }

        hp.num_pitch_slots      = kv.get_i32("iaamt.num_pitch_slots", 1);
        hp.semi_crf_head_dim    = kv.get_i32("iaamt.semi_crf_head_dim", 256);
        hp.semi_crf_length_penalty = kv.get_f32("iaamt.semi_crf_length_penalty", 0.0f);
        hp.semi_crf_length_scaling = kv.get_str("iaamt.semi_crf_length_scaling", "none");
        hp.num_instrument_classes  = kv.get_i32("iaamt.num_instrument_classes", 0);
        hp.has_slot_embedding      = kv.get_bool("iaamt.has_slot_embedding", false);
        hp.has_interval_instrument_head =
            kv.get_bool("iaamt.has_interval_instrument_head", false);
        hp.use_interval_boundary_head =
            kv.get_bool("iaamt.use_interval_boundary_head", true);

        // beat_chord consumes a MIDI roll, so it carries no CQT front end
        if (!is_beat_chord) {
            if (!hp.cqt_kernels_embedded) {
                err = "this GGUF was converted with --no-embed-cqt; the runtime needs "
                      "the precomputed CQT kernels";
                break;
            }
            if (hp.harmonics.empty() || hp.harmonics.size() != hp.harmonic_shifts.size()) {
                err = "missing or inconsistent iaamt.harmonics / iaamt.cqt.harmonic_shifts";
                break;
            }

            const std::vector<int> fft_sizes = kv.get_i32_array("iaamt.cqt.fft_sizes");
            const std::vector<int> bin_beg   = kv.get_i32_array("iaamt.cqt.stage_bin_start");
            const std::vector<int> bin_end   = kv.get_i32_array("iaamt.cqt.stage_bin_end");
            if (fft_sizes.empty() || fft_sizes.size() != bin_beg.size() ||
                fft_sizes.size() != bin_end.size()) {
                err = "missing or inconsistent iaamt.cqt.* stage metadata";
                break;
            }
            hp.cqt_stages.resize(fft_sizes.size());
            for (size_t i = 0; i < fft_sizes.size(); ++i) {
                hp.cqt_stages[i] = { fft_sizes[i], bin_beg[i], bin_end[i] };
            }
        }

        if (hp.hidden_size % hp.n_head != 0 || (hp.hidden_size / hp.n_head) % 2 != 0) {
            err = "hidden_size / n_head must be even for RoPE";
            break;
        }

        // ---- backend ----
        model.backend_cpu = ggml_backend_cpu_init();
        if (!model.backend_cpu) {
            err = "failed to initialize the CPU backend";
            break;
        }
        if (use_gpu) {
            ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
            if (dev) {
                model.backend = ggml_backend_dev_init(dev, nullptr);
            }
        }
        if (!model.backend) {
            model.backend = model.backend_cpu;
        }

        // ---- create tensors ----
        const int n_tensors = (int) gguf_get_n_tensors(gguf);
        {
            ggml_init_params params = {
                /* mem_size   = */ (size_t) (n_tensors + 1) * ggml_tensor_overhead(),
                /* mem_buffer = */ nullptr,
                /* no_alloc   = */ true,
            };
            model.ctx_data = ggml_init(params);
            if (!model.ctx_data) {
                err = "failed to create the tensor context";
                break;
            }
        }

        std::map<std::string, size_t> offsets;
        for (int i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(gguf, i);
            ggml_tensor * src = ggml_get_tensor(ctx_meta, name);
            ggml_tensor * dst = ggml_dup_tensor(model.ctx_data, src);
            ggml_set_name(dst, name);
            offsets[name] = gguf_get_data_offset(gguf) + gguf_get_tensor_offset(gguf, i);
        }

        model.buf_weights = ggml_backend_alloc_ctx_tensors(model.ctx_data, model.backend);
        if (!model.buf_weights) {
            err = "failed to allocate the weight buffer";
            break;
        }
        ggml_backend_buffer_set_usage(model.buf_weights, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        // ---- read data ----
        {
            std::ifstream fin(fname, std::ios::binary);
            if (!fin) {
                err = "failed to reopen " + fname;
                break;
            }
            std::vector<uint8_t> read_buf;
            const bool host = ggml_backend_buffer_is_host(model.buf_weights);
            bool read_ok = true;
            for (int i = 0; i < n_tensors && read_ok; ++i) {
                const char * name = gguf_get_tensor_name(gguf, i);
                ggml_tensor * cur = ggml_get_tensor(model.ctx_data, name);
                const size_t nbytes = ggml_nbytes(cur);
                fin.seekg(offsets[name], std::ios::beg);
                if (host) {
                    fin.read((char *) cur->data, nbytes);
                } else {
                    read_buf.resize(nbytes);
                    fin.read((char *) read_buf.data(), nbytes);
                    ggml_backend_tensor_set(cur, read_buf.data(), 0, nbytes);
                }
                read_ok = (bool) fin;
            }
            if (!read_ok) {
                err = "failed while reading tensor data from " + fname;
                break;
            }
        }

        // ---- bind ----
        bool missing = false;
        auto get = [&](const std::string & name, bool required = true) -> ggml_tensor * {
            ggml_tensor * t = ggml_get_tensor(model.ctx_data, name.c_str());
            if (!t && required) {
                err = "missing tensor: " + name;
                missing = true;
            }
            return t;
        };

        model.stem_conv1_w   = get("bb.stem.conv1.weight");
        model.stem_conv1_b   = get("bb.stem.conv1.bias");
        model.stem_conv2_w   = get("bb.stem.conv2.weight");
        model.stem_conv2_b   = get("bb.stem.conv2.bias");
        model.stem_freq_embd = get("bb.stem.freq_embd", !is_beat_chord);
        for (int i = 0; i < 4 && !missing; ++i) {
            model.stem_blocks[i].conv_w = get(fmt("bb.stem.blk.%d.conv.weight", i));
            model.stem_blocks[i].conv_b = get(fmt("bb.stem.blk.%d.conv.bias",   i));
            model.stem_blocks[i].norm_w = get(fmt("bb.stem.blk.%d.norm.weight", i));
            model.stem_blocks[i].norm_b = get(fmt("bb.stem.blk.%d.norm.bias",   i));
        }

        model.pitch_type_embd   = get("bb.pitch_type_embd");
        if (!is_beat_chord) {
            model.band_type_embd    = get("bb.band_type_embd");
            model.pitch_query_fc1_w = get("bb.pitch_query.fc1.weight");
            model.pitch_query_fc1_b = get("bb.pitch_query.fc1.bias");
            model.pitch_query_fc2_w = get("bb.pitch_query.fc2.weight");
            model.pitch_query_fc2_b = get("bb.pitch_query.fc2.bias");
        }

        model.layers.resize(hp.n_layer);
        for (int il = 0; il < hp.n_layer && !missing; ++il) {
            const char * axes[2] = { "time", "band" };
            iaamt_attn * dst[2]  = { &model.layers[il].time, &model.layers[il].band };
            for (int a = 0; a < 2; ++a) {
                const std::string p = fmt("bb.blk.%d.%s.", il, axes[a]);
                iaamt_attn & at = *dst[a];
                at.norm_q     = get(p + "attn_norm.weight");
                at.norm_kv    = get(p + "attn_norm_kv.weight");
                at.q_w        = get(p + "attn_q.weight");
                at.q_b        = get(p + "attn_q.bias");
                at.k_w        = get(p + "attn_k.weight");
                at.k_b        = get(p + "attn_k.bias");
                at.v_w        = get(p + "attn_v.weight");
                at.v_b        = get(p + "attn_v.bias");
                at.gate_w     = get(p + "attn_gate.weight");
                at.gate_b     = get(p + "attn_gate.bias");
                at.out_w      = get(p + "attn_out.weight");
                at.out_b      = get(p + "attn_out.bias");
                at.ffn_norm   = get(p + "ffn_norm.weight");
                at.ffn_up_w   = get(p + "ffn_up.weight");
                at.ffn_up_b   = get(p + "ffn_up.bias");
                at.ffn_down_w = get(p + "ffn_down.weight");
                at.ffn_down_b = get(p + "ffn_down.bias");
            }
        }

        model.output_norm = get("bb.output_norm.weight");
        model.up_conv_w   = get("bb.up_conv.weight", !is_beat_chord);
        model.up_conv_b   = get("bb.up_conv.bias",   !is_beat_chord);

        auto bind_adapter = [&](iaamt_adapter & ad, const std::string & p) {
            ad.norm_w = get(p + "norm.weight");
            ad.norm_b = get(p + "norm.bias");
            ad.fc1_w  = get(p + "fc1.weight");
            ad.fc1_b  = get(p + "fc1.bias");
            ad.fc2_w  = get(p + "fc2.weight");
            ad.fc2_b  = get(p + "fc2.bias");
        };

        auto bind_bc_head = [&](iaamt_bc_head & h, const std::string & p,
                                const std::vector<std::string> & projs) {
            h.norm_w = get(p + "norm.weight");
            h.norm_b = get(p + "norm.bias");
            h.fc_w   = get(p + "fc.weight");
            h.fc_b   = get(p + "fc.bias");
            h.n_proj = (int) projs.size();
            for (size_t i = 0; i < projs.size(); ++i) {
                h.proj_w[i] = get(p + projs[i] + ".weight");
                h.proj_b[i] = get(p + projs[i] + ".bias");
            }
        };
        const std::vector<std::string> beat_projs  = { "frame", "group_boundary" };
        const std::vector<std::string> chord_projs = {
            "boundary", "root_chord", "bass", "key_boundary", "key", "pitch" };

        if (is_velocity) {
            // The velocity head stops at the backbone; everything past it is a
            // per-note MLP evaluated on the CPU, so bind nothing else here.
            if (missing) {
                break;
            }
        } else if (is_beat_chord) {
            model.bc_conv1_w        = model.stem_conv1_w;
            model.bc_conv1_b        = model.stem_conv1_b;
            model.bc_conv2_w        = model.stem_conv2_w;
            model.bc_conv2_b        = model.stem_conv2_b;
            model.bc_chan_w         = get("bb.stem.channel_proj.weight");
            model.bc_chan_b         = get("bb.stem.channel_proj.bias");
            model.bc_pitch_embd     = get("bb.stem.pitch_embd");
            model.bc_input_proj_w   = get("bb.input_proj.weight");
            model.bc_input_proj_b   = get("bb.input_proj.bias");
            model.bc_pitch_pos_embd = get("bb.pitch_pos_embd");
            model.bc_global_tokens  = get("bb.global_tokens");
            model.bc_global_type    = get("bb.global_type_embd");
            model.bc_global_up_w    = get("bb.global_up_conv.weight");
            model.bc_global_up_b    = get("bb.global_up_conv.bias");
            model.bc_merge_w        = get("bb.global_token_merge.weight");
            model.bc_merge_b        = get("bb.global_token_merge.bias");

            bind_adapter(model.bc_beat_adapter,  "head.beat_adapter.");
            bind_adapter(model.bc_chord_adapter, "head.chord_adapter.");
            bind_bc_head(model.bc_beat_head,  "head.beat.",  beat_projs);
            bind_bc_head(model.bc_chord_head, "head.chord.", chord_projs);

            for (int layer : hp.inter_refine_layers) {
                iaamt_bc_refine r;
                r.layer = layer;
                const std::string p = fmt("head.inter.%d.", layer);
                r.up_conv_w = get(p + "global_up_conv.weight");
                r.up_conv_b = get(p + "global_up_conv.bias");
                r.merge_w   = get(p + "global_token_merge.weight");
                r.merge_b   = get(p + "global_token_merge.bias");
                bind_adapter(r.beat_adapter,  p + "beat_adapter.");
                bind_adapter(r.chord_adapter, p + "chord_adapter.");
                bind_bc_head(r.beat_head,  p + "beat.",  beat_projs);
                bind_bc_head(r.chord_head, p + "chord.", chord_projs);
                r.beat_down_w  = get(p + "beat_down_conv.weight");
                r.beat_down_b  = get(p + "beat_down_conv.bias");
                r.chord_down_w = get(p + "chord_down_conv.weight");
                r.chord_down_b = get(p + "chord_down_conv.bias");
                r.beat_fb_w    = get(p + "beat_feedback.weight");
                r.beat_fb_b    = get(p + "beat_feedback.bias");
                r.chord_fb_w   = get(p + "chord_feedback.weight");
                r.chord_fb_b   = get(p + "chord_feedback.bias");
                r.beat_gate    = get(p + "beat_feedback_gate");
                r.chord_gate   = get(p + "chord_feedback_gate");
                model.bc_refine.push_back(r);
            }
            if (missing) {
                break;
            }
        } else {

        bind_adapter(model.interval_adapter,   "head.interval_adapter.");
        bind_adapter(model.instrument_adapter, "head.instrument_adapter.");

        model.interval_scorer_w = get("head.interval_scorer.weight");
        model.interval_scorer_b = get("head.interval_scorer.bias");
        model.instrument_cls_w  = get("head.instrument_classifier.weight", false);
        model.instrument_cls_b  = get("head.instrument_classifier.bias",   false);
        model.slot_embd         = get("head.slot_embd.weight", hp.has_slot_embedding);

        if (hp.use_interval_boundary_head) {
            model.boundary_fc1_w = get("head.boundary.fc1.weight", false);
            model.boundary_fc1_b = get("head.boundary.fc1.bias",   false);
            model.boundary_fc2_w = get("head.boundary.fc2.weight", false);
            model.boundary_fc2_b = get("head.boundary.fc2.bias",   false);
            if (!model.boundary_fc1_w) {
                hp.use_interval_boundary_head = false;
            }
        }
        }   // !is_velocity

        if (missing) {
            break;
        }

        // ---- CQT tables to host memory ----
        if (is_beat_chord) {
            ok = true;
            break;
        }
        const int n_stages = (int) hp.cqt_stages.size();
        model.cqt_kernel_re.resize(n_stages);
        model.cqt_kernel_im.resize(n_stages);
        model.cqt_window.resize(n_stages);
        model.cqt_resampler.resize(std::max(0, n_stages - 1));

        auto fetch = [&](const std::string & name, std::vector<float> & out) -> bool {
            ggml_tensor * t = ggml_get_tensor(model.ctx_data, name.c_str());
            if (!t) {
                err = "missing tensor: " + name;
                return false;
            }
            if (t->type != GGML_TYPE_F32) {
                err = name + " must be f32";
                return false;
            }
            out.resize(ggml_nelements(t));
            ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));
            return true;
        };

        if (is_velocity) {
            auto & v = model.vel;
            const bool vel_ok =
                fetch("head.pitch_embd.weight",   v.pitch_embd) &&
                fetch("head.program_embd.weight", v.program_embd) &&
                fetch("head.drum_embd.weight",    v.drum_embd) &&
                fetch("head.stem_embd.weight",    v.stem_embd) &&
                fetch("head.duration.fc1.weight", v.dur_w1) &&
                fetch("head.duration.fc1.bias",   v.dur_b1) &&
                fetch("head.duration.fc2.weight", v.dur_w2) &&
                fetch("head.duration.fc2.bias",   v.dur_b2) &&
                fetch("head.note_query.fc1.weight", v.nq_w1) &&
                fetch("head.note_query.fc1.bias",   v.nq_b1) &&
                fetch("head.note_query.fc2.weight", v.nq_w2) &&
                fetch("head.note_query.fc2.bias",   v.nq_b2) &&
                fetch("head.local_proj.weight",  v.local_w) &&
                fetch("head.local_proj.bias",    v.local_b) &&
                fetch("head.local_attn.fc1.weight", v.att_w1) &&
                fetch("head.local_attn.fc1.bias",   v.att_b1) &&
                fetch("head.local_attn.fc2.weight", v.att_w2) &&
                fetch("head.local_attn.fc2.bias",   v.att_b2) &&
                fetch("head.note_fusion.norm.weight", v.fuse_norm_w) &&
                fetch("head.note_fusion.norm.bias",   v.fuse_norm_b) &&
                fetch("head.note_fusion.fc.weight",   v.fuse_w) &&
                fetch("head.note_fusion.fc.bias",     v.fuse_b) &&
                fetch("head.velocity.weight", v.vel_w) &&
                fetch("head.velocity.bias",   v.vel_b);
            if (!vel_ok) {
                break;
            }
        }

        if (hp.use_interval_boundary_head && !is_velocity) {
            if (!fetch("head.boundary.fc1.weight", model.boundary_w1) ||
                !fetch("head.boundary.fc1.bias",   model.boundary_b1) ||
                !fetch("head.boundary.fc2.weight", model.boundary_w2) ||
                !fetch("head.boundary.fc2.bias",   model.boundary_b2)) {
                break;
            }
        }

        bool cqt_ok = true;
        for (int i = 0; i < n_stages && cqt_ok; ++i) {
            cqt_ok &= fetch(fmt("cqt.kernel.%d.real", i), model.cqt_kernel_re[i]);
            cqt_ok &= fetch(fmt("cqt.kernel.%d.imag", i), model.cqt_kernel_im[i]);
            cqt_ok &= fetch(fmt("cqt.window.%d", i),      model.cqt_window[i]);
            if (i + 1 < n_stages) {
                cqt_ok &= fetch(fmt("cqt.resampler.%d.weight", i), model.cqt_resampler[i]);
            }
        }
        if (!cqt_ok) {
            break;
        }

        ok = true;
    } while (false);

    if (ctx_meta) {
        ggml_free(ctx_meta);
    }
    gguf_free(gguf);
    return ok;
}
