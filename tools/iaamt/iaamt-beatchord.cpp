// Beat / chord / key model: MIDI roll in, frame-level beat, chord and key out.
//
// Ports beat_chord/models/{stem,backbone,network}.py.  The backbone reuses the
// same dual-axis Transformer as the audio models but is fed a MIDI roll through
// a depthwise conv stem, carries four global tokens alongside the pitch tokens,
// and re-injects intermediate beat/chord predictions as extra tokens after the
// layers listed in `iaamt.inter_refine_layers`.

#include "iaamt.h"
#include "iaamt-graph.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>

namespace {

// Conv1d(C, C, kernel_size=k, stride=k) used to fold high-res probabilities
// back down to the token frame rate.
ggml_tensor * down_conv(ggml_context * ctx, ggml_tensor * w, ggml_tensor * b,
                        ggml_tensor * x, int stride) {
    // x is [C, T]; ggml_conv_1d wants [T, C], and it always builds an f16
    // im2col, so the kernel has to be f16 as well.
    x = ggml_cont(ctx, ggml_transpose(ctx, x));
    x = ggml_conv_1d(ctx, ggml_cast(ctx, w, GGML_TYPE_F16), x, stride, 0, 1);
    x = ggml_cont(ctx, ggml_transpose(ctx, x));         // [C_out, T_out]
    return ggml_add(ctx, x, ggml_reshape_2d(ctx, b, b->ne[0], 1));
}

// ConvTranspose1d(D, D, k=s=up) over G tokens, expressed as a matmul because
// the kernel and stride match, exactly as the audio up_conv does.
ggml_tensor * up_conv_tokens(ggml_context * ctx, ggml_tensor * w, ggml_tensor * b,
                             ggml_tensor * x, int64_t D, int64_t G, int64_t T) {
    const int64_t up = w->ne[0];
    ggml_tensor * k = ggml_reshape_2d(ctx, w, up * D, D);
    k = ggml_cont(ctx, ggml_transpose(ctx, k));                 // [D(in), up*D(out)]

    x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));       // [D, T, G]
    x = ggml_mul_mat(ctx, k, x);                                // [up*D, T, G]
    x = ggml_reshape_4d(ctx, x, up, D, T, G);
    x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));       // [D, up, T, G]
    x = ggml_reshape_3d(ctx, x, D, up * T, G);
    return ggml_add(ctx, x, b);
}

// _match_lowres_time_length: the strided down-conv can land a frame short of the
// token axis (ceil vs floor), so zero-pad or trim to exactly `target`.
ggml_tensor * match_time(ggml_context * ctx, ggml_tensor * x, int64_t target) {
    if (x->ne[1] == target) {
        return x;
    }
    if (x->ne[1] > target) {
        return ggml_cont(ctx, ggml_view_2d(ctx, x, x->ne[0], target, x->nb[1], 0));
    }
    return ggml_pad(ctx, x, 0, (int) (target - x->ne[1]), 0, 0);
}

// Merges G global tokens per frame into one: "b t g d -> b t (g d)" then Linear.
ggml_tensor * merge_globals(ggml_context * ctx, ggml_tensor * w, ggml_tensor * b,
                            ggml_tensor * x, int64_t D, int64_t G, int64_t T) {
    // x is [D, T, G]; the flatten needs d fastest inside g
    x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));       // [D, G, T]
    x = ggml_reshape_2d(ctx, x, D * G, T);
    return iaamt_linear(ctx, w, b, x);
}

// LayerNorm + Linear + GELU, then the head's output projections.
ggml_tensor * bc_head_shared(ggml_context * ctx, const iaamt_bc_head & h, ggml_tensor * x) {
    x = iaamt_layer_norm(ctx, x, h.norm_w, h.norm_b);
    x = iaamt_linear(ctx, h.fc_w, h.fc_b, x);
    return ggml_gelu_erf(ctx, x);
}

} // namespace

void iaamt_bc_build_roll(const iaamt_model & model,
                         const std::vector<iaamt_note> & notes,
                         int64_t window_start_sample,
                         int n_frames,
                         std::vector<float> & roll) {
    const iaamt_hparams & hp = model.hparams;
    const int P  = hp.pitch_query_count;
    const int C  = hp.num_input_channels;
    const int CC = C / 2;                     // instrument classes
    const int hop = hp.hop_length;

    roll.assign((size_t) C * n_frames * P, 0.0f);

    for (const iaamt_note & n : notes) {
        const int pitch_index = n.pitch - hp.pitch_min;
        if (pitch_index < 0 || pitch_index >= P) {
            continue;
        }
        // Without an instrument classifier the transcription output is a single
        // track, so notes land in the class their MIDI program maps to.
        const int class_id = n.is_drum ? 0 : std::min(n.program / 8, CC - 1);

        const double t0 = (double) (n.start_sample - window_start_sample) / hop;
        const double t1 = (double) (n.end_sample   - window_start_sample) / hop;
        const int f0 = std::max(0, (int) std::floor(t0));
        const int f1 = std::min(n_frames, (int) std::ceil(t1));
        if (f1 <= f0) {
            continue;
        }
        for (int f = f0; f < f1; ++f) {
            roll[((size_t) class_id * n_frames + f) * P + pitch_index] = 1.0f;
        }
        const int onset = (int) std::lround(t0);
        if (onset >= 0 && onset < n_frames) {
            roll[((size_t) (class_id + CC) * n_frames + onset) * P + pitch_index] = 1.0f;
        }
    }
}

ggml_cgraph * iaamt_bc_build_graph(iaamt_context * ctx0, int n_frames) {
    const iaamt_model   & model = *ctx0->model;
    const iaamt_hparams & hp    = model.hparams;

    const int64_t D  = hp.hidden_size;
    const int64_t P  = hp.pitch_query_count;
    const int64_t G  = hp.num_global_tokens;
    const int64_t C  = hp.num_input_channels;
    const int     nh = hp.n_head;
    const int     hd = hp.head_dim();
    const int     up = hp.time_downsample;

    ggml_init_params params = {
        ctx0->buf_compute_meta.size(), ctx0->buf_compute_meta.data(), true,
    };
    ggml_context * ctx = ggml_init(params);
    ggml_cgraph  * gf  = ggml_new_graph_custom(ctx, 16384, false);

    ctx0->inp_feats = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, P, n_frames, C, 1);
    ggml_set_name(ctx0->inp_feats, "roll");
    ggml_set_input(ctx0->inp_feats);

    // ---- MidiFrameStem: depthwise 7x7, depthwise 5x5, 1x1 mix ----
    // ggml_conv_2d_dw routes an f16 im2col into mul_mat's src1, which no backend
    // accepts.  The dedicated op works, and takes f32 kernels: CUDA asserts on
    // f32 and the CPU path reads the kernel as float without checking, so an
    // f16 kernel would be silently misread there.
    ggml_tensor * cur = ggml_conv_2d_dw_direct(ctx, model.bc_conv1_w, ctx0->inp_feats,
                                               1, 1, 3, 3, 1, 1);
    cur = iaamt_add_bias_chan(ctx, cur, model.bc_conv1_b);
    cur = ggml_conv_2d_dw_direct(ctx, model.bc_conv2_w, cur, 1, 1, 2, 2, 1, 1);
    cur = iaamt_add_bias_chan(ctx, cur, model.bc_conv2_b);
    cur = ggml_conv_2d(ctx, model.bc_chan_w, cur, 1, 1, 0, 0, 1, 1);
    cur = iaamt_add_bias_chan(ctx, cur, model.bc_chan_b);
    cur = ggml_add(ctx, cur, model.bc_pitch_embd);            // [P, 1, base_ch, 1]

    // time is halved three times; pitch is preserved throughout
    cur = iaamt_conv_block(ctx, model.stem_blocks[0], cur, 1, 2, true);
    cur = iaamt_conv_block(ctx, model.stem_blocks[1], cur, 1, 2, true);
    cur = iaamt_conv_block(ctx, model.stem_blocks[2], cur, 1, 2, true);
    cur = iaamt_conv_block(ctx, model.stem_blocks[3], cur, 1, 1, false);

    const int64_t T2 = cur->ne[1];

    // "b d t p -> b t p d"
    cur = ggml_cont(ctx, ggml_permute(ctx, cur, 1, 2, 0, 3));  // [stem_ch, P, T2]
    cur = iaamt_linear(ctx, model.bc_input_proj_w, model.bc_input_proj_b, cur);
    cur = ggml_add(ctx, cur, model.bc_pitch_pos_embd);        // [D, P, 1]
    cur = ggml_add(ctx, cur, model.pitch_type_embd);

    ggml_tensor * globals = ggml_add(ctx, model.bc_global_tokens, model.bc_global_type);
    globals = ggml_repeat_4d(ctx, ggml_reshape_3d(ctx, globals, D, G, 1), D, G, T2, 1);
    cur = ggml_concat(ctx, cur, globals, 1);                  // [D, P + G, T2]

    const int64_t n_pure = P + G;
    const int64_t n_max  = n_pure + 2;   // refinement adds a beat and a chord token

    ctx0->inp_pos_band = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_max);
    ggml_set_name(ctx0->inp_pos_band, "pos_token");
    ggml_set_input(ctx0->inp_pos_band);

    ctx0->inp_pos_time = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T2);
    ggml_set_name(ctx0->inp_pos_time, "pos_time");
    ggml_set_input(ctx0->inp_pos_time);

    for (int il = 0; il < hp.n_layer; ++il) {
        ggml_tensor * pos_tok = ggml_view_1d(ctx, ctx0->inp_pos_band, cur->ne[1], 0);

        cur = iaamt_transformer_block(ctx, model.layers[il].band, cur, pos_tok, nh, hd);
        cur = ggml_cont(ctx, ggml_permute(ctx, cur, 0, 2, 1, 3));
        cur = iaamt_transformer_block(ctx, model.layers[il].time, cur, ctx0->inp_pos_time, nh, hd);
        cur = ggml_cont(ctx, ggml_permute(ctx, cur, 0, 2, 1, 3));

        // ---- intermediate refinement ----
        const iaamt_bc_refine * ref = nullptr;
        for (const auto & r : model.bc_refine) {
            if (r.layer == il) {
                ref = &r;
            }
        }
        if (!ref) {
            continue;
        }

        // drop tokens a previous stage appended, so they never compound
        ggml_tensor * pure = ggml_cont(ctx,
            ggml_view_3d(ctx, cur, D, n_pure, T2, cur->nb[1], cur->nb[2], 0));
        ggml_tensor * gpart = ggml_cont(ctx,
            ggml_view_3d(ctx, pure, D, G, T2, pure->nb[1], pure->nb[2], P * pure->nb[1]));

        ggml_tensor * hi = up_conv_tokens(ctx, ref->up_conv_w, ref->up_conv_b, gpart, D, G, T2);
        const int64_t T_hi = std::min<int64_t>(n_frames, hi->ne[1]);
        hi = ggml_cont(ctx, ggml_view_3d(ctx, hi, D, T_hi, G, hi->nb[1], hi->nb[2], 0));
        ggml_tensor * gfeat = merge_globals(ctx, ref->merge_w, ref->merge_b, hi, D, G, T_hi);

        // beat branch: probabilities are concatenated in head order
        ggml_tensor * bh = iaamt_adapter_apply(ctx, ref->beat_adapter, gfeat);
        bh = bc_head_shared(ctx, ref->beat_head, bh);
        ggml_tensor * frame = iaamt_linear(ctx, ref->beat_head.proj_w[0],
                                           ref->beat_head.proj_b[0], bh);
        // frame is [2 + n_meter, T]: beat, downbeat, meter...
        ggml_tensor * beat_l = ggml_cont(ctx,
            ggml_view_2d(ctx, frame, 1, T_hi, frame->nb[1], 0));
        ggml_tensor * down_l = ggml_cont(ctx,
            ggml_view_2d(ctx, frame, 1, T_hi, frame->nb[1], frame->nb[0]));
        ggml_tensor * meter_l = ggml_cont(ctx,
            ggml_view_2d(ctx, frame, frame->ne[0] - 2, T_hi, frame->nb[1], 2 * frame->nb[0]));
        // SumHead: a downbeat is always a beat
        beat_l = ggml_add(ctx, beat_l, down_l);
        ggml_tensor * beat_p = ggml_concat(ctx,
            ggml_concat(ctx, ggml_sigmoid(ctx, beat_l), ggml_sigmoid(ctx, down_l), 0),
            ggml_soft_max(ctx, meter_l), 0);

        ggml_tensor * beat_low = down_conv(ctx, ref->beat_down_w, ref->beat_down_b, beat_p, up);
        beat_low = match_time(ctx, beat_low, T2);
        ggml_tensor * beat_tok = iaamt_linear(ctx, ref->beat_fb_w, ref->beat_fb_b, beat_low);
        beat_tok = ggml_mul(ctx, beat_tok, ggml_sigmoid(ctx, ref->beat_gate));

        // chord branch
        ggml_tensor * ch = iaamt_adapter_apply(ctx, ref->chord_adapter, gfeat);
        ch = bc_head_shared(ctx, ref->chord_head, ch);
        ggml_tensor * parts[6];
        for (int i = 0; i < 6; ++i) {
            parts[i] = iaamt_linear(ctx, ref->chord_head.proj_w[i],
                                    ref->chord_head.proj_b[i], ch);
        }
        // boundary, root_chord, bass, key_boundary, key, pitch
        ggml_tensor * chord_p = ggml_sigmoid(ctx, parts[0]);
        chord_p = ggml_concat(ctx, chord_p, ggml_soft_max(ctx, parts[1]), 0);
        chord_p = ggml_concat(ctx, chord_p, ggml_soft_max(ctx, parts[2]), 0);
        chord_p = ggml_concat(ctx, chord_p, ggml_sigmoid(ctx, parts[3]), 0);
        chord_p = ggml_concat(ctx, chord_p, ggml_soft_max(ctx, parts[4]), 0);
        chord_p = ggml_concat(ctx, chord_p, ggml_sigmoid(ctx, parts[5]), 0);

        ggml_tensor * chord_low = down_conv(ctx, ref->chord_down_w, ref->chord_down_b, chord_p, up);
        chord_low = match_time(ctx, chord_low, T2);
        ggml_tensor * chord_tok = iaamt_linear(ctx, ref->chord_fb_w, ref->chord_fb_b, chord_low);
        chord_tok = ggml_mul(ctx, chord_tok, ggml_sigmoid(ctx, ref->chord_gate));

        // append both as extra tokens: [D, T2] -> [D, 1, T2]
        beat_tok  = ggml_reshape_3d(ctx, beat_tok,  D, 1, T2);
        chord_tok = ggml_reshape_3d(ctx, chord_tok, D, 1, T2);
        cur = ggml_concat(ctx, ggml_concat(ctx, pure, beat_tok, 1), chord_tok, 1);
    }

    cur = iaamt_rms_norm_mul(ctx, cur, model.output_norm);

    ggml_tensor * gpart = ggml_cont(ctx,
        ggml_view_3d(ctx, cur, D, G, T2, cur->nb[1], cur->nb[2], P * cur->nb[1]));
    ggml_tensor * hi = up_conv_tokens(ctx, model.bc_global_up_w, model.bc_global_up_b,
                                      gpart, D, G, T2);
    const int64_t T_hi = std::min<int64_t>(n_frames, hi->ne[1]);
    hi = ggml_cont(ctx, ggml_view_3d(ctx, hi, D, T_hi, G, hi->nb[1], hi->nb[2], 0));
    ggml_tensor * gfeat = merge_globals(ctx, model.bc_merge_w, model.bc_merge_b, hi, D, G, T_hi);

    ggml_tensor * bh = iaamt_adapter_apply(ctx, model.bc_beat_adapter, gfeat);
    bh = bc_head_shared(ctx, model.bc_beat_head, bh);
    ctx0->out_proj = iaamt_linear(ctx, model.bc_beat_head.proj_w[0],
                                  model.bc_beat_head.proj_b[0], bh);
    ggml_set_name(ctx0->out_proj, "beat_frame");
    ggml_set_output(ctx0->out_proj);
    ggml_build_forward_expand(gf, ctx0->out_proj);

    ggml_tensor * ch = iaamt_adapter_apply(ctx, model.bc_chord_adapter, gfeat);
    ch = bc_head_shared(ctx, model.bc_chord_head, ch);
    ctx0->bc_out.clear();
    for (int i = 0; i < 6; ++i) {
        ggml_tensor * t = iaamt_linear(ctx, model.bc_chord_head.proj_w[i],
                                       model.bc_chord_head.proj_b[i], ch);
        static const char * names[6] = {
            "chord_boundary", "root_chord", "bass", "key_boundary", "key", "chord_pitch" };
        ggml_set_name(t, names[i]);
        ggml_set_output(t);
        ggml_build_forward_expand(gf, t);
        ctx0->bc_out.push_back(t);
    }

    ggml_free(ctx);
    return gf;
}

namespace {

int argmax_col(const std::vector<float> & v, int n_rows, int col) {
    int best = 0;
    for (int i = 1; i < n_rows; ++i) {
        if (v[(size_t) col * n_rows + i] > v[(size_t) col * n_rows + best]) {
            best = i;
        }
    }
    return best;
}

float sigmoidf(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

} // namespace

bool iaamt_bc_forward(iaamt_context * ctx,
                      const std::vector<float> & roll,
                      int n_frames,
                      iaamt_bc_out & out,
                      std::string & err) {
    const iaamt_model   & model = *ctx->model;
    const iaamt_hparams & hp    = model.hparams;

    ggml_cgraph * gf = iaamt_bc_build_graph(ctx, n_frames);
    if (!gf) {
        err = "failed to build the beat/chord graph";
        return false;
    }
    if (getenv("IAAMT_DEBUG_GRAPH")) {
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            ggml_tensor * n = ggml_graph_node(gf, i);
            fprintf(stderr, "%4d %-18s %-6s", i, ggml_op_name(n->op), ggml_type_name(n->type));
            for (int s = 0; s < GGML_MAX_SRC && n->src[s]; ++s) {
                fprintf(stderr, "  src%d=%s", s, ggml_type_name(n->src[s]->type));
            }
            fprintf(stderr, "\n");
        }
    }

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        err = "failed to allocate the beat/chord graph";
        return false;
    }

    ggml_backend_tensor_set(ctx->inp_feats, roll.data(), 0, roll.size() * sizeof(float));
    for (ggml_tensor * pos : { ctx->inp_pos_band, ctx->inp_pos_time }) {
        std::vector<int32_t> ids(pos->ne[0]);
        for (int64_t i = 0; i < pos->ne[0]; ++i) {
            ids[i] = (int32_t) i;
        }
        ggml_backend_tensor_set(pos, ids.data(), 0, ids.size() * sizeof(int32_t));
    }

    ggml_backend_cpu_set_n_threads(model.backend_cpu, ctx->n_threads);
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        err = "beat/chord graph evaluation failed";
        return false;
    }

    ggml_tensor * frame = ctx->out_proj;             // [2 + n_meter, T]
    const int n_row = (int) frame->ne[0];
    const int T     = (int) frame->ne[1];
    std::vector<float> fbuf((size_t) ggml_nelements(frame));
    ggml_backend_tensor_get(frame, fbuf.data(), 0, ggml_nbytes(frame));

    out.n_frames = T;
    out.beat.resize(T);
    out.downbeat.resize(T);
    out.meter.resize(T);
    for (int t = 0; t < T; ++t) {
        const float b = fbuf[(size_t) t * n_row + 0];
        const float d = fbuf[(size_t) t * n_row + 1];
        out.beat[t]     = sigmoidf(b + d);           // SumHead
        out.downbeat[t] = sigmoidf(d);
        int best = 0;
        for (int i = 1; i < n_row - 2; ++i) {
            if (fbuf[(size_t) t * n_row + 2 + i] > fbuf[(size_t) t * n_row + 2 + best]) {
                best = i;
            }
        }
        out.meter[t] = best;
    }

    auto fetch = [&](int idx, std::vector<float> & dst, int & rows) {
        ggml_tensor * t = ctx->bc_out[idx];
        rows = (int) t->ne[0];
        dst.resize((size_t) ggml_nelements(t));
        ggml_backend_tensor_get(t, dst.data(), 0, ggml_nbytes(t));
    };

    std::vector<float> buf;
    int rows = 0;

    fetch(0, buf, rows);
    out.chord_boundary.resize(T);
    for (int t = 0; t < T; ++t) {
        out.chord_boundary[t] = sigmoidf(buf[(size_t) t * rows]);
    }

    fetch(1, buf, rows);
    out.root_chord.resize(T);
    for (int t = 0; t < T; ++t) {
        out.root_chord[t] = argmax_col(buf, rows, t);
    }

    fetch(2, buf, rows);
    out.bass.resize(T);
    for (int t = 0; t < T; ++t) {
        out.bass[t] = argmax_col(buf, rows, t);
    }

    fetch(4, buf, rows);
    out.key.resize(T);
    for (int t = 0; t < T; ++t) {
        out.key[t] = argmax_col(buf, rows, t);
    }

    GGML_UNUSED(hp);
    return true;
}

bool iaamt_bc_write(const std::string & fname,
                    const iaamt_model & model,
                    const iaamt_bc_out & out,
                    int64_t window_start_sample,
                    std::string & err) {
    const iaamt_hparams & hp = model.hparams;
    std::ofstream f(fname);
    if (!f) {
        err = "failed to open " + fname + " for writing";
        return false;
    }

    static const char * PC[13] = { "C", "C#", "D", "D#", "E", "F", "F#", "G",
                                   "G#", "A", "A#", "B", "N" };

    const double sec_per_frame = (double) hp.hop_length / hp.sample_rate;
    const double base = (double) window_start_sample / hp.sample_rate;

    f << "# time_s\tbeat\tdownbeat\troot_chord\tbass\tkey\tmeter\n";
    for (int t = 0; t < out.n_frames; ++t) {
        // peak-pick: a local maximum above 0.5 marks a beat
        const bool is_peak =
            out.beat[t] > 0.5f &&
            (t == 0 || out.beat[t] >= out.beat[t - 1]) &&
            (t + 1 == out.n_frames || out.beat[t] > out.beat[t + 1]);
        if (!is_peak && out.chord_boundary[t] <= 0.5f) {
            continue;
        }
        const int bass = out.bass[t];
        const int key  = out.key[t];
        f << (base + t * sec_per_frame) << "\t"
          << (is_peak ? 1 : 0) << "\t"
          << (out.downbeat[t] > 0.5f ? 1 : 0) << "\t"
          << out.root_chord[t] << "\t"
          << PC[std::min(std::max(bass, 0), 12)] << "\t"
          << PC[std::min(std::max(key, 0), 12)] << "\t"
          << out.meter[t] << "\n";
    }
    return true;
}
