// llama-iaamt-cli: transcribe audio to MIDI with an instrument-agnostic-amt GGUF.

#if defined(_WIN32)
// miniaudio pulls in windows.h, whose min/max macros break std::min/std::max
#   define NOMINMAX
#   define WIN32_LEAN_AND_MEAN
#endif

#include "iaamt.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#define MINIAUDIO_IMPLEMENTATION
#define MA_NO_ENCODING
#define MA_NO_DEVICE_IO
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MA_NO_ENGINE
#define MA_NO_GENERATION
#define MA_API static
#include "miniaudio/miniaudio.h"

struct cli_params {
    std::string model;
    std::string audio;
    std::string out_midi;
    int   n_threads   = 4;
    bool  use_gpu     = true;
    int   window_ms   = 8000;
    int   stride_ms   = -1;     // defaults to window_ms / 2
    float note_bias   = 0.0f;
    float merge_onset_ms = 50.0f;
    float merge_gap_ms   = -1.0f;
    float min_note_ms    = 5.0f;
    int   velocity       = 100;
    float silence_dbfs   = -72.0f;
    bool  no_boundary    = false;
    bool  verbose        = false;
    std::string dump_prefix;    // writes window-0 intermediates as .npy
    std::string in_midi;        // velocity models: the notes to score
};

static void print_usage(const char * argv0) {
    printf("usage: %s -m MODEL.gguf -a AUDIO -o OUT.mid [options]\n\n", argv0);
    printf("  -m, --model PATH       iaamt GGUF (from convert_iaamt_to_gguf.py)\n");
    printf("  -a, --audio PATH       input audio (wav/mp3/flac)\n");
    printf("  -o, --output PATH      output MIDI file\n");
    printf("      --midi PATH        input MIDI (required by velocity models)\n");
    printf("  -t, --threads N        CPU threads (default 4)\n");
    printf("      --cpu              run the graph on the CPU\n");
    printf("      --window-ms N      analysis window (default 8000)\n");
    printf("      --stride-ms N      window hop (default window/2)\n");
    printf("      --note-bias F      added to every interval score (default 0)\n");
    printf("      --merge-onset-ms F merge notes starting within F ms (default 50)\n");
    printf("      --merge-gap-ms F   merge continuations across F ms (default: one hop)\n");
    printf("      --min-note-ms F    drop notes shorter than F ms (default 5)\n");
    printf("      --velocity N       MIDI velocity for every note (default 100)\n");
    printf("      --silence-dbfs F   skip windows quieter than F dBFS (default -72)\n");
    printf("      --no-boundary      skip the sub-frame boundary head\n");
    printf("  -v, --verbose          per-window progress\n");
}

static bool parse_args(int argc, char ** argv, cli_params & p) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char * what) -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: %s requires a value\n", what);
                return nullptr;
            }
            return argv[++i];
        };
        if (a == "-m" || a == "--model") {
            const char * v = next("--model");   if (!v) return false; p.model = v;
        } else if (a == "-a" || a == "--audio") {
            const char * v = next("--audio");   if (!v) return false; p.audio = v;
        } else if (a == "-o" || a == "--output") {
            const char * v = next("--output");  if (!v) return false; p.out_midi = v;
        } else if (a == "-t" || a == "--threads") {
            const char * v = next("--threads"); if (!v) return false; p.n_threads = atoi(v);
        } else if (a == "--cpu") {
            p.use_gpu = false;
        } else if (a == "--window-ms") {
            const char * v = next("--window-ms"); if (!v) return false; p.window_ms = atoi(v);
        } else if (a == "--stride-ms") {
            const char * v = next("--stride-ms"); if (!v) return false; p.stride_ms = atoi(v);
        } else if (a == "--note-bias") {
            const char * v = next("--note-bias"); if (!v) return false; p.note_bias = (float) atof(v);
        } else if (a == "--merge-onset-ms") {
            const char * v = next("--merge-onset-ms"); if (!v) return false; p.merge_onset_ms = (float) atof(v);
        } else if (a == "--merge-gap-ms") {
            const char * v = next("--merge-gap-ms"); if (!v) return false; p.merge_gap_ms = (float) atof(v);
        } else if (a == "--min-note-ms") {
            const char * v = next("--min-note-ms"); if (!v) return false; p.min_note_ms = (float) atof(v);
        } else if (a == "--velocity") {
            const char * v = next("--velocity"); if (!v) return false; p.velocity = atoi(v);
        } else if (a == "--silence-dbfs") {
            const char * v = next("--silence-dbfs"); if (!v) return false; p.silence_dbfs = (float) atof(v);
        } else if (a == "--midi") {
            const char * v = next("--midi"); if (!v) return false; p.in_midi = v;
        } else if (a == "--dump-prefix") {
            const char * v = next("--dump-prefix"); if (!v) return false; p.dump_prefix = v;
        } else if (a == "--no-boundary") {
            p.no_boundary = true;
        } else if (a == "-v" || a == "--verbose") {
            p.verbose = true;
        } else if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            exit(0);
        } else {
            fprintf(stderr, "error: unknown argument '%s'\n", a.c_str());
            return false;
        }
    }
    if (p.model.empty() || p.audio.empty() || p.out_midi.empty()) {
        print_usage(argv[0]);
        return false;
    }
    return true;
}

// Decodes to planar stereo at the model's sample rate.  Mono sources are
// duplicated into both channels, matching how the reference loads audio.
static bool decode_audio(const std::string & path,
                         int sample_rate,
                         std::vector<std::vector<float>> & pcm,
                         int64_t & n_samples) {
    ma_decoder_config cfg = ma_decoder_config_init(ma_format_f32, 2, sample_rate);
    ma_decoder decoder;
    if (ma_decoder_init_file(path.c_str(), &cfg, &decoder) != MA_SUCCESS) {
        return false;
    }
    ma_uint64 frame_count = 0;
    if (ma_decoder_get_length_in_pcm_frames(&decoder, &frame_count) != MA_SUCCESS ||
        frame_count == 0) {
        ma_decoder_uninit(&decoder);
        return false;
    }
    std::vector<float> interleaved((size_t) frame_count * 2);
    ma_uint64 read = 0;
    if (ma_decoder_read_pcm_frames(&decoder, interleaved.data(), frame_count, &read) != MA_SUCCESS) {
        ma_decoder_uninit(&decoder);
        return false;
    }
    ma_decoder_uninit(&decoder);

    n_samples = (int64_t) read;
    pcm.assign(2, std::vector<float>((size_t) n_samples));
    for (int64_t i = 0; i < n_samples; ++i) {
        pcm[0][i] = interleaved[(size_t) i * 2 + 0];
        pcm[1][i] = interleaved[(size_t) i * 2 + 1];
    }
    return true;
}

// Writes a .npy so intermediates can be diffed against the reference model.
static bool write_npy(const std::string & path,
                      const std::vector<float> & data,
                      const std::vector<int64_t> & shape) {
    std::string dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (";
    for (size_t i = 0; i < shape.size(); ++i) {
        dict += std::to_string(shape[i]);
        dict += ",";
    }
    dict += "), }";
    size_t hdr = 10 + dict.size() + 1;
    const size_t pad = (64 - (hdr % 64)) % 64;
    dict.append(pad, ' ');
    dict += "\n";

    std::vector<uint8_t> out;
    const uint8_t magic[] = { 0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0 };
    out.insert(out.end(), magic, magic + 8);
    out.push_back((uint8_t) (dict.size() & 0xff));
    out.push_back((uint8_t) (dict.size() >> 8));
    out.insert(out.end(), dict.begin(), dict.end());

    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        return false;
    }
    fwrite(out.data(), 1, out.size(), f);
    fwrite(data.data(), sizeof(float), data.size(), f);
    fclose(f);
    return true;
}

// _build_window_starts
static std::vector<int64_t> build_window_starts(int64_t total, int64_t window, int64_t stride) {
    std::vector<int64_t> starts;
    if (total <= window) {
        starts.push_back(0);
        return starts;
    }
    for (int64_t s = 0; s < total - window + 1; s += stride) {
        starts.push_back(s);
    }
    const int64_t last = total - window;
    if (starts.back() != last) {
        starts.push_back(last);
    }
    return starts;
}

int main(int argc, char ** argv) {
    cli_params params;
    if (!parse_args(argc, argv, params)) {
        return 1;
    }

    // progress is written with \r, so keep it unbuffered even through a pipe
    setvbuf(stdout, nullptr, _IONBF, 0);

    ggml_backend_load_all();

    iaamt_model model;
    std::string err;
    if (!iaamt_model_load(model, params.model, params.use_gpu, err)) {
        fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }
    const iaamt_hparams & hp = model.hparams;

    printf("model : %s\n", params.model.c_str());
    printf("        %d layers, %d heads, token dim %d, %d pitches x %d slot(s)\n",
           hp.n_layer, hp.n_head, hp.token_dim(), hp.pitch_query_count, hp.num_pitch_slots);
    printf("        HCQT %d bins over %d stages, %d Hz, hop %d\n",
           hp.cqt_n_bins, (int) hp.cqt_stages.size(), hp.sample_rate, hp.hop_length);
    printf("backend: %s\n", ggml_backend_name(model.backend));

    // ---- beat / chord / key: MIDI roll in, analysis out ----
    if (hp.model_type == IAAMT_MODEL_TYPE_BEAT_CHORD) {
        if (params.in_midi.empty()) {
            fprintf(stderr, "error: beat_chord models need --midi with the input notes\n");
            return 1;
        }
        std::vector<iaamt_note> notes;
        if (!iaamt_read_midi(params.in_midi, hp.sample_rate, notes, err)) {
            fprintf(stderr, "error: %s\n", err.c_str());
            return 1;
        }
        if (notes.empty()) {
            fprintf(stderr, "error: %s contains no notes\n", params.in_midi.c_str());
            return 1;
        }
        int64_t last = 0;
        for (const iaamt_note & n : notes) {
            last = std::max(last, n.end_sample);
        }
        const int n_frames = (int) ((last + hp.hop_length - 1) / hp.hop_length) + 1;
        printf("notes  : %zu read from %s (%.2f s)\n", notes.size(),
               params.in_midi.c_str(), (double) last / hp.sample_rate);

        iaamt_context * bctx = iaamt_ctx_init(model, params.n_threads);
        std::vector<float> roll;
        iaamt_bc_build_roll(model, notes, 0, n_frames, roll);

        iaamt_bc_out bc;
        if (!iaamt_bc_forward(bctx, roll, n_frames, bc, err)) {
            fprintf(stderr, "error: %s\n", err.c_str());
            iaamt_ctx_free(bctx);
            return 1;
        }
        iaamt_ctx_free(bctx);

        if (!iaamt_bc_write(params.out_midi, model, bc, 0, err)) {
            fprintf(stderr, "error: %s\n", err.c_str());
            return 1;
        }
        int n_beats = 0;
        for (int t = 0; t < bc.n_frames; ++t) {
            if (bc.beat[t] > 0.5f) {
                n_beats++;
            }
        }
        printf("frames : %d, %d above the beat threshold\n", bc.n_frames, n_beats);
        printf("wrote  : %s\n", params.out_midi.c_str());
        return 0;
    }

    std::vector<std::vector<float>> pcm;
    int64_t total_samples = 0;
    if (!decode_audio(params.audio, hp.sample_rate, pcm, total_samples)) {
        fprintf(stderr, "error: failed to decode %s\n", params.audio.c_str());
        return 1;
    }
    printf("audio  : %s (%.2f s)\n", params.audio.c_str(),
           (double) total_samples / hp.sample_rate);

    const int64_t window_samples =
        (int64_t) llround(params.window_ms * hp.sample_rate / 1000.0);
    const int64_t stride_samples =
        (int64_t) llround((params.stride_ms > 0 ? params.stride_ms : params.window_ms / 2)
                          * hp.sample_rate / 1000.0);
    if (window_samples <= 0 || stride_samples <= 0) {
        fprintf(stderr, "error: window and stride must be positive\n");
        return 1;
    }

    const std::vector<int64_t> starts =
        build_window_starts(total_samples, window_samples, stride_samples);

    const bool is_velocity   = (hp.model_type == IAAMT_MODEL_TYPE_VELOCITY);
    const bool is_beat_chord = (hp.model_type == IAAMT_MODEL_TYPE_BEAT_CHORD);
    std::vector<iaamt_note> in_notes;
    if (is_velocity || is_beat_chord) {
        if (params.in_midi.empty()) {
            fprintf(stderr, "error: %s models need --midi with the input notes\n",
                    hp.model_type.c_str());
            return 1;
        }
        if (!iaamt_read_midi(params.in_midi, hp.sample_rate, in_notes, err)) {
            fprintf(stderr, "error: %s\n", err.c_str());
            return 1;
        }
        printf("notes  : %zu read from %s\n", in_notes.size(), params.in_midi.c_str());
    }

    iaamt_decode_params dp;
    dp.note_bias         = params.note_bias;
    dp.merge_onset_ms    = params.merge_onset_ms;
    dp.merge_gap_ms      = params.merge_gap_ms;
    dp.velocity          = params.velocity;
    dp.min_note_ms       = params.min_note_ms;
    dp.use_boundary_head = !params.no_boundary;

    iaamt_stitcher st;
    iaamt_stitcher_init(st, model, dp, total_samples);

    iaamt_context * ctx = iaamt_ctx_init(model, params.n_threads);
    iaamt_ctx_set_debug(ctx, !params.dump_prefix.empty());

    const float silence_gate = params.silence_dbfs <= 0.0f
        ? powf(10.0f, params.silence_dbfs / 20.0f) : 0.0f;

    std::vector<std::vector<float>> window(2, std::vector<float>((size_t) window_samples));
    std::vector<float> feats;

    int n_intervals = 0;
    int n_skipped   = 0;
    const auto t0 = std::chrono::steady_clock::now();

    for (size_t wi = 0; wi < starts.size(); ++wi) {
        const int64_t start = starts[wi];
        const int64_t valid = std::min<int64_t>(window_samples, total_samples - start);

        for (int c = 0; c < 2; ++c) {
            std::fill(window[c].begin(), window[c].end(), 0.0f);
            std::copy(pcm[c].begin() + start, pcm[c].begin() + start + valid,
                      window[c].begin());
        }

        // silence gate: mean over channels, then RMS.  Velocity scoring never
        // skips -- a skipped window would silently leave its notes unscored.
        if (silence_gate > 0.0f && !is_velocity) {
            double acc = 0.0;
            for (int64_t i = 0; i < window_samples; ++i) {
                const double m = 0.5 * (window[0][i] + window[1][i]);
                acc += m * m;
            }
            if (std::sqrt(acc / (double) window_samples) < silence_gate) {
                n_skipped++;
                continue;
            }
        }

        const int n_frames = iaamt_cqt_apply(model, window, (int) window_samples, feats);
        const int crop     = n_frames;

        iaamt_window_out out;
        if (!iaamt_forward(ctx, feats, n_frames, crop, out, err)) {
            fprintf(stderr, "error: window %zu: %s\n", wi, err.c_str());
            iaamt_ctx_free(ctx);
            return 1;
        }

        if (!params.dump_prefix.empty() && wi == 0) {
            write_npy(params.dump_prefix + "hcqt.npy", feats,
                      { hp.num_audio_channels, n_frames, hp.cqt_n_bins });
            if (!out.query.empty()) {
                write_npy(params.dump_prefix + "query.npy", out.query,
                          { out.n_tracks, out.n_frames, out.head_dim });
                write_npy(params.dump_prefix + "key.npy", out.key,
                          { out.n_tracks, out.n_frames, out.head_dim });
                write_npy(params.dump_prefix + "diag.npy", out.diag,
                          { out.n_tracks, out.n_frames });
            }
            if (!out.pitch_features.empty()) {
                write_npy(params.dump_prefix + "pitch_features.npy", out.pitch_features,
                          { out.n_tracks, out.n_frames, out.feat_dim });
            }
            for (const auto & d : out.debug) {
                write_npy(params.dump_prefix + d.name + ".npy", d.data, d.ne);
            }
            printf("dumped window 0 (%zu intermediates) to %s*.npy\n",
                   out.debug.size(), params.dump_prefix.c_str());
        }

        // frame_valid_mask length = ceil(valid_samples / hop), clamped
        const int valid_model_frames = std::min<int>(
            out.n_frames, (int) ((valid + hp.hop_length - 1) / hp.hop_length));

        if (is_velocity) {
            // Each note is scored by the window whose stride slot holds its
            // onset, so every note gets the full look-ahead the offsets need.
            const int64_t assign_end = (wi + 1 == starts.size())
                ? start + valid
                : start + stride_samples;
            iaamt_velocity_apply(model, out, window, (int) window_samples,
                                 valid_model_frames, start, assign_end, in_notes);
        } else {
            n_intervals += iaamt_stitcher_consume(st, model, out, dp,
                                                  valid_model_frames, (int) valid, start);
        }

        if (params.verbose || (wi % 8 == 0)) {
            printf("\rwindow %zu/%zu  intervals=%d  skipped=%d",
                   wi + 1, starts.size(), n_intervals, n_skipped);
            fflush(stdout);
        }
    }
    printf("\rwindow %zu/%zu  intervals=%d  skipped=%d\n",
           starts.size(), starts.size(), n_intervals, n_skipped);

    iaamt_ctx_free(ctx);

    const std::vector<iaamt_note> notes = is_velocity
        ? in_notes
        : iaamt_stitcher_finalize(st, dp);
    const auto t1 = std::chrono::steady_clock::now();
    const double secs = std::chrono::duration<double>(t1 - t0).count();

    printf("decoded %zu note(s) in %.1f s (%.1fx realtime)\n",
           notes.size(), secs,
           ((double) total_samples / hp.sample_rate) / std::max(secs, 1e-9));

    // Summarise the semi-CRF scores rather than printing one per note: the
    // spread is what tells a reader whether the decode discriminated at all.
    // A run where min and max are nearly equal has ranked nothing, which is
    // worth seeing without having to dump every note.
    if (!is_velocity && !notes.empty()) {
        float lo = notes[0].crf_score;
        float hi = notes[0].crf_score;
        double sum = 0.0;
        for (const iaamt_note & n : notes) {
            lo = std::min(lo, n.crf_score);
            hi = std::max(hi, n.crf_score);
            sum += n.crf_score;
        }
        printf("crf    : score min %.3f, mean %.3f, max %.3f\n",
               lo, sum / (double) notes.size(), hi);
    }

    if (!iaamt_write_midi(params.out_midi, notes, hp.sample_rate, err)) {
        fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }
    printf("wrote  : %s\n", params.out_midi.c_str());
    return 0;
}
