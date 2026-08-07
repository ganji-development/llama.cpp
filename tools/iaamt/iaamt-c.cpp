// Implementation of the flat C ABI declared in iaamt-c.h.
//
// This is the same window loop iaamt-cli runs, with the file I/O taken out: the
// caller supplies decoded audio and receives notes, so nothing here reads or
// writes a file except the GGUF at open time. Keeping the loop here rather than
// exposing the per-window functions across the ABI is deliberate - the ordering
// of cqt/forward/consume, the valid-frame arithmetic at the tail, and the
// silence gate are all easy to get subtly wrong, and a second implementation of
// them in the consumer would drift from this one.

#include "iaamt-c.h"
#include "iaamt.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

namespace {

void set_err(char * err, size_t err_size, const std::string & msg) {
    if (err == nullptr || err_size == 0) {
        return;
    }
    const size_t n = std::min(msg.size(), err_size - 1);
    std::memcpy(err, msg.data(), n);
    err[n] = '\0';
}

} // namespace

struct iaamt_session {
    iaamt_model model;
};

int32_t iaamt_c_abi_version(void) {
    return IAAMT_C_ABI_VERSION;
}

void iaamt_c_default_params(iaamt_c_params * params) {
    if (params == nullptr) {
        return;
    }
    // Mirrors iaamt_decode_params and the CLI's cli_params, so a caller that
    // starts here and changes nothing gets exactly what the tool produces.
    // These are not free choices: the window length in particular changes the
    // transcription, because the semi-CRF decodes each window independently and
    // a longer one is not simply more of the same work. 8 s against 60 s on the
    // same bass stem is 408 notes against 27.
    const iaamt_decode_params dp;
    params->note_bias        = dp.note_bias;
    params->merge_onset_ms   = dp.merge_onset_ms;
    params->merge_gap_ms     = dp.merge_gap_ms;
    params->velocity         = dp.velocity;
    params->min_note_ms      = dp.min_note_ms;
    params->use_boundary_head = dp.use_boundary_head ? 1 : 0;
    params->window_ms        = 8000.0f;
    params->stride_ms        = 0.0f;    // half the window
    params->silence_dbfs     = -72.0f;
    params->n_threads        = 4;
}

iaamt_session * iaamt_c_open(const char * gguf_path,
                             int32_t      use_gpu,
                             char *       err,
                             size_t       err_size) {
    if (gguf_path == nullptr) {
        set_err(err, err_size, "gguf_path is null");
        return nullptr;
    }
    // No exception may cross this boundary: the caller is a different runtime
    // and unwinding into it is undefined.
    try {
        auto * session = new iaamt_session();
        std::string load_err;
        if (!iaamt_model_load(session->model, gguf_path, use_gpu != 0, load_err)) {
            set_err(err, err_size, load_err);
            delete session;
            return nullptr;
        }
        return session;
    } catch (const std::exception & e) {
        set_err(err, err_size, e.what());
        return nullptr;
    } catch (...) {
        set_err(err, err_size, "unknown error loading model");
        return nullptr;
    }
}

void iaamt_c_close(iaamt_session * session) {
    delete session;
}

int32_t iaamt_c_sample_rate(const iaamt_session * session) {
    return session == nullptr ? 0 : session->model.hparams.sample_rate;
}

int32_t iaamt_c_channels(const iaamt_session * session) {
    return session == nullptr ? 0 : session->model.hparams.input_audio_channels;
}

int32_t iaamt_c_is_transcription_model(const iaamt_session * session) {
    if (session == nullptr) {
        return 0;
    }
    return session->model.hparams.model_type == IAAMT_MODEL_TYPE_TRANSCRIPTION ? 1 : 0;
}

int32_t iaamt_c_transcribe(iaamt_session *        session,
                           const float * const *  channels,
                           int32_t                n_channels,
                           int64_t                n_samples,
                           const iaamt_c_params * params,
                           iaamt_c_note **        out_notes,
                           size_t *               out_count,
                           char *                 err,
                           size_t                 err_size) {
    if (session == nullptr || channels == nullptr || params == nullptr
        || out_notes == nullptr || out_count == nullptr) {
        set_err(err, err_size, "null argument");
        return 1;
    }
    if (n_channels <= 0 || n_samples <= 0) {
        set_err(err, err_size, "audio is empty");
        return 1;
    }
    for (int32_t c = 0; c < n_channels; ++c) {
        if (channels[c] == nullptr) {
            set_err(err, err_size, "null channel pointer");
            return 1;
        }
    }
    // A velocity or beat/chord checkpoint would run happily and return
    // something that is not a transcription, so refuse rather than mislead.
    if (!iaamt_c_is_transcription_model(session)) {
        set_err(err, err_size,
                "checkpoint is not a transcription model: " + session->model.hparams.model_type);
        return 1;
    }

    try {
        const iaamt_model &   model = session->model;
        const iaamt_hparams & hp    = model.hparams;

        const int64_t window_samples =
            (int64_t) llround((double) params->window_ms * hp.sample_rate / 1000.0);
        const int64_t stride_samples =
            (int64_t) llround((double) (params->stride_ms > 0.0f
                                            ? params->stride_ms
                                            : params->window_ms / 2.0f)
                              * hp.sample_rate / 1000.0);
        if (window_samples <= 0 || stride_samples <= 0) {
            set_err(err, err_size, "window and stride must be positive");
            return 1;
        }

        iaamt_decode_params dp;
        dp.note_bias         = params->note_bias;
        dp.merge_onset_ms    = params->merge_onset_ms;
        dp.merge_gap_ms      = params->merge_gap_ms;
        dp.velocity          = params->velocity;
        dp.min_note_ms       = params->min_note_ms;
        dp.use_boundary_head = params->use_boundary_head != 0;

        iaamt_stitcher st;
        iaamt_stitcher_init(st, model, dp, n_samples);

        iaamt_context * ctx = iaamt_ctx_init(const_cast<iaamt_model &>(model),
                                             params->n_threads);
        if (ctx == nullptr) {
            set_err(err, err_size, "failed to create inference context");
            return 1;
        }

        const int n_model_channels = hp.input_audio_channels;
        std::vector<std::vector<float>> window(
            (size_t) n_model_channels, std::vector<float>((size_t) window_samples));
        std::vector<float> feats;

        const float silence_gate = params->silence_dbfs < 0.0f
            ? powf(10.0f, params->silence_dbfs / 20.0f)
            : 0.0f;

        const std::vector<int64_t> starts =
            iaamt_build_window_starts(n_samples, window_samples, stride_samples);

        for (int64_t start : starts) {
            const int64_t valid = std::min<int64_t>(window_samples, n_samples - start);

            for (int c = 0; c < n_model_channels; ++c) {
                // A mono source against a stereo model repeats its last
                // channel rather than leaving one silent, which would halve the
                // energy the front end sees.
                const float * src = channels[std::min(c, n_channels - 1)];
                std::fill(window[(size_t) c].begin(), window[(size_t) c].end(), 0.0f);
                std::copy(src + start, src + start + valid, window[(size_t) c].begin());
            }

            if (silence_gate > 0.0f) {
                double acc = 0.0;
                for (int64_t i = 0; i < window_samples; ++i) {
                    double m = 0.0;
                    for (int c = 0; c < n_model_channels; ++c) {
                        m += window[(size_t) c][(size_t) i];
                    }
                    m /= (double) n_model_channels;
                    acc += m * m;
                }
                if (std::sqrt(acc / (double) window_samples) < silence_gate) {
                    continue;
                }
            }

            const int n_frames = iaamt_cqt_apply(model, window, (int) window_samples, feats);

            iaamt_window_out out;
            std::string fwd_err;
            if (!iaamt_forward(ctx, feats, n_frames, n_frames, out, fwd_err)) {
                iaamt_ctx_free(ctx);
                set_err(err, err_size, fwd_err);
                return 1;
            }

            const int valid_model_frames = std::min<int>(
                out.n_frames, (int) ((valid + hp.hop_length - 1) / hp.hop_length));

            iaamt_stitcher_consume(st, model, out, dp,
                                   valid_model_frames, (int) valid, start);
        }

        iaamt_ctx_free(ctx);

        const std::vector<iaamt_note> notes = iaamt_stitcher_finalize(st, dp);
        if (notes.empty()) {
            *out_notes = nullptr;
            *out_count = 0;
            return 0;
        }

        auto * result = static_cast<iaamt_c_note *>(
            std::malloc(notes.size() * sizeof(iaamt_c_note)));
        if (result == nullptr) {
            set_err(err, err_size, "out of memory");
            return 1;
        }
        for (size_t i = 0; i < notes.size(); ++i) {
            const iaamt_note & n = notes[i];
            result[i].pitch             = n.pitch;
            result[i].slot              = n.slot;
            result[i].start_sample      = n.start_sample;
            result[i].end_sample        = n.end_sample;
            result[i].velocity          = n.velocity;
            result[i].has_onset         = n.has_onset ? 1 : 0;
            result[i].has_offset        = n.has_offset ? 1 : 0;
            result[i].crf_score         = n.crf_score;
            result[i].onset_confidence  = n.onset_confidence;
            result[i].offset_confidence = n.offset_confidence;
        }
        *out_notes = result;
        *out_count = notes.size();
        return 0;
    } catch (const std::exception & e) {
        set_err(err, err_size, e.what());
        return 1;
    } catch (...) {
        set_err(err, err_size, "unknown error during transcription");
        return 1;
    }
}

void iaamt_c_free_notes(iaamt_c_note * notes) {
    std::free(notes);
}
