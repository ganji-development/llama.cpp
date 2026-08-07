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

// Resampling only: the decoders and device backends are not needed here, since
// the caller supplies decoded samples. This is the same resampler the CLI gets
// through ma_decoder, which is the point - one implementation, so every
// consumer of this model sees the same samples for the same source.
//
// NOMINMAX because miniaudio reaches windows.h, whose min/max macros break the
// explicitly qualified std::min above.
#define NOMINMAX
#define MA_NO_DECODING
#define MA_NO_ENCODING
#define MA_NO_DEVICE_IO
#define MA_NO_THREADING
#define MA_NO_GENERATION
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio/miniaudio.h"

// NOMINMAX alone is not enough: something earlier in this translation unit
// already reached windows.h without it, so by here the macros exist regardless.
#ifdef min
#  undef min
#endif
#ifdef max
#  undef max
#endif

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
                           int32_t                sample_rate,
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
    if (sample_rate <= 0) {
        set_err(err, err_size, "sample_rate must be positive");
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

        const int n_model_channels = hp.input_audio_channels;

        // Resampling comes before anything that depends on the length, because
        // the stitcher and the window grid are both in model-rate samples and
        // seeding them from the caller's rate would put every note in the wrong
        // place. Done once up front rather than per window, so a window
        // boundary never falls inside the converter's internal state.
        // `resampled` owns the converted audio when a conversion happened and
        // stays empty otherwise, in which case `source` points straight at the
        // caller's buffers and nothing is copied.
        std::vector<std::vector<float>> resampled;
        std::vector<const float *>      source((size_t) n_model_channels);
        int64_t                         source_samples = n_samples;

        if (sample_rate != hp.sample_rate) {
            // Interleave at the model's channel count first: ma_data_converter
            // works on frames, and doing the channel fan-out here means the
            // duplication of a mono source happens before conversion rather
            // than after, which is what the CLI's decoder also does.
            std::vector<float> in((size_t) n_samples * n_model_channels);
            for (int64_t i = 0; i < n_samples; ++i) {
                for (int c = 0; c < n_model_channels; ++c) {
                    in[(size_t) i * n_model_channels + c] =
                        channels[std::min(c, n_channels - 1)][i];
                }
            }

            ma_data_converter_config cfg = ma_data_converter_config_init(
                ma_format_f32, ma_format_f32,
                (ma_uint32) n_model_channels, (ma_uint32) n_model_channels,
                (ma_uint32) sample_rate, (ma_uint32) hp.sample_rate);

            // ma_data_converter_config_init_default leaves lpfOrder at 1, while
            // the decoder path the CLI uses gets MA_DEFAULT_RESAMPLER_LPF_ORDER
            // through ma_resampler_config_init. That is not a cosmetic
            // difference: a first-order low-pass is not enough anti-aliasing for
            // 2:1 decimation, and the partials that fold back land in the HCQT
            // as pitch energy that is not in the source. Measured on one bass
            // stem, the weak filter changed 408 notes into 403.
            cfg.resampling.linear.lpfOrder = MA_DEFAULT_RESAMPLER_LPF_ORDER;

            ma_data_converter conv;
            if (ma_data_converter_init(&cfg, nullptr, &conv) != MA_SUCCESS) {
                set_err(err, err_size, "failed to initialise resampler");
                return 1;
            }

            ma_uint64 in_frames  = (ma_uint64) n_samples;
            ma_uint64 out_frames = 0;
            ma_data_converter_get_expected_output_frame_count(&conv, in_frames, &out_frames);

            std::vector<float> out((size_t) out_frames * n_model_channels);
            const ma_result r = ma_data_converter_process_pcm_frames(
                &conv, in.data(), &in_frames, out.data(), &out_frames);
            ma_data_converter_uninit(&conv, nullptr);
            if (r != MA_SUCCESS) {
                set_err(err, err_size, "resampling failed");
                return 1;
            }

            source_samples = (int64_t) out_frames;
            resampled.assign((size_t) n_model_channels,
                             std::vector<float>((size_t) source_samples));
            for (int64_t i = 0; i < source_samples; ++i) {
                for (int c = 0; c < n_model_channels; ++c) {
                    resampled[(size_t) c][(size_t) i] =
                        out[(size_t) i * n_model_channels + c];
                }
            }
            for (int c = 0; c < n_model_channels; ++c) {
                source[(size_t) c] = resampled[(size_t) c].data();
            }
        } else {
            for (int c = 0; c < n_model_channels; ++c) {
                source[(size_t) c] = channels[std::min(c, n_channels - 1)];
            }
        }

        if (source_samples <= 0) {
            set_err(err, err_size, "audio is empty after resampling");
            return 1;
        }

        // Both of these are in model-rate samples, which is why they come after
        // the conversion.
        iaamt_stitcher st;
        iaamt_stitcher_init(st, model, dp, source_samples);

        iaamt_context * ctx = iaamt_ctx_init(const_cast<iaamt_model &>(model),
                                             params->n_threads);
        if (ctx == nullptr) {
            set_err(err, err_size, "failed to create inference context");
            return 1;
        }

        std::vector<std::vector<float>> window(
            (size_t) n_model_channels, std::vector<float>((size_t) window_samples));
        std::vector<float> feats;

        const float silence_gate = params->silence_dbfs < 0.0f
            ? powf(10.0f, params->silence_dbfs / 20.0f)
            : 0.0f;

        const std::vector<int64_t> starts =
            iaamt_build_window_starts(source_samples, window_samples, stride_samples);

        for (int64_t start : starts) {
            const int64_t valid = std::min<int64_t>(window_samples, source_samples - start);

            // The channel fan-out for a mono source already happened, either
            // during conversion or when `source` was pointed at the caller's
            // buffers, so every entry here is a real channel.
            for (int c = 0; c < n_model_channels; ++c) {
                const float * src = source[(size_t) c];
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
