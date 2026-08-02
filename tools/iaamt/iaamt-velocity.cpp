// Per-note velocity head.
//
// Ports VelocityPredictionModel's head from velocity/modeling/model.py.  The
// backbone runs in the ggml graph; everything here is a handful of small MLPs
// evaluated once per note, so it stays on the CPU.
//
// Only the velocity output is implemented.  `predict_stem_gain` is false in the
// published checkpoint (no global_audio_projection / stem_gain_head tensors),
// and stem gain only has meaning for a multi-stem batch.

#include "iaamt.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

// y = W x + b, with W stored row-major as [out, in]
void linear(const std::vector<float> & w,
            const std::vector<float> & b,
            const float * x,
            int n_in,
            float * y) {
    const int n_out = (int) b.size();
    for (int o = 0; o < n_out; ++o) {
        const float * row = &w[(size_t) o * n_in];
        float acc = b[o];
        for (int i = 0; i < n_in; ++i) {
            acc += row[i] * x[i];
        }
        y[o] = acc;
    }
}

float gelu(float x) {
    return 0.5f * x * (1.0f + std::erf(x * 0.70710678f));
}

// F.avg_pool1d(kernel=2*hop, stride=hop, padding=hop/2, ceil_mode=True) over
// squared samples, then linear resize to n_frames and 10*log10.
void log_energy(const std::vector<std::vector<float>> & pcm,
                int n_samples,
                int hop,
                int n_frames,
                bool absolute,
                std::vector<float> & out) {
    const int n_ch = (int) pcm.size();
    const int k    = std::max(1, hop * 2);
    const int pad  = hop / 2;
    const int n_pool = (n_samples + 2 * pad - k + hop - 1) / hop + 1;   // ceil_mode

    // three channels: left, right, mono
    std::vector<std::vector<float>> pooled(3, std::vector<float>(std::max(1, n_pool), 0.0f));
    for (int c = 0; c < 3; ++c) {
        for (int p = 0; p < n_pool; ++p) {
            const int base = p * hop - pad;
            double acc = 0.0;
            int    cnt = 0;
            for (int i = 0; i < k; ++i) {
                const int idx = base + i;
                if (idx < 0 || idx >= n_samples) {
                    continue;
                }
                float v;
                if (c < n_ch) {
                    v = pcm[c][idx];
                } else {
                    v = 0.0f;
                    for (int ch = 0; ch < n_ch; ++ch) {
                        v += pcm[ch][idx];
                    }
                    v /= (float) n_ch;
                }
                acc += (double) v * v;
                cnt++;
            }
            // avg_pool1d divides by the full kernel size, including padding
            pooled[c][p] = (float) (acc / (double) k);
            (void) cnt;
        }
    }

    // F.interpolate(mode="linear", align_corners=False)
    out.assign((size_t) n_frames * 3, 0.0f);
    for (int c = 0; c < 3; ++c) {
        for (int t = 0; t < n_frames; ++t) {
            const float src = ((float) t + 0.5f) * (float) n_pool / (float) n_frames - 0.5f;
            const float cl  = std::min(std::max(src, 0.0f), (float) (n_pool - 1));
            const int   lo  = (int) std::floor(cl);
            const int   hi  = std::min(lo + 1, n_pool - 1);
            const float a   = cl - (float) lo;
            const float v   = pooled[c][lo] + a * (pooled[c][hi] - pooled[c][lo]);
            float db = 10.0f * std::log10(std::max(v, 1e-10f));
            if (absolute) {
                // float dBFS: -100 dB maps to -2 and 0 dB to +2
                db = std::min(std::max((db + 50.0f) / 25.0f, -2.0f), 2.0f);
            }
            out[(size_t) t * 3 + c] = db;
        }
    }

    if (absolute) {
        return;
    }
    // relative mode: subtract the per-clip mean, then scale
    for (int c = 0; c < 3; ++c) {
        double mean = 0.0;
        for (int t = 0; t < n_frames; ++t) {
            mean += out[(size_t) t * 3 + c];
        }
        mean /= std::max(1, n_frames);
        for (int t = 0; t < n_frames; ++t) {
            float & v = out[(size_t) t * 3 + c];
            v = std::min(std::max(v - (float) mean, -60.0f), 60.0f) / 30.0f;
        }
    }
}

} // namespace

void iaamt_velocity_apply(const iaamt_model & model,
                          const iaamt_window_out & out,
                          const std::vector<std::vector<float>> & pcm,
                          int n_window_samples,
                          int valid_model_frames,
                          int64_t window_start_sample,
                          int64_t assign_end_sample,
                          std::vector<iaamt_note> & notes) {
    const iaamt_hparams & hp = model.hparams;
    const auto & v = model.vel;

    const int D  = out.feat_dim;
    const int T  = out.n_frames;
    const int P  = out.n_tracks;
    const int H  = hp.note_hidden_size;
    const int NO = (int) hp.local_frame_offsets.size();
    if (D <= 0 || T <= 0 || NO <= 0) {
        return;
    }

    std::vector<float> energy;
    log_energy(pcm, n_window_samples, hp.hop_length, T,
               hp.absolute_velocity_energy, energy);

    const int meta_dim  = 32 + 32 + 8 + 24 + 16;
    const int local_dim = D + 3;

    std::vector<float> meta(meta_dim);
    std::vector<float> dur_h(16);
    std::vector<float> dur(16);
    std::vector<float> query(H);
    std::vector<float> qh(H);
    std::vector<float> local(local_dim);
    std::vector<float> proj((size_t) NO * H);
    std::vector<float> score(NO);
    std::vector<float> attn(NO);
    std::vector<float> scratch(H);
    std::vector<float> att_h(H);
    std::vector<float> pooled(H);
    std::vector<float> fused(H);
    std::vector<float> fh(H);
    std::vector<float> nrm(H);
    std::vector<float> logits(v.vel_b.size());

    const float frames_per_second = (float) hp.sample_rate / (float) hp.hop_length;

    for (iaamt_note & note : notes) {
        if (note.start_sample < window_start_sample || note.start_sample >= assign_end_sample) {
            continue;
        }

        // ---- note metadata -> query ----
        const int pitch   = std::min(std::max(note.pitch, 0), 127);
        const int program = std::min(std::max(note.program, 0), 127);
        const int drum    = note.is_drum ? 1 : 0;
        // A single mixed input has no stem identity; the final embedding row is
        // the reserved "unknown" slot the reference uses for padded input.
        const int stem_class = hp.num_stem_classes;

        const double dur_s = std::max(0.0,
            (double) (note.end_sample - note.start_sample) / hp.sample_rate);
        const float log_dur = std::log1p((float) dur_s);
        linear(v.dur_w1, v.dur_b1, &log_dur, 1, dur_h.data());
        for (float & x : dur_h) {
            x = gelu(x);
        }
        linear(v.dur_w2, v.dur_b2, dur_h.data(), 16, dur.data());

        int off = 0;
        std::copy_n(&v.pitch_embd  [(size_t) pitch      * 32], 32, meta.begin() + off); off += 32;
        std::copy_n(&v.program_embd[(size_t) program    * 32], 32, meta.begin() + off); off += 32;
        std::copy_n(&v.drum_embd   [(size_t) drum       *  8],  8, meta.begin() + off); off +=  8;
        std::copy_n(&v.stem_embd   [(size_t) stem_class * 24], 24, meta.begin() + off); off += 24;
        std::copy_n(dur.begin(), 16, meta.begin() + off);

        linear(v.nq_w1, v.nq_b1, meta.data(), meta_dim, qh.data());
        for (float & x : qh) {
            x = gelu(x);
        }
        linear(v.nq_w2, v.nq_b2, qh.data(), H, query.data());

        // ---- gather frames around the onset ----
        const float onset = (float) ((double) (note.start_sample - window_start_sample)
                                     / hp.sample_rate) * frames_per_second;
        const int   pitch_index = std::min(std::max(note.pitch - hp.pitch_min, 0), P - 1);

        int n_valid = 0;
        for (int o = 0; o < NO; ++o) {
            const float pos = onset + hp.local_frame_offsets[o];
            const int   lo  = (int) std::floor(pos);
            const float a   = pos - (float) lo;
            const int   l   = std::min(std::max(lo,     0), T - 1);
            const int   u   = std::min(std::max(lo + 1, 0), T - 1);

            const float * fl = &out.pitch_features[((size_t) pitch_index * T + l) * D];
            const float * fu = &out.pitch_features[((size_t) pitch_index * T + u) * D];
            for (int i = 0; i < D; ++i) {
                local[i] = fl[i] + a * (fu[i] - fl[i]);
            }
            for (int i = 0; i < 3; ++i) {
                const float el = energy[(size_t) l * 3 + i];
                const float eu = energy[(size_t) u * 3 + i];
                local[D + i] = el + a * (eu - el);
            }

            float * pr = &proj[(size_t) o * H];
            linear(v.local_w, v.local_b, local.data(), local_dim, pr);

            const bool ok = pos >= 0.0f && pos < (float) valid_model_frames;
            if (ok) {
                n_valid++;
                // local_attention: Linear -> Tanh -> Linear, over proj + query
                for (int i = 0; i < H; ++i) {
                    scratch[i] = pr[i] + query[i];
                }
                linear(v.att_w1, v.att_b1, scratch.data(), H, att_h.data());
                for (float & x : att_h) {
                    x = std::tanh(x);
                }
                linear(v.att_w2, v.att_b2, att_h.data(), H, &score[o]);
            }
            attn[o] = ok ? 1.0f : 0.0f;
        }

        // masked softmax over the offsets
        if (n_valid == 0) {
            continue;   // onset lies outside the valid region of this window
        }
        float mx = -INFINITY;
        for (int o = 0; o < NO; ++o) {
            if (attn[o] > 0.0f) {
                mx = std::max(mx, score[o]);
            }
        }
        float sum = 0.0f;
        for (int o = 0; o < NO; ++o) {
            attn[o] = attn[o] > 0.0f ? std::exp(score[o] - mx) : 0.0f;
            sum += attn[o];
        }
        sum = std::max(sum, 1e-8f);

        std::fill(pooled.begin(), pooled.end(), 0.0f);
        for (int o = 0; o < NO; ++o) {
            const float wgt = attn[o] / sum;
            if (wgt == 0.0f) {
                continue;
            }
            const float * pr = &proj[(size_t) o * H];
            for (int i = 0; i < H; ++i) {
                pooled[i] += pr[i] * wgt;
            }
        }

        // note_fusion: LayerNorm -> Linear -> GELU
        for (int i = 0; i < H; ++i) {
            fh[i] = pooled[i] + query[i];
        }
        double mean = 0.0;
        for (int i = 0; i < H; ++i) {
            mean += fh[i];
        }
        mean /= H;
        double var = 0.0;
        for (int i = 0; i < H; ++i) {
            const double d = fh[i] - mean;
            var += d * d;
        }
        var /= H;
        const float inv = 1.0f / std::sqrt((float) var + 1e-5f);
        for (int i = 0; i < H; ++i) {
            nrm[i] = ((float) (fh[i] - mean)) * inv * v.fuse_norm_w[i] + v.fuse_norm_b[i];
        }
        linear(v.fuse_w, v.fuse_b, nrm.data(), H, fused.data());
        for (float & x : fused) {
            x = gelu(x);
        }

        linear(v.vel_w, v.vel_b, fused.data(), H, logits.data());

        // expected velocity over the 1..127 support
        const int n_cls = (int) logits.size();
        float lmax = logits[0];
        for (int i = 1; i < n_cls; ++i) {
            lmax = std::max(lmax, logits[i]);
        }
        double denom = 0.0;
        double acc   = 0.0;
        for (int i = 0; i < n_cls; ++i) {
            const double p = std::exp(logits[i] - lmax);
            denom += p;
            acc   += p * (double) (i + 1);
        }
        const int vel = (int) std::lround(acc / std::max(denom, 1e-12));
        note.velocity = std::min(std::max(vel, 1), 127);
    }
}
