// CPU HCQT front end.
//
// This mirrors AudioFeatureExtractor + RecursiveCQT from the reference
// implementation.  It runs on the CPU rather than in the ggml graph because the
// transform is a fixed preprocessing step over complex STFT frames, the same
// reason mtmd-audio.cpp computes mel spectrograms outside the graph.
//
// The STFT->CQT kernels and Hann windows are not in the checkpoint (they are
// non-persistent torch buffers); the converter precomputes them, so this file
// only consumes them.

#include "iaamt.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {

// ---------------------------------------------------------------------------
// radix-2 FFT, in-place over interleaved (re, im) pairs
// ---------------------------------------------------------------------------

struct fft_plan {
    int n = 0;
    std::vector<int>   rev;
    std::vector<float> tw_re;   // twiddles per stage, flattened
    std::vector<float> tw_im;

    void init(int n_) {
        if (n == n_) {
            return;
        }
        n = n_;
        rev.resize(n);
        int bits = 0;
        while ((1 << bits) < n) {
            bits++;
        }
        for (int i = 0; i < n; ++i) {
            int r = 0;
            for (int b = 0; b < bits; ++b) {
                if (i & (1 << b)) {
                    r |= 1 << (bits - 1 - b);
                }
            }
            rev[i] = r;
        }
        tw_re.resize(n / 2);
        tw_im.resize(n / 2);
        for (int i = 0; i < n / 2; ++i) {
            const double ang = -2.0 * M_PI * i / n;
            tw_re[i] = (float) std::cos(ang);
            tw_im[i] = (float) std::sin(ang);
        }
    }

    // data holds 2*n floats
    void run(float * data) const {
        for (int i = 0; i < n; ++i) {
            const int j = rev[i];
            if (i < j) {
                std::swap(data[2*i + 0], data[2*j + 0]);
                std::swap(data[2*i + 1], data[2*j + 1]);
            }
        }
        for (int len = 2; len <= n; len <<= 1) {
            const int half = len / 2;
            const int step = n / len;
            for (int i = 0; i < n; i += len) {
                for (int j = 0; j < half; ++j) {
                    const float wr = tw_re[j * step];
                    const float wi = tw_im[j * step];
                    float * a = data + 2 * (i + j);
                    float * b = data + 2 * (i + j + half);
                    const float br = b[0] * wr - b[1] * wi;
                    const float bi = b[0] * wi + b[1] * wr;
                    b[0] = a[0] - br;
                    b[1] = a[1] - bi;
                    a[0] += br;
                    a[1] += bi;
                }
            }
        }
    }
};

// torch.stft(center=True, pad_mode="reflect"): reflect-pad by n_fft/2 on both
// ends, then frame.  Reflection excludes the edge sample itself.
float padded_sample(const float * x, int n, int idx) {
    if (n == 1) {
        return x[0];
    }
    // fold idx into [0, n-1] by repeated reflection
    while (idx < 0 || idx >= n) {
        if (idx < 0) {
            idx = -idx;
        }
        if (idx >= n) {
            idx = 2 * (n - 1) - idx;
        }
    }
    return x[idx];
}

// Conv1d(1, 1, kernel_size=taps, stride=2, padding=taps/2, bias=False)
void resample_half(const std::vector<float> & in,
                   const std::vector<float> & filt,
                   std::vector<float> & out) {
    const int n    = (int) in.size();
    const int taps = (int) filt.size();
    const int pad  = taps / 2;
    const int n_out = (n + 2 * pad - taps) / 2 + 1;
    out.assign(std::max(0, n_out), 0.0f);
    for (int o = 0; o < n_out; ++o) {
        const int base = o * 2 - pad;
        float acc = 0.0f;
        for (int k = 0; k < taps; ++k) {
            const int idx = base + k;
            if (idx >= 0 && idx < n) {
                acc += in[idx] * filt[k];
            }
        }
        out[o] = acc;
    }
}

} // namespace

int iaamt_hparams::n_bands() const {
    // stem strides: block1 (t/2, f/1), block2 (t/2, f/2), block3 (t/2, f/2)
    // torch conv output: floor((in + 2*pad - k)/stride) + 1, pad=1, k=3
    auto conv_out = [](int in, int stride) {
        return (in + 2 - 3) / stride + 1;
    };
    int f = cqt_n_bins;
    f = conv_out(f, 1);
    f = conv_out(f, 2);
    f = conv_out(f, 2);
    return f;
}

int iaamt_cqt_apply(const iaamt_model & model,
                    const std::vector<std::vector<float>> & pcm,
                    int n_samples,
                    std::vector<float> & out) {
    const iaamt_hparams & hp = model.hparams;

    const int n_ch        = hp.input_audio_channels;
    const int n_harm      = (int) hp.harmonics.size();
    const int n_bins      = hp.cqt_n_bins;         // 312
    const int n_bins_big  = hp.cqt_n_bins_large;   // 396
    const int n_stages    = (int) hp.cqt_stages.size();
    const int target_frames = (n_samples + hp.hop_length - 1) / hp.hop_length;

    // magnitude CQT for every audio channel, padded up to n_bins_large
    // large[ch][bin * target_frames + t]
    std::vector<std::vector<float>> large(n_ch);

    fft_plan plan;
    std::vector<float> frame;
    std::vector<float> cur;
    std::vector<float> next;

    for (int ch = 0; ch < n_ch; ++ch) {
        large[ch].assign((size_t) n_bins_big * target_frames, 0.0f);

        cur.assign(pcm[ch].begin(), pcm[ch].begin() + n_samples);

        for (int s = 0; s < n_stages; ++s) {
            const iaamt_cqt_stage & stage = hp.cqt_stages[s];
            const int n_fft   = stage.n_fft;
            const int n_rfft  = n_fft / 2 + 1;
            const int n_freqs = stage.bin_end - stage.bin_beg;
            const int hop     = hp.hop_length >> s;
            const int n_cur   = (int) cur.size();

            plan.init(n_fft);
            frame.assign((size_t) 2 * n_fft, 0.0f);

            const float * win = model.cqt_window[s].data();
            const float * kre = model.cqt_kernel_re[s].data();
            const float * kim = model.cqt_kernel_im[s].data();

            // center=True gives 1 + n_cur/hop frames
            const int n_frames = n_cur > 0 ? 1 + n_cur / hop : 0;
            const int n_use    = std::min(n_frames, target_frames);

            for (int t = 0; t < n_use; ++t) {
                const int start = t * hop - n_fft / 2;
                for (int i = 0; i < n_fft; ++i) {
                    frame[2*i + 0] = padded_sample(cur.data(), n_cur, start + i) * win[i];
                    frame[2*i + 1] = 0.0f;
                }
                plan.run(frame.data());

                // cqt[k] = sum_j stft[j] * kernel[k][j]
                for (int k = 0; k < n_freqs; ++k) {
                    const float * kr = kre + (size_t) k * n_rfft;
                    const float * ki = kim + (size_t) k * n_rfft;
                    float acc_re = 0.0f;
                    float acc_im = 0.0f;
                    for (int j = 0; j < n_rfft; ++j) {
                        const float sr = frame[2*j + 0];
                        const float si = frame[2*j + 1];
                        acc_re += sr * kr[j] - si * ki[j];
                        acc_im += sr * ki[j] + si * kr[j];
                    }
                    const int bin = stage.bin_beg + k;
                    large[ch][(size_t) bin * target_frames + t] =
                        std::sqrt(acc_re * acc_re + acc_im * acc_im);
                }
            }

            if (s + 1 < n_stages) {
                resample_half(cur, model.cqt_resampler[s], next);
                cur.swap(next);
            }
        }
    }

    // harmonic interpolation + standardization
    // spec[c][h][f][t] -> out[(c*n_harm + h)][t][f]
    const size_t n_out = (size_t) n_bins * target_frames * n_ch * n_harm;
    out.assign(n_out, 0.0f);

    // gather first so mean/std can be computed over the whole window
    std::vector<float> spec(n_out);
    size_t widx = 0;
    for (int c = 0; c < n_ch; ++c) {
        for (int h = 0; h < n_harm; ++h) {
            const float shift = hp.harmonic_shifts[h];
            for (int f = 0; f < n_bins; ++f) {
                float pos = (float) f + shift;
                pos = std::min(std::max(pos, 0.0f), (float) (n_bins_big - 1));
                const int lo = (int) std::floor(pos);
                const int hi = std::min(lo + 1, n_bins_big - 1);
                const float alpha = pos - (float) lo;
                const float * a = &large[c][(size_t) lo * target_frames];
                const float * b = &large[c][(size_t) hi * target_frames];
                for (int t = 0; t < target_frames; ++t) {
                    float v = a[t] + alpha * (b[t] - a[t]);
                    if (hp.cqt_log_scale) {
                        v = std::log(v + 1e-8f);
                    }
                    spec[widx + (size_t) f * target_frames + t] = v;
                }
            }
            widx += (size_t) n_bins * target_frames;
        }
    }

    // torch: mean/std over every dim except batch, std with Bessel correction
    double sum = 0.0;
    for (size_t i = 0; i < n_out; ++i) {
        sum += spec[i];
    }
    const double mean = sum / (double) n_out;
    double var = 0.0;
    for (size_t i = 0; i < n_out; ++i) {
        const double d = spec[i] - mean;
        var += d * d;
    }
    var /= (double) (n_out > 1 ? n_out - 1 : 1);
    const float inv_std = 1.0f / std::max((float) std::sqrt(var), 1e-8f);

    // "b c h f t -> b (c h) t f": channel = c*n_harm + h, then [t][f]
    for (int c = 0; c < n_ch; ++c) {
        for (int h = 0; h < n_harm; ++h) {
            const int    chan = c * n_harm + h;
            const float * src = &spec[((size_t) c * n_harm + h) * n_bins * target_frames];
            float       * dst = &out[(size_t) chan * n_bins * target_frames];
            for (int f = 0; f < n_bins; ++f) {
                for (int t = 0; t < target_frames; ++t) {
                    dst[(size_t) t * n_bins + f] =
                        ((float) (src[(size_t) f * target_frames + t] - mean)) * inv_std;
                }
            }
        }
    }

    return target_frames;
}
