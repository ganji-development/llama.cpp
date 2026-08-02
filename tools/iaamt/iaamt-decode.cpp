// Semi-CRF decoding: interval scores -> Viterbi -> notes.
//
// Ports viterbiBackward from modeling/heads/semi_crf.py and the V1 note
// stitching from inference/v1_windowed.py.  Both run on the CPU: the Viterbi
// recursion is a serial dynamic program over each pitch track and does not map
// onto a ggml graph.

#include "iaamt.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace {

// score[end][begin] = dot(query[begin], key[end]), with interval_diag added on
// the diagonal.  Only end >= begin is ever read.
void build_score(const iaamt_window_out & out,
                 int track,
                 int len,
                 float note_bias,
                 const std::string & length_scaling,
                 float length_penalty,
                 std::vector<float> & score) {
    const int   D  = out.head_dim;
    const int   T  = out.n_frames;
    const float * q = &out.query[(size_t) track * T * D];
    const float * k = &out.key  [(size_t) track * T * D];
    const float * d = &out.diag [(size_t) track * T];

    score.assign((size_t) len * len, 0.0f);

    const bool scale_linear = (length_scaling == "linear");
    const bool scale_sqrt   = (length_scaling == "sqrt");

    for (int end = 0; end < len; ++end) {
        const float * ke = k + (size_t) end * D;
        float * row = &score[(size_t) end * len];
        for (int begin = 0; begin <= end; ++begin) {
            const float * qb = q + (size_t) begin * D;
            float acc = 0.0f;
            for (int i = 0; i < D; ++i) {
                acc += qb[i] * ke[i];
            }
            acc += note_bias;
            if (scale_linear || scale_sqrt) {
                float s = (float) (end - begin);
                if (scale_sqrt) {
                    s = std::sqrt(s);
                }
                acc *= s;
            }
            if (length_penalty != 0.0f) {
                acc -= (float) (end - begin) * length_penalty;
            }
            row[begin] = acc;
        }
        score[(size_t) end * len + end] += d[end];
    }
}

struct interval {
    int begin;
    int end;
};

// viterbiBackward with an all-zero noise score.
void viterbi_backward(const std::vector<float> & score,
                      int len,
                      int forced_start,
                      std::vector<interval> & out) {
    out.clear();
    if (len <= 0) {
        return;
    }
    if (len == 1) {
        if (score[0] > 0.0f) {
            out.push_back({0, 0});
        }
        return;
    }

    auto at = [&](int end, int begin) { return score[(size_t) end * len + begin]; };

    std::vector<float> q(len, 0.0f);
    // ptr[i] corresponds to position t = len - 2 - i
    std::vector<int> ptr(len - 1, -1);

    const float last = at(len - 1, len - 1);
    q[len - 1] = last > 0.0f ? last : 0.0f;

    for (int offset = 1; offset < len; ++offset) {
        const int t = len - offset - 1;

        // candidate 0: skip this position (noise score is zero)
        float best = q[t + 1];
        int   sel  = -1;
        for (int e = t + 1; e < len; ++e) {
            const float cand = q[e] + at(e, t);
            if (cand > best) {
                best = cand;
                sel  = e - (t + 1);
            }
        }
        ptr[len - 2 - t] = sel;

        const float diag = at(t, t);
        q[t] = best + (diag > 0.0f ? diag : 0.0f);
    }

    int pos = std::max(0, std::min(forced_start, len - 1));
    while (pos < len - 1) {
        const int sel = ptr[len - pos - 2];
        if (at(pos, pos) > 0.0f) {
            out.push_back({pos, pos});
        }
        if (sel < 0) {
            pos += 1;
        } else {
            const int end = sel + pos + 1;
            out.push_back({pos, end});
            pos = end;
        }
    }
    if (at(len - 1, len - 1) > 0.0f) {
        out.push_back({len - 1, len - 1});
    }

    // _sanitize_track_intervals: clamp, drop inversions, keep them disjoint
    std::sort(out.begin(), out.end(), [](const interval & a, const interval & b) {
        return a.begin != b.begin ? a.begin < b.begin : a.end < b.end;
    });
    std::vector<interval> clean;
    for (interval iv : out) {
        iv.begin = std::max(0, iv.begin);
        iv.end   = std::min(iv.end, len - 1);
        if (iv.end < iv.begin) {
            continue;
        }
        if (!clean.empty() && iv.begin <= clean.back().end) {
            iv.begin = clean.back().end + 1;
        }
        if (iv.end < iv.begin) {
            continue;
        }
        clean.push_back(iv);
    }
    out.swap(clean);
}

// torch.distributions.ContinuousBernoulli(logits).mean
float continuous_bernoulli_mean(float logits) {
    const float p = 1.0f / (1.0f + std::exp(-logits));
    if (std::fabs(p - 0.5f) <= 0.001f) {
        return 0.5f;
    }
    return p / (2.0f * p - 1.0f) + 1.0f / (std::log1p(-p) - std::log(p));
}

struct boundary_flag {
    bool  has_onset;
    bool  has_offset;
    float onset_off;
    float offset_off;
};

// IntervalBoundaryPredictor over concat(begin, end, begin*end)
bool predict_boundary(const iaamt_model & model,
                      const iaamt_window_out & out,
                      int track,
                      int begin,
                      int end,
                      boundary_flag & flag) {
    if (model.boundary_w1.empty() || out.interval_features.empty()) {
        return false;
    }
    const int T = out.n_frames;
    const int D = out.feat_dim;
    const float * fb = &out.interval_features[((size_t) track * T + begin) * D];
    const float * fe = &out.interval_features[((size_t) track * T + end)   * D];

    // fc1: [3D] -> [H]
    const int H = (int) model.boundary_b1.size();
    const float * w1 = model.boundary_w1.data();
    const float * b1 = model.boundary_b1.data();
    const float * w2 = model.boundary_w2.data();
    const float * b2 = model.boundary_b2.data();

    std::vector<float> h(H);
    for (int o = 0; o < H; ++o) {
        const float * row = w1 + (size_t) o * 3 * D;
        float acc = b1[o];
        for (int i = 0; i < D; ++i) {
            acc += row[i] * fb[i];
        }
        for (int i = 0; i < D; ++i) {
            acc += row[D + i] * fe[i];
        }
        for (int i = 0; i < D; ++i) {
            acc += row[2 * D + i] * fb[i] * fe[i];
        }
        // exact GELU, matching nn.GELU
        h[o] = 0.5f * acc * (1.0f + std::erf(acc * 0.70710678f));
    }

    float logits[4];
    for (int o = 0; o < 4; ++o) {
        const float * row = w2 + (size_t) o * H;
        float acc = b2[o];
        for (int i = 0; i < H; ++i) {
            acc += row[i] * h[i];
        }
        logits[o] = acc;
    }

    flag.has_onset  = logits[0] > 0.0f;
    flag.has_offset = logits[1] > 0.0f;
    flag.onset_off  = std::min(std::max((continuous_bernoulli_mean(logits[2]) - 0.005f) / 0.99f, 0.0f), 1.0f);
    flag.offset_off = std::min(std::max((continuous_bernoulli_mean(logits[3]) - 0.005f) / 0.99f, 0.0f), 1.0f);
    return true;
}

} // namespace

void iaamt_stitcher_init(iaamt_stitcher & st,
                         const iaamt_model & model,
                         const iaamt_decode_params & params,
                         int64_t total_samples) {
    const iaamt_hparams & hp = model.hparams;
    const int n_tracks = hp.pitch_query_count * hp.num_pitch_slots;
    const int sr = hp.sample_rate;

    st.hop_length    = hp.hop_length;
    st.sample_rate   = sr;
    st.total_samples = total_samples;
    st.n_slots       = hp.num_pitch_slots;
    st.velocity      = std::max(1, std::min(127, params.velocity));
    // the reference defaults merge_gap to a single hop
    st.merge_gap_samples = params.merge_gap_ms < 0.0f
        ? (int64_t) hp.hop_length
        : (int64_t) llround(params.merge_gap_ms * sr / 1000.0);
    st.merge_onset_samples = (int64_t) llround(params.merge_onset_ms * sr / 1000.0);

    st.last_closed.assign(n_tracks, 0);
    st.by_track.assign(n_tracks, {});
}

int iaamt_stitcher_consume(iaamt_stitcher & st,
                           const iaamt_model & model,
                           const iaamt_window_out & out,
                           const iaamt_decode_params & params,
                           int valid_model_frames,
                           int valid_audio_samples,
                           int64_t window_start_sample) {
    const iaamt_hparams & hp = model.hparams;
    const int n_tracks = out.n_tracks;
    const int len = std::min(valid_model_frames, out.n_frames);
    if (len <= 0) {
        return 0;
    }

    const int64_t window_model_start =
        (int64_t) llround((double) window_start_sample / (double) st.hop_length);

    std::vector<float>    score;
    std::vector<interval> intervals;
    int n_decoded = 0;

    for (int track = 0; track < n_tracks; ++track) {
        build_score(out, track, len, params.note_bias,
                    hp.semi_crf_length_scaling, hp.semi_crf_length_penalty, score);

        const int forced = (int) std::max<int64_t>(
            0, std::min<int64_t>(st.last_closed[track] - window_model_start, len - 1));

        viterbi_backward(score, len, forced, intervals);
        n_decoded += (int) intervals.size();

        const int pitch_index = track / st.n_slots;
        const int slot_index  = track % st.n_slots;

        for (const interval & iv : intervals) {
            boundary_flag flag{};
            bool have_flag = false;
            if (params.use_boundary_head && hp.use_interval_boundary_head) {
                have_flag = predict_boundary(model, out, track, iv.begin, iv.end, flag);
            }
            if (!have_flag) {
                // _resolve_interval_boundary_flags fallback
                flag.has_onset  = iv.begin > 0;
                flag.has_offset = iv.end < std::max(0, len - 1);
                flag.onset_off  = 0.0f;
                flag.offset_off = 1.0f;
                if (window_start_sample == 0 && iv.begin == 0) {
                    flag.has_onset = true;
                }
            }

            if (flag.has_offset) {
                st.last_closed[track] = window_model_start + iv.end;
            }

            int64_t start_sample = window_start_sample +
                (int64_t) llround(((double) iv.begin + flag.onset_off) * st.hop_length);
            int64_t end_sample = window_start_sample +
                std::min<int64_t>((int64_t) valid_audio_samples,
                                  (int64_t) llround(((double) iv.end + flag.offset_off) * st.hop_length));

            start_sample = std::max<int64_t>(0, std::min(start_sample, st.total_samples));
            end_sample   = std::max<int64_t>(start_sample + 1, std::min(end_sample, st.total_samples));

            const int64_t window_valid_end =
                std::min<int64_t>(st.total_samples, window_start_sample + valid_audio_samples);
            if (start_sample >= window_valid_end) {
                continue;
            }

            iaamt_note note;
            note.pitch        = 21 + pitch_index;
            note.slot         = slot_index;
            note.start_sample = start_sample;
            note.end_sample   = end_sample;
            note.velocity     = st.velocity;
            note.has_onset    = flag.has_onset;
            note.has_offset   = flag.has_offset;
            st.by_track[track].push_back(note);
        }
    }

    return n_decoded;
}

std::vector<iaamt_note> iaamt_stitcher_finalize(iaamt_stitcher & st,
                                                const iaamt_decode_params & params) {
    std::vector<iaamt_note> all;
    for (auto & track_notes : st.by_track) {
        if (!track_notes.empty()) {
            track_notes.back().has_offset = true;
        }
        all.insert(all.end(), track_notes.begin(), track_notes.end());
    }
    if (all.empty()) {
        return all;
    }

    // _merge_nearby_notes
    std::sort(all.begin(), all.end(), [](const iaamt_note & a, const iaamt_note & b) {
        if (a.pitch != b.pitch) return a.pitch < b.pitch;
        if (a.slot  != b.slot)  return a.slot  < b.slot;
        if (a.start_sample != b.start_sample) return a.start_sample < b.start_sample;
        return a.end_sample < b.end_sample;
    });

    std::vector<iaamt_note> merged;
    iaamt_note cur = all[0];
    for (size_t i = 1; i < all.size(); ++i) {
        const iaamt_note & n = all[i];
        const bool same_track = (n.pitch == cur.pitch && n.slot == cur.slot);
        const bool by_gap = same_track &&
                            n.start_sample <= cur.end_sample + st.merge_gap_samples &&
                            !n.has_onset;
        const bool by_onset = same_track &&
                              std::llabs(n.start_sample - cur.start_sample) <= st.merge_onset_samples;
        if (by_gap || by_onset) {
            cur.end_sample = std::max(cur.end_sample, n.end_sample);
            cur.velocity   = std::max(cur.velocity, n.velocity);
            cur.has_onset  = cur.has_onset || n.has_onset;
            cur.has_offset = by_gap ? n.has_offset : (cur.has_offset || n.has_offset);
            continue;
        }
        merged.push_back(cur);
        cur = n;
    }
    merged.push_back(cur);

    const int64_t min_samples =
        (int64_t) llround(params.min_note_ms * st.sample_rate / 1000.0);

    std::vector<iaamt_note> out;
    out.reserve(merged.size());
    for (const iaamt_note & n : merged) {
        if (!n.has_offset) {
            continue;
        }
        if (n.end_sample - n.start_sample < min_samples) {
            continue;
        }
        out.push_back(n);
    }

    std::sort(out.begin(), out.end(), [](const iaamt_note & a, const iaamt_note & b) {
        if (a.start_sample != b.start_sample) return a.start_sample < b.start_sample;
        if (a.pitch != b.pitch) return a.pitch < b.pitch;
        if (a.slot  != b.slot)  return a.slot  < b.slot;
        return a.end_sample < b.end_sample;
    });
    return out;
}
