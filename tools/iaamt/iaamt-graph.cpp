// ggml graph for the instrument-agnostic-amt backbone and transcription head.
//
// Layout note: torch tensors are described here in torch order, ggml shapes in
// ne order (ne0 fastest).  The HCQT input arrives as ne = [F, T, C, 1], which is
// torch [B=1, C, T, F] -- exactly what StemConv expects with ggml's dim0 mapped
// to torch's W (frequency) and dim1 to torch's H (time).

#include "iaamt.h"
#include "iaamt-graph.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static void dbg_mark(iaamt_context * ctx, ggml_cgraph * gf, const char * name, ggml_tensor * t) {
    if (!ctx->debug) {
        return;
    }
    ggml_set_name(t, name);
    ggml_set_output(t);
    ggml_build_forward_expand(gf, t);
    ctx->dbg.emplace_back(name, t);
}

static ggml_tensor * add_bias(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * b) {
    if (!b) {
        return cur;
    }
    return ggml_add(ctx, cur, b);
}

// bias over the channel axis of a [W, H, C, N] tensor
ggml_tensor * iaamt_add_bias_chan(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * b) {
    return ggml_add(ctx, cur, ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1));
}

ggml_tensor * iaamt_rms_norm_mul(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * gamma) {
    cur = ggml_rms_norm(ctx, cur, IAAMT_RMS_EPS);
    return ggml_mul(ctx, cur, gamma);
}

ggml_tensor * iaamt_layer_norm(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * w, ggml_tensor * b) {
    cur = ggml_norm(ctx, cur, IAAMT_NORM_EPS);
    cur = ggml_mul(ctx, cur, w);
    return ggml_add(ctx, cur, b);
}

ggml_tensor * iaamt_linear(ggml_context * ctx, ggml_tensor * w, ggml_tensor * b, ggml_tensor * x) {
    return add_bias(ctx, ggml_mul_mat(ctx, w, x), b);
}

// Conv2d + GroupNorm(4) [+ GELU]
ggml_tensor * iaamt_conv_block(ggml_context * ctx,
                               const iaamt_stem_block & blk,
                         ggml_tensor * cur,
                         int s0, int s1,
                         bool act) {
    cur = ggml_conv_2d(ctx, blk.conv_w, cur, s0, s1, 1, 1, 1, 1);
    cur = iaamt_add_bias_chan(ctx, cur, blk.conv_b);
    cur = ggml_group_norm(ctx, cur, 4, IAAMT_NORM_EPS);
    cur = ggml_mul(ctx, cur, ggml_reshape_4d(ctx, blk.norm_w, 1, 1, blk.norm_w->ne[0], 1));
    cur = ggml_add(ctx, cur, ggml_reshape_4d(ctx, blk.norm_b, 1, 1, blk.norm_b->ne[0], 1));
    if (act) {
        cur = ggml_gelu_erf(ctx, cur);
    }
    return cur;
}

// TaskFeatureAdapter: features + net(features)
ggml_tensor * iaamt_adapter_apply(ggml_context * ctx, const iaamt_adapter & ad, ggml_tensor * x) {
    ggml_tensor * h = iaamt_layer_norm(ctx, x, ad.norm_w, ad.norm_b);
    h = iaamt_linear(ctx, ad.fc1_w, ad.fc1_b, h);
    h = ggml_gelu_erf(ctx, h);
    h = iaamt_linear(ctx, ad.fc2_w, ad.fc2_b, h);
    return ggml_add(ctx, x, h);
}

// Self-attention with RoPE and a per-head sigmoid gate, then the FFN.
// `cur` is [D, S, B]; `pos` carries S positions.
ggml_tensor * iaamt_transformer_block(ggml_context * ctx,
                                const iaamt_attn & at,
                                ggml_tensor * cur,
                                ggml_tensor * pos,
                                int n_head,
                                int head_dim) {
    const int64_t D = cur->ne[0];
    const int64_t S = cur->ne[1];
    const int64_t B = cur->ne[2];

    // ---- attention ----
    {
        ggml_tensor * inp = cur;

        // norm_q feeds both the queries and the gates; norm_context feeds K/V.
        ggml_tensor * h = iaamt_rms_norm_mul(ctx, inp, at.norm_q);
        ggml_tensor * c = iaamt_rms_norm_mul(ctx, inp, at.norm_kv);

        ggml_tensor * q = iaamt_linear(ctx, at.q_w, at.q_b, h);
        ggml_tensor * k = iaamt_linear(ctx, at.k_w, at.k_b, c);
        ggml_tensor * v = iaamt_linear(ctx, at.v_w, at.v_b, c);

        // [hd*nh, S, B] -> [hd, nh, S, B] so ggml_rope sees S on ne2
        q = ggml_reshape_4d(ctx, q, head_dim, n_head, S, B);
        k = ggml_reshape_4d(ctx, k, head_dim, n_head, S, B);
        v = ggml_reshape_4d(ctx, v, head_dim, n_head, S, B);

        q = ggml_rope(ctx, q, pos, head_dim, GGML_ROPE_TYPE_NORMAL);
        k = ggml_rope(ctx, k, pos, head_dim, GGML_ROPE_TYPE_NORMAL);

        ggml_tensor * Q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3)); // [hd, S, nh, B]
        ggml_tensor * K = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
        ggml_tensor * V = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3)); // [S, hd, nh, B]

        ggml_tensor * kq = ggml_mul_mat(ctx, K, Q);                        // [S, S, nh, B]
        kq = ggml_soft_max_ext(ctx, kq, nullptr, 1.0f / sqrtf((float) head_dim), 0.0f);

        ggml_tensor * kqv = ggml_mul_mat(ctx, V, kq);                      // [hd, S, nh, B]

        // per-head sigmoid gate, computed from the normalized query input
        ggml_tensor * g = ggml_sigmoid(ctx, iaamt_linear(ctx, at.gate_w, at.gate_b, h)); // [nh, S, B]
        g = ggml_cont(ctx, ggml_permute(ctx, g, 1, 0, 2, 3));              // [S, nh, B]
        g = ggml_reshape_4d(ctx, g, 1, S, n_head, B);
        kqv = ggml_mul(ctx, kqv, g);

        kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));          // [hd, nh, S, B]
        kqv = ggml_reshape_3d(ctx, kqv, head_dim * n_head, S, B);

        cur = ggml_add(ctx, inp, iaamt_linear(ctx, at.out_w, at.out_b, kqv));
    }

    // ---- feed forward ----
    {
        ggml_tensor * h = iaamt_rms_norm_mul(ctx, cur, at.ffn_norm);
        h = iaamt_linear(ctx, at.ffn_up_w, at.ffn_up_b, h);
        h = ggml_gelu_erf(ctx, h);
        h = iaamt_linear(ctx, at.ffn_down_w, at.ffn_down_b, h);
        cur = ggml_add(ctx, cur, h);
    }

    GGML_UNUSED(D);
    return cur;
}

namespace {

ggml_cgraph * build_graph(iaamt_context * ctx0, int n_frames, int crop_length) {
    const iaamt_model   & model = *ctx0->model;
    const iaamt_hparams & hp    = model.hparams;

    const int D  = hp.token_dim();
    const int P  = hp.pitch_query_count;
    const int nh = hp.n_head;
    const int hd = hp.head_dim();
    const int S  = hp.num_pitch_slots;

    ggml_init_params params = {
        /* mem_size   = */ ctx0->buf_compute_meta.size(),
        /* mem_buffer = */ ctx0->buf_compute_meta.data(),
        /* no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    ggml_cgraph  * gf  = ggml_new_graph_custom(ctx, 16384, false);

    // ---- inputs ----
    ctx0->inp_feats = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
                                         hp.cqt_n_bins, n_frames, hp.num_audio_channels, 1);
    ggml_set_name(ctx0->inp_feats, "feats");
    ggml_set_input(ctx0->inp_feats);

    ctx0->inp_pitch = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, P);
    ggml_set_name(ctx0->inp_pitch, "pitch_feats");
    ggml_set_input(ctx0->inp_pitch);

    ggml_tensor * cur = ctx0->inp_feats;

    // ---- conv stem ----
    cur = ggml_conv_2d(ctx, model.stem_conv1_w, cur, 1, 1, 3, 3, 1, 1);
    cur = iaamt_add_bias_chan(ctx, cur, model.stem_conv1_b);
    cur = ggml_add(ctx, cur, model.stem_freq_embd);          // [F, 1, base_ch, 1]
    cur = ggml_conv_2d(ctx, model.stem_conv2_w, cur, 1, 1, 2, 2, 1, 1);
    cur = iaamt_add_bias_chan(ctx, cur, model.stem_conv2_b);

    // torch strides are (time, freq); ggml dim0 is freq, dim1 is time
    cur = iaamt_conv_block(ctx, model.stem_blocks[0], cur, 1, 2, true);
    cur = iaamt_conv_block(ctx, model.stem_blocks[1], cur, 2, 2, true);
    cur = iaamt_conv_block(ctx, model.stem_blocks[2], cur, 2, 2, true);
    cur = iaamt_conv_block(ctx, model.stem_blocks[3], cur, 1, 1, false);
    dbg_mark(ctx0, gf, "stem", cur);

    const int64_t n_band = cur->ne[0];
    const int64_t T2     = cur->ne[1];   // frames after the stem (T/8)

    // "b d t f -> b t f d": [F', T', D] -> [D, F', T']
    cur = ggml_cont(ctx, ggml_permute(ctx, cur, 1, 2, 0, 3));

    // ---- tokens ----
    ggml_tensor * band = ggml_add(ctx, cur, model.band_type_embd);

    ggml_tensor * pq = iaamt_linear(ctx, model.pitch_query_fc1_w, model.pitch_query_fc1_b, ctx0->inp_pitch);
    pq = ggml_gelu_erf(ctx, pq);
    pq = iaamt_linear(ctx, model.pitch_query_fc2_w, model.pitch_query_fc2_b, pq);   // [D, P]
    pq = ggml_add(ctx, pq, model.pitch_type_embd);
    pq = ggml_repeat_4d(ctx, ggml_reshape_3d(ctx, pq, D, P, 1), D, P, T2, 1);

    cur = ggml_concat(ctx, band, pq, 1);                     // [D, K, T']
    dbg_mark(ctx0, gf, "tokens", cur);
    const int64_t K = cur->ne[1];

    ctx0->inp_pos_band = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, K);
    ggml_set_name(ctx0->inp_pos_band, "pos_band");
    ggml_set_input(ctx0->inp_pos_band);

    ctx0->inp_pos_time = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T2);
    ggml_set_name(ctx0->inp_pos_time, "pos_time");
    ggml_set_input(ctx0->inp_pos_time);

    // ---- dual-axis transformer ----
    for (int il = 0; il < hp.n_layer; ++il) {
        // band axis: sequence over tokens, batch over time
        cur = iaamt_transformer_block(ctx, model.layers[il].band, cur, ctx0->inp_pos_band, nh, hd);
        dbg_mark(ctx0, gf, ("band" + std::to_string(il)).c_str(), cur);

        // time axis: [D, K, T'] -> [D, T', K]
        cur = ggml_cont(ctx, ggml_permute(ctx, cur, 0, 2, 1, 3));
        cur = iaamt_transformer_block(ctx, model.layers[il].time, cur, ctx0->inp_pos_time, nh, hd);
        cur = ggml_cont(ctx, ggml_permute(ctx, cur, 0, 2, 1, 3));
        dbg_mark(ctx0, gf, ("layer" + std::to_string(il)).c_str(), cur);
    }

    cur = iaamt_rms_norm_mul(ctx, cur, model.output_norm);
    dbg_mark(ctx0, gf, "final_norm", cur);

    // ---- pitch branch ----
    ggml_tensor * pitch = ggml_cont(ctx,
        ggml_view_3d(ctx, cur, D, P, T2, cur->nb[1], cur->nb[2], n_band * cur->nb[1]));

    // ConvTranspose1d(k=8, s=8) has no kernel overlap, so it is exactly a
    // matmul that fans each frame out into 8.  ggml_conv_transpose_1d would
    // work too, but it only accepts a 2D input and we carry 88 pitch tracks.
    //
    //   out[co, t*8 + k, p] = sum_ci x[ci, t, p] * W[k, co, ci]
    {
        const int64_t up = model.up_conv_w->ne[0];                 // kernel == stride
        ggml_tensor * w = ggml_reshape_2d(ctx, model.up_conv_w, up * D, D);
        w = ggml_cont(ctx, ggml_transpose(ctx, w));                // [D(in), up*D(out)]

        // the view above is [D, P, T']; the reshape below needs time on ne1
        pitch = ggml_cont(ctx, ggml_permute(ctx, pitch, 0, 2, 1, 3));   // [D, T', P]
        pitch = ggml_mul_mat(ctx, w, pitch);                       // [up*D, T', P]
        pitch = ggml_reshape_4d(ctx, pitch, up, D, T2, P);
        pitch = ggml_cont(ctx, ggml_permute(ctx, pitch, 1, 0, 2, 3));   // [D, up, T', P]
        pitch = ggml_reshape_3d(ctx, pitch, D, up * T2, P);        // [D, T'*8, P]
        pitch = ggml_add(ctx, pitch, model.up_conv_b);
    }

    dbg_mark(ctx0, gf, "upconv", pitch);

    // _match_time_length: trim to the audio frame count
    const int64_t crop = std::min<int64_t>(crop_length, pitch->ne[1]);
    pitch = ggml_cont(ctx,
        ggml_view_3d(ctx, pitch, D, crop, P, pitch->nb[1], pitch->nb[2], 0));

    if (hp.model_type == IAAMT_MODEL_TYPE_VELOCITY) {
        // The velocity head queries these frames per note on the CPU.
        ctx0->out_feat = pitch;
        ggml_set_name(ctx0->out_feat, "pitch_features");
        ggml_set_output(ctx0->out_feat);
        ggml_build_forward_expand(gf, ctx0->out_feat);
        ggml_free(ctx);
        return gf;
    }

    // ---- transcription head ----
    pitch = iaamt_adapter_apply(ctx, model.interval_adapter, pitch);

    if (S > 1 && model.slot_embd) {
        // features[..., p, :] + slot_embedding[s] -> track index p*S + s
        ggml_tensor * f4 = ggml_reshape_4d(ctx, pitch, D, crop, 1, P);
        f4 = ggml_repeat_4d(ctx, f4, D, crop, S, P);
        f4 = ggml_add(ctx, f4, ggml_reshape_4d(ctx, model.slot_embd, D, 1, S, 1));
        pitch = ggml_reshape_3d(ctx, ggml_cont(ctx, f4), D, crop, (int64_t) S * P);
    }

    ctx0->out_feat = pitch;
    ggml_set_name(ctx0->out_feat, "interval_features");
    ggml_set_output(ctx0->out_feat);

    ctx0->out_proj = iaamt_linear(ctx, model.interval_scorer_w, model.interval_scorer_b, pitch);
    ggml_set_name(ctx0->out_proj, "interval_proj");
    ggml_set_output(ctx0->out_proj);

    ggml_build_forward_expand(gf, ctx0->out_proj);
    ggml_build_forward_expand(gf, ctx0->out_feat);

    ggml_free(ctx);
    return gf;
}

} // namespace

iaamt_context * iaamt_ctx_init(iaamt_model & model, int n_threads) {
    iaamt_context * ctx = new iaamt_context();
    ctx->model     = &model;
    ctx->n_threads = n_threads;
    ctx->buf_compute_meta.resize(16384 * ggml_tensor_overhead() + ggml_graph_overhead_custom(16384, false));

    // The scheduler keeps a CPU fallback so ops the accelerator lacks still run.
    std::vector<ggml_backend_t>             backends;
    std::vector<ggml_backend_buffer_type_t> bufts;
    if (model.backend != model.backend_cpu) {
        backends.push_back(model.backend);
        bufts.push_back(ggml_backend_get_default_buffer_type(model.backend));
    }
    backends.push_back(model.backend_cpu);
    bufts.push_back(ggml_backend_get_default_buffer_type(model.backend_cpu));

    ctx->sched = ggml_backend_sched_new(backends.data(), bufts.data(),
                                        (int) backends.size(), 16384, false, true);
    return ctx;
}

void iaamt_ctx_set_debug(iaamt_context * ctx, bool enable) {
    ctx->debug = enable;
}

void iaamt_ctx_free(iaamt_context * ctx) {
    if (!ctx) {
        return;
    }
    if (ctx->sched) {
        ggml_backend_sched_free(ctx->sched);
    }
    delete ctx;
}

bool iaamt_forward(iaamt_context * ctx,
                   const std::vector<float> & feats,
                   int n_frames_in,
                   int crop_length,
                   iaamt_window_out & out,
                   std::string & err) {
    const iaamt_model   & model = *ctx->model;
    const iaamt_hparams & hp    = model.hparams;

    ctx->dbg.clear();
    ggml_cgraph * gf = build_graph(ctx, n_frames_in, crop_length);
    if (!gf) {
        err = "failed to build the compute graph";
        return false;
    }
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        err = "failed to allocate the compute graph";
        return false;
    }

    // ---- upload inputs ----
    {
        const size_t want = (size_t) hp.cqt_n_bins * n_frames_in * hp.num_audio_channels;
        if (feats.size() != want) {
            err = "HCQT feature buffer has the wrong size";
            return false;
        }
        ggml_backend_tensor_set(ctx->inp_feats, feats.data(), 0, want * sizeof(float));
    }
    {
        // PitchQueryEmbedding input: [p/128, sin(2*pi*p/12), cos(2*pi*p/12), 1]
        const int P = hp.pitch_query_count;
        std::vector<float> pf((size_t) 4 * P);
        for (int i = 0; i < P; ++i) {
            const float p = 21.0f + (float) i;
            pf[4*i + 0] = p / 128.0f;
            pf[4*i + 1] = sinf(2.0f * (float) M_PI * p / 12.0f);
            pf[4*i + 2] = cosf(2.0f * (float) M_PI * p / 12.0f);
            pf[4*i + 3] = 1.0f;
        }
        ggml_backend_tensor_set(ctx->inp_pitch, pf.data(), 0, pf.size() * sizeof(float));
    }
    for (ggml_tensor * pos : { ctx->inp_pos_band, ctx->inp_pos_time }) {
        std::vector<int32_t> ids(pos->ne[0]);
        for (int64_t i = 0; i < pos->ne[0]; ++i) {
            ids[i] = (int32_t) i;
        }
        ggml_backend_tensor_set(pos, ids.data(), 0, ids.size() * sizeof(int32_t));
    }

    ggml_backend_cpu_set_n_threads(model.backend_cpu, ctx->n_threads);
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        err = "graph evaluation failed";
        return false;
    }

    // ---- read back ----
    if (hp.model_type == IAAMT_MODEL_TYPE_VELOCITY) {
        ggml_tensor * pf = ctx->out_feat;          // [D, crop, P]
        out.n_frames = (int) pf->ne[1];
        out.n_tracks = (int) pf->ne[2];
        out.feat_dim = (int) pf->ne[0];
        out.pitch_features.resize((size_t) ggml_nelements(pf));
        ggml_backend_tensor_get(pf, out.pitch_features.data(), 0, ggml_nbytes(pf));
        return true;
    }

    ggml_tensor * proj = ctx->out_proj;   // [513, crop, tracks]
    ggml_tensor * feat = ctx->out_feat;   // [D,   crop, tracks]

    const int hdim   = hp.semi_crf_head_dim;
    const int crop   = (int) proj->ne[1];
    const int tracks = (int) proj->ne[2];

    out.n_frames = crop;
    out.n_tracks = tracks;
    out.head_dim = hdim;
    out.feat_dim = (int) feat->ne[0];

    std::vector<float> raw((size_t) ggml_nelements(proj));
    ggml_backend_tensor_get(proj, raw.data(), 0, ggml_nbytes(proj));

    out.query.resize((size_t) tracks * crop * hdim);
    out.key.resize((size_t) tracks * crop * hdim);
    out.diag.resize((size_t) tracks * crop);

    // IntervalScorer scales the query by 1/sqrt(head_dim)
    const float qscale = 1.0f / sqrtf((float) hdim);
    const int   row    = 2 * hdim + 1;
    for (int t = 0; t < tracks; ++t) {
        for (int f = 0; f < crop; ++f) {
            const float * src = &raw[((size_t) t * crop + f) * row];
            float * dq = &out.query[((size_t) t * crop + f) * hdim];
            float * dk = &out.key  [((size_t) t * crop + f) * hdim];
            for (int d = 0; d < hdim; ++d) {
                dq[d] = src[d] * qscale;
                dk[d] = src[hdim + d];
            }
            out.diag[(size_t) t * crop + f] = src[2 * hdim];
        }
    }

    if (hp.use_interval_boundary_head) {
        out.interval_features.resize((size_t) ggml_nelements(feat));
        ggml_backend_tensor_get(feat, out.interval_features.data(), 0, ggml_nbytes(feat));
    }

    for (const auto & [name, t] : ctx->dbg) {
        iaamt_window_out::debug_tensor d;
        d.name = name;
        for (int i = GGML_MAX_DIMS - 1; i >= 0; --i) {
            if (t->ne[i] > 1 || !d.ne.empty()) {
                d.ne.push_back(t->ne[i]);
            }
        }
        d.data.resize((size_t) ggml_nelements(t));
        ggml_backend_tensor_get(t, d.data.data(), 0, ggml_nbytes(t));
        out.debug.push_back(std::move(d));
    }

    return true;
}
