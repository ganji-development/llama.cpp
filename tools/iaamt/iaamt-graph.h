// Internal graph state and building blocks shared by the audio and MIDI-roll
// models.  Not part of the public interface in iaamt.h.

#pragma once

#include "iaamt.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <string>
#include <utility>
#include <vector>

struct iaamt_context {
    iaamt_model * model = nullptr;
    int n_threads = 4;

    ggml_backend_sched_t sched = nullptr;

    // graph inputs
    ggml_tensor * inp_feats     = nullptr;   // HCQT features, or the MIDI roll
    ggml_tensor * inp_pitch     = nullptr;   // [4, P] pitch-query features
    ggml_tensor * inp_pos_band  = nullptr;   // token-axis positions
    ggml_tensor * inp_pos_time  = nullptr;   // time-axis positions

    // graph outputs
    ggml_tensor * out_proj      = nullptr;
    ggml_tensor * out_feat      = nullptr;
    std::vector<ggml_tensor *> bc_out;       // beat/chord head projections

    bool debug = false;
    std::vector<std::pair<std::string, ggml_tensor *>> dbg;

    std::vector<uint8_t> buf_compute_meta;
};

// RMSNorm here is x / max(||x||_2, eps) * sqrt(D) * gamma, which is plain RMS
// normalization; eps only floors near-zero rows and is 2^-24 in the reference.
constexpr float IAAMT_RMS_EPS  = 1e-12f;
constexpr float IAAMT_NORM_EPS = 1e-5f;   // torch LayerNorm / GroupNorm default

ggml_tensor * iaamt_add_bias_chan(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * b);
ggml_tensor * iaamt_rms_norm_mul (ggml_context * ctx, ggml_tensor * cur, ggml_tensor * gamma);
ggml_tensor * iaamt_layer_norm   (ggml_context * ctx, ggml_tensor * cur,
                                  ggml_tensor * w, ggml_tensor * b);
ggml_tensor * iaamt_linear       (ggml_context * ctx, ggml_tensor * w, ggml_tensor * b,
                                  ggml_tensor * x);

// Conv2d + GroupNorm(4) [+ GELU]; s0 strides ne0, s1 strides ne1.
ggml_tensor * iaamt_conv_block(ggml_context * ctx, const iaamt_stem_block & blk,
                               ggml_tensor * cur, int s0, int s1, bool act);

// LayerNorm + Linear + GELU + Linear wrapped in a residual.
ggml_tensor * iaamt_adapter_apply(ggml_context * ctx, const iaamt_adapter & ad,
                                  ggml_tensor * x);

// Self-attention with RoPE and a per-head sigmoid gate, then the FFN.
// `cur` is [D, S, B]; `pos` carries S positions.
ggml_tensor * iaamt_transformer_block(ggml_context * ctx, const iaamt_attn & at,
                                      ggml_tensor * cur, ggml_tensor * pos,
                                      int n_head, int head_dim);

ggml_cgraph * iaamt_bc_build_graph(iaamt_context * ctx0, int n_frames);
