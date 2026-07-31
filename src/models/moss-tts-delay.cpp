#include "models.h"

namespace {

class llm_graph_input_moss_audio_channel : public llm_graph_input_i {
public:
    llm_graph_input_moss_audio_channel(uint32_t channel, uint32_t n_channels)
        : channel(channel), n_channels(n_channels) {}

    void set_input(const llama_ubatch * ubatch) override {
        GGML_ASSERT(tokens != nullptr);

        data.resize(ubatch->n_tokens, 0);
        if (ubatch->token_audio != nullptr) {
            GGML_ASSERT(ubatch->n_token_audio == n_channels);

            for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
                data[i] = ubatch->token_audio[(size_t) i*n_channels + channel];
            }
        }

        ggml_backend_tensor_set(tokens, data.data(), 0, data.size()*ggml_element_size(tokens));
    }

    bool can_reuse(const llm_graph_params & params) override {
        return
            tokens != nullptr &&
            tokens->ne[0] == params.ubatch.n_tokens &&
            (
                (params.ubatch.n_token_audio == n_channels && params.ubatch.token_audio != nullptr) ||
                (params.ubatch.n_token_audio == 0 && params.ubatch.token_audio == nullptr)
            );
    }

    ggml_tensor * tokens = nullptr;

private:
    const uint32_t channel;
    const uint32_t n_channels;
    std::vector<llama_token> data;
};

}

void llama_model_moss_tts_delay::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_N_VQ,             hparams.n_vq);
    ml.get_key(LLM_KV_AUDIO_VOCAB_SIZE, hparams.audio_vocab_size);
    ml.get_key(LLM_KV_AUDIO_PAD_CODE,   hparams.audio_pad_code);

    ml.get_key(LLM_KV_AUDIO_START_TOKEN_ID,                 hparams.audio_start_token_id,                false);
    ml.get_key(LLM_KV_AUDIO_END_TOKEN_ID,                   hparams.audio_end_token_id,                  false);
    ml.get_key(LLM_KV_AUDIO_USER_SLOT_TOKEN_ID,             hparams.audio_user_slot_token_id,            false);
    ml.get_key(LLM_KV_AUDIO_ASSISTANT_GEN_SLOT_TOKEN_ID,    hparams.audio_assistant_gen_slot_token_id,   false);
    ml.get_key(LLM_KV_AUDIO_ASSISTANT_DELAY_SLOT_TOKEN_ID,  hparams.audio_assistant_delay_slot_token_id, false);
    ml.get_key(LLM_KV_SAMPLING_RATE,                        hparams.sampling_rate,                       false);

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    switch (hparams.n_layer()) {
        case 28: type = hparams.n_embd == 1024 ? LLM_TYPE_0_6B : LLM_TYPE_1_7B; break;
        case 36: type = hparams.n_embd == 2560 ? LLM_TYPE_4B : LLM_TYPE_8B; break;
        case 40: type = LLM_TYPE_14B; break;
        case 64: type = LLM_TYPE_32B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_moss_tts_delay::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    const int64_t n_audio_vocab =
        hparams.audio_vocab_size > 0 ? std::max<int64_t>(hparams.audio_vocab_size + 1, hparams.audio_pad_code + 1) : 0;

    if (hparams.n_vq == 0) {
        throw std::runtime_error("n_vq must be > 0 for MOSS_TTS_DELAY");
    }
    if (n_audio_vocab == 0) {
        throw std::runtime_error("audio_vocab_size must be > 0 for MOSS_TTS_DELAY");
    }

    tok_embd_audio.resize(hparams.n_vq);
    output_audio.resize(hparams.n_vq);

    for (uint32_t i = 0; i < hparams.n_vq; ++i) {
        tok_embd_audio[i] = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_AUDIO, "weight", -1, i), {n_embd, n_audio_vocab}, 0);
    }

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    for (uint32_t i = 0; i < hparams.n_vq; ++i) {
        output_audio[i] = create_tensor(tn(LLM_TENSOR_OUTPUT_AUDIO, "weight", -1, i), {n_embd, n_audio_vocab}, 0);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_gqa, n_embd_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_moss_tts_delay::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_moss_tts_delay::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);
    GGML_ASSERT(hparams.n_vq == model.tok_embd_audio.size());

    ggml_tensor * cur;
    ggml_tensor * inpL = build_inp_embd(model.tok_embd);

    GGML_ASSERT(ubatch.token != nullptr);
    GGML_ASSERT(
        (ubatch.token_audio != nullptr && ubatch.n_token_audio == hparams.n_vq) ||
        (ubatch.token_audio == nullptr && ubatch.n_token_audio == 0));

    for (uint32_t i = 0; i < hparams.n_vq; ++i) {
        auto inp_audio = std::make_unique<llm_graph_input_moss_audio_channel>(i, hparams.n_vq);
        inp_audio->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_tokens);
        cb(inp_audio->tokens, "inp_audio_tokens", i);
        ggml_set_input(inp_audio->tokens);

        ggml_tensor * audio_embd = ggml_get_rows(ctx0, model.tok_embd_audio[i], inp_audio->tokens);
        cb(audio_embd, "audio_embd", i);

        inpL = ggml_add(ctx0, inpL, audio_embd);
        cb(inpL, "input_sum", i);

        res->add_input(std::move(inp_audio));
    }

    ggml_tensor * inp_pos = build_inp_pos();
    auto * inp_attn = build_attn_inp_kv();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL,
                model.layers[il].attn_norm, nullptr,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        {
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head, n_head_kv, il);

            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
            cb(Kcur, "Kcur_normed", il);

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur,   inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, nullptr,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   nullptr, nullptr,
                model.layers[il].ffn_gate, nullptr, nullptr,
                model.layers[il].ffn_down, nullptr, nullptr,
                nullptr,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = build_norm(inpL,
            model.output_norm, nullptr,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    GGML_ASSERT(hparams.n_vq == model.output_audio.size());

    ggml_tensor * logits = build_lora_mm(model.output, cur);
    cb(logits, "result_output_text", -1);

    for (uint32_t i = 0; i < hparams.n_vq; ++i) {
        ggml_tensor * audio_logits = build_lora_mm(model.output_audio[i], cur);
        cb(audio_logits, "result_output_audio", i);

        logits = ggml_concat(ctx0, logits, audio_logits, 0);
        cb(logits, "result_output_concat", i);
    }

    logits = ggml_cont(ctx0, logits);
    cb(logits, "result_output_cont", -1);

    res->t_logits = logits;
    cb(logits, "result_output", -1);

    ggml_build_forward_expand(gf, logits);
}
