#include "models.h"

// MOSS-Music audio tower.
//
// Unlike Whisper/Ultravox (Conv1d over time, mel bins as channels), MOSS-Music
// treats the mel spectrogram as a 2D image and downsamples frequency and time
// by 8x with three Conv2d layers, then flattens channel x frequency and
// projects to d_model. Reference: MossMusicEncoder in modeling_moss_music.py.
//
// The output is a widened row per audio token:
//     [ audio_adapter | deepstack_0 | deepstack_1 | deepstack_2 ]
// The decoder slices sub-block k+1 back out and adds it to the output of its
// layer k (see llm_build_moss_music), which reproduces the forward hooks the
// reference installs on the LLM decoder layers.
ggml_cgraph * clip_graph_moss_music::build() {
    const int n_frames = img.nx;
    const int n_mel    = img.ny;

    GGML_ASSERT(n_mel == hparams.n_mel_bins);

    // 3x conv2d, kernel 3, stride 2, pad 1 -> ceil(n/2) each time
    auto conv_out_len = [](int n) { return (n - 1) / 2 + 1; };
    const int n_pos = conv_out_len(conv_out_len(conv_out_len(n_frames)));
    const int n_freq = conv_out_len(conv_out_len(conv_out_len(n_mel)));

    GGML_ASSERT(model.position_embeddings->ne[1] >= n_pos);

    // reference builds [B, 1, n_mels, T], i.e. torch H = freq, W = time.
    // ggml_conv_2d wants [W, H, IC, N] and build_inp_raw already gives
    // [n_frames, n_mel, 1] = [time, freq, 1] -- so no transpose here
    // (the conformer stem does transpose, because NeMo orients it the other way).
    ggml_tensor * cur = build_inp_raw(1);

    // conv2d stem
    {
        for (size_t i = 0; i < model.conv2d_w.size(); ++i) {
            cur = ggml_conv_2d(ctx0, model.conv2d_w[i], cur, 2, 2, 1, 1, 1, 1);
            // bias is stored as [1, 1, OC] so it broadcasts over time and freq
            cur = ggml_add(ctx0, cur, model.conv2d_b[i]);
            // nn.GELU() is the exact erf form, not the tanh approximation
            cur = ggml_gelu_erf(ctx0, cur);
            cb(cur, "conv2d", (int) i);
        }

        GGML_ASSERT(cur->ne[0] == n_pos);
        GGML_ASSERT(cur->ne[1] == n_freq);

        // reference: x.permute(0, 3, 1, 2).flatten(2) on [B, C, F, T]
        //         -> [B, T, C, F] -> [B, T, C*F], i.e. index = c*F + f.
        // here cur is [T, F, C]; permute to [F, C, T] so that reshaping
        // ne[0]*ne[1] yields exactly index = c*F + f, one row per frame.
        cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 2, 0, 1, 3));
        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0] * cur->ne[1], cur->ne[2]);
        cb(cur, "stem_flat", -1);

        GGML_ASSERT(cur->ne[0] == model.stem_proj_w->ne[0]);

        cur = ggml_mul_mat(ctx0, model.stem_proj_w, cur);
        cur = ggml_add(ctx0, cur, model.stem_proj_b);
        cb(cur, "stem_proj", -1);
    }

    // sanity check (only check one layer, but it should be the same for all)
    GGML_ASSERT(model.layers[0].ln_1_w && model.layers[0].ln_1_b);
    GGML_ASSERT(model.layers[0].ln_2_w && model.layers[0].ln_2_b);
    GGML_ASSERT(model.layers[0].q_b);
    GGML_ASSERT(model.layers[0].v_b);
    GGML_ASSERT(!model.layers[0].k_b); // no bias for k

    ggml_tensor * pos_embd_selected = ggml_view_2d(
        ctx0, model.position_embeddings,
        model.position_embeddings->ne[0], n_pos,
        model.position_embeddings->nb[1], 0
    );

    // whisper encoder stack; collect every layer output so the deepstack
    // mergers can read the ones they need
    std::vector<ggml_tensor *> layers_out;
    cur = build_vit(
            cur, n_pos,
            NORM_TYPE_NORMAL,
            hparams.ffn_op,
            pos_embd_selected,
            nullptr,
            &layers_out);
    cb(cur, "after_transformer", -1);

    // main audio adapter: GatedMLP, no biases
    ggml_tensor * embeddings = build_ffn(cur,
            model.mm_ffn_up_w,   nullptr,
            model.mm_ffn_gate_w, nullptr,
            model.mm_ffn_down_w, nullptr,
            FFN_SILU, -1);
    cb(embeddings, "audio_adapter", -1);

    // deepstack mergers read pre-final-norm hidden states from the encoder
    // layers named in clip.audio.deepstack_layer_indexes
    for (int k = 0; k < hparams.n_deepstack_audio; ++k) {
        const int il = hparams.deepstack_layer_indexes[k];
        GGML_ASSERT(il >= 0 && il < (int) layers_out.size());

        ggml_tensor * ds = build_ffn(layers_out[il],
                model.ds_up_w[k],   nullptr,
                model.ds_gate_w[k], nullptr,
                model.ds_down_w[k], nullptr,
                FFN_SILU, k);
        cb(ds, "deepstack", k);

        // concat on the feature axis, matching the decoder's slice offsets
        embeddings = ggml_concat(ctx0, embeddings, ds, 0);
    }

    cb(embeddings, "projected", -1);

    ggml_build_forward_expand(gf, embeddings);

    return gf;
}
