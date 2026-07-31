# MOSS-Music GGUF Conversion

This document describes how to convert **MOSS-Music-8B-Thinking** to GGUF.

MOSS-Music is an **audio-understanding** model (audio in, text out). It is a
different architecture from MOSS-TTS-Delay, which is a text-to-speech model —
the two share nothing but a name prefix.

> [!IMPORTANT]
> Conversion is complete and verified. **Audio inference is not implemented yet.**
> The main GGUF runs as a text-only Qwen3 model today. See
> [Runtime status](#runtime-status) before using the mmproj file.

## Architecture

| Component | Shape | Goes to |
|-----------|-------|---------|
| Qwen3 backbone | 36 layers, hidden 4096, ffn 12288, 32 heads / 8 KV | main `.gguf` |
| Conv2d audio stem | 3 × `Conv2d(3×3, stride 2, pad 1)`, 480 channels | `mmproj-*.gguf` |
| Stem projection | `Linear(7680 → 1280)` | `mmproj-*.gguf` |
| Whisper encoder | 32 layers, d_model 1280, ffn 5120, 20 heads | `mmproj-*.gguf` |
| `audio_adapter` | GatedMLP `1280 → 8192 → 4096` | `mmproj-*.gguf` |
| `deepstack_audio_merger_list` | 3 × GatedMLP `1280 → 8192 → 4096` | `mmproj-*.gguf` |

Two details separate MOSS-Music from the Whisper-based audio models already
supported (Ultravox, Qwen2-Audio, Voxtral):

1. **The stem is Conv2d, not Conv1d.** Whisper convolves over time with the mel
   bins as channels. MOSS-Music treats the mel spectrogram as a 2D image
   (`[B, 1, n_mels, T]`) and downsamples both frequency and time by 8×. The
   480 channels × 16 remaining mel bins are then flattened to 7680 and projected
   to `d_model`.

2. **DeepStack injects into the LLM, not the encoder.** Qwen3-VL's DeepStack is
   internal to the vision tower. MOSS-Music takes encoder hidden states from
   layers `[8, 16, 24]`, runs each through its own GatedMLP, and **adds** the
   result to the output of decoder layers 0, 1 and 2 — at the audio token
   positions only. In the reference implementation this is done with PyTorch
   forward hooks (`_register_llm_deepstack_hooks` in `modeling_moss_music.py`).

`audio_encoder.out_proj` is `nn.Identity()` because `output_dim == d_model == 1280`,
so it contributes no weights.

## Usage

```bash
# LLM backbone -> Moss-Music-8.2B-F16.gguf
python convert_hf_to_gguf.py /path/to/MOSS-Music-8B-Thinking --outtype f16 --outfile ./out/

# audio encoder + projectors -> mmproj-Moss-Music-F16.gguf
python convert_hf_to_gguf.py /path/to/MOSS-Music-8B-Thinking --mmproj --outtype f16 --outfile ./out/
```

Both commands read the same checkpoint and partition its 902 tensors with no
overlap: 399 to the backbone, 503 to the mmproj.

The text model registers as architecture `moss-music`. Its backbone is byte-for-byte
a Qwen3 stack, so it reuses `llm_build_qwen3`; the distinct architecture name exists
so the runtime can tell that this checkpoint expects an audio mmproj.

## Tensor naming

| HF name | GGUF name |
|---------|-----------|
| `language_model.embed_tokens.weight` | `token_embd.weight` |
| `language_model.layers.{i}.*` | `blk.{i}.*` (standard Qwen3) |
| `language_model.norm.weight` | `output_norm.weight` |
| `lm_head.weight` | `output.weight` |
| `audio_encoder.conv{1,2,3}.*` | `a.conv2d.{1,2,3}.*` |
| `audio_encoder.stem_proj.*` | `a.stem_proj.*` |
| `audio_encoder.embed_positions` | `a.position_embd.weight` (precomputed, see below) |
| `audio_encoder.layers.{i}.*` | `a.blk.{i}.*` (shared with Ultravox) |
| `audio_encoder.layer_norm.*` | `a.post_ln.*` |
| `audio_adapter.{gate,up,down}_proj.weight` | `mm.a.{gate,up,down}.weight` |
| `deepstack_audio_merger_list.{k}.{gate,up,down}_proj.weight` | `mm.a.deepstack.{k}.{gate,up,down}.weight` |

### Layout conventions

**Conv2d weights pass through unchanged.** Torch stores them as
`[OC, IC, KH, KW]`; GGUF reverses the dimension order, giving
`ne = [KW, KH, IC, OC]` — exactly `ggml_conv_2d`'s expected kernel layout.

**Conv2d biases are reshaped to `[1, 1, OC]`** so they broadcast over the
frequency and time axes of `ggml_conv_2d`'s `[OW, OH, OC, N]` output. (Whisper's
Conv1d path unsqueezes to `[1, OC]` for the same reason.)

**Position embeddings are precomputed.** `SinusoidsPositionEmbedding` is a
non-trainable buffer that the reference model rebuilds on every forward pass.
Conversion emits the full `[1280, 1500]` table instead, so the runtime can slice
the first `seq_len` rows. The checkpoint's `inv_timescales` buffer is skipped —
the table is regenerated from the same formula. This mirrors what Qwen2.5-Omni
already does.

## Metadata

Standard audio keys (`clip.audio.embedding_length`, `.block_count`,
`.attention.head_count`, `.num_mel_bins`, …) are written as usual, with
`clip.projector_type = moss_music`. MOSS-Music adds:

| Key | Value | Meaning |
|-----|-------|---------|
| `clip.audio.downsample_rate` | 8 | total stem downsampling (2³) |
| `clip.audio.num_conv2d_layers` | 3 | conv layers in the stem |
| `clip.audio.adapter_hidden_size` | 8192 | GatedMLP hidden dim |
| `clip.audio.deepstack_layer_indexes` | `[8, 16, 24]` | **encoder** layers the mergers read from |
| `clip.audio.deepstack_num_inject` | 3 | number of **decoder** layers injected into (0 … n−1) |

The two DeepStack keys index different stacks — the first is source layers in the
audio encoder, the second is how many decoder layers receive an injection.

## Runtime status

Audio inference works. Run it with:

```bash
llama-mtmd-cli -m Moss-Music-8.2B-F16.gguf \
               --mmproj mmproj-Moss-Music-F16.gguf \
               --audio clip.mp3 -p "Describe this music."
```

### How the pieces fit

**Audio graph** — `tools/mtmd/models/moss-music.cpp`, projector type `moss_music`.
Conv2d stem, permute-and-flatten, `stem_proj`, sinusoid positions, the shared
Whisper layer stack via `build_vit`, final layer norm, GatedMLP adapter.

**DeepStack** reuses the mechanism Qwen3-VL already established, so it needed no
new core API. The mmproj emits one widened row per audio token:

```
[ audio_adapter | deepstack_0 | deepstack_1 | deepstack_2 ]   (4 x 4096 = 16384)
```

`llama_hparams::n_embd_inp()` widens the embedding row by
`moss-music.n_deepstack_layers`, `llama_batch.embd` carries it unchanged, and
`llm_build_moss_music` slices sub-block `k+1` back out and adds it to the output
of decoder layer `k` — reproducing the forward hooks the reference installs.
Text tokens are zero-padded to the same width by `build_inp_embd`, so the add is
a no-op for them and no audio-position mask is required.

> [!IMPORTANT]
> The main GGUF and the mmproj must be converted from the same checkpoint by the
> same version of the converter. `clip_n_mmproj_embd` and
> `llama_model_n_embd_inp` must agree (16384); mtmd refuses the pair otherwise.
> A main GGUF converted before `n_deepstack_layers` was emitted will be rejected.

**Preprocessing** — `mtmd_audio_preprocessor_moss`. Whisper mel parameters, but
HF's `center=True` / `pad_mode="reflect"` framing with no 30 s silence pad and no
30 s truncation, matching `WhisperFeatureExtractor`. Audio is chunked at 120 s,
the limit implied by `max_source_positions=1500` at 12.5 tokens/s.

### Not implemented

Time-marker interleaving (`enable_time_marker=True` in the reference processor),
which splices digit tokens between audio spans. It is off by default. Audio
*generation* is out of scope — this model is audio-understanding only.
