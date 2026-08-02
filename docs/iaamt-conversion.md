# Instrument-Agnostic AMT GGUF Conversion

This document describes how to convert the `best_*.pth` checkpoints published by
[instrument-agnostic-amt](https://github.com/anime-song/instrument-agnostic-amt)
to GGUF. That project extends [Transkun](https://github.com/Yujia-Yan/Transkun)'s
Neural Semi-CRF piano transcriber into a general, instrument-agnostic
audio-to-MIDI model.

> [!IMPORTANT]
> Conversion is complete and verified — all 11 published checkpoints round-trip
> bit-exactly. **All three model families run** via `llama-iaamt-cli`. These
> GGUFs are not loadable by `llama-cli` or `llama-server` — they are not
> language models and register no `llm_arch`. See
> [Runtime status](#runtime-status).

## Model families

Three architectures ship under the same `best_*.pth` naming. The converter tells
them apart from the state dict and writes `iaamt.model_type` accordingly.

| `iaamt.model_type` | Checkpoints | Input → output |
|--------------------|-------------|----------------|
| `transcription` | `best_model{,_bass,_bass_v2,_drums,_guitar,_guitar_v1_5,_other,_vocal,_vocal_harmony}.pth` | stereo audio → per-pitch note intervals |
| `velocity` | `best_velocity_model.pth` | stereo audio + note list → per-note MIDI velocity |
| `beat_chord` | `best_beat_chord_key.pth` | MIDI roll → beat / chord / key |

`transcription` and `velocity` share the V1 HCQT backbone verbatim; only the head
differs. `beat_chord` uses a different stem (depthwise conv over a 72-channel MIDI
roll, no CQT) but the same dual-axis Transformer block.

## Architecture

```
Audio [B, 2, T]
   -> AudioFeatureExtractor   Harmonic CQT x 5, 312 bins, 36 bins/octave -> [B, 10, T, 312]
   -> StemConv                2D CNN, time /8 and frequency /4            -> [B, 256, T/8, 78]
   -> Dual-axis Transformer   6 layers x (band-axis + time-axis)
      + PitchQueryEmbedding   88 learned MIDI-pitch query tokens
   -> ConvTranspose1d         time x8 back to frame rate
   -> head                    semi-CRF interval scorer / velocity / beat+chord
```

Each backbone layer holds **two** independent single-layer Transformers, one per
axis. The audio models store them as `[time, band]` and `beat_chord` as
`[pitch, time]`, so the index-to-axis mapping is model-dependent; the converter
names them by role instead, and `bb.blk.{L}.band` is always the token axis.
Both apply the token axis first. All use RMSNorm (weight only, no bias), RoPE on
Q/K, a per-head sigmoid gate on the attention output, and a GELU FFN with a 4x
expansion.

## Usage

The converter needs only `numpy` and the bundled `gguf-py` — it reads
`torch.save` archives directly, so **PyTorch is not required**. It is pure CPU
work: no CUDA, no ROCm, no GPU of any kind, so it runs anywhere including the
same machine that holds the checkpoints.

```bash
# one checkpoint
python convert_iaamt_to_gguf.py models/best_model.pth --outdir ./out

# every best_*.pth in a directory
python convert_iaamt_to_gguf.py models/ --outdir ./out

# half precision (norms, biases, gates and CQT kernels stay f32)
python convert_iaamt_to_gguf.py models/best_model.pth --outdir ./out --outtype f16
```

| Flag | Meaning |
|------|---------|
| `--outfile` | exact output path; only valid for a single input |
| `--outdir` | output directory, default `.` (name is `<stem>-<F32\|F16>.gguf`) |
| `--outtype` | `f32` (default) or `f16` |
| `--ema` | prefer `ema_state_dict` when the checkpoint carries one |
| `--no-embed-cqt` | omit the precomputed CQT kernels (see below) |

Output sizes are ~56 MB (`f32`) or ~29 MB (`f16`) per transcription model.

## Tensor naming

| Checkpoint key | GGUF name |
|----------------|-----------|
| `backbone.stem.conv{1,2}.*` | `bb.stem.conv{1,2}.*` |
| `backbone.stem.block{n}.0.*` | `bb.stem.blk.{n-1}.conv.*` |
| `backbone.stem.block{n}.1.*` | `bb.stem.blk.{n-1}.norm.*` (GroupNorm, 4 groups) |
| `backbone.stem.freq_embed` | `bb.stem.freq_embd` |
| `backbone.pitch_query_embed.mlp.{0,2}.*` | `bb.pitch_query.fc{1,2}.*` |
| `backbone.layers.{L}.{A}.layers.0.0.*` | `bb.blk.{L}.{time\|band}.attn_*` (by role, see above) |
| `backbone.layers.{L}.{A}.layers.0.1.net.{0,1,4}.*` | `bb.blk.{L}.{time\|band}.ffn_{norm,up,down}.*` |
| `backbone.final_norm.gamma` | `bb.output_norm.weight` |
| `backbone.up_conv.*` | `bb.up_conv.*` |
| `backbone.feature_extractor.cqt.resamplers.{i}.weight` | `cqt.resampler.{i}.weight` |
| `{interval,instrument,beat,chord}_adapter.net.{0,1,4}.*` | `head.{...}_adapter.{norm,fc1,fc2}.*` |
| `interval_scorer.proj.*` | `head.interval_scorer.*` |
| `interval_boundary_predictor.net.{0,3}.*` | `head.boundary.fc{1,2}.*` |
| `interval_instrument_predictor.net.{0,1,4}.*` | `head.interval_instrument.{norm,fc1,fc2}.*` |
| `instrument_classifier.*` | `head.instrument_classifier.*` |
| `velocity_head.*` | `head.velocity.*` |
| `inter_{beat,chord}_heads.{k}.*` | `head.inter.{k}.{beat,chord}.*` |

Within an attention block: `norm_q` → `attn_norm`, `norm_context` →
`attn_norm_kv`, `to_{q,k,v}` → `attn_{q,k,v}`, `to_gates` → `attn_gate`,
`to_out.0` → `attn_out`.

Weights pass through unchanged — GGUF reverses the dimension order, so a torch
`Conv2d` weight `[OC, IC, KH, KW]` lands as `ne = [KW, KH, IC, OC]`, exactly
`ggml_conv_2d`'s expected kernel layout.

### Dropped tensors

`rope.inv_freq` (one per attention block, 12 per transcription checkpoint) is not
emitted. It is the closed form `1 / 10000^(2i/head_dim)` over `head_dim = 32`,
which the runtime regenerates. Nothing else is dropped: every remaining
checkpoint tensor appears in the GGUF bit-for-bit.

## Precomputed CQT kernels

`RecursiveCQT` registers its STFT→CQT kernels and Hann windows as
**non-persistent** buffers, so they are absent from every checkpoint — only the 7
resampler FIR filters are stored. Rebuilding them at load time would mean porting
`scipy.signal` to C++, so conversion emits them instead. This is the same
approach `docs/moss-music-conversion.md` takes with `SinusoidsPositionEmbedding`.

For the shipped configuration (`sr=22050`, `hop=512`, `fmin=27.5`,
`bins_per_octave=36`, `filter_scale=0.475`, harmonics `1–5`) the plan resolves to
**312 CQT bins across 8 recursive stages**, each with `n_fft = 256`:

| Stage | Bins | Kernel shape (`[n_freqs, n_fft/2+1]`) |
|-------|------|----------------------------------------|
| 0 | 252–312 | `[60, 129]` |
| 1–7 | 36 each, descending | `[36, 129]` |

Each stage emits three tensors, ~322 KB total:

- `cqt.kernel.{i}.real`, `cqt.kernel.{i}.imag` — the complex kernel, split
- `cqt.window.{i}` — the periodic Hann window (`0.5 - 0.5·cos(2πn/N)`, matching
  `scipy.signal.get_window("hann", N)`, which is `fftbins=True` by default)

Stage 0 is the **highest** octave and runs at the full sample rate; each later
stage halves the rate via `cqt.resampler.{i}` and halves the hop. The stage's bin
range is `iaamt.cqt.stage_bin_{start,end}[i]`. Note the partial octave is
absorbed by stage 0 rather than becoming a ninth stage.

`--no-embed-cqt` skips these and sets `iaamt.cqt.kernels_embedded = false`; the
runtime then owns the reconstruction.

## Metadata

Every `model_config` entry is written under the `iaamt.` namespace, minus
training-only fields (`dropout`, `spec_augment_params`, `use_gradient_checkpoint`,
`harmonic_dropout_p`, `time_mask_*`, `inter_loss_weight`). So `iaamt.hidden_size`,
`iaamt.sample_rate`, `iaamt.hop_length`, `iaamt.harmonics`, … are all present.

Added on top:

| Key | Meaning |
|-----|---------|
| `iaamt.model_type` | `transcription`, `velocity` or `beat_chord` |
| `iaamt.source_checkpoint` | original `.pth` filename |
| `iaamt.has_slot_embedding` | whether pitch slots are used (`num_pitch_slots > 1`) |
| `iaamt.has_interval_instrument_head` | whether per-interval instrument classification exists |
| `iaamt.num_audio_channels` | `input_audio_channels × len(harmonics)`, i.e. 10 |
| `iaamt.input_audio_channels` | stereo (2); only written when `model_config` omits it |
| `iaamt.cqt.n_stages` | recursive CQT stages (8) |
| `iaamt.cqt.fft_sizes` | per-stage `n_fft` |
| `iaamt.cqt.stage_bin_start` / `.stage_bin_end` | per-stage CQT bin range |
| `iaamt.cqt.q` | `filter_scale / (2^(1/bins_per_octave) − 1)` |
| `iaamt.cqt.fmin_large` / `.actual_bins` / `.n_bins_large` | derived HCQT geometry |
| `iaamt.cqt.harmonic_shifts` | bin offsets for each harmonic, in bins |
| `iaamt.cqt.kernels_embedded` | whether the kernels above are present |
| `iaamt.beat_meter_classes`, `iaamt.chord_quality_map`, `iaamt.inference_config` | JSON blobs carried by `best_beat_chord_key.pth` |

## Checkpoint variation

The transcription checkpoints are not uniform, and the runtime must branch on
metadata rather than assuming one shape:

| Checkpoint | `num_pitch_slots` | `num_instrument_classes` | slot embd | interval-instr head |
|------------|-------------------|--------------------------|-----------|---------------------|
| `best_model`, `_bass`, `_guitar`, `_vocal` | 1 | 34 | no | no |
| `_bass_v2`, `_drums`, `_guitar_v1_5` | 1 | 36 | yes | yes |
| `_other`, `_vocal_harmony` | 3 | 35 | yes | yes |

## Runtime status

Transcription inference works. `tools/iaamt/` builds `llama-iaamt-cli`:

```bash
llama-iaamt-cli -m best_model-F32.gguf -a song.wav -o song.mid
```

| Flag | Meaning |
|------|---------|
| `-m/-a/-o` | model, input audio (wav/mp3/flac), output MIDI |
| `-t N` | CPU threads for the front end and any CPU-scheduled ops |
| `--cpu` | keep the whole graph on the CPU |
| `--window-ms` / `--stride-ms` | analysis window and hop (default 8000 / half) |
| `--note-bias F` | added to every interval score; raise to admit more notes |
| `--merge-onset-ms` / `--merge-gap-ms` | cross-window note stitching |
| `--min-note-ms` / `--velocity` | output filtering and fixed MIDI velocity |
| `--silence-dbfs F` | skip windows quieter than F dBFS (default −72) |
| `--no-boundary` | skip the sub-frame boundary head |
| `--dump-prefix P` | write window-0 intermediates as `P*.npy` for parity checks |

It links `ggml` alone — no `llama`, no `common`, no vocabulary, no KV cache.

### How the pieces fit

**HCQT front end** (`iaamt-cqt.cpp`, CPU) — 8-stage recursive STFT using the
embedded kernels, FIR half-band decimation between stages, harmonic
interpolation, whole-window standardization. This is fixed preprocessing over
complex STFT frames, so it sits outside the graph for the same reason
`mtmd-audio.cpp` computes mel spectrograms on the CPU.

**Backbone** (`iaamt-graph.cpp`) — Conv2d stem with 4-group GroupNorm, then six
dual-axis layers. Both axes are the same graph on a permuted tensor: band-axis
attention runs with tokens on `ne1` and time batched on `ne2`, and the time axis
swaps them. Two details are not in the standard attention helpers: RMSNorm here
is `x/‖x‖₂·√D·γ` (which `ggml_rms_norm` computes up to the epsilon placement),
and `to_gates` applies a per-head sigmoid gate to the attention output, which
needs an explicit broadcast multiply.

**Upsampling** — `ConvTranspose1d(k=8, s=8)` has no kernel overlap, so it is
exactly a matmul that fans each frame into eight. `ggml_conv_transpose_1d`
accepts only a 2-D input and this carries 88 pitch tracks, so the graph reshapes
the weight to `[D_in, 8·D_out]` and lets `ggml_mul_mat` batch over pitch.

**Semi-CRF decoding** (`iaamt-decode.cpp`, CPU) — `viterbiBackward` with a zero
noise score, run per pitch track. A serial dynamic program over 88 (or 88×3)
tracks does not map onto a ggml graph, so it stays a CPU post-pass over the
scorer's output, as does the boundary head, which is evaluated only for decoded
intervals. Cross-window stitching reproduces `WindowNoteStitcher`: each track
remembers where it was last closed and that seeds the next window's backtrace.

**MIDI** (`iaamt-midi.cpp`) — a minimal type-0 SMF writer, 480 ticks/quarter at
120 BPM.

### Velocity

`best_velocity_model` scores an existing note list against the audio it came
from.  It shares the transcription backbone verbatim, so the graph simply stops
before the semi-CRF head and hands the pitch features to a CPU pass that runs
the per-note MLPs (`iaamt-velocity.cpp`).

```bash
llama-iaamt-cli -m best_model-F32.gguf          -a stem.wav -o notes.mid
llama-iaamt-cli -m best_velocity_model-F32.gguf -a stem.wav --midi notes.mid -o out.mid
```

Each note is scored by the window whose stride slot contains its onset, so the
`local_frame_offsets` look-ahead is always inside the window.  A single mixed
input has no stem identity, so `stem_embedding`'s reserved final row (the
"unknown" slot the reference uses for padded input) is selected.

`predict_stem_gain` is false in the published checkpoint — it ships no
`global_audio_projection` or `stem_gain_head` tensors — and stem gain is only
meaningful for a multi-stem batch, so it is not implemented.

### Beat, chord and key

`best_beat_chord_key` consumes a MIDI roll rather than audio, so it takes
`--midi` and writes a tab-separated analysis instead of a MIDI file:

```bash
llama-iaamt-cli -m best_beat_chord_key-F32.gguf --midi notes.mid -o analysis.tsv
```

Its backbone reuses the same dual-axis Transformer, but note that the two
per-layer Transformers are stored in the **opposite order** from the audio
models: `layers.{L}.0` is the pitch axis here and the time axis there.  The
converter resolves this by naming them by role, so `bb.blk.{L}.band` is always
the token axis and the runtime stays uniform.

The intermediate refinement is implemented in full: after each layer in
`iaamt.inter_refine_layers`, the global tokens are upsampled and merged, run
through that stage's beat and chord heads, folded back down with a strided
Conv1d, projected and scaled by a learned sigmoid gate, then appended as two
extra tokens for the next layer.  Tokens a previous stage appended are stripped
first so they never compound.

**Decoding is simplified.** The reference post-processes these logits with a
dynamic-programming beat grid, meter grouping and chord smoothing
(`beat_chord/decoding/`, ~2400 lines).  This runtime emits frame-level results
directly: beats by local-maximum peak picking above 0.5, and chord, bass and key
by per-frame argmax.  That is enough to read the harmony and pulse, but it will
be noisier at boundaries than the reference decoder.

### Not implemented

Instrument classification.  The head is converted but unused, so transcription
writes a single-track MIDI and the beat/chord roll assigns notes to a class by
MIDI program rather than by the classifier.

## Verification

**Conversion** — every GGUF was re-read with `gguf.GGUFReader` and compared
tensor-by-tensor against the source `.pth` with `np.array_equal`: 3080 tensors
across all 11 checkpoints match exactly, with the 12-per-model `rope.inv_freq`
the only intentional omission. Re-run with:

```bash
python tests/iaamt_convert_roundtrip.py models/
```

**Graph** — `tests/iaamt_reference_forward.py` is a numpy port of the reference
modules that reads the same GGUF weights and the CLI's dumped HCQT, so any
mismatch isolates the ggml graph from the front end:

```bash
llama-iaamt-cli --cpu -m M.gguf -a a.wav -o a.mid --dump-prefix ./d_
python tests/iaamt_reference_forward.py M.gguf ./d_ --compare
```

Current agreement on `best_model`, over 345 frames × 88 tracks:

| tensor | max abs diff | rms |
|--------|--------------|-----|
| `query` | 9.5e-07 | 3.1e-08 |
| `key`   | 1.8e-05 | 4.9e-07 |
| `diag`  | 4.1e-06 | 8.0e-07 |

Every named intermediate (`stem`, `tokens`, `band{i}`, `layer{i}`,
`final_norm`, `upconv`) is dumped alongside, so a regression can be bisected to
one stage rather than bisected by hand.

**End to end, transcription** — on a synthesized 9 s stereo file containing five
known pitches, `best_model` recovers all five at the correct pitch with onsets
within 20 ms:

| decoded | expected |
|---------|----------|
| midi 60 @ 0.50–1.48 s | midi 60 @ 0.50–1.50 s |
| midi 64 @ 2.01–2.99 s | midi 64 @ 2.00–3.00 s |
| midi 67 @ 3.51–4.49 s | midi 67 @ 3.50–4.50 s |
| midi 72 @ 5.00–6.49 s | midi 72 @ 5.00–6.50 s |
| midi 55 @ 7.00–8.00 s | midi 55 @ 7.00–8.00 s |

`best_model_other` (3 pitch slots, 264 tracks) recovers the same five, which
exercises the slot-embedding path. `best_model_drums` returns nothing on pitched
tones, as expected for a drum model.

This validates the HCQT independently: the CQT kernels are precomputed by the
converter and had no reference to check against, but a bin-117 peak resolves to
261.6 Hz — exactly MIDI 60 — and the decoded pitches are correct, which they
could not be if the filterbank were misaligned.

**End to end, velocity** — on the same five pitches rendered at descending
levels, predicted velocity tracks level monotonically:

| level | 0 dB | −8 dB | −16 dB | −26 dB | −38 dB |
|-------|------|-------|--------|--------|--------|
| velocity | 75 | 62 | 53 | 36 | 23 |

**End to end, beat/chord** — on a synthesized I–IV–V–I in C at 120 BPM with
two-second bars, the model recovers beats at 0.5 s spacing, downbeats at 2.0,
4.0 and 6.0 s, a bass line of C → F → G → C, and key C throughout, returning to
the same `root_chord` class when the harmony returns to C.

All three families produce byte-identical output on CPU and on ROCm.

### Running on ROCm under Windows

`ggml-hip.dll` links `amdhip64_7.dll`, and Windows searches `System32` before
`PATH`.  If the AMD driver has installed its own copy there, it is loaded
instead of the SDK build the project was compiled against, and every call fails
in `ggml_backend_cuda_device_get_memory` with `hipMemGetInfo: invalid argument`
(stock `test-backend-ops` fails the same way).  Copy the SDK's runtime next to
the executable, which the loader searches first:

```powershell
Copy-Item "$env:ROCM_PATHinmdhip64_7.dll" .uildin```

### Backend op notes

Two ops needed care and are worth knowing about before extending the graph:

- `ggml_conv_transpose_1d` accepts only a 2-D input, so the pitch-track upsample
  is expressed as a matmul instead (kernel and stride are equal, so the two are
  identical).
- `ggml_conv_2d_dw` builds an f16 im2col and feeds it to `mul_mat`'s `src1`,
  which no backend accepts; `ggml_conv_2d_dw_direct` works. It wants an **f32**
  kernel: CUDA asserts on f32, while the CPU path reads the kernel as float
  without checking, so an f16 kernel is silently misread rather than rejected.
  `ggml_conv_1d`, by contrast, always builds an f16 im2col and so needs an f16
  kernel.
