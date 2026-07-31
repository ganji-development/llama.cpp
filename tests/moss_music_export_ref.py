#!/usr/bin/env python3
"""Numeric parity check for the MOSS-Music audio tower.

Runs the HuggingFace reference implementation of the audio encoder, the
audio_adapter and the DeepStack mergers on a WAV file, and compares the result
against the projector output dumped by llama.cpp.

Only the audio tower is loaded (~1.7 GB), never the 8B language model.

Usage:
    # 1. dump the llama.cpp side
    MTMD_DEBUG_EMBD_FILE=dump.bin llama-mtmd-cli \\
        -m model.gguf --mmproj mmproj.gguf --audio clip.wav -p "x" -n 1

    # 2. compare
    python tests/moss_music_export_ref.py \\
        --model-dir /path/to/MOSS-Music-8B-Thinking \\
        --audio clip.wav --dump dump.bin
"""

import argparse
import math
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def log(msg):
    print(msg, flush=True)


# --------------------------------------------------------------------------
# reference modules (mirrors modeling_moss_music.py)
# --------------------------------------------------------------------------

class SinusoidsPositionEmbedding(nn.Module):
    def __init__(self, num_positions: int, embedding_dim: int):
        super().__init__()
        max_timescale = 10000.0
        log_timescale_increment = math.log(max_timescale) / (embedding_dim // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(embedding_dim // 2).float())
        self.register_buffer("inv_timescales", inv_timescales, persistent=False)

    def forward(self, seq_len: int, device: torch.device):
        scaled_time = torch.arange(seq_len, device=device, dtype=self.inv_timescales.dtype).unsqueeze(1) \
            * self.inv_timescales.unsqueeze(0)
        return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1).unsqueeze(0)


class GatedMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.gate_proj = nn.Linear(input_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(input_size, hidden_size, bias=False)
        self.down_proj = nn.Linear(hidden_size, output_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def build_encoder(audio_cfg):
    """MossMusicEncoder, using the stock HF WhisperEncoderLayer."""
    from transformers.models.whisper.configuration_whisper import WhisperConfig
    from transformers.models.whisper.modeling_whisper import WhisperEncoderLayer

    wcfg = WhisperConfig(
        d_model=audio_cfg["d_model"],
        encoder_attention_heads=audio_cfg["encoder_attention_heads"],
        encoder_ffn_dim=audio_cfg["encoder_ffn_dim"],
        activation_function=audio_cfg.get("activation_function", "gelu"),
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
    )
    wcfg._attn_implementation = "eager"

    hidden = audio_cfg["downsample_hidden_size"]
    enc = nn.Module()
    enc.conv1 = nn.Conv2d(1, hidden, (3, 3), (2, 2), (1, 1))
    enc.conv2 = nn.Conv2d(hidden, hidden, (3, 3), (2, 2), (1, 1))
    enc.conv3 = nn.Conv2d(hidden, hidden, (3, 3), (2, 2), (1, 1))
    enc.stem_proj = nn.Linear(hidden * 16, audio_cfg["d_model"])
    enc.embed_positions = SinusoidsPositionEmbedding(audio_cfg["max_source_positions"], audio_cfg["d_model"])
    enc.layers = nn.ModuleList([WhisperEncoderLayer(wcfg) for _ in range(audio_cfg["encoder_layers"])])
    enc.layer_norm = nn.LayerNorm(audio_cfg["d_model"], eps=audio_cfg.get("layer_norm_eps", 1e-5))
    return enc


def encoder_forward(enc, mel, deepstack_idxs):
    """Mirrors MossMusicEncoder.forward for a single un-padded sequence."""
    gelu = nn.GELU()
    x = mel.unsqueeze(0).unsqueeze(1)          # [1, 1, n_mels, T]
    x = gelu(enc.conv1(x))
    x = gelu(enc.conv2(x))
    x = gelu(enc.conv3(x))
    x = x.permute(0, 3, 1, 2).contiguous().flatten(2)   # [1, T', C*F]
    x = enc.stem_proj(x)

    x = x + enc.embed_positions(x.shape[1], x.device).to(x.dtype)

    deepstack = []
    for i, layer in enumerate(enc.layers):
        x = layer(x, None, layer_head_mask=None, output_attentions=False)[0]
        if i in deepstack_idxs:
            deepstack.append(x)

    return enc.layer_norm(x), deepstack


# --------------------------------------------------------------------------

def load_mel(path, sample_rate, n_fft, hop_length, n_mels):
    """Reference mel via HF WhisperFeatureExtractor."""
    from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor
    import wave

    with wave.open(str(path), "rb") as w:
        if w.getframerate() != sample_rate:
            raise SystemExit(f"expected {sample_rate} Hz WAV, got {w.getframerate()}")
        if w.getnchannels() != 1:
            raise SystemExit("expected mono WAV")
        raw = w.readframes(w.getnframes())

    wav = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    fe = WhisperFeatureExtractor(feature_size=n_mels, sampling_rate=sample_rate,
                                 hop_length=hop_length, n_fft=n_fft)
    mel = fe._np_extract_fbank_features(wav[None, ...], device="cpu")[0]
    return torch.from_numpy(np.asarray(mel)).float(), wav


def read_dump(path):
    with open(path, "rb") as f:
        n_embd, n_tokens = struct.unpack("<ii", f.read(8))
        data = np.frombuffer(f.read(n_embd * n_tokens * 4), dtype="<f4")
    return data.reshape(n_tokens, n_embd)


def compare(name, ref, got, tol):
    ref = np.asarray(ref, dtype=np.float64)
    got = np.asarray(got, dtype=np.float64)
    if ref.shape != got.shape:
        log(f"  {name:22s} SHAPE MISMATCH ref={ref.shape} got={got.shape}")
        return False

    denom = max(np.abs(ref).max(), 1e-9)
    max_abs = np.abs(ref - got).max()
    rel = max_abs / denom
    num = float((ref * got).sum())
    cos = num / (np.linalg.norm(ref) * np.linalg.norm(got) + 1e-30)
    ok = rel <= tol
    log(f"  {name:22s} {'ok ' if ok else 'FAIL'}  max|d|={max_abs:.5f}  rel={rel:.5f}  cos={cos:.6f}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--audio", required=True, type=Path)
    ap.add_argument("--dump", required=True, type=Path, help="MTMD_DEBUG_EMBD_FILE output")
    ap.add_argument("--tol", type=float, default=0.02, help="relative tolerance (f16 mmproj)")
    args = ap.parse_args()

    import json
    from safetensors import safe_open

    cfg = json.loads((args.model_dir / "config.json").read_text())
    acfg = cfg["audio_config"]
    n_ds = min(len(acfg["deepstack_encoder_layer_indexes"]), cfg["deepstack_num_inject_layers"])
    ds_idxs = acfg["deepstack_encoder_layer_indexes"][:n_ds]

    # ---- gather only the audio-side tensors -------------------------------
    index = json.loads((args.model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    wanted = {k: v for k, v in index.items()
              if k.startswith(("audio_encoder.", "audio_adapter.", "deepstack_audio_merger_list."))}
    by_file = {}
    for k, fn in wanted.items():
        by_file.setdefault(fn, []).append(k)

    sd = {}
    for fn, keys in by_file.items():
        with safe_open(str(args.model_dir / fn), framework="pt") as f:
            for k in keys:
                sd[k] = f.get_tensor(k).float()
    log(f"loaded {len(sd)} audio-tower tensors")

    # ---- build and load ---------------------------------------------------
    enc = build_encoder(acfg)
    enc_sd = {k[len("audio_encoder."):]: v for k, v in sd.items() if k.startswith("audio_encoder.")}
    enc_sd.pop("embed_positions.inv_timescales", None)
    missing, unexpected = enc.load_state_dict(enc_sd, strict=False)
    missing = [m for m in missing if "inv_timescales" not in m]
    if missing or unexpected:
        log(f"  WARNING missing={missing[:4]} unexpected={unexpected[:4]}")

    d_model, hidden, out_dim = acfg["output_dim"], cfg["adapter_hidden_size"], cfg["hidden_size"]

    adapter = GatedMLP(d_model, hidden, out_dim)
    adapter.load_state_dict({k[len("audio_adapter."):]: v
                             for k, v in sd.items() if k.startswith("audio_adapter.")})

    mergers = []
    for i in range(n_ds):
        m = GatedMLP(d_model, hidden, out_dim)
        pfx = f"deepstack_audio_merger_list.{i}."
        m.load_state_dict({k[len(pfx):]: v for k, v in sd.items() if k.startswith(pfx)})
        mergers.append(m)

    # ---- reference forward ------------------------------------------------
    mel, wav = load_mel(args.audio, acfg.get("mel_sr", 16000), 400, 160, acfg["num_mel_bins"])
    log(f"audio: {len(wav)} samples, mel {tuple(mel.shape)}, deepstack layers {ds_idxs}")

    with torch.no_grad():
        enc_out, deepstack = encoder_forward(enc, mel, set(ds_idxs))
        ref_adapter = adapter(enc_out)[0]
        ref_ds = [mergers[i](deepstack[i])[0] for i in range(n_ds)]

    ref_full = torch.cat([ref_adapter] + ref_ds, dim=-1).numpy()

    # ---- compare ----------------------------------------------------------
    got = read_dump(args.dump)
    log(f"\nreference {ref_full.shape}  vs  llama.cpp {got.shape}")
    if ref_full.shape[0] != got.shape[0]:
        log(f"TOKEN COUNT MISMATCH: reference {ref_full.shape[0]}, llama.cpp {got.shape[0]}")
        return 1

    log("")
    ok = True
    ok &= compare("audio_adapter", ref_adapter.numpy(), got[:, :out_dim], args.tol)
    for i in range(n_ds):
        lo = out_dim * (i + 1)
        ok &= compare(f"deepstack[{i}] (enc L{ds_idxs[i]})", ref_ds[i].numpy(),
                      got[:, lo:lo + out_dim], args.tol)
    ok &= compare("FULL ROW", ref_full, got, args.tol)

    log("\n" + ("PARITY PASS" if ok else "PARITY FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
