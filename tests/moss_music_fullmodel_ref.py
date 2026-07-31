#!/usr/bin/env python3
"""Full-model parity for MOSS-Music: validates DeepStack injection into the LLM.

tests/moss_music_export_ref.py only covers the audio tower. This script exercises
the decoder path -- specifically that llm_build_moss_music slices deepstack merger
k out of the widened embedding row and adds it to decoder layer k at the right
offset. A wrong offset still passes encoder parity but corrupts generation.

Runs the HF reference end to end (audio -> logits) and prints the greedy
continuation plus the top logits, so the same prompt can be compared against
llama-mtmd-cli.

Usage:
    python tests/moss_music_fullmodel_ref.py --model-dir DIR --audio clip.wav \\
        [--prompt "Describe this audio."] [-n 24]
"""

import argparse
import sys
import wave
from pathlib import Path

import numpy as np
import torch


def log(m):
    print(m, flush=True)


def load_wav(path, sample_rate):
    with wave.open(str(path), "rb") as w:
        if w.getframerate() != sample_rate or w.getnchannels() != 1:
            raise SystemExit(f"need mono {sample_rate} Hz WAV")
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--audio", required=True, type=Path)
    ap.add_argument("--prompt", default="Describe this audio.")
    ap.add_argument("-n", "--n-predict", type=int, default=24)
    args = ap.parse_args()

    import json
    from transformers import AutoTokenizer
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor

    cfg = json.loads((args.model_dir / "config.json").read_text())
    acfg = cfg["audio_config"]
    sr = 16000

    tok = AutoTokenizer.from_pretrained(str(args.model_dir), trust_remote_code=True)

    log("loading full model in bfloat16 (this is slow on CPU)...")
    # config.json only maps AutoConfig/AutoProcessor, so pull the class directly
    model_cls = get_class_from_dynamic_module(
        "modeling_moss_music.MossMusicModel", str(args.model_dir))

    # The shipped remote code is inconsistent: MossMusicConfig keeps audio_config
    # as a plain dict, but MossMusicEncoder does attribute access on it. Promote
    # it to a namespace so the reference can actually be constructed.
    from types import SimpleNamespace
    from transformers import AutoConfig
    hf_cfg = AutoConfig.from_pretrained(str(args.model_dir), trust_remote_code=True)
    if isinstance(hf_cfg.audio_config, dict):
        hf_cfg.audio_config = SimpleNamespace(**hf_cfg.audio_config)

    model = model_cls.from_pretrained(
        str(args.model_dir),
        config=hf_cfg,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    log("model loaded")

    # mel, matching the converter/runtime path
    wav = load_wav(args.audio, sr)
    fe = WhisperFeatureExtractor(feature_size=acfg["num_mel_bins"], sampling_rate=sr,
                                 hop_length=160, n_fft=400)
    mel = torch.from_numpy(np.asarray(fe._np_extract_fbank_features(wav[None, ...], device="cpu")[0])).float()

    def conv_out(n):
        return (n - 1) // 2 + 1

    n_audio_tok = conv_out(conv_out(conv_out(mel.shape[-1])))
    log(f"audio: mel {tuple(mel.shape)} -> {n_audio_tok} audio tokens")

    # same wrapper mtmd uses: <|audio_bos|> <audio embeddings> <|audio_eos|>
    AUDIO_ID = 151654  # <|vision_pad|>, reused as <|AUDIO|> placeholder
    pre = tok(f"<|im_start|>user\n<|audio_bos|>", add_special_tokens=False).input_ids
    post = tok(f"<|audio_eos|>\n{args.prompt}<|im_end|>\n<|im_start|>assistant\n",
               add_special_tokens=False).input_ids
    input_ids = torch.tensor([pre + [AUDIO_ID] * n_audio_tok + post], dtype=torch.long)
    audio_mask = (input_ids == AUDIO_ID)
    log(f"prompt: {input_ids.shape[1]} tokens ({len(pre)} pre + {n_audio_tok} audio + {len(post)} post)")

    out_ids = []
    past = None
    cur_ids = input_ids
    with torch.no_grad():
        for step in range(args.n_predict):
            kwargs = dict(input_ids=cur_ids, past_key_values=past, use_cache=True)
            if step == 0:
                kwargs.update(
                    audio_data=mel.to(torch.bfloat16),
                    audio_data_seqlens=torch.tensor([mel.shape[-1]], dtype=torch.long),
                    audio_input_mask=audio_mask,
                )
            out = model(**kwargs)
            past = out.past_key_values
            logits = out.logits[0, -1].float()

            if step == 0:
                top = torch.topk(logits, 5)
                log("\nfirst-position top-5 logits (compare against llama.cpp):")
                for v, i in zip(top.values.tolist(), top.indices.tolist()):
                    log(f"   {i:>7}  {v:8.4f}  {tok.decode([i])!r}")

            nxt = int(torch.argmax(logits))
            out_ids.append(nxt)
            cur_ids = torch.tensor([[nxt]], dtype=torch.long)

    log("\n--- HF reference greedy output ---")
    log(tok.decode(out_ids))
    log("--- end ---")
    log(f"\ntoken ids: {out_ids}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
