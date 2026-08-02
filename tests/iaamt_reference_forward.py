#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Numpy reference forward pass for iaamt, used to validate the ggml graph.

Ported directly from the module definitions in the upstream project
(modeling/backbone.py, modeling/blocks/transformer.py, modeling/heads/*).  It
reads weights from the GGUF and the HCQT features dumped by

    llama-iaamt-cli --dump-prefix PREFIX ...

so a mismatch isolates the ggml graph rather than the CPU front end.  Every
named intermediate the CLI dumps has a matching ``PREFIX + "ref_*.npy"`` here,
which makes the divergence bisectable stage by stage.

Usage:
    llama-iaamt-cli --cpu -m M.gguf -a a.wav -o a.mid --dump-prefix ./d_
    python tests/iaamt_reference_forward.py M.gguf ./d_
    python tests/iaamt_reference_forward.py M.gguf ./d_ --compare

Requires numpy only.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path("E:/Ganji-Development/moss-music-llama.cpp")
sys.path.insert(1, str(REPO / "gguf-py"))
import gguf

PREFIX = sys.argv[2] if len(sys.argv) > 2 else "d_"
COMPARE = "--compare" in sys.argv
# GGUFReader already hands back arrays in torch (reversed-ne) order.
T = {t.name: np.array(t.data, dtype=np.float32)
     for t in gguf.GGUFReader(sys.argv[1]).tensors}


def w(name):
    return T[name]


def conv2d(x, k, b, stride, pad):
    """x [N,C,H,W], k [OC,IC,KH,KW] in torch order."""
    N, C, H, W = x.shape
    OC, IC, KH, KW = k.shape
    sh, sw = stride
    ph, pw = pad
    xp = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    OH = (H + 2 * ph - KH) // sh + 1
    OW = (W + 2 * pw - KW) // sw + 1
    cols = np.empty((N, IC * KH * KW, OH * OW), dtype=np.float32)
    idx = 0
    for ci in range(IC):
        for i in range(KH):
            for j in range(KW):
                cols[:, idx, :] = xp[:, ci, i:i + sh * OH:sh, j:j + sw * OW:sw].reshape(N, -1)
                idx += 1
    out = k.reshape(OC, -1) @ cols
    return (out.reshape(N, OC, OH, OW) + b.reshape(1, OC, 1, 1)).astype(np.float32)


def group_norm(x, g, gw, gb, eps=1e-5):
    N, C, H, W = x.shape
    xr = x.reshape(N, g, C // g * H * W)
    m = xr.mean(-1, keepdims=True)
    v = xr.var(-1, keepdims=True)
    xr = (xr - m) / np.sqrt(v + eps)
    return (xr.reshape(N, C, H, W) * gw.reshape(1, C, 1, 1) + gb.reshape(1, C, 1, 1)).astype(np.float32)


def gelu_np(x):
    """Exact erf GELU, matching nn.GELU (not the tanh approximation)."""
    import math
    vec = np.vectorize(math.erf)
    return (0.5 * x * (1.0 + vec(x / np.sqrt(2)))).astype(np.float32)


def rms_norm(x, gamma):
    d = x.shape[-1]
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n = np.maximum(n, 5.960464477539063e-08)
    return (x / n * np.sqrt(d) * gamma).astype(np.float32)


def layer_norm(x, ww, bb, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = x.var(-1, keepdims=True)
    return ((x - m) / np.sqrt(v + eps) * ww + bb).astype(np.float32)


def rope(x):
    """x [B, H, S, D] -> interleaved rotary."""
    B, H, S, D = x.shape
    inv = 1.0 / (10000.0 ** (np.arange(0, D, 2, dtype=np.float64) / D))
    ang = np.arange(S)[:, None] * inv[None, :]
    c = np.cos(ang)[None, None]
    s = np.sin(ang)[None, None]
    e = x[..., 0::2]
    o = x[..., 1::2]
    y = np.empty_like(x)
    y[..., 0::2] = e * c - o * s
    y[..., 1::2] = o * c + e * s
    return y.astype(np.float32)


def attention(x, p, nh, hd):
    """x [B, S, D]."""
    B, S, D = x.shape
    h = rms_norm(x, w(p + "attn_norm.weight"))
    c = rms_norm(x, w(p + "attn_norm_kv.weight"))
    q = h @ w(p + "attn_q.weight").T + w(p + "attn_q.bias")
    k = c @ w(p + "attn_k.weight").T + w(p + "attn_k.bias")
    v = c @ w(p + "attn_v.weight").T + w(p + "attn_v.bias")
    q = q.reshape(B, S, nh, hd).transpose(0, 2, 1, 3)
    k = k.reshape(B, S, nh, hd).transpose(0, 2, 1, 3)
    v = v.reshape(B, S, nh, hd).transpose(0, 2, 1, 3)
    q = rope(q)
    k = rope(k)
    att = q @ k.transpose(0, 1, 3, 2) / np.sqrt(hd)
    att = att - att.max(-1, keepdims=True)
    att = np.exp(att)
    att /= att.sum(-1, keepdims=True)
    o = att @ v                                        # [B, nh, S, hd]
    g = 1.0 / (1.0 + np.exp(-(h @ w(p + "attn_gate.weight").T + w(p + "attn_gate.bias"))))
    o = o * g.transpose(0, 2, 1)[..., None]
    o = o.transpose(0, 2, 1, 3).reshape(B, S, nh * hd)
    o = o @ w(p + "attn_out.weight").T + w(p + "attn_out.bias")
    x = x + o
    h = rms_norm(x, w(p + "ffn_norm.weight"))
    h = h @ w(p + "ffn_up.weight").T + w(p + "ffn_up.bias")
    h = gelu_np(h)
    h = h @ w(p + "ffn_down.weight").T + w(p + "ffn_down.bias")
    return x + h


hcqt = np.load(PREFIX + "hcqt.npy")            # [C, T, F]
C, Tn, F = hcqt.shape
x = hcqt[None]                                  # [1, C, T, F]
print("input", x.shape)

x = conv2d(x, w("bb.stem.conv1.weight"), w("bb.stem.conv1.bias"), (1, 1), (3, 3))
x = x + w("bb.stem.freq_embd").reshape(1, -1, 1, F)
x = conv2d(x, w("bb.stem.conv2.weight"), w("bb.stem.conv2.bias"), (1, 1), (2, 2))
for i, (st, act) in enumerate([((2, 1), True), ((2, 2), True), ((2, 2), True), ((1, 1), False)]):
    x = conv2d(x, w(f"bb.stem.blk.{i}.conv.weight"), w(f"bb.stem.blk.{i}.conv.bias"), st, (1, 1))
    x = group_norm(x, 4, w(f"bb.stem.blk.{i}.norm.weight"), w(f"bb.stem.blk.{i}.norm.bias"))
    if act:
        x = gelu_np(x)
print("stem", x.shape)
np.save(PREFIX+"ref_stem.npy", x[0].transpose(2,1,0))   # ggml ne order [F,T,C]

# b d t f -> b t f d
x = x.transpose(0, 2, 3, 1)
B, T2, NB, D = x.shape

pitches = np.arange(21, 21 + 88, dtype=np.float32)
pf = np.stack([pitches / 128.0,
               np.sin(2 * np.pi * pitches / 12.0),
               np.cos(2 * np.pi * pitches / 12.0),
               np.ones_like(pitches)], -1)
pq = pf @ w("bb.pitch_query.fc1.weight").T + w("bb.pitch_query.fc1.bias")
pq = gelu_np(pq)
pq = pq @ w("bb.pitch_query.fc2.weight").T + w("bb.pitch_query.fc2.bias")

tokens = np.concatenate([
    x + w("bb.band_type_embd").reshape(1, 1, 1, D),
    np.broadcast_to(pq[None, None] + w("bb.pitch_type_embd").reshape(1, 1, 1, D),
                    (B, T2, 88, D)),
], axis=2).astype(np.float32)
print("tokens", tokens.shape)
np.save(PREFIX+"ref_tokens.npy", tokens[0].transpose(0,1,2))

nh, hd = 12, 32
K = tokens.shape[2]
cur = tokens
for il in range(6):
    b = cur.reshape(B * T2, K, D)
    b = attention(b, f"bb.blk.{il}.band.", nh, hd)
    cur = b.reshape(B, T2, K, D)
    t = cur.transpose(0, 2, 1, 3).reshape(B * K, T2, D)
    t = attention(t, f"bb.blk.{il}.time.", nh, hd)
    cur = t.reshape(B, K, T2, D).transpose(0, 2, 1, 3)
    np.save(PREFIX+f"ref_layer{il}.npy", cur[0])   # [T2, K, D]
    print(f"  layer {il} done, |x| = {np.abs(cur).mean():.4f}")

cur = rms_norm(cur, w("bb.output_norm.weight"))
np.save(PREFIX+"ref_final_norm.npy", cur[0])
pitch = cur[:, :, NB:, :]                       # [B, T2, P, D]

# (b p) d t, ConvTranspose1d k=s=8
P = pitch.shape[2]
pt = pitch.transpose(0, 2, 3, 1).reshape(B * P, D, T2)
uw = w("bb.up_conv.weight")                      # torch [Cin, Cout, K]
# y[b, o, t*8 + k] = sum_c pt[b, c, t] * uw[c, o, k]
tmp = np.einsum("bct,cok->botk", pt, uw)
y = tmp.reshape(B * P, D, T2 * 8)
y = y + w("bb.up_conv.bias").reshape(1, D, 1)
crop = Tn
y = y[:, :, :crop]
pitch = y.reshape(B, P, D, crop).transpose(0, 3, 1, 2)   # [B, T, P, D]
print("pitch", pitch.shape)

h = layer_norm(pitch, w("head.interval_adapter.norm.weight"), w("head.interval_adapter.norm.bias"))
h = h @ w("head.interval_adapter.fc1.weight").T + w("head.interval_adapter.fc1.bias")
h = gelu_np(h)
h = h @ w("head.interval_adapter.fc2.weight").T + w("head.interval_adapter.fc2.bias")
feat = pitch + h

proj = feat @ w("head.interval_scorer.weight").T + w("head.interval_scorer.bias")
hd2 = 256
query = proj[..., :hd2] / np.sqrt(hd2)
key = proj[..., hd2:2 * hd2]
diag = proj[..., 2 * hd2]

print("diag  min %.3f max %.3f mean %.3f  >0: %d/%d"
      % (diag.min(), diag.max(), diag.mean(), (diag > 0).sum(), diag.size))
qk = np.einsum("btpd,bspd->bpts", query, key)
print("qk    min %.3f max %.3f mean %.3f  >0: %d"
      % (qk.min(), qk.max(), qk.mean(), (qk > 0).sum()))

np.save(PREFIX + "ref_query.npy", query[0].transpose(1, 0, 2))
np.save(PREFIX + "ref_key.npy", key[0].transpose(1, 0, 2))
np.save(PREFIX + "ref_diag.npy", diag[0].T)
print("saved reference query/key/diag")


if COMPARE:
    print()
    ok = True
    for name, cpp, ref in (
        ("query", PREFIX + "query.npy", PREFIX + "ref_query.npy"),
        ("key",   PREFIX + "key.npy",   PREFIX + "ref_key.npy"),
        ("diag",  PREFIX + "diag.npy",  PREFIX + "ref_diag.npy"),
    ):
        a = np.load(cpp)
        b = np.load(ref)
        delta = np.abs(a - b).max()
        # f32 accumulation over a 6-layer transformer; 1e-3 is generous but
        # still four orders below the score magnitudes the decoder thresholds on
        status = "ok  " if delta < 1e-3 else "FAIL"
        if delta >= 1e-3:
            ok = False
        print("%s %-6s maxdiff %.3e  rms %.3e"
              % (status, name, delta, np.sqrt(((a - b) ** 2).mean())))
    print()
    print("RESULT:", "graph matches the reference" if ok else "GRAPH DIVERGES")
    sys.exit(0 if ok else 1)
