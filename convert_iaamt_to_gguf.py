#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert instrument-agnostic-amt (Transkun-derived) ``.pth`` checkpoints to GGUF.

The upstream project is https://github.com/anime-song/instrument-agnostic-amt,
which extends Transkun's Neural Semi-CRF piano transcriber into an
instrument-agnostic model.  It ships three checkpoint families, all named
``best_*.pth``:

  transcription  audio -> note intervals   (best_model*.pth)
  velocity       audio + notes -> velocity (best_velocity_model.pth)
  beat_chord     MIDI roll -> beat/chord/key (best_beat_chord_key.pth)

All three write ``general.architecture = "iaamt"`` and are told apart by
``iaamt.model_type``.  See docs/iaamt-conversion.md.

The checkpoints are read without PyTorch: ``torch.save`` writes a zip of a
pickle plus raw little-endian storages, which numpy can map directly.  That
keeps conversion runnable anywhere gguf-py runs.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import os
import pickle
import re
import sys
import zipfile
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

if "NO_LOCAL_GGUF" not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent / "gguf-py"))
import gguf  # noqa: E402

logger = logging.getLogger("iaamt-to-gguf")

ARCH = "iaamt"

MODEL_TYPE_TRANSCRIPTION = "transcription"
MODEL_TYPE_VELOCITY = "velocity"
MODEL_TYPE_BEAT_CHORD = "beat_chord"


# ---------------------------------------------------------------------------
# torch.save() zip reader
# ---------------------------------------------------------------------------

# torch storage class name -> (numpy dtype, is_bfloat16)
_STORAGE_DTYPES: dict[str, tuple[np.dtype, bool]] = {
    "FloatStorage": (np.dtype("<f4"), False),
    "DoubleStorage": (np.dtype("<f8"), False),
    "HalfStorage": (np.dtype("<f2"), False),
    "BFloat16Storage": (np.dtype("<u2"), True),
    "LongStorage": (np.dtype("<i8"), False),
    "IntStorage": (np.dtype("<i4"), False),
    "ShortStorage": (np.dtype("<i2"), False),
    "CharStorage": (np.dtype("i1"), False),
    "ByteStorage": (np.dtype("u1"), False),
    "BoolStorage": (np.dtype("?"), False),
}


class _Opaque:
    """Stand-in for a torch/numpy class we only need to identify by name."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<opaque {self.name}>"

    def __call__(self, *args: Any, **kwargs: Any) -> "_Opaque":
        return self

    def __setstate__(self, state: Any) -> None:
        pass


class _StorageRef:
    __slots__ = ("key", "dtype", "is_bf16", "numel")

    def __init__(self, key: str, dtype: np.dtype, is_bf16: bool, numel: int) -> None:
        self.key = key
        self.dtype = dtype
        self.is_bf16 = is_bf16
        self.numel = numel


class _TensorRef:
    """A tensor's location inside the zip, resolved lazily by ``TorchCheckpoint``."""

    __slots__ = ("storage", "offset", "shape", "strides")

    def __init__(
        self,
        storage: _StorageRef,
        offset: int,
        shape: Sequence[int],
        strides: Sequence[int],
    ) -> None:
        self.storage = storage
        self.offset = int(offset)
        self.shape = tuple(int(dim) for dim in shape)
        self.strides = tuple(int(stride) for stride in strides)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"TensorRef{self.shape}"


def _rebuild_tensor(storage: _StorageRef, offset: int, size: Any, stride: Any, *_: Any) -> _TensorRef:
    return _TensorRef(storage, offset, size, stride)


def _rebuild_parameter(data: Any, *_: Any) -> Any:
    return data


class _Unpickler(pickle.Unpickler):
    """Resolve torch's rebuild hooks; make every other torch class inert."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "torch._utils":
            if name in ("_rebuild_tensor", "_rebuild_tensor_v2", "_rebuild_tensor_v3"):
                return _rebuild_tensor
            if name in ("_rebuild_parameter", "_rebuild_parameter_with_state"):
                return _rebuild_parameter
        if module.startswith(("torch", "numpy")):
            return _Opaque(f"{module}.{name}")
        try:
            return super().find_class(module, name)
        except Exception:
            # Training-time classes (argparse.Namespace subclasses, W&B config,
            # ...) show up in the ``config`` blob and are never needed here.
            return _Opaque(f"{module}.{name}")

    def persistent_load(self, pid: Any) -> _StorageRef:
        if not (isinstance(pid, tuple) and pid and pid[0] == "storage"):
            raise ValueError(f"unsupported persistent id in checkpoint: {pid!r}")
        _, storage_type, key, _location, numel = pid
        type_name = getattr(storage_type, "name", str(storage_type)).rsplit(".", 1)[-1]
        try:
            dtype, is_bf16 = _STORAGE_DTYPES[type_name]
        except KeyError:
            raise ValueError(f"unsupported torch storage type: {type_name}") from None
        return _StorageRef(str(key), dtype, is_bf16, int(numel))


class TorchCheckpoint:
    """Read a ``torch.save`` zip archive without importing torch."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._zip = zipfile.ZipFile(path)
        names = self._zip.namelist()
        pickles = [name for name in names if name.endswith("data.pkl")]
        if not pickles:
            raise ValueError(
                f"{path} is not a zip-format torch checkpoint "
                "(legacy tar checkpoints are not supported)"
            )
        self._prefix = pickles[0][: -len("data.pkl")]
        self.root = _Unpickler(io.BytesIO(self._zip.read(pickles[0]))).load()
        if not isinstance(self.root, Mapping):
            raise ValueError(f"{path}: expected a dict at the top level")

    def close(self) -> None:
        self._zip.close()

    def __enter__(self) -> "TorchCheckpoint":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def materialize(self, ref: _TensorRef) -> np.ndarray:
        storage = ref.storage
        raw = self._zip.read(f"{self._prefix}data/{storage.key}")
        flat = np.frombuffer(raw, dtype=storage.dtype)
        if flat.size != storage.numel:
            raise ValueError(
                f"storage {storage.key}: expected {storage.numel} elements, got {flat.size}"
            )
        item = storage.dtype.itemsize
        if ref.shape:
            view = np.lib.stride_tricks.as_strided(
                flat[ref.offset :],
                shape=ref.shape,
                strides=tuple(stride * item for stride in ref.strides),
            )
        else:
            # 0-d tensor (e.g. a learned scalar gate); GGUF has no rank-0 tensors.
            view = flat[ref.offset : ref.offset + 1]
        array = np.array(view, copy=True)
        if storage.is_bf16:
            array = (array.astype(np.uint32) << 16).view(np.float32)
        return array


# ---------------------------------------------------------------------------
# checkpoint structure helpers
# ---------------------------------------------------------------------------


def _plain(value: Any) -> Any:
    """Drop the opaque placeholders the unpickler leaves in ``config`` blobs."""
    if isinstance(value, _Opaque):
        return None
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return None


def select_state_dict(root: Mapping[str, Any], *, prefer_ema: bool) -> dict[str, _TensorRef]:
    candidates: list[str] = []
    if prefer_ema:
        candidates.append("ema_state_dict")
    candidates += ["model_state_dict", "state_dict"]
    for key in candidates:
        value = root.get(key)
        if isinstance(value, Mapping) and value:
            return _normalize_keys(value)
    if all(isinstance(v, _TensorRef) for v in root.values()):
        return _normalize_keys(root)
    raise ValueError("checkpoint contains no recognizable state dict")


def _normalize_keys(state: Mapping[str, Any]) -> dict[str, _TensorRef]:
    """Strip DDP/`head.` prefixes so flat and nested layouts converge."""
    out: dict[str, _TensorRef] = {}
    for raw_key, value in state.items():
        if not isinstance(value, _TensorRef):
            continue
        key = str(raw_key)
        if key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("head."):
            key = key[len("head.") :]
        out[key] = value
    return out


def extract_model_config(root: Mapping[str, Any]) -> dict[str, Any]:
    raw = root.get("model_config")
    if not isinstance(raw, Mapping):
        run_config = root.get("config")
        if isinstance(run_config, Mapping):
            raw = run_config.get("model_config")
    if not isinstance(raw, Mapping):
        raise ValueError("checkpoint does not contain model_config")
    return {str(k): _plain(v) for k, v in raw.items()}


def detect_model_type(state: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    if "velocity_head.weight" in state:
        return MODEL_TYPE_VELOCITY
    if "chord_head.root_chord.weight" in state or "beat_head.frame_proj.weight" in state:
        return MODEL_TYPE_BEAT_CHORD
    if "interval_scorer.proj.weight" in state:
        return MODEL_TYPE_TRANSCRIPTION
    raise ValueError(
        "unrecognized checkpoint: no velocity, beat/chord or interval-scorer head found "
        f"(model_config keys: {sorted(config)[:8]})"
    )


# ---------------------------------------------------------------------------
# CQT precomputation
#
# RecursiveCQT registers its kernels and STFT windows as *non-persistent*
# buffers, so they are absent from every checkpoint -- only the 7 resampler FIR
# filters are stored.  Rebuilding them at load time would mean porting
# scipy.signal to C++, so conversion emits them instead.  This mirrors what the
# MOSS-Music converter does with SinusoidsPositionEmbedding.
# ---------------------------------------------------------------------------


def _periodic_hann(length: int) -> np.ndarray:
    """``scipy.signal.get_window("hann", N)`` -- periodic, not symmetric."""
    if length < 1:
        return np.ones(0, dtype=np.float64)
    if length == 1:
        return np.ones(1, dtype=np.float64)
    n = np.arange(length, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * n / length)


class CQTPlan:
    """Numpy port of ``RecursiveCQT.__init__``'s kernel/stage construction."""

    def __init__(
        self,
        *,
        sample_rate: int,
        hop_length: int,
        fmin: float,
        n_bins: int,
        bins_per_octave: int,
        filter_scale: float,
    ) -> None:
        if hop_length <= 0 or (hop_length & (hop_length - 1)) != 0:
            raise ValueError("hop_length must be a positive power of two")
        self.sample_rate = int(sample_rate)
        self.hop_length = int(hop_length)
        self.fmin = float(fmin)
        self.n_bins = int(n_bins)
        self.bins_per_octave = int(bins_per_octave)
        self.q = float(filter_scale) / (2.0 ** (1.0 / self.bins_per_octave) - 1.0)

        all_freqs = self.fmin * 2.0 ** (np.arange(self.n_bins) / self.bins_per_octave)

        # A partial octave is absorbed by the top stage rather than becoming one.
        if self.n_bins <= self.bins_per_octave:
            self.stage_ranges = [(0, self.n_bins)]
        else:
            remainder = self.n_bins % self.bins_per_octave
            top_bins = self.bins_per_octave + (remainder if remainder else 0)
            lower_bins = self.n_bins - top_bins
            self.stage_ranges = [
                (start, start + self.bins_per_octave)
                for start in range(0, lower_bins, self.bins_per_octave)
            ]
            self.stage_ranges.append((lower_bins, self.n_bins))
        self.n_octaves = len(self.stage_ranges)
        if self.hop_length < 2 ** (self.n_octaves - 1):
            raise ValueError("hop_length is too small for the number of recursive stages")

        self.fft_sizes: list[int] = []
        self.kernels: list[np.ndarray] = []
        self.windows: list[np.ndarray] = []
        self.stage_bins: list[tuple[int, int]] = []

        current_sr = float(self.sample_rate)
        # Stages run highest octave first, matching the forward pass.
        for octave_start, octave_end in reversed(self.stage_ranges):
            if octave_start >= octave_end:
                continue
            freqs = all_freqs[octave_start:octave_end]
            if float(freqs[-1]) > current_sr / 2.0:
                raise ValueError("requested CQT bins exceed Nyquist for this stage")
            fft_size = 2 ** int(math.ceil(math.log2(self.q * current_sr / float(freqs[0]))))
            self.fft_sizes.append(fft_size)
            self.stage_bins.append((int(octave_start), int(octave_end)))
            self.kernels.append(self._kernel(freqs, current_sr, fft_size))
            self.windows.append(_periodic_hann(fft_size).astype(np.float32))
            current_sr /= 2.0

    def _kernel(self, freqs: np.ndarray, fs: float, n_fft: int) -> np.ndarray:
        kernel = np.zeros((len(freqs), n_fft // 2 + 1), dtype=np.complex64)
        for k, freq in enumerate(freqs):
            length = min(self.q * fs / float(freq), float(n_fft))
            win = _periodic_hann(int(math.ceil(length)))
            time_idx = np.arange(len(win), dtype=np.float64)
            time_kernel = win * np.exp(2j * np.pi * float(freq) * time_idx / fs)
            norm = np.abs(time_kernel).sum()
            if norm > 1e-8:
                time_kernel = time_kernel / norm
            padded = np.zeros(n_fft, dtype=np.complex128)
            start = (n_fft - len(time_kernel)) // 2
            padded[start : start + len(time_kernel)] = time_kernel
            kernel[k, :] = np.fft.fft(padded)[: n_fft // 2 + 1]
        return kernel


def feature_extractor_geometry(config: Mapping[str, Any]) -> dict[str, Any]:
    """Reproduce ``AudioFeatureExtractor``'s derived HCQT dimensions."""
    harmonics = [float(h) for h in config.get("harmonics", (1.0, 2.0, 3.0, 4.0, 5.0))]
    bins_per_octave = int(config["cqt_bins_per_octave"])
    cqt_n_bins = int(config["cqt_n_bins"])
    fmin = float(config["cqt_fmin"])
    sample_rate = int(config["sample_rate"])
    channels = int(config.get("input_audio_channels", 2))

    min_h, max_h = min(harmonics), max(harmonics)
    fmin_large = fmin * min_h
    n_bins_large = math.ceil(cqt_n_bins + bins_per_octave * math.log2(max_h / min_h))
    max_valid_bins = math.floor(
        bins_per_octave * math.log2((sample_rate / 2.0) / fmin_large) + 1
    )
    actual_bins = min(n_bins_large, max_valid_bins)
    if actual_bins <= 0:
        raise ValueError("CQT configuration has no bins below Nyquist")
    shifts = [bins_per_octave * math.log2(h / min_h) for h in harmonics]
    return {
        "harmonics": harmonics,
        "harmonic_shifts": shifts,
        "fmin_large": fmin_large,
        "n_bins_large": n_bins_large,
        "actual_cqt_bins": actual_bins,
        "num_audio_channels": channels * len(harmonics),
        "input_audio_channels": channels,
    }


# ---------------------------------------------------------------------------
# tensor naming
# ---------------------------------------------------------------------------

# Which axis each of the two per-layer Transformers attends over.  The V1
# backbone stores [time, band] while the beat/chord backbone stores
# [pitch, time], so the index-to-axis map is model-dependent.  Both apply the
# token-axis transformer first, so naming by role keeps the runtime uniform.
_AXIS_BY_INDEX = {
    MODEL_TYPE_TRANSCRIPTION: {"0": "time", "1": "band"},
    MODEL_TYPE_VELOCITY:      {"0": "time", "1": "band"},
    MODEL_TYPE_BEAT_CHORD:    {"0": "band", "1": "time"},
}

_ATTN_SUFFIX = {
    "norm_q.gamma": "attn_norm.weight",
    "norm_context.gamma": "attn_norm_kv.weight",
    "to_q.weight": "attn_q.weight",
    "to_q.bias": "attn_q.bias",
    "to_k.weight": "attn_k.weight",
    "to_k.bias": "attn_k.bias",
    "to_v.weight": "attn_v.weight",
    "to_v.bias": "attn_v.bias",
    "to_gates.weight": "attn_gate.weight",
    "to_gates.bias": "attn_gate.bias",
    "to_out.0.weight": "attn_out.weight",
    "to_out.0.bias": "attn_out.bias",
    # RoPE inverse frequencies are a closed form of (head_dim, base=10000);
    # the runtime regenerates them, so they are dropped here.
    "rope.inv_freq": None,
}

_FFN_SUFFIX = {
    "net.0.gamma": "ffn_norm.weight",
    "net.1.weight": "ffn_up.weight",
    "net.1.bias": "ffn_up.bias",
    "net.4.weight": "ffn_down.weight",
    "net.4.bias": "ffn_down.bias",
}

# LayerNorm + Linear + GELU + Dropout + Linear, used by every "adapter" module.
_ADAPTER_SLOT = {"0": "norm", "1": "fc1", "4": "fc2"}

_LAYER_RE = re.compile(r"^backbone\.layers\.(\d+)\.([01])\.layers\.0\.([01])\.(.+)$")

_STATIC_RULES: list[tuple[re.Pattern[str], str | None]] = [
    (re.compile(r"^backbone\.band_type_embed$"), "bb.band_type_embd"),
    (re.compile(r"^backbone\.pitch_type_embed$"), "bb.pitch_type_embd"),
    (re.compile(r"^backbone\.pitch_pos_embed$"), "bb.pitch_pos_embd"),
    (re.compile(r"^backbone\.global_token$"), "bb.global_token"),
    (re.compile(r"^backbone\.global_tokens$"), "bb.global_tokens"),
    (re.compile(r"^backbone\.global_type_embed$"), "bb.global_type_embd"),
    (re.compile(r"^backbone\.feature_extractor\.cqt\.resamplers\.(\d+)\.weight$"),
     r"cqt.resampler.\1.weight"),
    (re.compile(r"^backbone\.stem\.freq_embed$"), "bb.stem.freq_embd"),
    (re.compile(r"^backbone\.stem\.pitch_embed$"), "bb.stem.pitch_embd"),
    (re.compile(r"^backbone\.stem\.(conv1|conv2|channel_proj)\.(weight|bias)$"),
     r"bb.stem.\1.\2"),
    (re.compile(r"^backbone\.stem\.block(\d)\.0\.(weight|bias)$"), None),   # handled below
    (re.compile(r"^backbone\.stem\.block(\d)\.1\.(weight|bias)$"), None),   # handled below
    (re.compile(r"^backbone\.pitch_query_embed\.mlp\.0\.(weight|bias)$"),
     r"bb.pitch_query.fc1.\1"),
    (re.compile(r"^backbone\.pitch_query_embed\.mlp\.2\.(weight|bias)$"),
     r"bb.pitch_query.fc2.\1"),
    (re.compile(r"^backbone\.input_proj\.(weight|bias)$"), r"bb.input_proj.\1"),
    (re.compile(r"^backbone\.final_norm\.gamma$"), "bb.output_norm.weight"),
    (re.compile(r"^backbone\.up_conv\.(weight|bias)$"), r"bb.up_conv.\1"),
    (re.compile(r"^backbone\.global_up_conv\.(weight|bias)$"), r"bb.global_up_conv.\1"),
    (re.compile(r"^backbone\.global_token_merge\.(weight|bias)$"),
     r"bb.global_token_merge.\1"),

    # --- transcription head ---
    (re.compile(r"^slot_embedding\.weight$"), "head.slot_embd.weight"),
    (re.compile(r"^interval_scorer\.proj\.(weight|bias)$"), r"head.interval_scorer.\1"),
    (re.compile(r"^interval_boundary_predictor\.net\.0\.(weight|bias)$"),
     r"head.boundary.fc1.\1"),
    (re.compile(r"^interval_boundary_predictor\.net\.3\.(weight|bias)$"),
     r"head.boundary.fc2.\1"),
    (re.compile(r"^instrument_classifier\.(weight|bias)$"),
     r"head.instrument_classifier.\1"),

    # --- velocity head ---
    (re.compile(r"^(pitch|program|drum|stem)_embedding\.weight$"), r"head.\1_embd.weight"),
    (re.compile(r"^duration_encoder\.0\.(weight|bias)$"), r"head.duration.fc1.\1"),
    (re.compile(r"^duration_encoder\.2\.(weight|bias)$"), r"head.duration.fc2.\1"),
    (re.compile(r"^note_query\.0\.(weight|bias)$"), r"head.note_query.fc1.\1"),
    (re.compile(r"^note_query\.3\.(weight|bias)$"), r"head.note_query.fc2.\1"),
    (re.compile(r"^local_audio_projection\.(weight|bias)$"), r"head.local_proj.\1"),
    (re.compile(r"^local_attention\.0\.(weight|bias)$"), r"head.local_attn.fc1.\1"),
    (re.compile(r"^local_attention\.2\.(weight|bias)$"), r"head.local_attn.fc2.\1"),
    (re.compile(r"^note_fusion\.0\.(weight|bias)$"), r"head.note_fusion.norm.\1"),
    (re.compile(r"^note_fusion\.1\.(weight|bias)$"), r"head.note_fusion.fc.\1"),
    (re.compile(r"^velocity_head\.(weight|bias)$"), r"head.velocity.\1"),
    (re.compile(r"^global_audio_projection\.0\.(weight|bias)$"), r"head.global_proj.\1"),
    (re.compile(r"^stem_gain_head\.0\.(weight|bias)$"), r"head.stem_gain.fc1.\1"),
    (re.compile(r"^stem_gain_head\.3\.(weight|bias)$"), r"head.stem_gain.fc2.\1"),
    (re.compile(r"^stem_gain_head\.5\.(weight|bias)$"), r"head.stem_gain.fc3.\1"),

    # --- beat / chord / key head ---
    (re.compile(r"^(beat|chord)_head\.shared\.0\.(weight|bias)$"), r"head.\1.norm.\2"),
    (re.compile(r"^(beat|chord)_head\.shared\.1\.(weight|bias)$"), r"head.\1.fc.\2"),
    (re.compile(r"^beat_head\.frame_proj\.(weight|bias)$"), r"head.beat.frame.\1"),
    (re.compile(r"^beat_head\.group_boundary_proj\.(weight|bias)$"),
     r"head.beat.group_boundary.\1"),
    (re.compile(r"^chord_head\.(boundary|root_chord|bass|key_boundary|key|pitch)\.(weight|bias)$"),
     r"head.chord.\1.\2"),

    # --- beat/chord intermediate refinement stages ---
    (re.compile(r"^inter_global_up_convs\.(\d+)\.(weight|bias)$"),
     r"head.inter.\1.global_up_conv.\2"),
    (re.compile(r"^inter_global_token_merges\.(\d+)\.(weight|bias)$"),
     r"head.inter.\1.global_token_merge.\2"),
    (re.compile(r"^inter_(beat|chord)_down_convs\.(\d+)\.(weight|bias)$"),
     r"head.inter.\2.\1_down_conv.\3"),
    (re.compile(r"^inter_(beat|chord)_feedback_projs\.(\d+)\.(weight|bias)$"),
     r"head.inter.\2.\1_feedback.\3"),
    (re.compile(r"^inter_feedback_gates_(beat|chord)\.(\d+)$"),
     r"head.inter.\2.\1_feedback_gate"),
    (re.compile(r"^inter_beat_heads\.(\d+)\.frame_proj\.(weight|bias)$"),
     r"head.inter.\1.beat.frame.\2"),
    (re.compile(r"^inter_beat_heads\.(\d+)\.group_boundary_proj\.(weight|bias)$"),
     r"head.inter.\1.beat.group_boundary.\2"),
    (re.compile(r"^inter_chord_heads\.(\d+)\.(boundary|root_chord|bass|key_boundary|key|pitch)\.(weight|bias)$"),
     r"head.inter.\1.chord.\2.\3"),
]

_STEM_BLOCK_CONV = re.compile(r"^backbone\.stem\.block(\d)\.0\.(weight|bias)$")
_STEM_BLOCK_NORM = re.compile(r"^backbone\.stem\.block(\d)\.1\.(weight|bias)$")
_ADAPTER_RE = re.compile(r"^(interval|instrument|beat|chord)_adapter\.net\.(\d)\.(weight|bias)$")
_INTER_ADAPTER_RE = re.compile(
    r"^inter_(beat|chord)_adapters\.(\d+)\.net\.(\d)\.(weight|bias)$"
)
_INTER_HEAD_SHARED_RE = re.compile(
    r"^inter_(beat|chord)_heads\.(\d+)\.shared\.([01])\.(weight|bias)$"
)
_IIP_RE = re.compile(r"^interval_instrument_predictor\.net\.(\d)\.(weight|bias)$")
_IIP_SLOT = {"0": "norm", "1": "fc1", "4": "fc2"}


class UnmappedTensor(Exception):
    pass


def map_tensor_name(key: str, model_type: str = MODEL_TYPE_TRANSCRIPTION) -> str | None:
    """Return the GGUF name for a checkpoint key, or ``None`` to drop it."""

    match = _LAYER_RE.match(key)
    if match:
        layer, axis_idx, sub_idx, suffix = match.groups()
        table = _ATTN_SUFFIX if sub_idx == "0" else _FFN_SUFFIX
        if suffix not in table:
            raise UnmappedTensor(key)
        mapped = table[suffix]
        if mapped is None:
            return None
        axis = _AXIS_BY_INDEX[model_type][axis_idx]
        return f"bb.blk.{int(layer)}.{axis}.{mapped}"

    match = _STEM_BLOCK_CONV.match(key)
    if match:
        return f"bb.stem.blk.{int(match.group(1)) - 1}.conv.{match.group(2)}"

    match = _STEM_BLOCK_NORM.match(key)
    if match:
        return f"bb.stem.blk.{int(match.group(1)) - 1}.norm.{match.group(2)}"

    match = _ADAPTER_RE.match(key)
    if match:
        task, slot, param = match.groups()
        if slot not in _ADAPTER_SLOT:
            raise UnmappedTensor(key)
        return f"head.{task}_adapter.{_ADAPTER_SLOT[slot]}.{param}"

    match = _INTER_ADAPTER_RE.match(key)
    if match:
        task, stage, slot, param = match.groups()
        if slot not in _ADAPTER_SLOT:
            raise UnmappedTensor(key)
        return f"head.inter.{int(stage)}.{task}_adapter.{_ADAPTER_SLOT[slot]}.{param}"

    match = _INTER_HEAD_SHARED_RE.match(key)
    if match:
        task, stage, slot, param = match.groups()
        return f"head.inter.{int(stage)}.{task}.{'norm' if slot == '0' else 'fc'}.{param}"

    match = _IIP_RE.match(key)
    if match:
        slot, param = match.groups()
        if slot not in _IIP_SLOT:
            raise UnmappedTensor(key)
        return f"head.interval_instrument.{_IIP_SLOT[slot]}.{param}"

    for pattern, replacement in _STATIC_RULES:
        if replacement is None:
            continue
        if pattern.match(key):
            return pattern.sub(replacement, key)

    raise UnmappedTensor(key)


# ---------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------

# model_config fields that are training-only and carry no inference meaning.
_SKIP_CONFIG_KEYS = {
    "use_gradient_checkpoint",
    "spec_augment_params",
    "harmonic_dropout_p",
    "dropout",
    "time_mask_prob",
    "time_mask_duration_ms",
    "inter_loss_weight",
}


def _write_scalar(writer: gguf.GGUFWriter, key: str, value: Any) -> bool:
    if isinstance(value, bool):
        writer.add_bool(key, value)
    elif isinstance(value, int):
        writer.add_int32(key, value)
    elif isinstance(value, float):
        writer.add_float32(key, value)
    elif isinstance(value, str):
        writer.add_string(key, value)
    else:
        return False
    return True


def write_config_metadata(writer: gguf.GGUFWriter, config: Mapping[str, Any]) -> None:
    for name in sorted(config):
        if name in _SKIP_CONFIG_KEYS:
            continue
        value = config[name]
        if value is None:
            continue
        key = f"{ARCH}.{name}"
        if _write_scalar(writer, key, value):
            continue
        if isinstance(value, (list, tuple)) and value:
            flat = [_plain(v) for v in value]
            if all(isinstance(v, bool) for v in flat):
                continue  # GGUF has no bool array we need here
            if all(isinstance(v, int) and not isinstance(v, bool) for v in flat):
                writer.add_array(key, [int(v) for v in flat])
                continue
            if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in flat):
                writer.add_array(key, [float(v) for v in flat])
                continue
            if all(isinstance(v, str) for v in flat):
                writer.add_array(key, list(flat))
                continue
            if all(isinstance(v, (list, tuple)) for v in flat):
                # e.g. inter_refine_layers-style nested lists -> JSON
                writer.add_string(key, json.dumps(flat))
                continue
        logger.debug("skipping unrepresentable model_config entry: %s", name)


def write_cqt_metadata(
    writer: gguf.GGUFWriter,
    plan: CQTPlan,
    geom: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    writer.add_int32(f"{ARCH}.cqt.n_stages", plan.n_octaves)
    writer.add_array(f"{ARCH}.cqt.fft_sizes", [int(v) for v in plan.fft_sizes])
    writer.add_array(f"{ARCH}.cqt.stage_bin_start", [int(a) for a, _ in plan.stage_bins])
    writer.add_array(f"{ARCH}.cqt.stage_bin_end", [int(b) for _, b in plan.stage_bins])
    writer.add_float32(f"{ARCH}.cqt.q", float(plan.q))
    writer.add_float32(f"{ARCH}.cqt.fmin_large", float(geom["fmin_large"]))
    writer.add_int32(f"{ARCH}.cqt.actual_bins", int(geom["actual_cqt_bins"]))
    writer.add_int32(f"{ARCH}.cqt.n_bins_large", int(geom["n_bins_large"]))
    writer.add_array(
        f"{ARCH}.cqt.harmonic_shifts", [float(v) for v in geom["harmonic_shifts"]]
    )
    writer.add_int32(f"{ARCH}.num_audio_channels", int(geom["num_audio_channels"]))
    if "input_audio_channels" not in config:
        # Defaulted to stereo by AudioFeatureExtractor; make it explicit.
        writer.add_int32(f"{ARCH}.input_audio_channels", int(geom["input_audio_channels"]))


def add_cqt_tensors(writer: gguf.GGUFWriter, plan: CQTPlan) -> int:
    count = 0
    for index, (kernel, window) in enumerate(zip(plan.kernels, plan.windows)):
        # [n_freqs, n_fft/2+1] complex -> two f32 tensors, GGUF ne = [n_rfft, n_freqs]
        writer.add_tensor(f"cqt.kernel.{index}.real", np.ascontiguousarray(kernel.real, dtype=np.float32))
        writer.add_tensor(f"cqt.kernel.{index}.imag", np.ascontiguousarray(kernel.imag, dtype=np.float32))
        writer.add_tensor(f"cqt.window.{index}", np.ascontiguousarray(window, dtype=np.float32))
        count += 3
    return count


# ---------------------------------------------------------------------------
# conversion
# ---------------------------------------------------------------------------

# Small/precision-critical tensors that stay f32 even with --outtype f16.
_KEEP_F32 = (
    "cqt.",
    "bb.output_norm",
    "head.interval_scorer",
    "head.slot_embd",
)


def _keep_f32(name: str) -> bool:
    if name.startswith(_KEEP_F32):
        return True
    return name.endswith((".bias", "norm.weight", "attn_gate.weight", "_feedback_gate"))


def iter_tensors(
    checkpoint: TorchCheckpoint,
    state: Mapping[str, _TensorRef],
    *,
    ftype: str,
    model_type: str,
) -> Iterator[tuple[str, np.ndarray]]:
    dropped: list[str] = []
    for key in state:
        name = map_tensor_name(key, model_type)
        if name is None:
            dropped.append(key)
            continue
        data = checkpoint.materialize(state[key])
        if data.dtype != np.float32 and np.issubdtype(data.dtype, np.floating):
            data = data.astype(np.float32)
        if ftype == "f16" and data.dtype == np.float32 and data.ndim >= 2 and not _keep_f32(name):
            data = data.astype(np.float16)
        yield name, np.ascontiguousarray(data)
    if dropped:
        shown = ", ".join(sorted(dropped)[:4])
        if len(dropped) > 4:
            shown += ", ..."
        logger.info("dropped %d regenerable tensor(s): %s", len(dropped), shown)


def convert(
    src: Path,
    dst: Path,
    *,
    ftype: str,
    prefer_ema: bool,
    embed_cqt: bool,
) -> None:
    with TorchCheckpoint(src) as checkpoint:
        root = checkpoint.root
        config = extract_model_config(root)
        state = select_state_dict(root, prefer_ema=prefer_ema)
        model_type = detect_model_type(state, config)
        logger.info("%s: %s, %d tensors", src.name, model_type, len(state))

        writer = gguf.GGUFWriter(path=None, arch=ARCH)
        writer.add_type("iaamt")
        writer.add_name(src.stem)
        writer.add_description(
            "instrument-agnostic-amt checkpoint converted from "
            f"{src.name} ({model_type})"
        )
        writer.add_file_type(
            gguf.LlamaFileType.MOSTLY_F16 if ftype == "f16" else gguf.LlamaFileType.ALL_F32
        )
        writer.add_string(f"{ARCH}.model_type", model_type)
        writer.add_string(f"{ARCH}.source_checkpoint", src.name)
        write_config_metadata(writer, config)

        # Structural facts the state dict encodes but model_config does not.
        # write_config_metadata already emitted every key model_config carries,
        # so only fill in the ones an older checkpoint may be missing.
        if "num_pitch_slots" not in config:
            writer.add_int32(f"{ARCH}.num_pitch_slots", 1)
        if "encoder_num_layers" not in config and "num_layers" in config:
            writer.add_int32(f"{ARCH}.encoder_num_layers", int(config["num_layers"]))
        writer.add_bool(
            f"{ARCH}.has_interval_instrument_head",
            any(k.startswith("interval_instrument_predictor.") for k in state),
        )
        writer.add_bool(f"{ARCH}.has_slot_embedding", "slot_embedding.weight" in state)

        # Auxiliary vocabularies live outside model_config in the beat checkpoint.
        for extra in ("beat_meter_classes", "chord_quality_map", "inference_config"):
            value = _plain(root.get(extra))
            if value is not None:
                writer.add_string(f"{ARCH}.{extra}", json.dumps(value, sort_keys=True))

        n_cqt = 0
        if model_type != MODEL_TYPE_BEAT_CHORD:
            geom = feature_extractor_geometry(config)
            plan = CQTPlan(
                sample_rate=int(config["sample_rate"]),
                hop_length=int(config["hop_length"]),
                fmin=geom["fmin_large"],
                n_bins=geom["actual_cqt_bins"],
                bins_per_octave=int(config["cqt_bins_per_octave"]),
                filter_scale=float(config["cqt_filter_scale"]),
            )
            n_resamplers = sum(
                1 for k in state if k.startswith("backbone.feature_extractor.cqt.resamplers.")
            )
            if n_resamplers != plan.n_octaves - 1:
                raise ValueError(
                    f"CQT plan derived {plan.n_octaves} stages ({plan.n_octaves - 1} "
                    f"resamplers) but the checkpoint stores {n_resamplers}"
                )
            write_cqt_metadata(writer, plan, geom, config)
            if embed_cqt:
                n_cqt = add_cqt_tensors(writer, plan)
            writer.add_bool(f"{ARCH}.cqt.kernels_embedded", embed_cqt)

        n_written = 0
        for name, data in iter_tensors(checkpoint, state, ftype=ftype,
                                       model_type=model_type):
            writer.add_tensor(name, data)
            n_written += 1

        dst.parent.mkdir(parents=True, exist_ok=True)
        writer.open_output_file(dst)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=True)
        writer.close()

    logger.info(
        "wrote %s (%d weight tensors + %d precomputed CQT tensors)",
        dst, n_written, n_cqt,
    )


def default_output(src: Path, outdir: Path, ftype: str) -> Path:
    return outdir / f"{src.stem}-{ftype.upper()}.gguf"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert instrument-agnostic-amt .pth checkpoints to GGUF",
    )
    parser.add_argument(
        "checkpoints", type=Path, nargs="+",
        help="one or more best_*.pth files, or directories to scan for them",
    )
    parser.add_argument(
        "--outfile", type=Path,
        help="output .gguf path (only valid with a single input checkpoint)",
    )
    parser.add_argument(
        "--outdir", type=Path, default=Path("."),
        help="directory for generated .gguf files (default: current directory)",
    )
    parser.add_argument("--outtype", choices=["f32", "f16"], default="f32")
    parser.add_argument(
        "--ema", action="store_true",
        help="prefer ema_state_dict when the checkpoint stores one",
    )
    parser.add_argument(
        "--no-embed-cqt", action="store_true",
        help="omit precomputed CQT kernels/windows (the runtime must rebuild them)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    sources: list[Path] = []
    for entry in args.checkpoints:
        if entry.is_dir():
            sources.extend(sorted(entry.glob("best_*.pth")))
        else:
            sources.append(entry)
    if not sources:
        parser.error("no checkpoints found")
    if args.outfile and len(sources) > 1:
        parser.error("--outfile requires exactly one input checkpoint")

    failures = 0
    for src in sources:
        dst = args.outfile or default_output(src, args.outdir, args.outtype)
        try:
            convert(
                src, dst,
                ftype=args.outtype,
                prefer_ema=args.ema,
                embed_cqt=not args.no_embed_cqt,
            )
        except UnmappedTensor as exc:
            failures += 1
            logger.error("%s: no GGUF name for checkpoint tensor %s", src.name, exc)
        except Exception as exc:
            failures += 1
            logger.error("%s: %s", src.name, exc)
            if args.verbose:
                raise
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
