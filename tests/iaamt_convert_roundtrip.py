#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify that convert_iaamt_to_gguf.py preserves every checkpoint tensor.

Converts one or more ``best_*.pth`` files in a temporary directory, reads the
GGUF back with ``gguf.GGUFReader`` and compares each tensor against the source
checkpoint bit-for-bit.  Only ``rope.inv_freq`` may be absent -- it is a closed
form the runtime regenerates.

Usage:
    python tests/iaamt_convert_roundtrip.py models/
    python tests/iaamt_convert_roundtrip.py models/best_model.pth --keep out/

Requires numpy only; the source checkpoints are read without PyTorch.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(1, str(REPO_ROOT / "gguf-py"))

import gguf  # noqa: E402
from convert_iaamt_to_gguf import (  # noqa: E402
    MODEL_TYPE_BEAT_CHORD,
    CQTPlan,
    TorchCheckpoint,
    convert,
    detect_model_type,
    extract_model_config,
    feature_extractor_geometry,
    map_tensor_name,
    select_state_dict,
)

# The only checkpoint tensors conversion is allowed to drop.
REGENERABLE_SUFFIX = "rope.inv_freq"


def check_one(pth: Path, outdir: Path) -> list[str]:
    problems: list[str] = []
    dst = outdir / f"{pth.stem}-F32.gguf"
    convert(pth, dst, ftype="f32", prefer_ema=False, embed_cqt=True)

    with TorchCheckpoint(pth) as checkpoint:
        config = extract_model_config(checkpoint.root)
        state = select_state_dict(checkpoint.root, prefer_ema=False)
        model_type = detect_model_type(state, config)

        tensors = {t.name: t for t in gguf.GGUFReader(dst).tensors}
        expected: set[str] = set()

        for key, ref in state.items():
            name = map_tensor_name(key, model_type)
            if name is None:
                if not key.endswith(REGENERABLE_SUFFIX):
                    problems.append(f"{key}: dropped but not regenerable")
                continue
            expected.add(name)
            if name not in tensors:
                problems.append(f"{key} -> {name}: missing from GGUF")
                continue
            src = checkpoint.materialize(ref).astype(np.float32)
            dst_data = np.asarray(tensors[name].data)
            if dst_data.size != src.size:
                problems.append(
                    f"{name}: {src.size} elements in checkpoint, {dst_data.size} in GGUF"
                )
                continue
            if not np.array_equal(src, dst_data.reshape(src.shape)):
                delta = np.abs(src - dst_data.reshape(src.shape)).max()
                problems.append(f"{name}: values differ, max|delta| = {delta}")

        unexpected = sorted(
            name for name in tensors
            if name not in expected and not name.startswith("cqt.")
        )
        if unexpected:
            problems.append(f"GGUF holds unmapped tensors: {unexpected}")

        n_cqt = sum(1 for name in tensors if name.startswith("cqt.kernel"))
        if model_type == MODEL_TYPE_BEAT_CHORD:
            if n_cqt:
                problems.append(f"beat_chord model should carry no CQT kernels, got {n_cqt}")
        else:
            problems += check_cqt(config, tensors, n_cqt)

    print(
        f"{'FAIL' if problems else 'ok  '} {pth.name:32s} {model_type:13s} "
        f"tensors={len(tensors)}"
    )
    return [f"{pth.name}: {p}" for p in problems]


def check_cqt(config, tensors, n_cqt: int) -> list[str]:
    """The kernels are precomputed, so at least assert they are well-formed."""
    problems: list[str] = []
    geom = feature_extractor_geometry(config)
    plan = CQTPlan(
        sample_rate=int(config["sample_rate"]),
        hop_length=int(config["hop_length"]),
        fmin=geom["fmin_large"],
        n_bins=geom["actual_cqt_bins"],
        bins_per_octave=int(config["cqt_bins_per_octave"]),
        filter_scale=float(config["cqt_filter_scale"]),
    )
    if n_cqt != 2 * plan.n_octaves:
        problems.append(f"expected {2 * plan.n_octaves} kernel tensors, got {n_cqt}")

    covered = sum(end - start for start, end in plan.stage_bins)
    if covered != geom["actual_cqt_bins"]:
        problems.append(
            f"stages cover {covered} bins but the HCQT needs {geom['actual_cqt_bins']}"
        )

    for index, window in enumerate(plan.windows):
        # A periodic Hann window of length N sums to exactly N/2.
        if abs(float(window.sum()) - len(window) / 2.0) > 1e-3:
            problems.append(f"cqt.window.{index} is not a periodic Hann window")

    for index, kernel in enumerate(plan.kernels):
        if not np.isfinite(kernel).all():
            problems.append(f"cqt.kernel.{index} contains non-finite values")
        # Rows are L1-normalized in the time domain, so |FFT| never exceeds 1.
        elif np.abs(kernel).max() > 1.0 + 1e-4:
            problems.append(
                f"cqt.kernel.{index} exceeds unit gain: {np.abs(kernel).max()}"
            )

    for index in range(plan.n_octaves):
        for name in (
            f"cqt.kernel.{index}.real",
            f"cqt.kernel.{index}.imag",
            f"cqt.window.{index}",
        ):
            if name not in tensors:
                problems.append(f"{name} missing from GGUF")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoints", type=Path, nargs="+",
        help="best_*.pth files, or directories to scan",
    )
    parser.add_argument(
        "--keep", type=Path,
        help="write the GGUFs here and keep them instead of using a temp dir",
    )
    args = parser.parse_args()

    sources: list[Path] = []
    for entry in args.checkpoints:
        sources.extend(sorted(entry.glob("best_*.pth")) if entry.is_dir() else [entry])
    if not sources:
        parser.error("no checkpoints found")

    outdir = args.keep or Path(tempfile.mkdtemp(prefix="iaamt-roundtrip-"))
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        problems: list[str] = []
        for pth in sources:
            problems += check_one(pth, outdir)
    finally:
        if args.keep is None:
            shutil.rmtree(outdir, ignore_errors=True)

    if problems:
        print(f"\n{len(problems)} problem(s):")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print(f"\nall {len(sources)} checkpoint(s) round-trip exactly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
