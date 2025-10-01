#!/usr/bin/env python3
"""Validator for Gate G3 (Samples)."""
from __future__ import annotations

import json
from pathlib import Path

import math
import numpy as np

REQUIRED_FILES = [
    "Vp_small.npy",
    "Vs_small.npy",
    "rho_small.npy",
    "lambda_small.npy",
    "mu_small.npy",
    "meta_small.json",
    "metrics.json",
]

EVIDENCE_TAGS = ["[E-sample-shape]", "[E-sample-dx]", "[E-sample-size]", "[E-sample-provenance]"]


class ValidationError(RuntimeError):
    pass


def load_meta(path: Path) -> dict:
    meta = json.loads(path.read_text())
    if "provenance" not in meta:
        raise ValidationError("meta_small missing provenance")
    for key in ("x0", "x1", "z0", "z1", "stride", "dx", "dz"):
        if key not in meta["provenance"]:
            raise ValidationError(f"provenance missing {key}")
    return meta


def validate_arrays(samples_dir: Path, meta: dict) -> None:
    nz, nx = int(meta["nz"]), int(meta["nx"])
    for name in ("Vp", "Vs", "rho", "lambda", "mu"):
        arr = np.load(samples_dir / f"{name}_small.npy")
        if arr.shape != (nz, nx):
            raise ValidationError(f"{name}_small shape {arr.shape} != {(nz, nx)}")
        if not np.isfinite(arr).all():
            raise ValidationError(f"{name}_small contains non finite values")


def validate_metrics(path: Path, meta: dict) -> None:
    metrics = json.loads(path.read_text())
    stride = meta["provenance"]["stride"]
    dx = meta["provenance"]["dx"]
    dz = meta["provenance"]["dz"]
    if not math.isclose(metrics.get("dx_prime", -1.0), dx * stride, rel_tol=1e-6):
        raise ValidationError("dx_prime incorrect")
    if not math.isclose(metrics.get("dz_prime", -1.0), dz * stride, rel_tol=1e-6):
        raise ValidationError("dz_prime incorrect")
    if metrics.get("total_bytes", 10 ** 9) > 50 * 2 ** 20:
        raise ValidationError("Sample size exceeds 50 MB")
    if metrics.get("evidence") != EVIDENCE_TAGS:
        raise ValidationError("Evidence tags mismatch")


def main() -> None:
    samples_dir = Path("02-data/marmousi2/samples")
    for name in REQUIRED_FILES:
        if not (samples_dir / name).exists():
            raise ValidationError(f"Missing {name}")
    meta = load_meta(samples_dir / "meta_small.json")
    validate_arrays(samples_dir, meta)
    validate_metrics(samples_dir / "metrics.json", meta)
    print("Gate G3 passed", " ".join(EVIDENCE_TAGS))


if __name__ == "__main__":
    try:
        main()
    except ValidationError as exc:
        print(f"Gate G3 failure: {exc}")
        raise SystemExit(1)
