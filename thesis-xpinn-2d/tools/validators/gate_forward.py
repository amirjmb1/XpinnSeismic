#!/usr/bin/env python3
"""Validator for Gate G1 (Forward Elastic)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

EVIDENCE_TAGS = ["[E-forward-cfl]", "[E-forward-P]", "[E-forward-S]", "[E-forward-sponge]", "[E-forward-nonan]"]


class ValidationError(RuntimeError):
    pass


def load_metrics(path: Path) -> dict:
    data = json.loads(path.read_text())
    for key in ("cfl", "arrival_error_p", "arrival_error_s", "reflection_ratio"):
        if key not in data:
            raise ValidationError(f"metrics missing {key}")
    return data


def check_arrays(out_dir: Path) -> None:
    for name in ("shot_vx.npy", "shot_vz.npy", "vx.npy", "vz.npy"):
        arr = np.load(out_dir / name)
        if not np.isfinite(arr).all():
            raise ValidationError(f"{name} contains non-finite values")


def main() -> None:
    out_dir = Path("04-experiments/forward_elastic_2d/out/shot_000")
    metrics = load_metrics(out_dir / "metrics.json")
    if metrics["cfl"] > 0.45:
        raise ValidationError(f"CFL {metrics['cfl']:.3f} exceeds 0.45")
    if metrics["arrival_error_p"] > 1.0:
        raise ValidationError(f"P arrival error {metrics['arrival_error_p']:.2f} > 1 sample")
    if metrics["arrival_error_s"] > 1.0:
        raise ValidationError(f"S arrival error {metrics['arrival_error_s']:.2f} > 1 sample")
    if metrics["reflection_ratio"] > 0.07:
        raise ValidationError(f"Reflection ratio {metrics['reflection_ratio']:.3f} exceeds 7%")
    if metrics.get("evidence") != EVIDENCE_TAGS:
        raise ValidationError("Evidence tags missing")
    if metrics.get("data_source") != "marmousi2" and os.getenv("ALLOW_SYNTHETIC_FOR_G1") != "1":
        raise ValidationError("Real Marmousi2 assets required for Gate G1")
    check_arrays(out_dir)
    quicklook = out_dir.parent / "quicklooks/shot_vx.png"
    if not quicklook.exists() or quicklook.stat().st_size < 10 * 1024:
        raise ValidationError("Shot quicklook missing or too small")
    print("Gate G1 passed", " ".join(EVIDENCE_TAGS))


if __name__ == "__main__":
    try:
        main()
    except ValidationError as exc:
        print(f"Gate G1 failure: {exc}")
        raise SystemExit(1)
