#!/usr/bin/env python3
"""Validator for Gate G0 (Ingest)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REQUIRED_FILES = [
    "Vp.npy",
    "Vs.npy",
    "rho.npy",
    "lambda.npy",
    "mu.npy",
    "meta.json",
    "stats.json",
    "metrics.json",
]

EVIDENCE_TAGS = ["[E-ingest-shape]", "[E-ingest-units]", "[E-ingest-quicklooks]", "[E-lame-invertible]"]


class ValidationError(RuntimeError):
    pass


def assert_file(path: Path) -> None:
    if not path.exists():
        raise ValidationError(f"Missing required file: {path}")


def load_meta(path: Path) -> dict:
    data = json.loads(path.read_text())
    required = ["nx", "nz", "dx", "dz", "units"]
    for key in required:
        if key not in data:
            raise ValidationError(f"meta.json missing {key}")
    if data.get("units", {}).get("space") != "m":
        raise ValidationError("meta units not SI")
    if data.get("units", {}).get("velocity") != "m/s":
        raise ValidationError("velocity units not SI")
    if data.get("units", {}).get("density") != "kg/m^3":
        raise ValidationError("density units not SI")
    return data


def validate_arrays(processed_dir: Path, meta: dict) -> None:
    nz, nx = int(meta["nz"]), int(meta["nx"])
    for name in ("Vp", "Vs", "rho", "lambda", "mu"):
        arr = np.load(processed_dir / f"{name}.npy")
        if arr.shape != (nz, nx):
            raise ValidationError(f"{name}.npy wrong shape {arr.shape}, expected {(nz, nx)}")
        if not np.isfinite(arr).all():
            raise ValidationError(f"{name}.npy contains non-finite values")
    lam = np.load(processed_dir / "lambda.npy")
    mu = np.load(processed_dir / "mu.npy")
    rho = np.load(processed_dir / "rho.npy")
    if np.any(mu < -1e-6):
        raise ValidationError("mu contains negative values")
    if np.any(lam + 2 * mu < -1e-6):
        raise ValidationError("lambda + 2 mu < 0")
    vp = np.sqrt(np.maximum(lam + 2 * mu, 0.0) / np.maximum(rho, 1e-12))
    vs = np.sqrt(np.maximum(mu, 0.0) / np.maximum(rho, 1e-12))
    vp_ref = np.load(processed_dir / "Vp.npy")
    vs_ref = np.load(processed_dir / "Vs.npy")
    rel_mae_vp = float(np.mean(np.abs(vp - vp_ref) / np.maximum(vp_ref, 1e-6)))
    rel_mae_vs = float(np.mean(np.abs(vs - vs_ref) / np.maximum(vs_ref, 1e-6)))
    if rel_mae_vp > 0.05 or rel_mae_vs > 0.05:
        raise ValidationError(f"Lamé inversion rel MAE too high: vp {rel_mae_vp:.3f}, vs {rel_mae_vs:.3f}")


def validate_metrics(metrics_path: Path) -> None:
    metrics = json.loads(metrics_path.read_text())
    if metrics.get("rel_mae_vp", 1.0) > 0.05:
        raise ValidationError("metrics rel_mae_vp exceeds threshold")
    if metrics.get("rel_mae_vs", 1.0) > 0.05:
        raise ValidationError("metrics rel_mae_vs exceeds threshold")
    if metrics.get("evidence") != EVIDENCE_TAGS:
        raise ValidationError("evidence tags missing or incorrect")


def validate_quicklooks(quicklook_dir: Path) -> None:
    required = ["Vp.png", "Vs.png", "rho.png"]
    for name in required:
        path = quicklook_dir / name
        if not path.exists():
            raise ValidationError(f"Missing quicklook {path}")
        if path.stat().st_size < 10 * 1024:
            raise ValidationError(f"Quicklook {path} too small to be informative")


def main() -> None:
    processed_dir = Path("02-data/marmousi2/processed")
    for name in REQUIRED_FILES:
        assert_file(processed_dir / name)
    meta = load_meta(processed_dir / "meta.json")
    validate_arrays(processed_dir, meta)
    validate_metrics(processed_dir / "metrics.json")
    validate_quicklooks(Path("02-data/marmousi2/processed/quicklooks"))
    print("Gate G0 passed", " ".join(EVIDENCE_TAGS))


if __name__ == "__main__":
    try:
        main()
    except ValidationError as exc:
        print(f"Gate G0 failure: {exc}")
        raise SystemExit(1)
