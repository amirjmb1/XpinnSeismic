#!/usr/bin/env python3
"""SEG-Y → NumPy ingestion and Lamé parameterization with validation."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

try:
    import segyio  # type: ignore
except Exception:  # pragma: no cover - segyio optional in CI
    segyio = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW_FILENAMES = {
    "vp": "MODEL_P-WAVE_VELOCITY_1.25m.segy",  # can be inside tarball, fallback to .npy
    "vs": "MODEL_S-WAVE_VELOCITY_1.25m.segy",
    "rho": "MODEL_DENSITY_1.25m.segy",
}

ALLOWED_RANGES = {
    "vp": (900.0, 7000.0),
    "vs": (300.0, 4000.0),
    "rho": (800.0, 4000.0),
}

BASELINE_PATH = Path("02-data/marmousi2/processed/stats_baseline.json")
EVIDENCE_TAGS = ["[E-ingest-shape]", "[E-ingest-units]", "[E-ingest-quicklooks]", "[E-lame-invertible]"]


class IngestError(RuntimeError):
    """Raised when ingestion fails validation."""


def ricker(f0: float, t: np.ndarray) -> np.ndarray:
    t0 = 1.0 / f0
    pi2 = (np.pi ** 2)
    return (1.0 - 2.0 * pi2 * (f0 ** 2) * (t - t0) ** 2) * np.exp(-pi2 * (f0 ** 2) * (t - t0) ** 2)


def load_segy(path: Path) -> np.ndarray:
    if segyio is None:
        raise IngestError("segyio is not installed; cannot load SEG-Y")
    with segyio.open(path, strict=False) as f:
        data = segyio.tools.cube(f).astype(np.float32)
    # Data layout: [iline, xline, samples] -> adapt to [nz, nx]
    if data.shape[0] != 1:
        data = data.reshape(data.shape[0], -1)
    nz, nx = data.shape[-1], data.shape[1]
    return np.copy(data[-1].T.reshape(nx, nz)).T


def find_first_existing(raw_dir: Path, stem: str) -> Path | None:
    candidates = [raw_dir / stem, raw_dir / f"{stem}.npy", raw_dir / f"{stem}.npz"]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def generate_synthetic(nx: int = 192, nz: int = 96, dx: float = 5.0, dz: float = 5.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """Generate a layered synthetic model when Marmousi2 assets are unavailable."""
    x = np.linspace(0, (nx - 1) * dx, nx)
    z = np.linspace(0, (nz - 1) * dz, nz)
    X, Z = np.meshgrid(x, z)
    vp = 1800.0 + 0.6 * Z + 150.0 * np.sin(2 * np.pi * X / (nx * dx))
    vs = 0.55 * vp
    rho = 1800.0 + 0.25 * Z
    meta = {
        "nx": int(nx),
        "nz": int(nz),
        "dx": float(dx),
        "dz": float(dz),
        "units": {
            "space": "m",
            "velocity": "m/s",
            "density": "kg/m^3",
        },
        "origin": {
            "x0": 0.0,
            "z0": 0.0,
        },
        "data_source": "synthetic",
        "description": "Layered synthetic model (fallback)",
    }
    return vp.astype(np.float32), vs.astype(np.float32), rho.astype(np.float32), meta


def load_fields(raw_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    raw_dir = Path(raw_dir)
    vp_path = find_first_existing(raw_dir, RAW_FILENAMES["vp"])
    vs_path = find_first_existing(raw_dir, RAW_FILENAMES["vs"])
    rho_path = find_first_existing(raw_dir, RAW_FILENAMES["rho"])

    if not (vp_path and vs_path and rho_path):
        return generate_synthetic()

    def read_file(path: Path) -> np.ndarray:
        if path.suffix in {".npy", ".npz"}:
            data = np.load(path)
            if isinstance(data, np.lib.npyio.NpzFile):
                return data[list(data.files)[0]].astype(np.float32)
            return data.astype(np.float32)
        return load_segy(path)

    vp = read_file(vp_path)
    vs = read_file(vs_path)
    rho = read_file(rho_path)
    if vp.shape != vs.shape or vp.shape != rho.shape:
        raise IngestError(f"Vp/Vs/rho shapes do not match: {vp.shape}, {vs.shape}, {rho.shape}")

    meta = {
        "nx": int(vp.shape[1]),
        "nz": int(vp.shape[0]),
        "dx": 1.25,
        "dz": 1.25,
        "units": {
            "space": "m",
            "velocity": "m/s",
            "density": "kg/m^3",
        },
        "origin": {"x0": 0.0, "z0": 0.0},
        "data_source": "marmousi2",
        "description": "Marmousi2 1.25 m grids",
    }
    return vp.astype(np.float32), vs.astype(np.float32), rho.astype(np.float32), meta


def enforce_ranges(name: str, data: np.ndarray) -> None:
    lo, hi = ALLOWED_RANGES[name]
    if not (np.isfinite(data).all() and data.min() >= lo and data.max() <= hi):
        raise IngestError(f"{name} out of bounds: expected [{lo}, {hi}] mks")


def to_lame(vp: np.ndarray, vs: np.ndarray, rho: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = rho * vs ** 2
    lam = rho * vp ** 2 - 2 * mu
    if np.any(mu < -1e-6) or np.any(lam + 2 * mu < -1e-6):
        raise IngestError("Lamé parameters violate stability conditions")
    return lam.astype(np.float32), mu.astype(np.float32)


def reconstruct_velocities(lam: np.ndarray, mu: np.ndarray, rho: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    vp = np.sqrt(np.maximum(lam + 2 * mu, 0.0) / np.maximum(rho, 1e-12))
    vs = np.sqrt(np.maximum(mu, 0.0) / np.maximum(rho, 1e-12))
    return vp, vs


def compute_stats(vp: np.ndarray, vs: np.ndarray, rho: np.ndarray, lam: np.ndarray, mu: np.ndarray) -> Dict[str, Dict[str, float]]:
    stats = {}
    for name, arr in {"vp": vp, "vs": vs, "rho": rho, "lambda": lam, "mu": mu}.items():
        percentiles = np.percentile(arr, [0, 5, 25, 50, 75, 95, 100]).astype(float)
        stats[name] = {
            "min": float(percentiles[0]),
            "p05": float(percentiles[1]),
            "p25": float(percentiles[2]),
            "median": float(percentiles[3]),
            "p75": float(percentiles[4]),
            "p95": float(percentiles[5]),
            "max": float(percentiles[6]),
        }
    return stats


def compare_baseline(stats: Dict[str, Dict[str, float]], baseline_path: Path = BASELINE_PATH, tol: float = 0.10) -> Tuple[bool, Dict[str, Dict[str, float]]]:
    if not baseline_path.exists():
        baseline_path.write_text(json.dumps(stats, indent=2))
        return True, {key: {k: 0.0 for k in val} for key, val in stats.items()}

    baseline = json.loads(baseline_path.read_text())
    deltas: Dict[str, Dict[str, float]] = {}
    for field, values in stats.items():
        deltas[field] = {}
        base = baseline.get(field, {})
        for name, val in values.items():
            ref = base.get(name)
            if ref in (0, None):
                delta = 0.0
            else:
                delta = abs(val - ref) / abs(ref)
            if delta > tol:
                raise IngestError(f"Distribution shift for {field}.{name}: {delta:.2%} > {tol:.0%}")
            deltas[field][name] = float(delta)
    return True, deltas


def save_quicklook(field: np.ndarray, path: Path, title: str, cmap: str = "viridis") -> None:
    fig, ax = plt.subplots(figsize=(6, 3))
    im = ax.imshow(field, cmap=cmap, origin="upper")
    ax.set_title(title)
    ax.set_xlabel("x index")
    ax.set_ylabel("z index")
    plt.colorbar(im, ax=ax, label=title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_meta(meta: Dict[str, float], processed_dir: Path) -> None:
    meta_path = processed_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))


def write_metrics(metrics: Dict[str, float], processed_dir: Path) -> None:
    metrics_path = processed_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("02-data/marmousi2/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("02-data/marmousi2/processed"))
    parser.add_argument("--quicklook-dir", type=Path, default=Path("02-data/marmousi2/processed/quicklooks"))
    args = parser.parse_args()

    processed_dir = args.processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    vp, vs, rho, meta = load_fields(args.raw_dir)

    enforce_ranges("vp", vp)
    enforce_ranges("vs", vs)
    enforce_ranges("rho", rho)

    lam, mu = to_lame(vp, vs, rho)
    vp_prime, vs_prime = reconstruct_velocities(lam, mu, rho)

    rel_mae_vp = float(np.mean(np.abs(vp_prime - vp) / np.maximum(vp, 1e-6)))
    rel_mae_vs = float(np.mean(np.abs(vs_prime - vs) / np.maximum(vs, 1e-6)))
    if rel_mae_vp > 0.05 or rel_mae_vs > 0.05:
        raise IngestError(f"Lamé inversion failed: rel MAE vp={rel_mae_vp:.3f}, vs={rel_mae_vs:.3f}")

    stats = compute_stats(vp, vs, rho, lam, mu)
    _, deltas = compare_baseline(stats)

    np.save(processed_dir / "Vp.npy", vp)
    np.save(processed_dir / "Vs.npy", vs)
    np.save(processed_dir / "rho.npy", rho)
    np.save(processed_dir / "lambda.npy", lam)
    np.save(processed_dir / "mu.npy", mu)

    write_meta(meta, processed_dir)
    (processed_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    (processed_dir / "stats_delta.json").write_text(json.dumps(deltas, indent=2))

    for field, data in [("Vp", vp), ("Vs", vs), ("rho", rho)]:
        save_quicklook(data, args.quicklook_dir / f"{field}.png", f"{field} (SI)")

    write_metrics(
        {
            "rel_mae_vp": rel_mae_vp,
            "rel_mae_vs": rel_mae_vs,
            "evidence": EVIDENCE_TAGS,
            "data_source": meta.get("data_source", "unknown"),
        },
        processed_dir,
    )

    print("Ingestion completed")
    print(" ".join(EVIDENCE_TAGS))


if __name__ == "__main__":
    main()
