#!/usr/bin/env python3
"""Create CI-friendly samples from processed Marmousi2 grids."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np

DEFAULT_PROVENANCE = {
    "x0": 0,
    "x1": 128,
    "z0": 0,
    "z1": 64,
    "stride": 2,
}

EVIDENCE_TAGS = ["[E-sample-shape]", "[E-sample-dx]", "[E-sample-size]", "[E-sample-provenance]"]


def crop_and_stride(arr: np.ndarray, provenance: Dict[str, int]) -> np.ndarray:
    x0, x1 = provenance["x0"], provenance["x1"]
    z0, z1 = provenance["z0"], provenance["z1"]
    stride = provenance["stride"]
    cropped = arr[z0:z1, x0:x1]
    return cropped[::stride, ::stride]


def save_sample(processed_dir: Path, samples_dir: Path, provenance: Dict[str, int]) -> Dict[str, float]:
    meta = json.loads((processed_dir / "meta.json").read_text())
    dx = float(meta["dx"])
    dz = float(meta["dz"])
    stride = provenance["stride"]

    samples_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    total_bytes = 0
    for name in ("Vp", "Vs", "rho", "lambda", "mu"):
        data = np.load(processed_dir / f"{name}.npy")
        sampled = crop_and_stride(data, provenance)
        path = samples_dir / f"{name}_small.npy"
        np.save(path, sampled)
        total_bytes += path.stat().st_size
        summary[name] = {
            "shape": sampled.shape,
            "min": float(sampled.min()),
            "max": float(sampled.max()),
        }

    meta_small = {
        "dx": dx * stride,
        "dz": dz * stride,
        "nx": int(summary["Vp"]["shape"][1]),
        "nz": int(summary["Vp"]["shape"][0]),
        "units": meta["units"],
        "provenance": {
            **provenance,
            "dx": dx,
            "dz": dz,
        },
        "data_source": meta.get("data_source", "unknown"),
    }
    (samples_dir / "meta_small.json").write_text(json.dumps(meta_small, indent=2))
    metrics = {
        "total_bytes": total_bytes,
        "dx_prime": meta_small["dx"],
        "dz_prime": meta_small["dz"],
        "evidence": EVIDENCE_TAGS,
    }
    (samples_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("02-data/marmousi2/processed"))
    parser.add_argument("--samples-dir", type=Path, default=Path("02-data/marmousi2/samples"))
    parser.add_argument("--provenance", type=str, default=json.dumps(DEFAULT_PROVENANCE), help="JSON string describing crop")
    args = parser.parse_args()

    provenance = json.loads(args.provenance)
    metrics = save_sample(args.processed_dir, args.samples_dir, provenance)
    print("Samples created", metrics)
    print(" ".join(EVIDENCE_TAGS))


if __name__ == "__main__":
    main()
