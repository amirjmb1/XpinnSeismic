#!/usr/bin/env python3
"""Generate camera-ready figures from forward modeling outputs."""
from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EVIDENCE_TAGS = ["[E-figs-shot]", "[E-figs-snap]", "[E-figs-meta]"]


def compute_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def shot_figure(out_dir: Path, shot: np.ndarray, dt: float, dx: float) -> Path:
    path = out_dir / "fig_elastic_shot.png"
    t = np.arange(shot.shape[0]) * dt
    x = np.arange(shot.shape[1]) * dx
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(
        shot,
        aspect="auto",
        cmap="seismic",
        origin="lower",
        extent=[x.min(), x.max(), t.min(), t.max()],
    )
    ax.set_xlabel("Offset (m)")
    ax.set_ylabel("Time (s)")
    ax.set_title("Elastic shot gather (Vx)")
    plt.colorbar(im, ax=ax, label="Amplitude")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def snapshot_figure(out_dir: Path, snapshot: np.ndarray) -> Path:
    path = out_dir / "fig_elastic_snap.png"
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(snapshot, cmap="seismic", origin="upper")
    ax.set_xlabel("x index")
    ax.set_ylabel("z index")
    ax.set_title("Elastic final snapshot (Vx)")
    plt.colorbar(im, ax=ax, label="Amplitude")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward-dir", type=Path, default=Path("04-experiments/forward_elastic_2d/out/shot_000"))
    parser.add_argument("--fig-dir", type=Path, default=Path("05-writing/figs-src/out"))
    args = parser.parse_args()

    args.fig_dir.mkdir(parents=True, exist_ok=True)

    shot = np.load(args.forward_dir / "shot_vx.npy")
    vx_snapshot = np.load(args.forward_dir / "vx.npy")
    meta = json.loads((args.forward_dir / "meta.json").read_text())
    dt = meta["dt"]
    dx = meta["dx"]

    shot_path = shot_figure(args.fig_dir, shot, dt, dx)
    snap_path = snapshot_figure(args.fig_dir, vx_snapshot)

    metadata = {
        "shot": {
            "path": str(shot_path),
            "sha256": compute_hash(shot_path),
            "source": str(args.forward_dir / "shot_vx.npy"),
        },
        "snapshot": {
            "path": str(snap_path),
            "sha256": compute_hash(snap_path),
            "source": str(args.forward_dir / "vx.npy"),
        },
        "evidence": EVIDENCE_TAGS,
    }
    (args.fig_dir / "figs_meta.json").write_text(json.dumps(metadata, indent=2))

    print("Figures generated")
    print(" ".join(EVIDENCE_TAGS))


if __name__ == "__main__":
    main()
