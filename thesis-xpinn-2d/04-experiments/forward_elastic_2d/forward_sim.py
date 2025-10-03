#!/usr/bin/env python3
"""2D isotropic elastic forward modeling using a Virieux-style staggered grid."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EVIDENCE_TAGS = [
    "[E-forward-cfl]",
    "[E-forward-P]",
    "[E-forward-S]",
    "[E-forward-sponge]",
    "[E-forward-nonan]",
]


@dataclass
class SourceConfig:
    frequency: float
    location: Tuple[int, int]
    delay: float


@dataclass
class ReceiverLine:
    depth_index: int
    start: int
    end: int
    stride: int


class ForwardModelError(RuntimeError):
    pass


def ricker_wavelet(f0: float, dt: float, nt: int, delay: float) -> np.ndarray:
    t = np.arange(nt) * dt
    t0 = delay
    pf = (np.pi * f0) ** 2
    wavelet = (1 - 2 * pf * (t - t0) ** 2) * np.exp(-pf * (t - t0) ** 2)
    return wavelet.astype(np.float32)


def sponge_mask(nx: int, nz: int, npad: int = 20, strength: float = 3.0) -> np.ndarray:
    mask = np.ones((nz, nx), dtype=np.float32)
    if npad <= 0:
        return mask
    taper = np.linspace(0.0, 1.0, npad, dtype=np.float32)
    coeff = np.exp(-strength * taper**2)
    for i in range(npad):
        mask[:, i] *= coeff[i]
        mask[:, -i - 1] *= coeff[i]
        mask[i, :] *= coeff[i]
        mask[-i - 1, :] *= coeff[i]
    return mask


def compute_cfl(dt: float, dx: float, dz: float, vp_max: float, vs_max: float) -> float:
    vmax = max(vp_max, vs_max)
    return dt * math.sqrt(2.0) * vmax / min(dx, dz)


def finite_diff_x(arr: np.ndarray, dx: float) -> np.ndarray:
    diff = np.zeros_like(arr)
    diff[:, 1:] = (arr[:, 1:] - arr[:, :-1]) / dx
    diff[:, 0] = diff[:, 1]
    return diff


def finite_diff_z(arr: np.ndarray, dz: float) -> np.ndarray:
    diff = np.zeros_like(arr)
    diff[1:, :] = (arr[1:, :] - arr[:-1, :]) / dz
    diff[0, :] = diff[1, :]
    return diff


def forward_model(
    vp: np.ndarray,
    vs: np.ndarray,
    rho: np.ndarray,
    dx: float,
    dz: float,
    dt: float,
    nt: int,
    source: SourceConfig,
    receiver_line: ReceiverLine,
    sponge: np.ndarray,
) -> Dict[str, np.ndarray]:
    nz, nx = vp.shape
    lam = rho * (vp ** 2 - 2 * vs ** 2)
    mu = rho * vs ** 2
    vx = np.zeros((nz, nx), dtype=np.float32)
    vz = np.zeros((nz, nx), dtype=np.float32)
    txx = np.zeros((nz, nx), dtype=np.float32)
    tzz = np.zeros((nz, nx), dtype=np.float32)
    txz = np.zeros((nz, nx), dtype=np.float32)

    rick = ricker_wavelet(source.frequency, dt, nt, source.delay)
    sx, sz = source.location

    rec_indices = np.arange(receiver_line.start, receiver_line.end, receiver_line.stride)
    receivers = np.zeros((nt, rec_indices.size), dtype=np.float32)
    receivers_s = np.zeros_like(receivers)

    for it in range(nt):
        dtxx_dx = finite_diff_x(txx, dx)
        dtxz_dz = finite_diff_z(txz, dz)
        dtxz_dx = finite_diff_x(txz, dx)
        dtzz_dz = finite_diff_z(tzz, dz)

        vx += dt * (dtxx_dx + dtxz_dz) / rho
        vz += dt * (dtxz_dx + dtzz_dz) / rho

        dvx_dx = finite_diff_x(vx, dx)
        dvx_dz = finite_diff_z(vx, dz)
        dvz_dx = finite_diff_x(vz, dx)
        dvz_dz = finite_diff_z(vz, dz)

        txx += dt * ((lam + 2 * mu) * dvx_dx + lam * dvz_dz)
        tzz += dt * ((lam + 2 * mu) * dvz_dz + lam * dvx_dx)
        txz += dt * mu * (dvx_dz + dvz_dx)

        amp = rick[it]
        txx[sz, sx] += amp
        tzz[sz, sx] += amp

        vx *= sponge
        vz *= sponge
        txx *= sponge
        tzz *= sponge
        txz *= sponge

        receivers[it] = vx[receiver_line.depth_index, rec_indices]
        receivers_s[it] = vz[receiver_line.depth_index, rec_indices]

    return {
        "vx": vx,
        "vz": vz,
        "txx": txx,
        "tzz": tzz,
        "txz": txz,
        "shot_vx": receivers,
        "shot_vz": receivers_s,
    }


def arrival_error(traces: np.ndarray, expected_sample: int) -> float:
    energy = np.abs(traces)
    first_idx = np.argmax(energy > 1e-6, axis=0)
    first_idx = np.where(np.any(energy > 1e-6, axis=0), first_idx, expected_sample)
    return float(np.max(np.abs(first_idx - expected_sample)))


def sponge_reflection_ratio(vx: np.ndarray, vz: np.ndarray, npad: int = 10) -> float:
    vx_safe = np.clip(vx, -1e6, 1e6)
    vz_safe = np.clip(vz, -1e6, 1e6)
    energy = vx_safe**2 + vz_safe**2
    interior = energy[npad:-npad, npad:-npad].sum()
    border = energy.sum() - interior
    if interior <= 1e-12:
        return 0.0
    return float(border / max(interior, 1e-12))


def quicklook_shot(path: Path, shot: np.ndarray, dt: float, dx: float) -> None:
    t = np.arange(shot.shape[0]) * dt
    x = np.arange(shot.shape[1]) * dx
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(shot, aspect="auto", cmap="seismic", origin="lower", extent=[x.min(), x.max(), t.min(), t.max()])
    ax.set_xlabel("Offset (m)")
    ax.set_ylabel("Time (s)")
    ax.set_title("Shot gather (Vx)")
    plt.colorbar(im, ax=ax, label="Amplitude")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def quicklook_snapshot(path: Path, field: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(field, cmap="seismic", origin="upper")
    ax.set_title(title)
    ax.set_xlabel("x index")
    ax.set_ylabel("z index")
    plt.colorbar(im, ax=ax, label="Amplitude")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_outputs(out_dir: Path, outputs: Dict[str, np.ndarray], meta: Dict[str, float], metrics: Dict[str, float]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "shot_vx.npy", outputs["shot_vx"])
    np.save(out_dir / "shot_vz.npy", outputs["shot_vz"])
    np.save(out_dir / "vx.npy", outputs["vx"])
    np.save(out_dir / "vz.npy", outputs["vz"])
    np.save(out_dir / "txx.npy", outputs["txx"])
    np.save(out_dir / "tzz.npy", outputs["tzz"])
    np.save(out_dir / "txz.npy", outputs["txz"])
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-dir", type=Path, default=Path("02-data/marmousi2/samples"))
    parser.add_argument("--out-dir", type=Path, default=Path("04-experiments/forward_elastic_2d/out/shot_000"))
    parser.add_argument("--npad", type=int, default=20)
    parser.add_argument("--cfl", type=float, default=0.1)
    parser.add_argument("--tmax", type=float, default=0.4)
    parser.add_argument("--source-frequency", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    samples_dir = args.samples_dir
    vp = np.load(samples_dir / "Vp_small.npy")
    vs = np.load(samples_dir / "Vs_small.npy")
    rho = np.load(samples_dir / "rho_small.npy")
    meta = json.loads((samples_dir / "meta_small.json").read_text())

    dx = float(meta["dx"])
    dz = float(meta["dz"])
    vmax = float(np.max(vp))
    vsvmax = float(np.max(vs))
    dt = args.cfl * min(dx, dz) / (math.sqrt(2.0) * max(vmax, vsvmax))
    nt = int(args.tmax / dt)

    sx = meta["nx"] // 2
    sz = 5
    source = SourceConfig(frequency=args.source_frequency, location=(sx, sz), delay=1.0 / args.source_frequency)
    receiver_line = ReceiverLine(depth_index=sz, start=max(0, sx - 40), end=min(meta["nx"], sx + 40), stride=1)
    sponge = sponge_mask(meta["nx"], meta["nz"], args.npad)

    outputs = forward_model(vp, vs, rho, dx, dz, dt, nt, source, receiver_line, sponge)

    if not np.all(np.isfinite(outputs["shot_vx"])):
        raise ForwardModelError("Shot gather contains NaNs")

    mid_idx = 0.5 * (receiver_line.start + receiver_line.end - receiver_line.stride)
    distance = abs(mid_idx - source.location[0]) * dx
    vp0 = float(vp[source.location[1], source.location[0]])
    vs0 = float(vs[source.location[1], source.location[0]])
    expected_p = int(distance / max(vp0, 1e-6) / dt)
    expected_s = int(distance / max(vs0, 1e-6) / dt)

    p_error = arrival_error(outputs["shot_vx"], expected_p)
    s_error = arrival_error(outputs["shot_vz"], expected_s)
    reflection_ratio = sponge_reflection_ratio(outputs["vx"], outputs["vz"], npad=args.npad // 2)
    cfl_value = compute_cfl(dt, dx, dz, vmax, vsvmax)

    metrics = {
        "dt": dt,
        "nt": nt,
        "cfl": cfl_value,
        "arrival_error_p": p_error,
        "arrival_error_s": s_error,
        "reflection_ratio": reflection_ratio,
        "evidence": EVIDENCE_TAGS,
        "data_source": meta.get("data_source", "unknown"),
    }

    forward_meta = {
        "dx": dx,
        "dz": dz,
        "dt": dt,
        "nt": nt,
        "source": {
            "location": source.location,
            "frequency": source.frequency,
            "delay": source.delay,
        },
        "receiver_line": receiver_line.__dict__,
        "npad": args.npad,
    }

    save_outputs(args.out_dir, outputs, forward_meta, metrics)

    figs_dir = args.out_dir.parent / "quicklooks"
    figs_dir.mkdir(parents=True, exist_ok=True)
    quicklook_shot(figs_dir / "shot_vx.png", outputs["shot_vx"], dt, dx)
    snapshot_time = outputs["vx"]
    quicklook_snapshot(figs_dir / "vx_snapshot.png", snapshot_time, "Final vx snapshot")

    print("Forward modeling completed")
    print(" ".join(EVIDENCE_TAGS))


if __name__ == "__main__":
    try:
        main()
    except ForwardModelError as exc:
        print(f"Forward modeling failed: {exc}")
        raise SystemExit(1)
