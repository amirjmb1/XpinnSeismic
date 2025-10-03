#!/usr/bin/env python3
"""Fetch Marmousi2 release assets into the raw data folder."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Optional

import requests

ASSETS = {
    "MODEL_P-WAVE_VELOCITY_1.25m.segy.tar.gz": "P-wave velocity (Vp)",
    "MODEL_S-WAVE_VELOCITY_1.25m.segy.tar.gz": "S-wave velocity (Vs)",
    "MODEL_DENSITY_1.25m.segy.tar.gz": "Density (rho)",
}

DEFAULT_TAG = "marmousi2-v1"
DEFAULT_REPO = os.environ.get("MARMOUSI_RELEASE_REPO", "mwe-repo/marmousi-assets")
CHUNK_SIZE = 2 ** 20


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_asset(session: requests.Session, url: str, dest: Path) -> None:
    with session.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        tmp_path = dest.with_suffix(dest.suffix + ".part")
        with tmp_path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
        tmp_path.replace(dest)


def fetch_assets(raw_dir: Path, repo: str, tag: str, assets: Iterable[str], token: Optional[str]) -> list[dict[str, str]]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    headers = {"User-Agent": "xpinn-orchestrator"}
    if token:
        headers["Authorization"] = f"token {token}"
    session.headers.update(headers)
    provenance: list[dict[str, str]] = []
    for asset in assets:
        url = f"https://github.com/{repo}/releases/download/{tag}/{asset}"
        dest = raw_dir / asset
        if dest.exists():
            sha = compute_sha256(dest)
            provenance.append({"asset": asset, "url": url, "sha256": sha, "status": "cached"})
            continue
        print(f"Downloading {asset} from {url}")
        try:
            download_asset(session, url, dest)
        except requests.HTTPError as exc:
            provenance.append({"asset": asset, "url": url, "status": "missing", "error": str(exc)})
            continue
        sha = compute_sha256(dest)
        provenance.append({"asset": asset, "url": url, "sha256": sha, "status": "downloaded"})
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("02-data/marmousi2/raw"), help="Output directory for raw assets")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="GitHub repository owner/name containing the release")
    parser.add_argument("--tag", default=DEFAULT_TAG, help="Release tag to fetch from")
    parser.add_argument("--assets", nargs="*", default=list(ASSETS.keys()), help="Specific assets to fetch")
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN"), help="GitHub token for authenticated requests")
    parser.add_argument("--provenance", type=Path, default=Path("02-data/marmousi2/raw/provenance.json"), help="Location to store provenance metadata")
    args = parser.parse_args()

    provenance = fetch_assets(args.raw_dir, args.repo, args.tag, args.assets, args.token)
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    args.provenance.write_text(json.dumps({"repo": args.repo, "tag": args.tag, "assets": provenance}, indent=2))
    print(f"Fetched {len(provenance)} assets. Metadata written to {args.provenance}")


if __name__ == "__main__":
    main()
