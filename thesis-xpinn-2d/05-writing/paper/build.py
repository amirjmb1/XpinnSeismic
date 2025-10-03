#!/usr/bin/env python3
"""Build the LaTeX paper using a self-contained Tectonic binary."""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Tuple

import requests

EVIDENCE_TAGS = ["[E-paper-build]", "[E-paper-cites]", "[E-paper-figs]"]
TECTONIC_VERSION = "tectonic-0.14.1-x86_64-unknown-linux-musl"
TECTONIC_URL = f"https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.14.1/{TECTONIC_VERSION}.tar.gz"


class BuildError(RuntimeError):
    pass


def ensure_tectonic(cache_dir: Path) -> Path:
    bin_path = cache_dir / "tectonic"
    if bin_path.exists():
        return bin_path
    cache_dir.mkdir(parents=True, exist_ok=True)
    import tarfile
    response = requests.get(TECTONIC_URL, timeout=60)
    response.raise_for_status()
    tar_path = cache_dir / "tectonic.tar.gz"
    tar_path.write_bytes(response.content)
    with tarfile.open(tar_path) as tf:
        members = [m for m in tf.getmembers() if m.isfile() and (m.name.endswith("/tectonic") or m.name == "tectonic")]
        if not members:
            raise BuildError("Tectonic binary not found in archive")
        member = members[0]
        with tf.extractfile(member) as src, bin_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    bin_path.chmod(bin_path.stat().st_mode | stat.S_IEXEC)
    return bin_path


def run_tectonic(binary: Path, workdir: Path, tex_file: Path) -> None:
    cmd = [str(binary.resolve()), "--keep-logs", "--synctex", tex_file.name]
    result = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
    (workdir / "build.log").write_text(result.stdout + "\n" + result.stderr)
    if result.returncode != 0:
        raise BuildError("Tectonic build failed")


def parse_metrics(tex_path: Path) -> Tuple[int, int]:
    content = tex_path.read_text()
    citation_count = len(re.findall(r"\\cite[pt]?{", content))
    figure_refs = len(re.findall(r"\\ref{fig:", content))
    return citation_count, figure_refs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-dir", type=Path, default=Path("05-writing/paper"))
    parser.add_argument("--output", type=Path, default=Path("05-writing/paper/paper.pdf"))
    parser.add_argument("--cache-dir", type=Path, default=Path("05-writing/paper/.tectonic"))
    args = parser.parse_args()

    tex_path = args.paper_dir / "paper.tex"
    if not tex_path.exists():
        raise BuildError("paper.tex not found")

    binary = ensure_tectonic(args.cache_dir)
    run_tectonic(binary, args.paper_dir, tex_path)

    citation_count, figure_refs = parse_metrics(tex_path)
    metrics = {
        "citation_count": citation_count,
        "figure_refs": figure_refs,
        "evidence": EVIDENCE_TAGS,
    }
    (args.paper_dir / "paper_metrics.json").write_text(json.dumps(metrics, indent=2))

    if not args.output.exists():
        raise BuildError("paper.pdf missing after build")

    print("Paper built successfully")
    print(" ".join(EVIDENCE_TAGS))


if __name__ == "__main__":
    try:
        main()
    except BuildError as exc:
        print(f"Paper build failed: {exc}")
        raise SystemExit(1)
