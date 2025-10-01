#!/usr/bin/env python3
"""Validator for Gate G4 (Figures)."""
from __future__ import annotations

import json
from pathlib import Path

EVIDENCE_TAGS = ["[E-figs-shot]", "[E-figs-snap]", "[E-figs-meta]"]


class ValidationError(RuntimeError):
    pass


def assert_size(path: Path, min_bytes: int = 10_000) -> None:
    if not path.exists():
        raise ValidationError(f"Missing figure {path}")
    if path.stat().st_size < min_bytes:
        raise ValidationError(f"Figure {path} too small")


def main() -> None:
    fig_dir = Path("05-writing/figs-src/out")
    shot = fig_dir / "fig_elastic_shot.png"
    snap = fig_dir / "fig_elastic_snap.png"
    assert_size(shot)
    assert_size(snap)
    meta_path = fig_dir / "figs_meta.json"
    if not meta_path.exists():
        raise ValidationError("Missing figs_meta.json")
    metadata = json.loads(meta_path.read_text())
    if metadata.get("evidence") != EVIDENCE_TAGS:
        raise ValidationError("Evidence tags missing")
    for key in ("shot", "snapshot"):
        info = metadata.get(key)
        if not info:
            raise ValidationError(f"Metadata missing {key}")
        for field in ("path", "sha256", "source"):
            if field not in info:
                raise ValidationError(f"Metadata {key} missing {field}")
    print("Gate G4 passed", " ".join(EVIDENCE_TAGS))


if __name__ == "__main__":
    try:
        main()
    except ValidationError as exc:
        print(f"Gate G4 failure: {exc}")
        raise SystemExit(1)
