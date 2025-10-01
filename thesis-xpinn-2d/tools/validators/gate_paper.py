#!/usr/bin/env python3
"""Validator for Gate G2 (Paper)."""
from __future__ import annotations

import json
from pathlib import Path

EVIDENCE_TAGS = ["[E-paper-build]", "[E-paper-cites]", "[E-paper-figs]"]


class ValidationError(RuntimeError):
    pass


def main() -> None:
    paper_dir = Path("05-writing/paper")
    pdf = paper_dir / "paper.pdf"
    metrics_path = paper_dir / "paper_metrics.json"
    if not pdf.exists():
        raise ValidationError("paper.pdf missing")
    if pdf.stat().st_size < 50 * 1024:
        raise ValidationError("paper.pdf too small to be substantive")
    if not metrics_path.exists():
        raise ValidationError("paper_metrics.json missing")
    metrics = json.loads(metrics_path.read_text())
    if metrics.get("citation_count", 0) < 5:
        raise ValidationError("Fewer than 5 citations")
    if metrics.get("figure_refs", 0) < 2:
        raise ValidationError("Figures not referenced in text")
    if metrics.get("evidence") != EVIDENCE_TAGS:
        raise ValidationError("Evidence tags missing")
    print("Gate G2 passed", " ".join(EVIDENCE_TAGS))


if __name__ == "__main__":
    try:
        main()
    except ValidationError as exc:
        print(f"Gate G2 failure: {exc}")
        raise SystemExit(1)
