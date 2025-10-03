# Paper Build Instructions

Run the builder to produce `paper.pdf`:

```bash
python build.py
```

The script downloads a self-contained Tectonic binary if needed, compiles `paper.tex`, and writes `paper_metrics.json` with citation and figure reference counts.
