# Thesis XPINN 2D Elastic Stack

This repository implements a fully automated research stack for two-dimensional isotropic elastic wave modeling and XPINN-based inversion on the Marmousi2 model.

## Project Layout

```
thesis-xpinn-2d/
  00-admin/                  # Acceptance report, critique
  01-litmap/                 # Literature assets (placeholder)
  02-data/marmousi2/         # Data roots (raw, processed, samples)
  04-experiments/            # Ingestion, forward solver, inversion scaffold
  05-writing/                # Figures and paper sources
  tools/                     # Fetch utilities and validators
  environment/               # Environment specifications
```

## One-click Orchestrator

The GitHub Actions workflow `.github/workflows/orchestrator.yml` runs a gated pipeline:

1. Fetch release assets
2. Create the Python environment
3. Ingest SEG-Y → NumPy and Lamé parameters (Gate G0)
4. Produce CI samples (Gate G3)
5. Run elastic forward modeling (Gate G1)
6. Generate figures (Gate G4)
7. Compile the paper (Gate G2)

Each job runs unit tests (where defined) and validator scripts that exit non-zero on failure, preventing downstream stages. On CI systems without Marmousi2 assets, Gate G1 blocks unless `ALLOW_SYNTHETIC_FOR_G1=1` is provided.

To execute locally with synthetic fallback:

```bash
python 04-experiments/data_ingest/ingest.py
python 04-experiments/data_ingest/make_samples.py
python 04-experiments/forward_elastic_2d/forward_sim.py --tmax 0.6
python 05-writing/figs-src/make_figures.py
ALLOW_SYNTHETIC_FOR_G1=1 python tools/validators/gate_forward.py
```

To build the paper (downloads a local Tectonic binary):

```bash
python 05-writing/paper/build.py
python tools/validators/gate_paper.py
```

## Data via Release Assets

Use `tools/data_fetch.py` to download the Marmousi2 release assets into `02-data/marmousi2/raw/`:

```bash
python tools/data_fetch.py --repo <owner>/<repo> --tag marmousi2-v1
```

If the assets are absent, ingestion falls back to a layered synthetic model for unit tests, but Gate G1 enforces real-data availability unless explicitly overridden.

## Artifacts & Rebuild

Continuous integration runs the orchestrator end-to-end and publishes regenerated quicklooks, simulation outputs, figures, and `paper.pdf` as workflow artifacts. The prepared archive `bootstrap-snapshot.zip` bundles the processed arrays, CI samples, forward-model results, and paper exports from the pre-source-only tree; upload this file to the GitHub Release **bootstrap-snapshot** (tag `bootstrap-snapshot`) to keep the bootstrap outputs available alongside the code history. Re-run `tools/data_fetch.py` once Marmousi2 release assets are available to rebuild the full stack locally or in CI.

## Environment

Install dependencies with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r environment/requirements.txt
```

## Licensing

Only derived NumPy arrays, metrics, and figures are tracked; raw SEG-Y assets remain outside version control.
