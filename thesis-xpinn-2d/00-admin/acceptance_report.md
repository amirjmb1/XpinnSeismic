# Acceptance Report

| Gate | Evidence Tags | Key Metrics |
| --- | --- | --- |
| G0 Ingest | [E-ingest-shape] [E-ingest-units] [E-ingest-quicklooks] [E-lame-invertible] | `02-data/marmousi2/processed/metrics.json` |
| G3 Samples | [E-sample-shape] [E-sample-dx] [E-sample-size] [E-sample-provenance] | `02-data/marmousi2/samples/metrics.json` |
| G1 Forward | [E-forward-cfl] [E-forward-P] [E-forward-S] [E-forward-sponge] [E-forward-nonan] | `04-experiments/forward_elastic_2d/out/shot_000/metrics.json` |
| G4 Figures | [E-figs-shot] [E-figs-snap] [E-figs-meta] | `05-writing/figs-src/out/figs_meta.json` |
| G2 Paper | [E-paper-build] [E-paper-cites] [E-paper-figs] | `05-writing/paper/paper_metrics.json` |

Each validator exits non-zero if thresholds are violated, ensuring subsequent stages do not start.
