# Weave: news recommendation, filter bubbles, and fatigue

This repository contains an agent-based simulation of news recommendation
systems. It reproduces the circle-world model from Aridor, Goncalves, and
Sikdar, then extends it to MIND-small news articles and reader fatigue.

## What is included

- `src/`: data preparation, simulation, recommenders, metrics, and reporting.
- `scripts/`: data download, experiment, analysis, and validation entry points.
- `configs/`: configurations for the final experiment families.
- `reports/`: committed tables, figures, and machine-readable result artifacts.
- `config.yaml`: shared defaults for data preparation and simulations.

Large input data is intentionally not committed. The scripts expect MIND-small
files under `data/raw/train/` and `data/raw/dev/`.

## Setup

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download and prepare MIND-small:

```bash
python scripts/download_mind.py
python -m src.data.prepare
```

The preparation step writes `articles.csv`, `embeddings.npy`, and
`user_histories.json` to `data/processed/`. Use
`python -m src.data.prepare --fake-embeddings` for a fast pipeline check.

## Reproduce the experiments

Run the circle-world reproduction:

```bash
python scripts/reproduce_aridor.py --workers 4
```

Run the news simulation:

```bash
python scripts/run_news.py --workers 4
```

Run all final experiment families:

```bash
bash scripts/run_final.sh
```

Set `QUICK=1` for a short validation run. Final outputs are written under
`results/` and `reports/final/`.

To rebuild report tables and figures from a saved run:

```bash
python scripts/analyse_results.py <run-directory> --out <output-directory>
```

Additional validation entry points are available in `scripts/`, including
reader-choice validation, fatigue estimation, and report rebuilding.

## Tests

The original test suite is retained in the local `scratch/tests/` archive for
reference during submission cleanup. The main reproducibility path is the
scripted workflow above.

## Data and generated files

Raw and processed data, cached Python files, experiment runs, and the
submission-cleanup archive are excluded by `.gitignore`. The MIND dataset is
subject to its own license; download it only after reviewing the terms at
https://msnews.github.io/.
