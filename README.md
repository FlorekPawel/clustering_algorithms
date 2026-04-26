# clustering_algorithms

Clustering benchmark project comparing multiple clustering algorithms on datasets from the Clustering Algorithms https://clustering-benchmarks.gagolewski.com/weave/suite-v1.html collection.

The repository contains:
- reusable Python modules for loading data and running experiments,
- a full experiment pipeline over all batteries,
- generated metrics tables and plots,
- a notebook and a Beamer presentation with analysis.

## What Is Included

Algorithms used in experiments:
- `KMeans`
- `GaussianMixture`
- `Genie` (multiple `gini_threshold` values)
- `AgglomerativeClustering` (`single`, `average`, `complete`, `ward`)
- `DBSCAN` (grid over `eps` and `min_samples`)
- `SpectralClustering`

Dataset batteries (local `data/`):
- `fcps`
- `graves`
- `sipu`
- `uci`

## Requirements

- Python `>=3.13`
- [`uv`](https://docs.astral.sh/uv/)

## Quick Start

Install dependencies and pre-commit hooks:

```bash
make install
```

This runs:
- `uv sync --all-groups`
- `uv run pre-commit install`

## Run Experiments

Run the full experiment pipeline:

```bash
make run-experiments
```

Equivalent command:

```bash
PYTHONPATH=src uv run python scripts/run_all_experiments.py
```

By default, results are saved to:
- `outputs/all_experiments.pkl`

You can also run with custom options:

```bash
PYTHONPATH=src uv run python -m clustering_algorithms.run_all \
  --data-dir data \
  --output outputs/all_experiments.pkl \
  --random-state 42 \
  --log-level INFO
```

> [!NOTE]
> If expected data structure is missing, the loader can download and prepare datasets from `gagolews/clustering-data-v1` automatically.

## Main Package API

Core modules are in `src/clustering_algorithms/`:
- `data_loader.py` - data availability checks, download, and dataset loading
- `clustering_model.py` - algorithm suite and per-dataset experiment execution
- `run_all.py` - CLI pipeline over all batteries and datasets

Minimal usage example:

```python
from clustering_algorithms.data_loader import load_data_and_labels
from clustering_algorithms.clustering_model import ClusteringModel

X, y = load_data_and_labels("fcps", "atom", data_dir="data")
model = ClusteringModel(n_clusters=2, random_state=42)
labels = model.run_all(X, y_true=y)
print(labels.keys())
```

## Repository Layout

```text
src/clustering_algorithms/   # package source code
scripts/                     # script entry points
data/                        # dataset batteries
outputs/                     # generated tables/plots/results
notebooks/                   # exploratory analysis
presentation/                # Beamer presentation
```

## Analysis Artifacts

- Metrics tables: `outputs/tables/`
- Plots: `outputs/plots/`
- Notebook: `notebooks/analysis.ipynb`
- Slides: `presentation/presentation.tex`

## Development Helpers

Available `make` targets:
- `make help`
- `make install`
- `make pre-commit`
- `make pre-commit-all`
- `make run-experiments`
- `make clean`
