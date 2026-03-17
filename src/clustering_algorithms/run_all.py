"""Run clustering experiments for all available datasets."""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import clustbench
import numpy as np
from tqdm.auto import tqdm

from clustering_algorithms.clustering_model import run_experiments_for_datasets
from clustering_algorithms.data_loader import ensure_data_available, load_data_and_labels

logger = logging.getLogger(__name__)


def run_all_experiments(
    data_dir: Path | str | None = None,
    random_state: int = 42,
    output_path: Path | str | None = None,
) -> dict[str, dict[str, object]]:
    """Download data if needed and run all experiments for all datasets."""
    logger.info("Starting experiment pipeline")
    path = ensure_data_available(data_dir=data_dir)

    batteries = clustbench.get_battery_names(path=path)
    datasets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for battery in tqdm(batteries, desc="Batteries"):
        dataset_names = clustbench.get_dataset_names(battery, path=path)
        for dataset_name in tqdm(
            dataset_names,
            desc=f"{battery} datasets",
            leave=False,
        ):
            x, y_true = load_data_and_labels(battery, dataset_name, data_dir=path)
            key = f"{battery}/{dataset_name}"
            datasets[key] = (np.asarray(x), np.asarray(y_true))

    logger.info("Collected %s datasets. Running clustering experiments...", len(datasets))
    results = run_experiments_for_datasets(
        datasets=datasets,
        random_state=random_state,
    )

    if output_path is not None:
        resolved_output = Path(output_path)
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        with resolved_output.open("wb") as file:
            pickle.dump(results, file)
        logger.info("Saved experiment results to %s", resolved_output)

    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run all clustering experiments.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory with clustering-data-v1 content. If missing, data is downloaded.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs") / "all_experiments.pkl",
        help="Where to store experiment results as a pickle file.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for stochastic algorithms.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    results = run_all_experiments(
        data_dir=args.data_dir,
        random_state=args.random_state,
        output_path=args.output,
    )
    print(f"Finished {len(results)} dataset experiments.")
    print(f"Results saved to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
