"""Clustering model runner with all required algorithms for experiments."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from genieclust import Genie
from numpy.typing import ArrayLike, NDArray
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

GENIE_G_VALUES = (0.1, 0.3, 0.5, 0.7, 0.9)
AGGLOMERATIVE_LINKAGES = ("single", "average", "complete", "ward")
DEFAULT_DBSCAN_EPS_VALUES = (0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5)
DEFAULT_DBSCAN_MIN_SAMPLES_VALUES = (3, 5, 10)


@dataclass(slots=True)
class ClusteringModel:
    """Run all clustering algorithms required in the assignment."""

    n_clusters: int
    random_state: int = 42

    @staticmethod
    def infer_n_clusters_from_reference(y_true: ArrayLike) -> int:
        """Infer known number of clusters from reference labels.

        Follows the assignment statement: K = np.max(y_true).
        """
        y = np.asarray(y_true)
        if y.size == 0:
            msg = "Reference labels cannot be empty."
            raise ValueError(msg)
        return int(np.max(y))

    def run_all(
        self,
        data: NDArray[np.floating],
        y_true: ArrayLike | None = None,
        dbscan_eps_values: tuple[float, ...] = DEFAULT_DBSCAN_EPS_VALUES,
        dbscan_min_samples_values: tuple[int, ...] = DEFAULT_DBSCAN_MIN_SAMPLES_VALUES,
    ) -> dict[str, NDArray[np.int_]]:
        """Fit all configured clustering algorithms and return labels."""
        labels, _ = self.run_all_with_details(
            data=data,
            y_true=y_true,
            dbscan_eps_values=dbscan_eps_values,
            dbscan_min_samples_values=dbscan_min_samples_values,
        )
        return labels

    def run_all_with_details(
        self,
        data: NDArray[np.floating],
        y_true: ArrayLike | None = None,
        dbscan_eps_values: tuple[float, ...] = DEFAULT_DBSCAN_EPS_VALUES,
        dbscan_min_samples_values: tuple[int, ...] = DEFAULT_DBSCAN_MIN_SAMPLES_VALUES,
    ) -> tuple[dict[str, NDArray[np.int_]], dict[str, dict[str, float | int]]]:
        """Fit all algorithms and return labels with DBSCAN configuration metadata."""
        x = np.asarray(data)
        target_clusters = (
            self.infer_n_clusters_from_reference(y_true) if y_true is not None else self.n_clusters
        )

        results: dict[str, NDArray[np.int_]] = {}
        logger.debug("Running clustering suite with target K=%s", target_clusters)

        results["kmeans"] = self._fit_predict(
            KMeans(
                n_clusters=target_clusters,
                n_init="auto",
                random_state=self.random_state,
            ),
            x,
        )

        results["gaussian_mixture"] = self._fit_predict(
            GaussianMixture(n_components=target_clusters, random_state=self.random_state),
            x,
        )

        for g in GENIE_G_VALUES:
            genie = Genie(n_clusters=target_clusters, gini_threshold=g)
            results[f"genie_g_{g:.1f}"] = self._fit_predict(genie, x)

        for linkage in AGGLOMERATIVE_LINKAGES:
            agglomerative = AgglomerativeClustering(n_clusters=target_clusters, linkage=linkage)
            results[f"agglomerative_{linkage}"] = self._fit_predict(agglomerative, x)

        for eps in dbscan_eps_values:
            for min_samples in dbscan_min_samples_values:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = self._fit_predict(dbscan, x)
                config_name = f"dbscan_eps_{eps:.1f}_min_samples_{min_samples}"
                results[config_name] = labels

        results["spectral"] = self._fit_predict(
            SpectralClustering(
                n_clusters=target_clusters,
                assign_labels="kmeans",
                random_state=self.random_state,
                affinity="nearest_neighbors",
            ),
            x,
        )

        return results

    @staticmethod
    def _fit_predict(model: object, x: NDArray[np.floating]) -> NDArray[np.int_]:
        fit_predict = getattr(model, "fit_predict", None)
        if callable(fit_predict):
            labels = fit_predict(x)
            return np.asarray(labels, dtype=int)

        fit = getattr(model, "fit", None)
        predict = getattr(model, "predict", None)
        if callable(fit) and callable(predict):
            fit(x)
            labels = predict(x)
            return np.asarray(labels, dtype=int)

        raise TypeError(f"Model {type(model).__name__} does not support fit/predict operations.")


def run_all_clustering_algorithms(
    data: NDArray[np.floating],
    n_clusters: int,
    y_true: ArrayLike | None = None,
    random_state: int = 42,
    dbscan_eps_values: tuple[float, ...] = DEFAULT_DBSCAN_EPS_VALUES,
    dbscan_min_samples_values: tuple[int, ...] = DEFAULT_DBSCAN_MIN_SAMPLES_VALUES,
) -> dict[str, NDArray[np.int_]]:
    """Convenience function for running all required clustering algorithms."""
    model = ClusteringModel(
        n_clusters=n_clusters,
        random_state=random_state,
    )
    return model.run_all(
        data=data,
        y_true=y_true,
        dbscan_eps_values=dbscan_eps_values,
        dbscan_min_samples_values=dbscan_min_samples_values,
    )


def run_experiments_for_datasets(
    datasets: dict[str, tuple[NDArray[np.floating], ArrayLike]],
    random_state: int = 42,
    dbscan_eps_values: tuple[float, ...] = DEFAULT_DBSCAN_EPS_VALUES,
    dbscan_min_samples_values: tuple[int, ...] = DEFAULT_DBSCAN_MIN_SAMPLES_VALUES,
    show_progress: bool = True,
) -> dict[str, dict[str, object]]:
    """Run all clustering algorithms for each dataset.

    Input format:
    `datasets[name] = (X, y_true)`
    """
    results: dict[str, dict[str, object]] = {}

    for dataset_name, (x, y_true) in tqdm(
        datasets.items(),
        desc="Datasets",
        total=len(datasets),
        disable=not show_progress,
    ):
        inferred_k = ClusteringModel.infer_n_clusters_from_reference(y_true)
        logger.info("Running dataset %s with inferred K=%s", dataset_name, inferred_k)
        model = ClusteringModel(n_clusters=inferred_k, random_state=random_state)
        labels = model.run_all_with_details(
            data=x,
            y_true=y_true,
            dbscan_eps_values=dbscan_eps_values,
            dbscan_min_samples_values=dbscan_min_samples_values,
        )
        results[dataset_name] = {
            "k": inferred_k,
            "labels": labels,
            "true_labels": np.asarray(y_true, dtype=int),
        }

    return results
