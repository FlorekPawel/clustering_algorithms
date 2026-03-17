"""Clustering model runner with all required algorithms for experiments."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from genieclust import Genie
from numpy.typing import ArrayLike, NDArray
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.metrics import silhouette_score
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
    ) -> tuple[dict[str, NDArray[np.int_]], dict[str, float | int]]:
        """Fit all algorithms and return labels with DBSCAN tuning metadata."""
        x = np.asarray(data)
        target_clusters = (
            self.infer_n_clusters_from_reference(y_true) if y_true is not None else self.n_clusters
        )

        results: dict[str, NDArray[np.int_]] = {}
        logger.debug("Running clustering suite with target K=%s", target_clusters)

        results["kmeans"] = self._fit_predict(
            KMeans(n_clusters=target_clusters, n_init="auto", random_state=self.random_state),
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

        dbscan_labels, dbscan_details = self._fit_dbscan_with_tuning(
            x=x,
            target_clusters=target_clusters,
            eps_values=dbscan_eps_values,
            min_samples_values=dbscan_min_samples_values,
        )
        results["dbscan"] = dbscan_labels
        logger.debug(
            "DBSCAN tuned: eps=%s min_samples=%s estimated_k=%s noise_ratio=%.4f",
            dbscan_details["eps"],
            dbscan_details["min_samples"],
            dbscan_details["n_estimated_clusters"],
            dbscan_details["noise_ratio"],
        )

        results["spectral"] = self._fit_predict(
            SpectralClustering(
                n_clusters=target_clusters,
                assign_labels="kmeans",
                random_state=self.random_state,
                affinity="nearest_neighbors",
            ),
            x,
        )

        return results, dbscan_details

    def _fit_dbscan_with_tuning(
        self,
        x: NDArray[np.floating],
        target_clusters: int,
        eps_values: tuple[float, ...],
        min_samples_values: tuple[int, ...],
    ) -> tuple[NDArray[np.int_], dict[str, float | int]]:
        best_labels: NDArray[np.int_] | None = None
        best_details: dict[str, float | int] | None = None
        best_score: tuple[float, float, float, float] | None = None

        for eps in eps_values:
            for min_samples in min_samples_values:
                labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(x)
                labels = np.asarray(labels, dtype=int)

                non_noise_mask = labels != -1
                non_noise_count = int(np.sum(non_noise_mask))
                n_noise = int(labels.size - non_noise_count)
                unique_non_noise = np.unique(labels[non_noise_mask])
                n_estimated_clusters = int(unique_non_noise.size)
                noise_ratio = n_noise / float(labels.size)
                cluster_gap = abs(n_estimated_clusters - target_clusters)

                silhouette = -1.0
                if n_estimated_clusters >= 2 and non_noise_count >= 2:
                    silhouette = float(silhouette_score(x[non_noise_mask], labels[non_noise_mask]))

                # DBSCAN is treated separately: prioritize matching known K,
                # then fewer noise points, then better silhouette.
                score = (cluster_gap, noise_ratio, -silhouette, float(min_samples))

                if best_score is None or score < best_score:
                    best_score = score
                    best_labels = labels
                    best_details = {
                        "eps": float(eps),
                        "min_samples": int(min_samples),
                        "n_estimated_clusters": n_estimated_clusters,
                        "n_noise_points": n_noise,
                        "noise_ratio": float(noise_ratio),
                        "target_clusters": int(target_clusters),
                    }

        if best_labels is None or best_details is None:
            msg = "DBSCAN tuning failed to produce a valid result."
            raise RuntimeError(msg)

        return best_labels, best_details

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
        labels, dbscan_details = model.run_all_with_details(
            data=x,
            y_true=y_true,
            dbscan_eps_values=dbscan_eps_values,
            dbscan_min_samples_values=dbscan_min_samples_values,
        )
        results[dataset_name] = {
            "k": inferred_k,
            "labels": labels,
            "dbscan": dbscan_details,
        }

    return results
