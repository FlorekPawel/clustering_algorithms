"""Data download and loading utilities for clustering-data-v1."""

from __future__ import annotations

import io
import json
import logging
import shutil
import urllib.request
import zipfile
from pathlib import Path

import clustbench

logger = logging.getLogger(__name__)
DEFAULT_OWNER = "gagolews"
DEFAULT_REPO = "clustering-data-v1"
DEFAULT_ALLOWED_DIRS = ("sipu", "uci", "fcps", "graves")
DEFAULT_DATASETS_TO_DROP = ("birch1", "birch2", "worms_2", "worms_64")


def _default_data_dir() -> Path:
    return Path(__file__).resolve().parent / "data"


def _has_expected_structure(data_dir: Path, allowed_dirs: tuple[str, ...]) -> bool:
    if not data_dir.exists() or not data_dir.is_dir():
        return False

    return all((data_dir / directory).is_dir() for directory in allowed_dirs)


def _download_and_filter_repository(
    destination: Path,
    owner: str,
    repo: str,
    allowed_dirs: tuple[str, ...],
    datasets_to_drop: tuple[str, ...],
) -> None:
    base_api = f"https://api.github.com/repos/{owner}/{repo}"
    logger.debug("Fetching repository metadata: %s", base_api)

    with urllib.request.urlopen(base_api) as response:  # noqa: S310
        repo_info = json.loads(response.read().decode("utf-8"))

    default_branch = repo_info["default_branch"]
    zip_url = f"https://codeload.github.com/{owner}/{repo}/zip/refs/heads/{default_branch}"
    logger.debug("Downloading data archive from branch %s", default_branch)

    with urllib.request.urlopen(zip_url) as response:  # noqa: S310
        zip_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        root_name = archive.namelist()[0].split("/")[0]
        temp_extract_dir = destination.parent / f".{repo}_tmp_extract"

        if temp_extract_dir.exists():
            shutil.rmtree(temp_extract_dir)
        temp_extract_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Extracting archive to temporary directory: %s", temp_extract_dir)
        archive.extractall(temp_extract_dir)

    extracted_repo_dir = temp_extract_dir / root_name
    if destination.exists():
        shutil.rmtree(destination)
    shutil.move(str(extracted_repo_dir), str(destination))
    shutil.rmtree(temp_extract_dir)

    allowed_dir_set = set(allowed_dirs)
    datasets_to_drop_set = set(datasets_to_drop)

    for item in list(destination.iterdir()):
        if item.is_dir() and item.name in allowed_dir_set:
            for subitem in list(item.iterdir()):
                name = subitem.name
                if name.split(".")[0] in datasets_to_drop_set:
                    subitem.unlink()
            continue

        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def ensure_data_available(
    data_dir: Path | str | None = None,
    owner: str = DEFAULT_OWNER,
    repo: str = DEFAULT_REPO,
    allowed_dirs: tuple[str, ...] = DEFAULT_ALLOWED_DIRS,
    datasets_to_drop: tuple[str, ...] = DEFAULT_DATASETS_TO_DROP,
) -> Path:
    """Ensure the clustering benchmark data exists locally.

    Downloads and filters the repository only when required directories are missing.
    """
    destination = Path(data_dir) if data_dir is not None else _default_data_dir()
    destination.parent.mkdir(parents=True, exist_ok=True)

    if _has_expected_structure(destination, allowed_dirs):
        logger.debug("Data already available in %s. Skipping download.", destination)
        return destination

    logger.info("Data not found in %s. Downloading from GitHub...", destination)
    _download_and_filter_repository(
        destination=destination,
        owner=owner,
        repo=repo,
        allowed_dirs=allowed_dirs,
        datasets_to_drop=datasets_to_drop,
    )
    logger.info("Data prepared at %s", destination)
    return destination


def load_data_and_labels(
    battery: str,
    dataset_name: str,
    data_dir: Path | str | None = None,
):
    """Load dataset data and the first label vector.

    Returns `(data, labels)` where labels are equivalent to `dataset.labels[0]`.
    """
    path = ensure_data_available(data_dir=data_dir)
    logger.debug("Loading dataset %s/%s from %s", battery, dataset_name, path)
    dataset = clustbench.load_dataset(battery, dataset_name, path=path)
    return dataset.data, dataset.labels[0]
