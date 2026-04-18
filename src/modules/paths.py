from __future__ import annotations

import os
from pathlib import Path


def get_app_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_dataset_dir() -> Path:
    """
    Return the project's built-in dataset directory.

    This app intentionally uses the dataset/ folder that lives inside the
    project root as the single source of truth, instead of honoring an
    external DATAPATH override.
    """
    return (get_app_root() / "dataset").resolve()


def ensure_datapath() -> Path:
    dataset_dir = get_dataset_dir()
    os.environ["DATAPATH"] = str(dataset_dir)
    return dataset_dir


def get_runtime_artifacts_dir() -> Path:
    return get_dataset_dir() / "runtime_artifacts"
