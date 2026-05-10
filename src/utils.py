"""
utils.py — Shared utilities for the Customer Review Intelligence Platform.

Provides logging configuration, common file I/O helpers, and metric
reporting functions used across all modules.
"""

import logging
import os
import json
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a logger with a consistent format for the project.

    Args:
        name: Logger name (typically __name__ of the caller module).
        level: Logging level (default INFO).

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def get_data_path(filename: str) -> Path:
    """
    Resolve a filename relative to the project data directory.

    Args:
        filename: File name inside data/.

    Returns:
        Absolute :class:`pathlib.Path`.
    """
    return DATA_DIR / filename


def get_model_path(filename: str) -> Path:
    """
    Resolve a filename relative to the project models directory.

    Args:
        filename: File name inside models/.

    Returns:
        Absolute :class:`pathlib.Path`.
    """
    return MODELS_DIR / filename


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(model: Any, filename: str) -> Path:
    """
    Persist a scikit-learn-compatible model with joblib.

    Args:
        model: Trained model object.
        filename: Target filename inside models/ (e.g. "lr_tfidf.joblib").

    Returns:
        Path to saved file.
    """
    path = get_model_path(filename)
    joblib.dump(model, path)
    get_logger(__name__).info("Model saved → %s", path)
    return path


def load_model(filename: str) -> Any:
    """
    Load a joblib-persisted model from the models directory.

    Args:
        filename: File name inside models/.

    Returns:
        Loaded model object.
    """
    path = get_model_path(filename)
    get_logger(__name__).info("Loading model ← %s", path)
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Metrics reporting
# ---------------------------------------------------------------------------

def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                                  target_names: Optional[list] = None) -> Dict:
    """
    Display and return a classification report as a dict.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        target_names: Optional list of class names.

    Returns:
        Dictionary containing per-class metrics + macro averages.
    """
    from sklearn.metrics import classification_report, accuracy_score
    logger = get_logger(__name__)
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    logger.info("Accuracy: %.4f", acc)
    logger.info("Classification Report:\n%s",
                classification_report(y_true, y_pred,
                                      target_names=target_names,
                                      zero_division=0))
    report["accuracy_score"] = acc
    return report


def save_metrics(metrics: Dict, filename: str) -> None:
    """
    Persist a metrics dictionary as a JSON file in the models directory.

    Args:
        metrics: Dictionary of metric name → value.
        filename: Target filename (e.g. "distilbert_metrics.json").
    """
    path = get_model_path(filename)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    get_logger(__name__).info("Metrics saved → %s", path)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_reviews(filepath: Optional[str] = None,
                 nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Load the Amazon Fine Food Reviews CSV.

    Args:
        filepath: Path to Reviews.csv. Defaults to data/Reviews.csv.
        nrows: Optional row limit for quick experiments.

    Returns:
        :class:`pandas.DataFrame` with raw review data.
    """
    if filepath is None:
        filepath = get_data_path("Reviews.csv")
    logger = get_logger(__name__)
    logger.info("Loading dataset from %s (nrows=%s) …", filepath, nrows)
    df = pd.read_csv(filepath, nrows=nrows)
    logger.info("Loaded %d rows × %d cols", *df.shape)
    return df
