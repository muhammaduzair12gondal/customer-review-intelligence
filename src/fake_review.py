"""
fake_review.py — Feature engineering and ML models for fake review detection.

Models:
    * :class:`FakeReviewDetector` — wraps both Random Forest and XGBoost.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from utils import get_logger, save_model, load_model

logger = get_logger(__name__)


class FakeReviewDetector:
    """
    Binary classifier that predicts whether a review is suspicious/fake.

    Uses an ensemble of Random Forest and XGBoost (optional) models with
    hand-crafted behavioural and linguistic features.

    Args:
        use_xgboost: If True, also train an XGBoost classifier for comparison.
        rf_params: Keyword arguments forwarded to RandomForestClassifier.
        xgb_params: Keyword arguments forwarded to XGBClassifier.
    """

    DEFAULT_RF_PARAMS: Dict = {
        "n_estimators": 300,
        "max_depth": 8,
        "min_samples_leaf": 20,
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": 42,
    }

    DEFAULT_XGB_PARAMS: Dict = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
    }

    FEATURE_COLS = [
        "review_length", "word_count", "avg_word_length",
        "exclamation_count", "capital_ratio",
        "helpfulness_ratio", "rating_deviation",
        "verified_purchase", "duplicate_text",
    ]

    def __init__(self,
                 use_xgboost: bool = True,
                 rf_params: Optional[Dict] = None,
                 xgb_params: Optional[Dict] = None) -> None:
        self.use_xgboost = use_xgboost
        self.scaler = StandardScaler()

        self.rf_model = RandomForestClassifier(
            **(rf_params or self.DEFAULT_RF_PARAMS)
        )
        if use_xgboost:
            try:
                from xgboost import XGBClassifier
                self.xgb_model = XGBClassifier(**(xgb_params or self.DEFAULT_XGB_PARAMS))
            except ImportError:
                logger.warning("XGBoost not installed — skipping XGB model.")
                self.use_xgboost = False
                self.xgb_model   = None
        else:
            self.xgb_model = None

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "FakeReviewDetector":
        """
        Train both classifiers on feature matrix X and label vector y.

        Args:
            X: DataFrame or array with feature columns.
            y: Binary label array (1 = suspicious, 0 = genuine).

        Returns:
            Self (for chaining).
        """
        X_arr = self._prepare(X, fit=True)
        logger.info("Training Random Forest on %d samples …", len(X_arr))
        self.rf_model.fit(X_arr, y)

        if self.use_xgboost and self.xgb_model is not None:
            logger.info("Training XGBoost …")
            self.xgb_model.fit(X_arr, y,
                               eval_set=[(X_arr, y)],
                               verbose=False)
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_proba_rf(self, X: pd.DataFrame) -> np.ndarray:
        """Return Random Forest probability of class=1 (suspicious)."""
        return self.rf_model.predict_proba(self._prepare(X))[:, 1]

    def predict_proba_xgb(self, X: pd.DataFrame) -> np.ndarray:
        """Return XGBoost probability of class=1 (suspicious)."""
        if not self.use_xgboost:
            raise RuntimeError("XGBoost model not available.")
        return self.xgb_model.predict_proba(self._prepare(X))[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict class label using RF model at given threshold."""
        return (self.predict_proba_rf(X) >= threshold).astype(int)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, prefix: str = "fake_review") -> None:
        """Save scaler, RF, and optionally XGB to models/."""
        save_model(self.scaler,   f"{prefix}_scaler.joblib")
        save_model(self.rf_model, f"{prefix}_rf.joblib")
        if self.use_xgboost and self.xgb_model is not None:
            save_model(self.xgb_model, f"{prefix}_xgb.joblib")

    @classmethod
    def load(cls, prefix: str = "fake_review") -> "FakeReviewDetector":
        """Load a saved detector."""
        obj = cls(use_xgboost=False)
        obj.scaler   = load_model(f"{prefix}_scaler.joblib")
        obj.rf_model = load_model(f"{prefix}_rf.joblib")
        try:
            obj.xgb_model   = load_model(f"{prefix}_xgb.joblib")
            obj.use_xgboost = True
        except Exception:
            pass
        return obj

    # ── Internals ─────────────────────────────────────────────────────────────

    def _prepare(self, X: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Select feature columns and scale."""
        cols = [c for c in self.FEATURE_COLS if c in X.columns]
        X_sel = X[cols].fillna(0).values.astype(float)
        if fit:
            return self.scaler.fit_transform(X_sel)
        return self.scaler.transform(X_sel)


# ── SHAP helper ───────────────────────────────────────────────────────────────

def shap_importance_plot(detector: FakeReviewDetector,
                          X: pd.DataFrame,
                          n_samples: int = 500,
                          model: str = "rf") -> None:
    """
    Generate a SHAP bar chart of feature importances.

    Args:
        detector: Fitted :class:`FakeReviewDetector`.
        X: Feature DataFrame (same columns used in training).
        n_samples: Number of rows to use for SHAP computation.
        model: 'rf' or 'xgb'.
    """
    import shap
    import matplotlib.pyplot as plt
    from utils import get_model_path

    clf = detector.rf_model if model == "rf" else detector.xgb_model
    X_arr = detector._prepare(X.head(n_samples))
    cols = [c for c in detector.FEATURE_COLS if c in X.columns]

    explainer   = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_arr)

    # For RF binary output shap_values is a list [class0, class1]
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    shap.summary_plot(sv, X_arr, feature_names=cols, show=False, plot_type="bar")
    plt.title(f"SHAP Feature Importance — {model.upper()}")
    plt.tight_layout()
    save_path = str(get_model_path(f"shap_fake_review_{model}.png"))
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    logger.info("SHAP plot saved → %s", save_path)
