"""
sentiment_model.py — Three-model sentiment classification pipeline.

Implements:
    1. VADER Baseline
    2. TF-IDF + Logistic Regression
    3. DistilBERT Fine-tuned (HuggingFace)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from utils import get_logger, save_model, load_model, get_model_path, save_metrics

logger = get_logger(__name__)

LABEL_NAMES = ["Negative", "Neutral", "Positive"]

# ── 1. VADER ─────────────────────────────────────────────────────────────────

class VADERSentimentModel:
    """
    Rule-based sentiment classifier using VADER compound scores.

    Thresholds:
        compound >= 0.05  → Positive (2)
        compound <= -0.05 → Negative (0)
        otherwise         → Neutral  (1)
    """

    def __init__(self) -> None:
        self._analyzer = SentimentIntensityAnalyzer()

    def predict_one(self, text: str) -> Tuple[int, float]:
        """
        Predict sentiment for a single review.

        Args:
            text: Raw or cleaned review text.

        Returns:
            Tuple of (label: int, compound_score: float).
        """
        scores = self._analyzer.polarity_scores(str(text))
        compound = scores["compound"]
        if compound >= 0.05:
            label = 2
        elif compound <= -0.05:
            label = 0
        else:
            label = 1
        return label, compound

    def predict(self, texts: pd.Series) -> np.ndarray:
        """
        Batch prediction.

        Args:
            texts: Series of review strings.

        Returns:
            Integer numpy array of predicted labels.
        """
        logger.info("Running VADER inference on %d texts …", len(texts))
        results = texts.apply(lambda t: self.predict_one(t)[0])
        return results.values

    def predict_proba_compound(self, texts: pd.Series) -> np.ndarray:
        """Return raw compound scores (not true probabilities)."""
        return texts.apply(lambda t: self.predict_one(t)[1]).values


# ── 2. TF-IDF + Logistic Regression ─────────────────────────────────────────

class TFIDFLogisticModel:
    """
    TF-IDF bigram vectoriser + Logistic Regression for 3-class sentiment.

    Attributes:
        vectorizer: Fitted :class:`TfidfVectorizer`.
        classifier: Fitted :class:`LogisticRegression`.
    """

    def __init__(self,
                 max_features: int = 50_000,
                 ngram_range: Tuple[int, int] = (1, 2),
                 C: float = 1.0,
                 max_iter: int = 1000,
                 class_weight: Optional[str] = "balanced") -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            min_df=2,
        )
        self.classifier = LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            solver="lbfgs",
            n_jobs=-1,
        )

    def fit(self, X_train: pd.Series, y_train: np.ndarray) -> "TFIDFLogisticModel":
        """
        Fit the vectoriser and classifier.

        Args:
            X_train: Series of cleaned text.
            y_train: Integer label array.

        Returns:
            Self (for method chaining).
        """
        logger.info("Fitting TF-IDF vectoriser …")
        X_vec = self.vectorizer.fit_transform(X_train)
        logger.info("Vocabulary size: %d", len(self.vectorizer.vocabulary_))
        logger.info("Training Logistic Regression …")
        self.classifier.fit(X_vec, y_train)
        return self

    def predict(self, X: pd.Series) -> np.ndarray:
        """Predict class labels."""
        X_vec = self.vectorizer.transform(X)
        return self.classifier.predict(X_vec)

    def predict_proba(self, X: pd.Series) -> np.ndarray:
        """Return class probability matrix (N × 3)."""
        X_vec = self.vectorizer.transform(X)
        return self.classifier.predict_proba(X_vec)

    def save(self, prefix: str = "tfidf_lr") -> None:
        """Persist vectoriser and classifier to models/."""
        save_model(self.vectorizer, f"{prefix}_vectorizer.joblib")
        save_model(self.classifier, f"{prefix}_classifier.joblib")

    @classmethod
    def load(cls, prefix: str = "tfidf_lr") -> "TFIDFLogisticModel":
        """Load a previously saved model."""
        obj = cls()
        obj.vectorizer  = load_model(f"{prefix}_vectorizer.joblib")
        obj.classifier  = load_model(f"{prefix}_classifier.joblib")
        return obj


# ── 3. DistilBERT Fine-tuned ──────────────────────────────────────────────────

class DistilBERTSentimentModel:
    """
    Fine-tuned DistilBERT for 3-class sentiment classification.

    Uses HuggingFace Trainer API with automatic GPU/CPU detection.

    Args:
        model_name: HuggingFace model identifier.
        num_labels: Number of output classes.
        max_length: Tokeniser max sequence length.
        output_dir: Directory to save fine-tuned weights.
    """

    MODEL_NAME = "distilbert-base-uncased"
    NUM_LABELS = 3

    def __init__(self,
                 model_name: str = MODEL_NAME,
                 num_labels: int = NUM_LABELS,
                 max_length: int = 128,
                 output_dir: Optional[str] = None) -> None:
        import torch
        from transformers import (AutoTokenizer,
                                   AutoModelForSequenceClassification,
                                   TrainingArguments, Trainer)

        self.model_name   = model_name
        self.num_labels   = num_labels
        self.max_length   = max_length
        self.output_dir   = output_dir or str(get_model_path("distilbert_sentiment"))
        self.device       = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("DistilBERT will run on: %s", self.device.upper())

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )

    def _tokenize(self, texts: List[str]) -> Dict:
        return self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors=None,
        )

    def fine_tune(self,
                  train_texts: List[str], train_labels: List[int],
                  eval_texts: List[str],  eval_labels: List[int],
                  num_epochs: int = 3,
                  batch_size: int = 16,
                  learning_rate: float = 2e-5) -> None:
        """
        Fine-tune DistilBERT on the training set.

        Args:
            train_texts: Cleaned training review strings.
            train_labels: Integer label list (0/1/2).
            eval_texts: Validation review strings.
            eval_labels: Validation labels.
            num_epochs: Number of training epochs.
            batch_size: Per-device batch size.
            learning_rate: AdamW learning rate.
        """
        import torch
        from torch.utils.data import Dataset
        from transformers import TrainingArguments, Trainer
        import evaluate

        class ReviewDataset(Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels    = labels

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
                item["labels"] = torch.tensor(self.labels[idx])
                return item

        logger.info("Tokenising %d train / %d eval samples …",
                    len(train_texts), len(eval_texts))
        train_enc = self._tokenize(train_texts)
        eval_enc  = self._tokenize(eval_texts)

        train_dataset = ReviewDataset(train_enc, train_labels)
        eval_dataset  = ReviewDataset(eval_enc,  eval_labels)

        metric = evaluate.load("f1")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels,
                                  average="macro")

        args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            learning_rate=learning_rate,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_steps=200,
            warmup_ratio=0.1,
            fp16=(self.device == "cuda"),
            report_to="none",
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )

        logger.info("Starting DistilBERT fine-tuning for %d epochs …", num_epochs)
        trainer.train()
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        logger.info("Model saved → %s", self.output_dir)

    def predict(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Run batch inference.

        Args:
            texts: List of review strings.
            batch_size: Inference batch size.

        Returns:
            Integer array of predicted class labels.
        """
        import torch

        self.model.eval()
        self.model.to(self.device)
        all_preds: List[int] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc   = self.tokenizer(batch, truncation=True, padding=True,
                                   max_length=self.max_length, return_tensors="pt")
            enc   = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self.model(**enc).logits
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

        return np.array(all_preds)

    def predict_proba(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Return softmax probabilities (N × 3)."""
        import torch
        import torch.nn.functional as F

        self.model.eval()
        self.model.to(self.device)
        all_probs: List[np.ndarray] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc   = self.tokenizer(batch, truncation=True, padding=True,
                                   max_length=self.max_length, return_tensors="pt")
            enc   = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self.model(**enc).logits
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

        return np.vstack(all_probs)

    @classmethod
    def from_pretrained(cls, model_dir: Optional[str] = None) -> "DistilBERTSentimentModel":
        """Load fine-tuned weights from disk."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        obj = cls.__new__(cls)
        import torch
        obj.device    = "cuda" if torch.cuda.is_available() else "cpu"
        model_dir     = model_dir or str(get_model_path("distilbert_sentiment"))
        obj.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        obj.model     = AutoModelForSequenceClassification.from_pretrained(model_dir)
        obj.max_length = 128
        logger.info("Loaded DistilBERT from %s", model_dir)
        return obj


# ── SHAP explainability helper ────────────────────────────────────────────────

def explain_tfidf_lr_with_shap(model: TFIDFLogisticModel,
                                 texts: List[str],
                                 n_samples: int = 200) -> None:
    """
    Generate SHAP summary and beeswarm plots for the TF-IDF + LR model.

    Args:
        model: Fitted :class:`TFIDFLogisticModel`.
        texts: List of cleaned text samples to explain.
        n_samples: Number of background samples for LinearExplainer.
    """
    import shap
    import matplotlib.pyplot as plt

    logger.info("Computing SHAP values for TF-IDF+LR …")
    X_vec = model.vectorizer.transform(texts[:n_samples])
    explainer = shap.LinearExplainer(model.classifier, X_vec,
                                      feature_names=model.vectorizer.get_feature_names_out())
    shap_values = explainer(X_vec)

    shap.summary_plot(shap_values[:, :, 2],  # Positive class
                      X_vec,
                      feature_names=model.vectorizer.get_feature_names_out(),
                      show=False, plot_type="bar")
    plt.title("SHAP — Top Features for Positive Class")
    plt.tight_layout()
    plt.savefig(str(get_model_path("shap_tfidf_lr.png")), dpi=150, bbox_inches="tight")
    plt.show()
    logger.info("SHAP plot saved.")
