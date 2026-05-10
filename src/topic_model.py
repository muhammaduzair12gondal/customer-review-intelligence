"""
topic_model.py — BERTopic wrapper for topic discovery on food reviews.

Provides training, saving/loading, and visualization helpers for the
BERTopic pipeline.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils import get_logger, get_model_path

logger = get_logger(__name__)


class ReviewTopicModel:
    """
    BERTopic-based topic model tailored for Amazon food reviews.

    Args:
        n_topics: Target number of topics (use 'auto' for automatic).
        min_topic_size: Minimum cluster size for HDBSCAN.
        embedding_model: Sentence-transformer model for embeddings.
    """

    DEFAULT_EMBEDDING = "all-MiniLM-L6-v2"

    def __init__(self,
                 n_topics: Optional[int] = None,
                 min_topic_size: int = 50,
                 embedding_model: str = DEFAULT_EMBEDDING) -> None:
        from bertopic import BERTopic

        self.n_topics = n_topics or "auto"
        self.min_topic_size = min_topic_size
        self.embedding_model_name = embedding_model

        self.model = BERTopic(
            nr_topics=self.n_topics,
            min_topic_size=min_topic_size,
            embedding_model=embedding_model,
            verbose=True,
        )
        logger.info("BERTopic initialised (min_topic_size=%d, embedding=%s)",
                     min_topic_size, embedding_model)

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(self, docs: List[str]) -> Tuple[List[int], np.ndarray]:
        """
        Fit BERTopic on a list of document strings.

        Args:
            docs: List of cleaned review texts.

        Returns:
            Tuple of (topic assignments per doc, topic probabilities matrix).
        """
        logger.info("Fitting BERTopic on %d documents …", len(docs))
        topics, probs = self.model.fit_transform(docs)
        n_found = len(set(topics)) - (1 if -1 in topics else 0)
        logger.info("Found %d topics.", n_found)
        return topics, probs

    def transform(self, docs: List[str]) -> Tuple[List[int], np.ndarray]:
        """
        Assign new documents to existing topics.

        Args:
            docs: List of review texts.

        Returns:
            Tuple of (topic assignments, probabilities).
        """
        return self.model.transform(docs)

    # ── Topic inspection ──────────────────────────────────────────────────────

    def get_topic_info(self) -> pd.DataFrame:
        """Return the BERTopic topic info DataFrame."""
        return self.model.get_topic_info()

    def get_top_words(self, topic_id: int, n: int = 10) -> List[Tuple[str, float]]:
        """
        Get the top representative words for a topic.

        Args:
            topic_id: Topic number.
            n: Number of words to return.

        Returns:
            List of (word, weight) tuples.
        """
        words = self.model.get_topic(topic_id)
        return words[:n]

    def set_topic_labels(self, labels: Dict[int, str]) -> None:
        """
        Manually label topics (e.g. {0: 'taste complaints', 1: 'delivery praise'}).

        Args:
            labels: Dict mapping topic_id → human label string.
        """
        self.model.set_topic_labels(labels)
        logger.info("Set %d custom topic labels.", len(labels))

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, name: str = "bertopic_model") -> None:
        """Save BERTopic model to models/<name>/."""
        save_dir = get_model_path(name)
        self.model.save(str(save_dir), serialization="safetensors",
                        save_ctfidf=True, save_embedding_model=self.embedding_model_name)
        logger.info("BERTopic saved → %s", save_dir)

    @classmethod
    def load(cls, name: str = "bertopic_model") -> "ReviewTopicModel":
        """Load a saved BERTopic model."""
        from bertopic import BERTopic
        load_path = get_model_path(name)
        obj = cls.__new__(cls)
        obj.model = BERTopic.load(str(load_path))
        logger.info("BERTopic loaded ← %s", load_path)
        return obj

    # ── Visualisations ────────────────────────────────────────────────────────

    def plot_topic_map(self):
        """Return interactive Plotly topic visualisation figure."""
        return self.model.visualize_topics()

    def plot_barchart(self, top_n_topics: int = 10):
        """Return Plotly bar chart of top-N topic keywords."""
        return self.model.visualize_barchart(top_n_topics=top_n_topics)

    def plot_heatmap(self):
        """Return topic similarity heatmap (Plotly)."""
        return self.model.visualize_heatmap()

    def plot_topics_over_time(self, docs: List[str],
                                timestamps: List,
                                n_bins: int = 20):
        """
        Build and plot topic distribution over time.

        Args:
            docs: Review texts used during fit.
            timestamps: Per-doc timestamps (datetime or numeric).
            n_bins: Number of time bins.

        Returns:
            Plotly figure.
        """
        topics_over_time = self.model.topics_over_time(
            docs, timestamps, nr_bins=n_bins
        )
        return self.model.visualize_topics_over_time(topics_over_time)
