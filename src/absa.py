"""
absa.py — Aspect-Based Sentiment Analysis for food reviews.

Uses spaCy dependency parsing to extract aspect terms from a predefined
lexicon, then classifies each aspect mention as positive/negative/neutral
using VADER compound scores on contextual windows.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from utils import get_logger

logger = get_logger(__name__)

# ── Food domain aspect lexicon ────────────────────────────────────────────────

ASPECT_KEYWORDS: Dict[str, List[str]] = {
    "taste":        ["taste", "flavor", "flavour", "delicious", "yummy", "bland",
                     "sweet", "salty", "sour", "bitter", "savory", "savoury"],
    "price":        ["price", "cost", "cheap", "expensive", "affordable", "value",
                     "worth", "overpriced", "bargain"],
    "packaging":    ["packaging", "package", "box", "container", "sealed",
                     "wrapped", "damaged", "broken", "leaking"],
    "delivery":     ["delivery", "shipping", "shiping", "arrived", "fast", "slow", "late",
                     "courier", "tracking"],
    "quality":      ["quality", "fresh", "stale", "expired", "organic",
                     "natural", "pure", "authentic", "genuine"],
    "smell":        ["smell", "aroma", "odor", "odour", "scent", "fragrant",
                     "rancid", "musty"],
}

ASPECT_LIST = list(ASPECT_KEYWORDS.keys())


# ── Core ABSA class ───────────────────────────────────────────────────────────

class AspectBasedSentimentAnalyzer:
    """
    Rule-based Aspect-Based Sentiment Analyser for food reviews.

    Steps:
        1. Detect aspect keywords in the review text.
        2. Extract a contextual window (5 tokens) around each keyword.
        3. Score the window with VADER to assign pos/neg/neutral sentiment.

    Args:
        context_window: Number of words on each side of the keyword to include.
    """

    def __init__(self, context_window: int = 5) -> None:
        self._vader = SentimentIntensityAnalyzer()
        self._context_window = context_window

        # Build reverse lookup: keyword → aspect
        self._keyword_to_aspect: Dict[str, str] = {}
        for aspect, keywords in ASPECT_KEYWORDS.items():
            for kw in keywords:
                self._keyword_to_aspect[kw.lower()] = aspect

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze_review(self, review_id: int, text: str) -> List[Dict]:
        """
        Extract aspect sentiments from a single review.

        Args:
            review_id: Row identifier (integer).
            text: Raw or cleaned review text.

        Returns:
            List of dicts with keys:
                review_id, aspect, sentiment, confidence, context_snippet.
        """
        if not isinstance(text, str):
            return []

        tokens = re.findall(r"\b\w+\b", text.lower())
        results: List[Dict] = []
        seen_aspects: set = set()

        for idx, token in enumerate(tokens):
            if token in self._keyword_to_aspect:
                aspect = self._keyword_to_aspect[token]
                if aspect in seen_aspects:
                    continue  # report each aspect once per review
                seen_aspects.add(aspect)

                # Extract context window
                start = max(0, idx - self._context_window)
                end   = min(len(tokens), idx + self._context_window + 1)
                snippet = " ".join(tokens[start:end])

                scores = self._vader.polarity_scores(snippet)
                compound = scores["compound"]

                if compound >= 0.05:
                    sentiment, confidence = "positive", round(compound, 4)
                elif compound <= -0.05:
                    sentiment, confidence = "negative", round(abs(compound), 4)
                else:
                    sentiment, confidence = "neutral", round(1 - abs(compound), 4)

                results.append({
                    "review_id":       review_id,
                    "aspect":          aspect,
                    "sentiment":       sentiment,
                    "confidence":      confidence,
                    "context_snippet": snippet,
                })

        return results

    def analyze_batch(self, df: pd.DataFrame,
                       text_col: str = "Text",
                       id_col: str = "Id") -> pd.DataFrame:
        """
        Run ABSA on an entire DataFrame.

        Args:
            df: DataFrame containing review texts.
            text_col: Name of the text column.
            id_col: Name of the review ID column.

        Returns:
            Long-format DataFrame with columns:
                [review_id, aspect, sentiment, confidence, context_snippet].
        """
        logger.info("Running ABSA on %d reviews …", len(df))
        records: List[Dict] = []
        for _, row in df.iterrows():
            records.extend(
                self.analyze_review(row[id_col], row[text_col])
            )
        result_df = pd.DataFrame(records)
        logger.info("ABSA complete. Found %d aspect mentions.", len(result_df))
        return result_df

    def get_aspect_summary(self, absa_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate aspect sentiment counts into a pivot table.

        Args:
            absa_df: Output of :meth:`analyze_batch`.

        Returns:
            DataFrame with aspects as rows and sentiment counts as columns.
        """
        if absa_df.empty:
            return pd.DataFrame()
        pivot = (
            absa_df.groupby(["aspect", "sentiment"])
                   .size()
                   .unstack(fill_value=0)
                   .reindex(columns=["negative", "neutral", "positive"], fill_value=0)
        )
        pivot["total"] = pivot.sum(axis=1)
        pivot = pivot.sort_values("total", ascending=False)
        return pivot


# ── spaCy dependency-aware extractor (optional, richer output) ────────────────

class SpacyABSAAnalyzer:
    """
    Dependency-parsing ABSA using spaCy.

    For each noun/noun-phrase in the dependency tree, checks whether it
    belongs to a known aspect and scores its sentiment using the governing
    opinion adjectives.

    Args:
        model: spaCy model name (must be installed).
    """

    def __init__(self, model: str = "en_core_web_sm") -> None:
        try:
            import spacy
            self._nlp = spacy.load(model)
        except OSError:
            raise RuntimeError(
                f"spaCy model '{model}' not found. "
                f"Run: python -m spacy download {model}"
            )
        self._vader = SentimentIntensityAnalyzer()
        self._keyword_to_aspect: Dict[str, str] = {}
        for aspect, keywords in ASPECT_KEYWORDS.items():
            for kw in keywords:
                self._keyword_to_aspect[kw.lower()] = aspect

    def analyze_review(self, review_id: int, text: str) -> List[Dict]:
        """
        Parse a review with spaCy and return aspect sentiments.

        Args:
            review_id: Unique review identifier.
            text: Raw review string.

        Returns:
            List of aspect-sentiment dicts (same schema as
            :meth:`AspectBasedSentimentAnalyzer.analyze_review`).
        """
        doc = self._nlp(text[:512])  # cap at 512 chars for speed
        results: List[Dict] = []
        seen: set = set()

        for token in doc:
            lw = token.lemma_.lower()
            if lw in self._keyword_to_aspect and lw not in seen:
                aspect = self._keyword_to_aspect[lw]
                seen.add(lw)

                # Collect opinion modifiers from the subtree
                opinion_words = [
                    child.text for child in token.subtree
                    if child.dep_ in ("amod", "advmod", "neg")
                ]
                context = f"{token.text} {' '.join(opinion_words)}"
                scores = self._vader.polarity_scores(context)
                compound = scores["compound"]

                if compound >= 0.05:
                    sentiment, confidence = "positive", round(compound, 4)
                elif compound <= -0.05:
                    sentiment, confidence = "negative", round(abs(compound), 4)
                else:
                    sentiment, confidence = "neutral", round(1 - abs(compound), 4)

                results.append({
                    "review_id":       review_id,
                    "aspect":          aspect,
                    "sentiment":       sentiment,
                    "confidence":      confidence,
                    "context_snippet": context,
                })

        return results
