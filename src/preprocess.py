"""
preprocess.py — Text cleaning and feature engineering pipeline.

All public functions are importable by notebooks and the FastAPI backend.
"""

import re
import string
import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from utils import get_logger

# ---------------------------------------------------------------------------
# Module logger
# ---------------------------------------------------------------------------
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# NLTK lazy downloads
# ---------------------------------------------------------------------------

def _ensure_nltk_data() -> None:
    """Download required NLTK corpora if not already present."""
    import nltk
    for resource in ("stopwords", "wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


_ensure_nltk_data()

_STOP_WORDS: set = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()

# ---------------------------------------------------------------------------
# Sentiment label creation
# ---------------------------------------------------------------------------

LABEL_MAP = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}
LABEL_NAMES = {0: "Negative", 1: "Neutral", 2: "Positive"}


def create_sentiment_labels(scores: pd.Series) -> pd.Series:
    """
    Convert 1-5 star scores to 3-class sentiment labels.

    Mapping:
        * 1-2  → 0 (Negative)
        * 3    → 1 (Neutral)
        * 4-5  → 2 (Positive)

    Args:
        scores: Integer Series of raw star ratings.

    Returns:
        Integer Series of sentiment labels.
    """
    return scores.map(LABEL_MAP)


# ---------------------------------------------------------------------------
# Fake review flag
# ---------------------------------------------------------------------------

def create_fake_review_flag(df: pd.DataFrame) -> pd.Series:
    """
    Create a binary 'suspicious' flag for potential fake reviews.

    A review is flagged suspicious when:
        * review body has fewer than 15 characters, OR
        * rating deviation > 3.0 AND denominator > 2 AND helpfulness_ratio < 0.1

    Args:
        df: DataFrame containing HelpfulnessNumerator,
            HelpfulnessDenominator, Score, ProductId, and Text columns.

    Returns:
        Boolean Series (True = suspicious).
    """
    text = df["Text"].fillna("")
    short_review = text.str.len() < 15

    denom = df["HelpfulnessDenominator"].replace(0, np.nan)
    ratio = (df["HelpfulnessNumerator"] / denom).fillna(0.0)

    mean_product_score = df.groupby("ProductId")["Score"].transform("mean")
    rating_deviation = (df["Score"] - mean_product_score).abs()

    outlier_low_helpfulness = (rating_deviation > 3.0) & (df["HelpfulnessDenominator"] > 2) & (ratio < 0.1)

    flag = short_review | outlier_low_helpfulness
    n_flagged = flag.sum()
    logger.info("Fake review flag: %d suspicious reviews (%.2f%%)",
                n_flagged, 100 * n_flagged / len(df))
    return flag


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str,
               remove_stopwords: bool = True,
               lemmatize: bool = True) -> str:
    """
    Clean a raw review string.

    Steps:
        1. Lower-case
        2. Remove HTML tags
        3. Remove URLs
        4. Remove punctuation & digits
        5. Tokenise on whitespace
        6. Optionally remove stop-words
        7. Optionally lemmatize

    Args:
        text: Raw review text.
        remove_stopwords: If True, strip English stop-words.
        lemmatize: If True, apply WordNet lemmatization.

    Returns:
        Cleaned string of space-joined tokens.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Lower-case
    text = text.lower()

    # 2. Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # 3. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 4. Remove punctuation and digits
    text = text.translate(str.maketrans("", "", string.punctuation + string.digits))

    # 5. Tokenise
    tokens: List[str] = text.split()

    # 6. Remove stop-words
    if remove_stopwords:
        tokens = [t for t in tokens if t not in _STOP_WORDS]

    # 7. Lemmatize
    if lemmatize:
        tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def clean_text_series(series: pd.Series,
                       remove_stopwords: bool = True,
                       lemmatize: bool = True) -> pd.Series:
    """
    Apply :func:`clean_text` to an entire Series using a vectorised approach.

    Args:
        series: Pandas Series of raw text strings.
        remove_stopwords: Passed to :func:`clean_text`.
        lemmatize: Passed to :func:`clean_text`.

    Returns:
        Series of cleaned strings.
    """
    logger.info("Cleaning %d text samples …", len(series))
    cleaned = series.fillna("").apply(
        lambda t: clean_text(t, remove_stopwords=remove_stopwords, lemmatize=lemmatize)
    )
    empty_count = (cleaned == "").sum()
    if empty_count:
        logger.warning("%d reviews became empty strings after cleaning.", empty_count)
    return cleaned


# ---------------------------------------------------------------------------
# Feature engineering for fake review detection
# ---------------------------------------------------------------------------

def engineer_fake_review_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute numerical features used for fake review detection.

    Engineered columns:
        * review_length      — character count of Text
        * word_count         — number of whitespace-delimited words
        * avg_word_length    — mean word length
        * exclamation_count  — count of '!' characters
        * capital_ratio      — fraction of alphabetical chars that are upper-case
        * helpfulness_ratio  — HelpfulnessNumerator / HelpfulnessDenominator
        * rating_deviation   — abs(Score - mean product score)
        * verified_purchase  — proxy: helpfulness_ratio > 0.5
        * duplicate_text     — True if Text duplicated in dataset

    Args:
        df: Input DataFrame with at least: Text, Score, ProductId,
            HelpfulnessNumerator, HelpfulnessDenominator.

    Returns:
        DataFrame with only the engineered feature columns (same index).
    """
    text = df["Text"].fillna("")

    review_length = text.str.len()
    word_count = text.str.split().str.len().fillna(0)
    avg_word_length = text.apply(
        lambda t: np.mean([len(w) for w in t.split()]) if t.split() else 0.0
    )
    exclamation_count = text.str.count(r"!")
    capital_ratio = text.apply(
        lambda t: sum(1 for c in t if c.isupper()) / max(sum(1 for c in t if c.isalpha()), 1)
    )

    denom = df["HelpfulnessDenominator"].replace(0, np.nan)
    helpfulness_ratio = (df["HelpfulnessNumerator"] / denom).fillna(0.0)

    mean_product_score = df.groupby("ProductId")["Score"].transform("mean")
    rating_deviation = (df["Score"] - mean_product_score).abs()

    verified_purchase = (helpfulness_ratio > 0.5).astype(int)
    duplicate_text = text.duplicated(keep=False).astype(int)

    features = pd.DataFrame(
        {
            "review_length": review_length,
            "word_count": word_count,
            "avg_word_length": avg_word_length,
            "exclamation_count": exclamation_count,
            "capital_ratio": capital_ratio,
            "helpfulness_ratio": helpfulness_ratio,
            "rating_deviation": rating_deviation,
            "verified_purchase": verified_purchase,
            "duplicate_text": duplicate_text,
        },
        index=df.index,
    )
    logger.info("Feature engineering complete. Shape: %s", features.shape)
    return features


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------

def run_full_pipeline(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute the complete preprocessing pipeline on a raw Reviews DataFrame.

    Steps:
        1. Drop rows with null Text or Score
        2. Drop exact duplicates (all columns)
        3. Create 3-class sentiment labels
        4. Create fake review flag
        5. Clean text

    Args:
        df: Raw DataFrame as loaded from Reviews.csv.

    Returns:
        Tuple of (processed_df, feature_df) where:
            * processed_df includes 'sentiment_label', 'is_suspicious', 'clean_text'
            * feature_df contains fake-review numerical features
    """
    logger.info("Starting full preprocessing pipeline … Input shape: %s", df.shape)

    # 1. Drop nulls in critical columns
    df = df.dropna(subset=["Text", "Score"]).copy()
    logger.info("After null drop: %d rows", len(df))

    # 2. Drop duplicates
    before = len(df)
    df = df.drop_duplicates()
    logger.info("Dropped %d exact duplicates. Remaining: %d", before - len(df), len(df))

    # 3. Sentiment labels
    df["sentiment_label"] = create_sentiment_labels(df["Score"])

    # 4. Fake review flag
    df["is_suspicious"] = create_fake_review_flag(df)

    # 5. Clean text
    df["clean_text"] = clean_text_series(df["Text"])

    # 6. Feature engineering
    feature_df = engineer_fake_review_features(df)

    logger.info("Pipeline complete. Final shape: %s", df.shape)
    return df, feature_df
