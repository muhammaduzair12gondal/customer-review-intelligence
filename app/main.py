"""
main.py — FastAPI backend for the Customer Review Intelligence Platform.

Endpoints:
    POST /analyze  — full pipeline analysis of a single review
    GET  /health   — health check
"""

import sys
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add src/ to path
SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
sys.path.insert(0, SRC_DIR)

from utils import get_logger, get_model_path
from preprocess import clean_text, engineer_fake_review_features
from sentiment_model import TFIDFLogisticModel, VADERSentimentModel
from absa import AspectBasedSentimentAnalyzer
from fake_review import FakeReviewDetector

logger = get_logger("api")

# ═════════════════════════════════════════════════════════════════════════════
# Pydantic schemas
# ═════════════════════════════════════════════════════════════════════════════

class ReviewRequest(BaseModel):
    """Request body for /analyze."""
    text: str = Field(..., min_length=1, description="Review text to analyse")


class AspectSentiment(BaseModel):
    """Individual aspect result."""
    aspect: str
    sentiment: str


class AnalysisResponse(BaseModel):
    """Response from /analyze."""
    sentiment: str
    sentiment_confidence: float
    aspects: List[AspectSentiment]
    is_fake_probability: float
    top_topics: List[str]


# ═════════════════════════════════════════════════════════════════════════════
# Application factory
# ═════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Customer Review Intelligence API",
    description="Unified NLP pipeline for sentiment, ABSA, fake detection & topics.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model holders (populated on startup) ──────────────────────────────────

_models: Dict = {}


@app.on_event("startup")
async def load_models() -> None:
    """Load all trained models into memory at application startup."""
    logger.info("Loading models …")

    # 1. Sentiment — try TF-IDF+LR first, fall back to VADER
    try:
        _models["sentiment"] = TFIDFLogisticModel.load()
        _models["sentiment_type"] = "tfidf_lr"
        logger.info("Loaded TF-IDF + LR sentiment model.")
    except Exception:
        logger.warning("TF-IDF+LR model not found — using VADER fallback.")
        _models["sentiment"] = VADERSentimentModel()
        _models["sentiment_type"] = "vader"

    # 2. ABSA
    _models["absa"] = AspectBasedSentimentAnalyzer(context_window=3)
    logger.info("ABSA analyser ready.")

    # 3. Fake review detector
    try:
        _models["fake"] = FakeReviewDetector.load()
        logger.info("Loaded fake review detector.")
    except Exception:
        _models["fake"] = None
        logger.warning("Fake review detector not found — will return 0.0.")

    # 4. BERTopic
    try:
        from topic_model import ReviewTopicModel
        _models["topics"] = ReviewTopicModel.load("bertopic_model")
        logger.info("Loaded BERTopic model.")
    except Exception:
        _models["topics"] = None
        logger.warning("BERTopic model not found — topics will be empty.")

    logger.info("All models loaded ✓")


# ═════════════════════════════════════════════════════════════════════════════
# Endpoints
# ═════════════════════════════════════════════════════════════════════════════

LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}


@app.get("/health")
async def health_check() -> Dict:
    """Return API health status."""
    return {
        "status": "healthy",
        "sentiment_model": _models.get("sentiment_type", "none"),
        "fake_detector":   "loaded" if _models.get("fake") else "unavailable",
        "topic_model":     "loaded" if _models.get("topics") else "unavailable",
    }


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_review(req: ReviewRequest) -> AnalysisResponse:
    """
    Full pipeline analysis of a single review.

    Returns sentiment, aspect sentiments, fake probability, and matched topics.
    """
    text = req.text

    # ── Sentiment ─────────────────────────────────────────────────────────
    sentiment_model = _models["sentiment"]
    if _models["sentiment_type"] == "tfidf_lr":
        cleaned = clean_text(text)
        proba = sentiment_model.predict_proba(pd.Series([cleaned]))[0]
        label_idx = int(proba.argmax())
        confidence = float(proba.max())
    else:
        label_idx, compound = sentiment_model.predict_one(text)
        confidence = round(min(abs(compound), 1.0), 4)

    sentiment = LABEL_MAP[label_idx]

    # ── ABSA ──────────────────────────────────────────────────────────────
    absa = _models["absa"]
    aspect_results = absa.analyze_review(0, text)
    aspects = [
        AspectSentiment(aspect=a["aspect"], sentiment=a["sentiment"])
        for a in aspect_results
    ]

    # ── Fake review probability ───────────────────────────────────────────
    fake_prob = 0.0
    if _models.get("fake"):
        # Build a minimal DataFrame row
        dummy_row = pd.DataFrame([{
            "Text": text,
            "Score": {0: 1, 1: 3, 2: 5}[label_idx],
            "ProductId": "unknown",
            "HelpfulnessNumerator": 0,
            "HelpfulnessDenominator": 0,
        }])
        features = engineer_fake_review_features(dummy_row)
        try:
            fake_prob = float(_models["fake"].predict_proba_rf(features)[0])
        except Exception:
            fake_prob = 0.0

    # ── Topics ────────────────────────────────────────────────────────────
    # 1. Hybrid Smart Topic Extractor (E-commerce priority)
    text_lower = text.lower()
    top_topics: List[str] = []
    
    if any(w in text_lower for w in ["shipping", "delivery", "arrived", "late", "package", "box", "tracking"]):
        top_topics.append("Shipping & Logistics")
    if any(w in text_lower for w in ["price", "expensive", "cheap", "worth", "money", "cost", "value"]):
        top_topics.append("Pricing & Value")
    if any(w in text_lower for w in ["customer service", "support", "refund", "return", "manager", "representative"]):
        top_topics.append("Customer Service")
    if any(w in text_lower for w in ["quality", "broken", "defective", "sturdy", "material", "durability", "taste", "flavor"]):
        top_topics.append("Product Quality")

    # 2. Fallback to BERTopic only if heuristic finds nothing AND text is long enough
    if not top_topics and _models.get("topics"):
        try:
            cleaned_for_topic = clean_text(text)
            if len(cleaned_for_topic.split()) >= 10:
                topic_ids, probs = _models["topics"].transform([cleaned_for_topic])
                info = _models["topics"].get_topic_info()
                for i, tid in enumerate(topic_ids):
                    prob_val = 1.0
                    if probs is not None and len(probs) > 0:
                        prob_item = probs[i]
                        prob_val = float(max(prob_item)) if hasattr(prob_item, "__iter__") else float(prob_item)
                    
                    if tid >= 0 and prob_val > 0.40:
                        row = info[info["Topic"] == tid]
                        if not row.empty:
                            top_topics.append(row.iloc[0].get("Name", f"Topic {tid}"))

        except Exception:
            pass

    return AnalysisResponse(
        sentiment=sentiment,
        sentiment_confidence=round(confidence, 4),
        aspects=aspects,
        is_fake_probability=round(fake_prob, 4),
        top_topics=top_topics[:5],
    )


# ═════════════════════════════════════════════════════════════════════════════
# Run with:  uvicorn app.main:app --reload --port 8000
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
