---
title: Review Intelligence
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Customer Review Intelligence Platform

A production-ready NLP analytics platform designed to extract deep insights from customer feedback. This system combines sentiment analysis, rule-based and neural aspect extraction, fake review detection, and topic discovery in a unified, premium interface.

## Core Features

- **Sentiment Classification:** Multi-stage pipeline using DistilBERT for high-accuracy sentiment scoring.
- **Authenticity Check:** Advanced "Fake Review" detection using a Random Forest + XGBoost ensemble trained on behavioral heuristics.
- **Aspect-Based Sentiment (ABSA):** Identifies specific product features (e.g., taste, packaging) and scores sentiment for each.
- **Topic Modeling:** Dynamic topic cluster discovery using BERTopic for high-level dataset summarization.
- **Integrated Dashboard:** A clean, "humanized" light-themed Streamlit interface for both single-review and batch analysis.

## Project Structure

```text
customer-review-intelligence/
├── app/
│   ├── main.py              # FastAPI Backend Engine
│   └── dashboard.py         # Streamlit Human-Centric UI
├── src/
│   ├── preprocess.py        # Text cleaning and Feature Engineering
│   ├── sentiment_model.py   # Sentiment classification (BERT/TF-IDF)
│   ├── absa.py              # Aspect Extraction logic
│   ├── fake_review.py       # Authenticity Detector classes
│   └── topic_model.py       # BERTopic integration
├── notebooks/               # Research and Model Training (01-05)
├── models/                  # Persisted ML model artifacts
└── requirements.txt         # Production dependencies
```

## Setup Instructions

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

2. **Train Models:**
   Run the notebooks in `notebooks/` sequentially from 01 to 05 to generate all model artifacts.

3. **Launch the Platform:**
   
   Start the Backend:
   ```bash
   uvicorn app.main:app --port 8000
   ```

   Start the Frontend:
   ```bash
   streamlit run app/dashboard.py
   ```

## Design Philosophy

This platform prioritizes **human-centric design**, utilizing a clean light theme, Inter typography, and intuitive visualizations to make complex NLP insights accessible and actionable.

---
Built by [Muhammad Uzair](https://github.com/muhammaduzair12gondal)
