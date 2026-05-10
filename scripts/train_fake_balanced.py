import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from preprocess import create_fake_review_flag, engineer_fake_review_features
from fake_review import FakeReviewDetector
from utils import get_logger

logger = get_logger("training_fake")

def main():
    data_path = "data/Reviews.csv"
    if not os.path.exists(data_path):
        logger.error(f"Data not found at {data_path}")
        return

    logger.info("Loading full dataset ...")
    df = pd.read_csv(data_path)
    
    logger.info("Labeling suspicious reviews ...")
    df["is_suspicious"] = create_fake_review_flag(df)
    
    # Create balanced dataset
    df_fake = df[df["is_suspicious"] == True]
    df_real = df[df["is_suspicious"] == False].sample(n=min(5000, len(df)), random_state=42)
    
    df_balanced = pd.concat([df_fake, df_real]).sample(frac=1, random_state=42)
    logger.info(f"Balanced dataset created: {len(df_balanced)} samples ({len(df_fake)} suspicious)")

    logger.info("Engineering features ...")
    X = engineer_fake_review_features(df_balanced)
    y = df_balanced["is_suspicious"].values.astype(int)

    logger.info("Training model ...")
    detector = FakeReviewDetector(use_xgboost=True)
    detector.fit(X, y)
    
    logger.info("Saving model ...")
    detector.save()
    logger.info("Retraining complete ✓")

if __name__ == "__main__":
    main()
