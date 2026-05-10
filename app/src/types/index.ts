/**
 * API Request / Response Types
 * Backend engineer can adjust these to match actual API contract
 */

export interface AnalyzeReviewRequest {
  text: string;
}

export interface Aspect {
  aspect: string;
  sentiment: number; // -1.0 to 1.0
}

export interface AnalyzeReviewResponse {
  sentiment: "positive" | "neutral" | "negative";
  sentiment_confidence: number; // 0.0 to 1.0
  is_fake_probability: number; // 0.0 to 1.0
  aspects: Aspect[];
  top_topics: string[];
}

export type TabView = "analyze" | "insights";

/**
 * Dataset Insights Types
 * Pre-calculated / static data expected from backend
 */
export interface DatasetSummary {
  totalReviews: number;
  averageRating: number;
  flaggedReviews: number;
}

export interface RatingDistribution {
  rating: number;
  count: number;
}

export interface SentimentComposition {
  name: string;
  value: number;
}
