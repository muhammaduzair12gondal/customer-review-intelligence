import { useState } from "react";
import { analyzeReview } from "@/services/api";
import type { AnalyzeReviewResponse } from "@/types";
import SentimentMetric from "./SentimentMetric";
import AuthenticityMetric from "./AuthenticityMetric";
import AspectsList from "./AspectsList";
import TopicsList from "./TopicsList";

export default function AnalyzeReview() {
  const [reviewText, setReviewText] = useState("");
  const [result, setResult] = useState<AnalyzeReviewResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async () => {
    if (!reviewText.trim()) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await analyzeReview(reviewText);
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An unexpected error occurred");
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      handleSubmit();
    }
  };

  return (
    <div className="analyze-review-view">
      {/* Input Section */}
      <section className="input-section">
        <textarea
          className="review-textarea"
          placeholder="Paste a customer review here... (Ctrl+Enter to submit)"
          value={reviewText}
          onChange={(e) => setReviewText(e.target.value)}
          onKeyDown={handleKeyDown}
          rows={6}
        />
        <button
          className="submit-btn"
          onClick={handleSubmit}
          disabled={loading || !reviewText.trim()}
        >
          {loading ? "Analyzing..." : "Analyze Review"}
        </button>
      </section>

      {/* Loading State */}
      {loading && (
        <div className="loading-state">
          <div className="spinner" />
          <p>Analyzing your review...</p>
        </div>
      )}

      {/* Error State */}
      {error && (
        <div className="error-state">
          <p>{error}</p>
        </div>
      )}

      {/* Results Section */}
      {result && !loading && (
        <section className="results-section">
          <div className="metrics-row">
            <SentimentMetric
              sentiment={result.sentiment}
              confidence={result.sentiment_confidence}
            />
            <AuthenticityMetric
              fakeProbability={result.is_fake_probability}
            />
          </div>

          <div className="details-row">
            <AspectsList aspects={result.aspects} />
            <TopicsList topics={result.top_topics} />
          </div>
        </section>
      )}
    </div>
  );
}
