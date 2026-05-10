interface SentimentMetricProps {
  sentiment: string;
  confidence: number;
}

export default function SentimentMetric({ sentiment, confidence }: SentimentMetricProps) {
  const confidencePercent = Math.round(confidence * 100);

  return (
    <div className="metric-card sentiment-metric">
      <h3>Sentiment</h3>
      <div className="metric-value sentiment-value" data-sentiment={sentiment}>
        {sentiment}
      </div>
      <div className="metric-confidence">
        <span className="confidence-label">Confidence</span>
        <span className="confidence-percent">{confidencePercent}%</span>
        <div className="confidence-bar">
          <div
            className="confidence-fill"
            style={{ width: `${confidencePercent}%` }}
          />
        </div>
      </div>
    </div>
  );
}
