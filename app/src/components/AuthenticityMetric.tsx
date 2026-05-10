interface AuthenticityMetricProps {
  fakeProbability: number;
}

export default function AuthenticityMetric({ fakeProbability }: AuthenticityMetricProps) {
  const percent = Math.round(fakeProbability * 100);

  return (
    <div className="metric-card authenticity-metric">
      <h3>Authenticity Check</h3>
      <div className="metric-value authenticity-value" data-risk={percent > 70 ? "high" : percent > 40 ? "medium" : "low"}>
        {percent}%
      </div>
      <div className="metric-label">Fake Probability</div>
      <div className="confidence-bar">
        <div
          className="confidence-fill authenticity-fill"
          style={{ width: `${percent}%` }}
        />
      </div>
    </div>
  );
}
