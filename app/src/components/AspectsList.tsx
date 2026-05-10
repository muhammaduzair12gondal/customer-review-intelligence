import type { Aspect } from "@/types";

interface AspectsListProps {
  aspects: Aspect[];
}

export default function AspectsList({ aspects }: AspectsListProps) {
  const getSentimentLabel = (score: number) => {
    if (score >= 0.3) return "positive";
    if (score <= -0.3) return "negative";
    return "neutral";
  };

  return (
    <div className="aspects-list">
      <h3>Extracted Aspects</h3>
      {aspects.length === 0 ? (
        <p className="empty-state">No aspects detected</p>
      ) : (
        <ul>
          {aspects.map((item) => (
            <li key={item.aspect} className="aspect-item">
              <span className="aspect-name">{item.aspect}</span>
              <span
                className="aspect-sentiment-badge"
                data-sentiment={getSentimentLabel(item.sentiment)}
              >
                {item.sentiment > 0 ? "+" : ""}
                {item.sentiment.toFixed(2)}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
