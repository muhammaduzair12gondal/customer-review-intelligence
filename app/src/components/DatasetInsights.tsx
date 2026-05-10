import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend,
} from "recharts";
import type { DatasetSummary, RatingDistribution, SentimentComposition } from "@/types";

/**
 * Mock data — replace with actual API call when backend endpoint is ready
 * Example: const { data } = useSWR('/insights', fetchDatasetInsights)
 */
const mockSummary: DatasetSummary = {
  totalReviews: 12483,
  averageRating: 4.2,
  flaggedReviews: 342,
};

const mockRatingDistribution: RatingDistribution[] = [
  { rating: 1, count: 420 },
  { rating: 2, count: 680 },
  { rating: 3, count: 1520 },
  { rating: 4, count: 3890 },
  { rating: 5, count: 5973 },
];

const mockSentimentComposition: SentimentComposition[] = [
  { name: "Positive", value: 8430 },
  { name: "Neutral", value: 2610 },
  { name: "Negative", value: 1443 },
];

const PIE_COLORS = ["#22c55e", "#6b7280", "#ef4444"];

export default function DatasetInsights() {
  return (
    <div className="dataset-insights-view">
      {/* Summary Cards */}
      <section className="summary-cards">
        <div className="summary-card">
          <h4>Total Reviews</h4>
          <p className="summary-value">{mockSummary.totalReviews.toLocaleString()}</p>
        </div>
        <div className="summary-card">
          <h4>Average Rating</h4>
          <p className="summary-value">{mockSummary.averageRating.toFixed(1)}</p>
        </div>
        <div className="summary-card">
          <h4>Flagged Reviews</h4>
          <p className="summary-value">{mockSummary.flaggedReviews.toLocaleString()}</p>
        </div>
      </section>

      {/* Charts */}
      <section className="charts-section">
        <div className="chart-container">
          <h3>Rating Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={mockRatingDistribution}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="rating" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-container">
          <h3>Sentiment Composition</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={mockSentimentComposition}
                cx="50%"
                cy="50%"
                outerRadius={100}
                fill="#8884d8"
                dataKey="value"
                label
              >
                {mockSentimentComposition.map((_entry, index) => (
                  <Cell key={index} fill={PIE_COLORS[index % PIE_COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </section>
    </div>
  );
}
