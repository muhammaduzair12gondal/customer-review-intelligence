interface TopicsListProps {
  topics: string[];
}

export default function TopicsList({ topics }: TopicsListProps) {
  return (
    <div className="topics-list">
      <h3>Detected Topics</h3>
      {topics.length === 0 ? (
        <p className="empty-state">No topics detected</p>
      ) : (
        <div className="topics-badges">
          {topics.map((topic, index) => (
            <span key={index} className="topic-badge">
              {topic}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
