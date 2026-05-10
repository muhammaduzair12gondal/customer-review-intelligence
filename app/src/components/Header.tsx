interface HeaderProps {
  activeTab: string;
  onTabChange: (tab: "analyze" | "insights") => void;
}

export default function Header({ activeTab, onTabChange }: HeaderProps) {
  return (
    <header>
      <h1>Customer Review Intelligence</h1>
      <nav>
        <button
          onClick={() => onTabChange("analyze")}
          data-active={activeTab === "analyze"}
        >
          Analyze Review
        </button>
        <button
          onClick={() => onTabChange("insights")}
          data-active={activeTab === "insights"}
        >
          Dataset Insights
        </button>
      </nav>
    </header>
  );
}
