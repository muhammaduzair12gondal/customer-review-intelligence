import { useState } from "react";
import type { TabView } from "@/types";
import Header from "@/components/Header";
import AnalyzeReview from "@/components/AnalyzeReview";
import DatasetInsights from "@/components/DatasetInsights";
import "./App.css";

function App() {
  const [activeTab, setActiveTab] = useState<TabView>("analyze");

  return (
    <div className="app-container">
      <Header activeTab={activeTab} onTabChange={setActiveTab} />
      <main className="main-content">
        {activeTab === "analyze" ? <AnalyzeReview /> : <DatasetInsights />}
      </main>
    </div>
  );
}

export default App;
