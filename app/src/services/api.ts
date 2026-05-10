import type { AnalyzeReviewRequest, AnalyzeReviewResponse } from "@/types";

/**
 * API Service Module
 * Centralized place for all API calls.
 * Backend engineer can adjust BASE_URL and endpoints here.
 */

const BASE_URL = "http://localhost:8000";

export async function analyzeReview(text: string): Promise<AnalyzeReviewResponse> {
  const payload: AnalyzeReviewRequest = { text };

  const response = await fetch(`${BASE_URL}/analyze`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error(`API Error: ${response.status} ${response.statusText}`);
  }

  return response.json();
}

// TODO: Add dataset insights endpoint when ready
// export async function fetchDatasetInsights(): Promise<DatasetInsightsResponse> {
//   const response = await fetch(`${BASE_URL}/insights`);
//   if (!response.ok) throw new Error(`API Error: ${response.status}`);
//   return response.json();
// }
