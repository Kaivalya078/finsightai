/**
 * FinSight AI - Main App
 * =======================
 * Single-page React UI for the RAG demo.
 * 
 * State management:
 *   - result: the full /chat response (or null)
 *   - loading: true while waiting for API
 *   - error: error message string (or null)
 * 
 * All state lives here and is passed down as props.
 */

import { useState } from "react";
import { askQuestion } from "./api";
import QuestionBox from "./components/QuestionBox";
import AnswerView from "./components/AnswerView";
import EvidenceList from "./components/EvidenceList";
import "./App.css";

export default function App() {
  // --- State ---
  const [result, setResult] = useState(null);    // API response
  const [loading, setLoading] = useState(false);  // Loading spinner
  const [error, setError] = useState(null);       // Error message

  // --- Handler: when user asks a question ---
  const handleAsk = async (question) => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await askQuestion(question);
      setResult(data);
    } catch (err) {
      setError(err.message || "Something went wrong. Is the backend running?");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="app-header">
        <div className="logo">
          <span className="logo-icon">◈</span>
          <h1>FinSight AI</h1>
        </div>
        <p className="subtitle">Indian Financial Document Analyzer</p>
      </header>

      {/* Main Content */}
      <main className="app-main">
        {/* Question Input */}
        <QuestionBox onAsk={handleAsk} loading={loading} />

        {/* Error Message */}
        {error && (
          <div className="error-box">
            <span className="error-icon">⚠</span>
            <p>{error}</p>
          </div>
        )}

        {/* Answer + Evidence */}
        {result && (
          <div className="results">
            <AnswerView
              answer={result.answer}
              citations={result.citations}
            />
            <EvidenceList evidence={result.evidence} />
          </div>
        )}

        {/* Empty State */}
        {!result && !loading && !error && (
          <div className="empty-state">
            <p>Ask a question about the loaded financial document.</p>
            <p className="hint">Try: "What are the risk factors?" or "Who are the promoters?"</p>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <p>Powered by RAG · GPT-4o-mini · Phase 2 Demo</p>
      </footer>
    </div>
  );
}
