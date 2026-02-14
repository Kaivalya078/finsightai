/**
 * QuestionBox Component
 * =====================
 * Input field + Ask button for submitting questions.
 * 
 * Props:
 *   onAsk(question) - callback when user submits a question
 *   loading         - boolean, disables input while waiting
 */

import { useState } from "react";

export default function QuestionBox({ onAsk, loading }) {
    const [question, setQuestion] = useState("");

    // Handle form submission
    const handleSubmit = (e) => {
        e.preventDefault(); // Prevent page reload
        const trimmed = question.trim();
        if (!trimmed) return; // Don't submit empty questions
        onAsk(trimmed);
    };

    return (
        <form className="question-box" onSubmit={handleSubmit}>
            <div className="input-wrapper">
                <input
                    type="text"
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    placeholder="Ask a question about the document..."
                    disabled={loading}
                    autoFocus
                />
                <button type="submit" disabled={loading || !question.trim()}>
                    {loading ? (
                        <span className="spinner" />
                    ) : (
                        "Ask"
                    )}
                </button>
            </div>
        </form>
    );
}
