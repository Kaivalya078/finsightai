/**
 * FinSight AI - API Client
 * ========================
 * Handles communication with the FastAPI backend.
 * 
 * Only one function: askQuestion()
 * Sends POST /chat and returns the response.
 */

// Backend URL (Vite dev server runs on 5173, FastAPI on 8000)
const API_BASE = "http://localhost:8000";

/**
 * Send a question to the RAG backend and get a grounded answer.
 * 
 * @param {string} question - The user's question
 * @returns {Promise<{answer: string, citations: string[], evidence: Array}>}
 * @throws {Error} If the request fails
 */
export async function askQuestion(question) {
    const response = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ question }),
    });

    // Handle HTTP errors
    if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        const message = errorData?.detail || `Server error (${response.status})`;
        throw new Error(message);
    }

    // Parse and return the JSON response
    return response.json();
}
