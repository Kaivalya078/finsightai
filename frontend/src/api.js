/**
 * FinSight AI - API Client
 * ========================
 * Handles communication with the FastAPI backend.
 */

// Base URL from Vite env variable, fallback to localhost
const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

/**
 * Check backend health status.
 * Returns a promise resolving to the health JSON.
 */
export async function checkHealth() {
    const response = await fetch(`${API_BASE}/health`);
    if (!response.ok) {
        throw new Error(`Health check failed: ${response.status}`);
    }
    return response.json();
}

/**
 * Send a question to the RAG backend and get a grounded answer.
 * Includes a timeout of 30 seconds.
 */
export async function askQuestion(question) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 120000); // 2 min — RAG pipeline can be slow
    try {
        const response = await fetch(`${API_BASE}/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question }),
            signal: controller.signal,
        });
        if (!response.ok) {
            const errorData = await response.json().catch(() => null);
            const message = errorData?.detail || `Server error (${response.status})`;
            throw new Error(message);
        }
        return response.json();
    } finally {
        clearTimeout(timeoutId);
    }
}
