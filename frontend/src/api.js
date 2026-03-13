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
export async function askQuestion(question, sessionId = null) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 120000);
    try {
        const body = { question };
        if (sessionId) body.session_id = sessionId;
        const response = await fetch(`${API_BASE}/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
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

/**
 * Upload a PDF for session-scoped retrieval.
 * Returns { session_id, chunks, company, year, document_type }
 */
export async function uploadPdf(file, companyName, year) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('company_name', companyName);
    if (year) formData.append('year', year);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 300000); // 5 min
    try {
        const response = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData,
            signal: controller.signal,
        });
        if (!response.ok) {
            const errorData = await response.json().catch(() => null);
            throw new Error(errorData?.detail || `Upload failed (${response.status})`);
        }
        return response.json();
    } finally {
        clearTimeout(timeoutId);
    }
}
