/**
 * FinSight AI - API Client
 * ========================
 * Handles communication with the FastAPI backend.
 * All protected requests include the JWT Authorization header.
 * On 401 responses, the token is cleared and the user is redirected to /login.
 */

import { getToken, clearSession } from './utils/auth';

// Base URL from Vite env variable, fallback to localhost
const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

// ── Helpers ──────────────────────────────────────────────────

function authHeaders() {
    const token = getToken();
    const headers = { "Content-Type": "application/json" };
    if (token) headers["Authorization"] = `Bearer ${token}`;
    return headers;
}

/**
 * Global 401 interceptor (addendum §4).
 * Call this after every fetch to protected endpoints.
 */
function handle401(response) {
    if (response.status === 401) {
        clearSession();
        window.location.href = "/login";
    }
    return response;
}

// ── Auth Endpoints (public) ─────────────────────────────────

export async function registerUser(name, email, password) {
    const response = await fetch(`${API_BASE}/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, email, password }),
    });
    if (!response.ok) {
        const err = await response.json().catch(() => null);
        throw new Error(err?.detail || `Registration failed (${response.status})`);
    }
    return response.json();
}

export async function loginUser(email, password) {
    const response = await fetch(`${API_BASE}/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
    });
    if (!response.ok) {
        const err = await response.json().catch(() => null);
        throw new Error(err?.detail || `Login failed (${response.status})`);
    }
    return response.json();
}

// ── Health ───────────────────────────────────────────────────

export async function checkHealth() {
    const response = await fetch(`${API_BASE}/health`);
    if (!response.ok) {
        throw new Error(`Health check failed: ${response.status}`);
    }
    return response.json();
}

// ── Chat (protected) ────────────────────────────────────────

export async function askQuestion(question, sessionId = null, conversationId = null) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 120000);
    try {
        const body = { question };
        if (sessionId) body.session_id = sessionId;
        if (conversationId) body.conversation_id = conversationId;

        const response = await fetch(`${API_BASE}/chat`, {
            method: "POST",
            headers: authHeaders(),
            body: JSON.stringify(body),
            signal: controller.signal,
        });
        handle401(response);
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

// ── Upload (protected) ──────────────────────────────────────

export async function uploadPdf(file, companyName, year) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('company_name', companyName);
    if (year) formData.append('year', year);

    const token = getToken();
    const headers = {};
    if (token) headers["Authorization"] = `Bearer ${token}`;

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 300000);
    try {
        const response = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            headers,
            body: formData,
            signal: controller.signal,
        });
        handle401(response);
        if (!response.ok) {
            const errorData = await response.json().catch(() => null);
            throw new Error(errorData?.detail || `Upload failed (${response.status})`);
        }
        return response.json();
    } finally {
        clearTimeout(timeoutId);
    }
}

// ── Conversations (protected) ───────────────────────────────

export async function getConversations(page = 1, limit = 50) {
    const response = await fetch(
        `${API_BASE}/conversations?page=${page}&limit=${limit}`,
        { headers: authHeaders() },
    );
    handle401(response);
    if (!response.ok) {
        const err = await response.json().catch(() => null);
        throw new Error(err?.detail || `Failed to load conversations (${response.status})`);
    }
    return response.json();
}

export async function getConversation(id) {
    const response = await fetch(`${API_BASE}/conversations/${id}`, {
        headers: authHeaders(),
    });
    handle401(response);
    if (!response.ok) {
        const err = await response.json().catch(() => null);
        throw new Error(err?.detail || `Failed to load conversation (${response.status})`);
    }
    return response.json();
}

export async function deleteConversationApi(id) {
    const response = await fetch(`${API_BASE}/conversations/${id}`, {
        method: "DELETE",
        headers: authHeaders(),
    });
    handle401(response);
    if (!response.ok) {
        const err = await response.json().catch(() => null);
        throw new Error(err?.detail || `Failed to delete conversation (${response.status})`);
    }
    return response.json();
}

export async function renameConversationApi(id, title) {
    const response = await fetch(`${API_BASE}/conversations/${id}`, {
        method: "PATCH",
        headers: authHeaders(),
        body: JSON.stringify({ title }),
    });
    handle401(response);
    if (!response.ok) {
        const err = await response.json().catch(() => null);
        throw new Error(err?.detail || `Failed to rename conversation (${response.status})`);
    }
    return response.json();
}
