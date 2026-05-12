/**
 * Authentication Utility
 * =======================
 * JWT token management — replaces the old hardcoded-credentials module.
 */

const TOKEN_KEY = 'cognifin_token';
const USER_KEY  = 'cognifin_user';

// ── Token helpers ──────────────────────────────────────────────

export function getToken() {
    return localStorage.getItem(TOKEN_KEY);
}

export function setToken(token) {
    localStorage.setItem(TOKEN_KEY, token);
}

export function clearToken() {
    localStorage.removeItem(TOKEN_KEY);
}

// ── Session helpers ────────────────────────────────────────────

export function saveSession(user, token) {
    setToken(token);
    localStorage.setItem(USER_KEY, JSON.stringify(user));
}

export function loadSession() {
    try {
        const token = getToken();
        if (!token) return null;

        const raw = localStorage.getItem(USER_KEY);
        if (!raw) return null;

        const user = JSON.parse(raw);
        return { user, token };
    } catch {
        return null;
    }
}

export function clearSession() {
    clearToken();
    localStorage.removeItem(USER_KEY);
}
