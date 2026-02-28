/**
 * Static Authentication Utility
 * ==============================
 * Hardcoded credentials for demo purposes.
 * Replace with real auth in production.
 */

const STATIC_USERS = [
    {
        email: 'demo@cognifin.ai',
        password: 'demo123',
        name: 'Demo User',
        avatar: 'DU'
    }
];

const AUTH_KEY = 'cognifin_auth';

/**
 * Validate credentials against static users
 */
export function validateCredentials(email, password) {
    const user = STATIC_USERS.find(
        (u) => u.email.toLowerCase() === email.toLowerCase() && u.password === password
    );

    if (!user) return null;

    return {
        email: user.email,
        name: user.name,
        avatar: user.avatar,
    };
}

/**
 * Save auth session to localStorage
 */
export function saveSession(user) {
    const session = {
        isAuthenticated: true,
        user,
        timestamp: new Date().toISOString(),
    };
    localStorage.setItem(AUTH_KEY, JSON.stringify(session));
}

/**
 * Load auth session from localStorage
 */
export function loadSession() {
    try {
        const raw = localStorage.getItem(AUTH_KEY);
        if (!raw) return null;
        const session = JSON.parse(raw);
        if (session.isAuthenticated) return session;
        return null;
    } catch {
        return null;
    }
}

/**
 * Clear auth session
 */
export function clearSession() {
    localStorage.removeItem(AUTH_KEY);
}
