/**
 * AuthContext
 * ===========
 * Provides authentication state and methods to the entire app.
 * Uses JWT tokens from the backend (replaces old static-credential auth).
 */

import { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { loginUser as apiLogin, registerUser as apiRegister } from '../api';
import { saveSession, loadSession, clearSession } from '../utils/auth';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
    const [user, setUser] = useState(null);
    const [isAuthenticated, setIsAuthenticated] = useState(false);
    const [isLoading, setIsLoading] = useState(true);

    // Restore session from localStorage on mount
    useEffect(() => {
        const session = loadSession();
        if (session?.user && session?.token) {
            setUser(session.user);
            setIsAuthenticated(true);
        }
        setIsLoading(false);
    }, []);

    // Login via backend POST /login
    const login = useCallback(async (email, password) => {
        const data = await apiLogin(email, password);
        // data = { token, user: { id, name, email } }
        saveSession(data.user, data.token);
        setUser(data.user);
        setIsAuthenticated(true);
        return data.user;
    }, []);

    // Register via backend POST /register
    const register = useCallback(async (name, email, password) => {
        const data = await apiRegister(name, email, password);
        // Registration returns token too, but we don't auto-login
        return data;
    }, []);

    // Logout — clear everything
    const logout = useCallback(() => {
        clearSession();
        setUser(null);
        setIsAuthenticated(false);
    }, []);

    // Login with an existing token (used by Google OAuth callback)
    const loginWithToken = useCallback((token, userData) => {
        saveSession(userData, token);
        setUser(userData);
        setIsAuthenticated(true);
    }, []);

    return (
        <AuthContext.Provider
            value={{ user, isAuthenticated, isLoading, login, register, logout, loginWithToken }}
        >
            {children}
        </AuthContext.Provider>
    );
}

export function useAuth() {
    const context = useContext(AuthContext);
    if (!context) throw new Error('useAuth must be used within an AuthProvider');
    return context;
}
