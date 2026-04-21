/**
 * Register Page
 * ==============
 * User registration with name, email, password, confirm password.
 * Shares the same glassmorphism style as LoginPage (auth.css).
 */

import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { User, Mail, Lock, Eye, EyeOff, ArrowRight, TrendingUp, ArrowLeft } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import '../styles/auth.css';

export default function RegisterPage() {
    const [name, setName] = useState('');
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const [showPassword, setShowPassword] = useState(false);
    const [error, setError] = useState('');
    const [isLoading, setIsLoading] = useState(false);

    const { register } = useAuth();
    const navigate = useNavigate();

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');

        // Client-side validation
        if (password.length < 6) {
            setError('Password must be at least 6 characters.');
            return;
        }
        if (password !== confirmPassword) {
            setError('Passwords do not match.');
            return;
        }

        setIsLoading(true);
        try {
            await register(name, email, password);
            navigate('/login');
        } catch (err) {
            setError(err.message || 'Registration failed.');
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="auth-page">
            {/* Background Effects */}
            <div className="auth-bg-effects">
                <div className="auth-gradient-orb auth-orb-1"></div>
                <div className="auth-gradient-orb auth-orb-2"></div>
                <div className="auth-grid-pattern"></div>
            </div>

            {/* Back to Home */}
            <Link to="/" className="auth-back-link">
                <ArrowLeft size={18} />
                <span>Back to Home</span>
            </Link>

            {/* Register Card */}
            <div className="auth-card">
                {/* Logo */}
                <div className="auth-logo">
                    <div className="auth-logo-icon">
                        <TrendingUp size={24} />
                    </div>
                    <h1>Cognifin</h1>
                </div>

                {/* Heading */}
                <div className="auth-heading">
                    <h2>Create Account</h2>
                    <p>Get started with your financial analysis</p>
                </div>

                {/* Form */}
                <form onSubmit={handleSubmit} className="auth-form">
                    {/* Name Field */}
                    <div className="auth-field">
                        <label htmlFor="name">Name</label>
                        <div className={`auth-input-wrapper ${name ? 'has-value' : ''}`}>
                            <User size={18} className="auth-input-icon" />
                            <input
                                id="name"
                                type="text"
                                value={name}
                                onChange={(e) => setName(e.target.value)}
                                placeholder="Your full name"
                                required
                                autoFocus
                                autoComplete="name"
                                minLength={2}
                            />
                        </div>
                    </div>

                    {/* Email Field */}
                    <div className="auth-field">
                        <label htmlFor="email">Email</label>
                        <div className={`auth-input-wrapper ${email ? 'has-value' : ''}`}>
                            <Mail size={18} className="auth-input-icon" />
                            <input
                                id="email"
                                type="email"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                                placeholder="you@example.com"
                                required
                                autoComplete="email"
                            />
                        </div>
                    </div>

                    {/* Password Field */}
                    <div className="auth-field">
                        <label htmlFor="password">Password</label>
                        <div className={`auth-input-wrapper ${password ? 'has-value' : ''}`}>
                            <Lock size={18} className="auth-input-icon" />
                            <input
                                id="password"
                                type={showPassword ? 'text' : 'password'}
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                placeholder="Min. 6 characters"
                                required
                                minLength={6}
                                autoComplete="new-password"
                            />
                            <button
                                type="button"
                                className="auth-toggle-password"
                                onClick={() => setShowPassword(!showPassword)}
                                tabIndex={-1}
                            >
                                {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                            </button>
                        </div>
                    </div>

                    {/* Confirm Password Field */}
                    <div className="auth-field">
                        <label htmlFor="confirmPassword">Confirm Password</label>
                        <div className={`auth-input-wrapper ${confirmPassword ? 'has-value' : ''}`}>
                            <Lock size={18} className="auth-input-icon" />
                            <input
                                id="confirmPassword"
                                type={showPassword ? 'text' : 'password'}
                                value={confirmPassword}
                                onChange={(e) => setConfirmPassword(e.target.value)}
                                placeholder="Re-enter password"
                                required
                                autoComplete="new-password"
                            />
                        </div>
                    </div>

                    {/* Error Message */}
                    {error && (
                        <div className="auth-error">
                            <span>⚠</span> {error}
                        </div>
                    )}

                    {/* Submit Button */}
                    <button
                        type="submit"
                        className="auth-submit-btn"
                        disabled={isLoading || !name || !email || !password || !confirmPassword}
                    >
                        {isLoading ? (
                            <div className="auth-spinner"></div>
                        ) : (
                            <>
                                <span>Create Account</span>
                                <ArrowRight size={18} />
                            </>
                        )}
                    </button>
                </form>

                {/* Link to Login */}
                <div className="auth-hint">
                    <p>Already have an account?</p>
                    <Link to="/login" className="auth-link">Sign in →</Link>
                </div>
            </div>
        </div>
    );
}
