/**
 * LoadingIndicator Component
 * ===========================
 * Displays rotating contextual loading messages inside an assistant
 * message bubble while the backend processes a query.
 *
 * Features:
 * - Rotates through messages every ~1.5s with fade animation
 * - Detects company names in the query for contextual messages
 * - Styled as an assistant message to match ChatGPT-like experience
 */

import { useState, useEffect, useMemo } from 'react';

// ── Known NIFTY 50 companies for simple keyword matching ─────────
const KNOWN_COMPANIES = [
    'Reliance', 'TCS', 'Infosys', 'HDFC', 'ICICI', 'Wipro', 'HCL',
    'Bharti Airtel', 'Airtel', 'SBI', 'Bajaj', 'Kotak', 'Axis',
    'L&T', 'Larsen', 'Asian Paints', 'Maruti', 'Titan', 'Sun Pharma',
    'UltraTech', 'NTPC', 'Power Grid', 'Nestle', 'Tata Motors', 'Tata Steel',
    'Tata', 'ITC', 'Adani', 'Hindustan Unilever', 'HUL', 'IndusInd',
    'Mahindra', 'Tech Mahindra', 'Cipla', 'Grasim', 'Divis', 'BPCL',
    'Eicher', 'Hero', 'Britannia', 'JSW', 'ONGC', 'Coal India',
    'Apollo', 'Dr Reddy', 'Hindalco', 'LTIMindtree',
];

const DEFAULT_MESSAGES = [
    '🔍 Searching in corpus...',
    '📊 Retrieving financial data...',
    '📈 Analyzing companies...',
    '🧠 Generating insights...',
];

/**
 * Extract company names from query via simple case-insensitive keyword match.
 */
function detectCompanies(query) {
    if (!query) return [];
    const lower = query.toLowerCase();
    return KNOWN_COMPANIES.filter((c) => lower.includes(c.toLowerCase()));
}

/**
 * Build the message list: contextual messages for detected companies
 * first, then the default messages.
 */
function buildMessages(query) {
    const companies = detectCompanies(query);
    const contextual = [];

    for (const company of companies) {
        contextual.push(`🔍 Searching ${company} data...`);
    }

    if (companies.length >= 2) {
        contextual.push('📊 Comparing financials...');
    }

    return [...contextual, ...DEFAULT_MESSAGES];
}

export default function LoadingIndicator({ query }) {
    const messages = useMemo(() => buildMessages(query), [query]);
    const [index, setIndex] = useState(0);
    const [visible, setVisible] = useState(true);

    useEffect(() => {
        setIndex(0);
        setVisible(true);
    }, [query]);

    useEffect(() => {
        const interval = setInterval(() => {
            // Fade out → switch message → fade in
            setVisible(false);
            setTimeout(() => {
                setIndex((prev) => (prev + 1) % messages.length);
                setVisible(true);
            }, 250); // matches CSS fade-out duration
        }, 1500);

        return () => clearInterval(interval);
    }, [messages.length]);

    return (
        <div className="chat-msg chat-msg-assistant">
            <div className="chat-msg-avatar">
                <span>FA</span>
            </div>
            <div className="chat-msg-content">
                <div className="chat-msg-header">
                    <span className="chat-msg-role">Cognifin · Analyst</span>
                </div>
                <div className="chat-msg-text chat-loading-indicator">
                    <span className={`loading-message ${visible ? 'visible' : ''}`}>
                        {messages[index]}
                    </span>
                    <span className="loading-dots">
                        <span className="loading-dot" />
                        <span className="loading-dot" />
                        <span className="loading-dot" />
                    </span>
                </div>
            </div>
        </div>
    );
}
