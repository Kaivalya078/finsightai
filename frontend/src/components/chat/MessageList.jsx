/**
 * MessageList Component
 * ======================
 * Premium bubble layout:
 *   - USER  → right-aligned gold bubble
 *   - AI    → left-aligned navy card with gold left-border
 * Word-by-word typewriter for new AI messages (ChatGPT style).
 */

import { useEffect, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { Copy, Check, ChevronDown, ChevronUp } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';

const WORD_INTERVAL_MS = 22;

function formatTime(timestamp) {
    if (!timestamp) return '';
    const d = new Date(timestamp);
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

export default function MessageList({ messages, isLoading }) {
    const bottomRef = useRef(null);

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, isLoading]);

    return (
        <div className="chat-messages">
            {messages.map((msg, index) => (
                <MessageBubble
                    key={msg.id}
                    message={msg}
                    isLatest={index === messages.length - 1}
                />
            ))}

            {/* Loading skeleton — left side, matches AI style */}
            {isLoading && (
                <div className="chat-msg chat-msg-assistant">
                    <div className="chat-msg-avatar">
                        <span>FA</span>
                    </div>
                    <div className="chat-msg-content">
                        <div className="chat-msg-header">
                            <span className="chat-msg-role">Cognifin · Analyst</span>
                        </div>
                        <div className="chat-msg-text">
                            <div className="chat-skeleton">
                                <div className="skeleton-line skeleton-line-1" />
                                <div className="skeleton-line skeleton-line-2" />
                                <div className="skeleton-line skeleton-line-3" />
                            </div>
                        </div>
                    </div>
                </div>
            )}

            <div ref={bottomRef} />
        </div>
    );
}

function MessageBubble({ message, isLatest }) {
    const isUser = message.role === 'user';
    const { user } = useAuth();
    const userName = user?.name || 'You';
    const userAvatar = user?.avatar || 'ME';
    const [copied, setCopied] = useState(false);
    const [showEvidence, setShowEvidence] = useState(false);
    const bubbleRef = useRef(null);

    // ── Typewriter state ──────────────────────────────────────
    const words = message.content.split(' ');
    const [displayedCount, setDisplayedCount] = useState(
        isLatest && !isUser ? 0 : words.length
    );
    const isStreaming = displayedCount < words.length;
    const displayedText = words.slice(0, displayedCount).join(' ');

    useEffect(() => {
        if (isUser || !isLatest || displayedCount >= words.length) return;
        const timer = setInterval(() => {
            setDisplayedCount((prev) => {
                if (prev >= words.length) { clearInterval(timer); return prev; }
                return prev + 1;
            });
        }, WORD_INTERVAL_MS);
        return () => clearInterval(timer);
    }, [isLatest, isUser, words.length]); // eslint-disable-line

    // Auto-scroll while streaming
    useEffect(() => {
        if (isStreaming) {
            bubbleRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
        }
    }, [displayedCount, isStreaming]);

    const handleCopy = () => {
        navigator.clipboard.writeText(message.content);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    const evidence = message.metadata?.evidence || [];
    const citations = message.metadata?.citations || [];

    return (
        <div
            ref={bubbleRef}
            className={`chat-msg ${isUser ? 'chat-msg-user' : 'chat-msg-assistant'}`}
        >
            {/* Avatar */}
            <div className="chat-msg-avatar">
                <span>{isUser ? userAvatar : 'FA'}</span>
            </div>

            {/* Content */}
            <div className="chat-msg-content">
                {/* Name + timestamp */}
                <div className="chat-msg-header">
                    <span className="chat-msg-role">
                        {isUser ? userName : 'Cognifin · Analyst'}
                    </span>
                    {!isStreaming && (
                        <span className="chat-msg-time">{formatTime(message.timestamp)}</span>
                    )}
                </div>

                {/* Bubble */}
                {isUser ? (
                    <div className="chat-msg-text">{message.content}</div>
                ) : (
                    <div className="chat-msg-text chat-msg-markdown">
                        <ReactMarkdown>{displayedText}</ReactMarkdown>
                        {isStreaming && <span className="stream-cursor" />}
                    </div>
                )}

                {/* Actions — only after streaming done */}
                {!isUser && !isStreaming && (
                    <div className="chat-msg-actions">
                        {citations.length > 0 && (
                            <div className="chat-citations">
                                <span className="chat-citations-label">Sources:</span>
                                {citations.map((id) => (
                                    <span key={id} className="chat-citation-badge">{id}</span>
                                ))}
                            </div>
                        )}

                        <div className="chat-msg-btns">
                            <button className="chat-action-btn" onClick={handleCopy}>
                                {copied ? <Check size={13} /> : <Copy size={13} />}
                                <span>{copied ? 'Copied' : 'Copy'}</span>
                            </button>

                            {evidence.length > 0 && (
                                <button
                                    className="chat-action-btn"
                                    onClick={() => setShowEvidence(!showEvidence)}
                                >
                                    {showEvidence ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
                                    <span>Evidence ({evidence.length})</span>
                                </button>
                            )}
                        </div>

                        {showEvidence && evidence.length > 0 && (
                            <div className="chat-evidence-panel">
                                {evidence.map((item) => (
                                    <EvidenceChunk key={item.chunk_id} item={item} />
                                ))}
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}

function EvidenceChunk({ item }) {
    const [expanded, setExpanded] = useState(false);
    return (
        <div className={`chat-evidence-item ${expanded ? 'expanded' : ''}`}>
            <button className="chat-evidence-header" onClick={() => setExpanded(!expanded)}>
                <span className="chat-evidence-id">{item.chunk_id}</span>
                <span className="chat-evidence-preview">
                    {expanded ? '' : item.snippet.slice(0, 80) + '...'}
                </span>
                {expanded ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
            </button>
            {expanded && (
                <div className="chat-evidence-body">
                    <p>{item.snippet}</p>
                </div>
            )}
        </div>
    );
}
