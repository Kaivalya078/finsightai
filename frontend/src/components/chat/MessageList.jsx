/**
 * MessageList Component
 * ======================
 * Premium bubble layout:
 *   - USER  → right-aligned gold bubble
 *   - AI    → left-aligned navy card with gold left-border
 * Word-by-word typewriter for new AI messages (ChatGPT style).
 * Markdown is rendered only after the full response is received.
 *
 * Includes: MetadataBar, PipelinePanel, FollowUpChips
 */

import { useEffect, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Copy, Check, ChevronDown, ChevronUp, FileText } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import PDFViewerModal from './PDFViewerModal';
import LoadingIndicator from './LoadingIndicator';

const WORD_INTERVAL_MS = 22;

function formatTime(timestamp) {
    if (!timestamp) return '';
    const d = new Date(timestamp);
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

export default function MessageList({ messages, isLoading, lastQuery, onFollowUp }) {
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
                    onFollowUp={onFollowUp}
                />
            ))}

            {/* Loading indicator — appears as assistant message */}
            {isLoading && <LoadingIndicator query={lastQuery} />}

            <div ref={bottomRef} />
        </div>
    );
}

function MessageBubble({ message, isLatest, onFollowUp }) {
    const isUser = message.role === 'user';
    const { user } = useAuth();
    const userName = user?.name || 'You';
    const userAvatar = user?.avatar || 'ME';
    const [copied, setCopied] = useState(false);
    const [showEvidence, setShowEvidence] = useState(false);
    const [showPipeline, setShowPipeline] = useState(false);
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
    const pipeline = message.metadata?.pipeline || {};
    const followUps = message.metadata?.follow_ups || [];

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
                        {isStreaming ? (
                            /* Plain text while streaming — no markdown rendering */
                            <>
                                <span>{displayedText}</span>
                                <span className="stream-cursor" />
                            </>
                        ) : (
                            /* Full markdown rendering after streaming completes */
                            <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                {message.content}
                            </ReactMarkdown>
                        )}
                    </div>
                )}

                {/* Metadata Bar — confidence, latency, sources */}
                {!isUser && !isStreaming && pipeline.confidence != null && (
                    <MetadataBar pipeline={pipeline} />
                )}

                {/* Actions — only after streaming done */}
                {!isUser && !isStreaming && (
                    <div className="chat-msg-actions">
                        {citations.length > 0 && (
                            <div className="chat-citations">
                                <span className="chat-citations-label">Sources:</span>
                                {/* Deduplicate by document_label */}
                                {(() => {
                                    const seen = new Set();
                                    return evidence
                                        .filter(item => {
                                            const key = item.document_label || item.chunk_id;
                                            if (seen.has(key)) return false;
                                            seen.add(key);
                                            return citations.includes(item.chunk_id);
                                        })
                                        .map(item => (
                                            <span key={item.chunk_id} className="chat-citation-badge">
                                                <FileText size={11} />
                                                {item.document_label
                                                    ? `${item.document_label}${item.page_number > 0 ? ` · p.${item.page_number}` : ''}`
                                                    : item.chunk_id
                                                }
                                            </span>
                                        ));
                                })()
                                }
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

                            {pipeline.latency_breakdown && (
                                <button
                                    className="chat-action-btn"
                                    onClick={() => setShowPipeline(!showPipeline)}
                                >
                                    {showPipeline ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
                                    <span>Pipeline</span>
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

                        {showPipeline && pipeline.latency_breakdown && (
                            <PipelinePanel
                                breakdown={pipeline.latency_breakdown}
                                model={pipeline.model}
                            />
                        )}
                    </div>
                )}

                {/* Follow-up question chips */}
                {!isUser && !isStreaming && followUps.length > 0 && onFollowUp && (
                    <div className="chat-follow-ups">
                        {followUps.map((q, i) => (
                            <button
                                key={i}
                                className="follow-up-chip"
                                onClick={() => onFollowUp(q)}
                            >
                                {q}
                            </button>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}

// =============================================================================
// MetadataBar — Confidence badge, latency, sources count
// =============================================================================

function MetadataBar({ pipeline }) {
    const conf = pipeline.confidence || 0;
    const label = conf >= 0.75 ? 'high' : conf >= 0.45 ? 'medium' : 'low';
    const icon = conf >= 0.75 ? '🟢' : conf >= 0.45 ? '🟡' : '🔴';
    const latency = pipeline.latency_ms ? (pipeline.latency_ms / 1000).toFixed(1) : null;
    const sources = pipeline.sources_used || 0;

    return (
        <div className="chat-meta-bar">
            <span className={`meta-badge meta-confidence meta-${label}`}>
                {icon} {Math.round(conf * 100)}% confidence
            </span>
            {latency && (
                <span className="meta-badge meta-latency">⚡ {latency}s</span>
            )}
            <span className="meta-badge meta-sources">📄 {sources} sources</span>
            {pipeline.intent && pipeline.intent !== 'lookup' && (
                <span className="meta-badge meta-intent">{pipeline.intent}</span>
            )}
            {pipeline.cached && (
                <span className="meta-badge meta-cached">⚡ cached</span>
            )}
        </div>
    );
}

// =============================================================================
// PipelinePanel — Step-by-step latency breakdown with bar chart
// =============================================================================

const STAGE_LABELS = {
    retrieval:              { icon: '🔍', label: 'Retrieval (FAISS + BM25)' },
    reranking:              { icon: '🔄', label: 'Reranking' },
    intelligent_retrieve:   { icon: '🧠', label: 'Intelligent Retrieval' },
    context_assembly:       { icon: '📋', label: 'Context Assembly' },
};

function PipelinePanel({ breakdown, model }) {
    const stages = Object.entries(breakdown)
        .filter(([k]) => k !== 'total')
        .sort((a, b) => b[1] - a[1]); // largest first
    const total = breakdown.total || 1;

    // Generation time = total minus all tracked stages
    const trackedMs = stages.reduce((sum, [, ms]) => sum + ms, 0);
    const genMs = Math.max(0, total - trackedMs);

    return (
        <div className="chat-pipeline-panel">
            <div className="pipeline-header">RAG Pipeline Breakdown</div>

            {stages.map(([stage, ms]) => {
                const info = STAGE_LABELS[stage] || { icon: '⚙️', label: stage };
                const pct = Math.min(100, (ms / total) * 100);
                const color = ms < 100 ? 'var(--pl-fast)' : ms < 500 ? 'var(--pl-mid)' : 'var(--pl-slow)';
                return (
                    <div key={stage} className="pipeline-stage">
                        <span className="pipeline-icon">{info.icon}</span>
                        <span className="pipeline-label">{info.label}</span>
                        <div className="pipeline-bar-wrap">
                            <div
                                className="pipeline-bar"
                                style={{ width: `${pct}%`, background: color }}
                            />
                        </div>
                        <span className="pipeline-ms">{Math.round(ms)}ms</span>
                    </div>
                );
            })}

            {/* Generation stage (computed) */}
            <div className="pipeline-stage">
                <span className="pipeline-icon">🤖</span>
                <span className="pipeline-label">Generation ({model || 'LLM'})</span>
                <div className="pipeline-bar-wrap">
                    <div
                        className="pipeline-bar"
                        style={{
                            width: `${Math.min(100, (genMs / total) * 100)}%`,
                            background: 'linear-gradient(90deg, #fbbf24, #f59e0b)',
                        }}
                    />
                </div>
                <span className="pipeline-ms">{Math.round(genMs)}ms</span>
            </div>

            <div className="pipeline-total">
                Total: <strong>{(total / 1000).toFixed(2)}s</strong>
            </div>
        </div>
    );
}

// =============================================================================
// EvidenceChunk — Expandable evidence item with PDF viewer
// =============================================================================

function EvidenceChunk({ item }) {
    const [expanded, setExpanded] = useState(false);
    const [pdfOpen, setPdfOpen] = useState(false);

    // Build a human-readable label: "DocumentLabel · Page N" or fall back to chunk_id
    const label = item.document_label
        ? `${item.document_label}${item.page_number > 0 ? ` · Page ${item.page_number}` : ''}`
        : item.chunk_id;

    // Backend serves PDFs from /pdfs/<company>/<year>.pdf
    // pdf_filename may contain stale Colab paths from cached responses
    // (e.g. "content/drive/MyDrive/data/ADANIPORTS/2023.pdf")
    // so we always extract just the last 2 segments: "COMPANY/YEAR.pdf"
    const pdfUrl = (() => {
        if (!item.pdf_filename) return null;
        const segments = item.pdf_filename.replace(/\\/g, '/').split('/').filter(Boolean);
        const cleanPath = segments.length >= 2
            ? `${segments[segments.length - 2]}/${segments[segments.length - 1]}`
            : segments[segments.length - 1] || null;
        if (!cleanPath) return null;
        return `${import.meta.env.VITE_API_BASE || 'http://localhost:8000'}/pdfs/${cleanPath}`;
    })();

    return (
        <>
            <div className={`chat-evidence-item ${expanded ? 'expanded' : ''}`}>
                <button className="chat-evidence-header" onClick={() => setExpanded(!expanded)}>
                    <span className="chat-evidence-id">
                        <FileText size={12} style={{ marginRight: 4, flexShrink: 0 }} />
                        {label}
                    </span>
                    <span className="chat-evidence-preview">
                        {expanded ? '' : item.snippet.slice(0, 80) + '...'}
                    </span>
                    {expanded ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
                </button>
                {expanded && (
                    <div className="chat-evidence-body">
                        <p>{item.snippet}</p>
                        {pdfUrl && (
                            <button
                                className="chat-evidence-pdf-link"
                                onClick={() => setPdfOpen(true)}
                            >
                                <FileText size={12} />
                                Open PDF{item.page_number > 0 ? ` (Page ${item.page_number})` : ''}
                            </button>
                        )}
                    </div>
                )}
            </div>

            {pdfOpen && pdfUrl && (
                <PDFViewerModal
                    pdfUrl={pdfUrl}
                    pageNumber={item.page_number || 1}
                    chunkText={item.snippet}
                    onClose={() => setPdfOpen(false)}
                />
            )}
        </>
    );
}
