/**
 * ChatPage
 * =========
 * Main chat interface with sidebar, header, message area, and keyboard shortcuts.
 * Supports session-scoped PDF upload with a progress overlay.
 */

import { useState, useCallback, useEffect, useRef } from 'react';
import { Toaster, toast } from 'react-hot-toast';
import { askQuestion, checkHealth, uploadPdf } from '../api';
import useConversations from '../hooks/useConversations';
import Sidebar from '../components/chat/Sidebar';
import ChatHeader from '../components/chat/ChatHeader';
import MessageList from '../components/chat/MessageList';
import ChatInput from '../components/chat/ChatInput';
import { Sparkles, TrendingUp, BarChart2, Users, DollarSign } from 'lucide-react';
import '../styles/chat.css';
import '../styles/api.css';

// ── Upload progress stages (simulated while HTTP call runs) ──────
const UPLOAD_STAGES = [
    { label: 'Reading PDF pages',          pct: 8  },
    { label: 'Splitting into chunks',      pct: 30 },
    { label: 'Generating embeddings',      pct: 65 },
    { label: 'Building vector index',      pct: 90 },
    { label: 'Finalising session',         pct: 98 },
];

// ── Upload Progress Overlay component ────────────────────────────
function UploadProgressOverlay({ fileName, stageIndex, pct, done, error }) {
    return (
        <div className="upload-overlay">
            <div className="upload-overlay-card">
                <div className="upload-overlay-title">
                    {done ? '✓ Analysis ready' : 'Processing PDF'}
                </div>
                <div className="upload-overlay-file">{fileName}</div>

                {error ? (
                    <div className="upload-overlay-error">{error}</div>
                ) : (
                    <>
                        {/* Progress bar */}
                        <div className="upload-progress-bar-wrap">
                            <div
                                className="upload-progress-bar-fill"
                                style={{ width: `${pct}%` }}
                            />
                        </div>
                        <div className="upload-progress-pct">{pct}%</div>

                        {/* Stage indicators */}
                        <div className="upload-stages">
                            {UPLOAD_STAGES.map((s, i) => (
                                <div
                                    key={s.label}
                                    className={`upload-stage ${
                                        i < stageIndex ? 'done' :
                                        i === stageIndex ? 'active' : ''
                                    }`}
                                >
                                    <span className="upload-stage-dot" />
                                    <span className="upload-stage-label">{s.label}</span>
                                </div>
                            ))}
                        </div>

                        {done && (
                            <div className="upload-overlay-done">
                                Start asking questions about this PDF!
                            </div>
                        )}
                    </>
                )}
            </div>
        </div>
    );
}

export default function ChatPage() {
    const {
        conversations,
        activeConversation,
        activeId,
        createConversation,
        addMessage,
        setConversationId,
        selectConversation,
        deleteConversation,
        renameConversation,
        refreshConversations,
    } = useConversations();

    const [isLoading, setIsLoading] = useState(false);
    const [lastQuery, setLastQuery] = useState('');
    const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
    const [backendDown, setBackendDown] = useState(false);
    const [bannerDismissed, setBannerDismissed] = useState(false);

    // Per-conversation session state (convId → { sessionId, fileName })
    const [sessions, setSessions] = useState({});

    // Upload progress state
    const [uploading, setUploading] = useState(false);
    const [uploadFileName, setUploadFileName] = useState('');
    const [uploadStage, setUploadStage] = useState(0);
    const [uploadPct, setUploadPct] = useState(0);
    const [uploadDone, setUploadDone] = useState(false);
    const [uploadError, setUploadError] = useState(null);
    const stageTimer = useRef(null);

    // Active session for current chat
    const activeSession = activeId ? sessions[activeId] : null;

    // --- Health Check on Mount ---
    useEffect(() => {
        checkHealth()
            .then(() => setBackendDown(false))
            .catch(() => {
                setBackendDown(true);
                setBannerDismissed(false);
            });
    }, []);

    // --- Keyboard Shortcuts ---
    useEffect(() => {
        const handleKeyboard = (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'n') {
                e.preventDefault();
                handleNewChat();
                toast('New chat created', { icon: '💬', duration: 1500 });
            }
            if ((e.ctrlKey || e.metaKey) && e.key === 'b') {
                e.preventDefault();
                setSidebarCollapsed((prev) => !prev);
            }
            if ((e.ctrlKey || e.metaKey) && e.key === 'Backspace' && activeId) {
                e.preventDefault();
                deleteConversation(activeId);
                toast('Chat deleted', { icon: '🗑️', duration: 1500 });
            }
        };
        window.addEventListener('keydown', handleKeyboard);
        return () => window.removeEventListener('keydown', handleKeyboard);
    }, [activeId, deleteConversation]);

    // --- Send Message ---
    const handleSend = useCallback(
        async (question) => {
            // Use existing active conversation id (may be null for first message)
            const convId = activeId;
            const conversationId = activeConversation?.id && activeConversation.id !== 'pending'
                ? activeConversation.id
                : null;

            addMessage(convId, 'user', question);
            setLastQuery(question);
            setIsLoading(true);

            // Pass session_id if this chat has one
            const sessionId = sessions[convId]?.sessionId || null;

            try {
                const data = await askQuestion(question, sessionId, conversationId);

                // If backend created a new conversation, store its id
                if (data.conversation_id && data.conversation_id !== conversationId) {
                    setConversationId(data.conversation_id);
                }

                addMessage(data.conversation_id || convId, 'assistant', data.answer, {
                    citations: data.citations || [],
                    evidence: data.evidence || [],
                    pipeline: data.metadata || {},
                    follow_ups: data.follow_ups || [],
                });
            } catch (err) {
                const errorMsg = err.message || 'Something went wrong. Is the backend running?';
                addMessage(convId, 'assistant', `⚠️ Error: ${errorMsg}`, {});
                toast.error(errorMsg, { duration: 4000 });
            } finally {
                setIsLoading(false);
            }
        },
        [activeId, activeConversation, addMessage, setConversationId, sessions]
    );

    // --- Upload PDF ---
    const handleUpload = useCallback(
        async (file, companyName, year) => {
            // Make sure we have a conversation to attach the session to
            let convId = activeId;
            if (!convId) convId = createConversation();

            // Start progress overlay
            setUploadFileName(file.name);
            setUploadStage(0);
            setUploadPct(UPLOAD_STAGES[0].pct);
            setUploadDone(false);
            setUploadError(null);
            setUploading(true);

            // Simulate stage progression while HTTP call runs
            let stageIdx = 0;
            const advanceStage = () => {
                stageIdx += 1;
                if (stageIdx < UPLOAD_STAGES.length) {
                    setUploadStage(stageIdx);
                    setUploadPct(UPLOAD_STAGES[stageIdx].pct);
                    // Each stage takes progressively longer (back-loading for embeddings)
                    const delays = [3000, 6000, 10000, 5000];
                    stageTimer.current = setTimeout(advanceStage, delays[stageIdx - 1] ?? 4000);
                }
            };
            stageTimer.current = setTimeout(advanceStage, 2500);

            try {
                const result = await uploadPdf(file, companyName, year);
                clearTimeout(stageTimer.current);

                // Lock progress at 100% and show done
                setUploadStage(UPLOAD_STAGES.length);
                setUploadPct(100);
                setUploadDone(true);

                // Store session for this conversation only
                setSessions(prev => ({
                    ...prev,
                    [convId]: { sessionId: result.session_id, fileName: file.name },
                }));

                // Add a system message confirming the upload
                addMessage(
                    convId,
                    'assistant',
                    `✅ **${file.name}** has been processed — ${result.chunks} chunks indexed.\n\nYou can now ask questions about this document. It's only available in this chat.`,
                    {}
                );

                // Auto-close overlay after 2s
                setTimeout(() => setUploading(false), 2000);

            } catch (err) {
                clearTimeout(stageTimer.current);
                setUploadError(err.message || 'Upload failed. Please try again.');
                setTimeout(() => setUploading(false), 4000);
                toast.error('Upload failed: ' + (err.message || 'Unknown error'), { duration: 4000 });
            }
        },
        [activeId, createConversation, addMessage]
    );

    const handleClearUpload = () => {
        if (!activeId) return;
        setSessions(prev => {
            const next = { ...prev };
            delete next[activeId];
            return next;
        });
        toast('PDF session cleared — back to global corpus', { icon: '🗂️', duration: 2000 });
    };

    const handleNewChat = () => {
        selectConversation(null);
    };

    return (
        <div className="chat-page">
            <Toaster
                position="top-center"
                toastOptions={{
                    style: {
                        background: '#1a1a2e',
                        color: '#fff',
                        border: '1px solid rgba(255,255,255,0.08)',
                        borderRadius: '12px',
                        fontSize: '14px',
                    },
                }}
            />

            {/* Health Banner */}
            {backendDown && !bannerDismissed && (
                <div className="health-banner">
                    <span className="health-banner-icon">⚠️</span>
                    <span className="health-banner-text">
                        <strong>Backend unavailable.</strong> Make sure the FastAPI server is running at{' '}
                        <code>http://localhost:8000</code> before sending questions.
                    </span>
                    <button
                        className="health-banner-dismiss"
                        onClick={() => setBannerDismissed(true)}
                        title="Dismiss"
                    >
                        ×
                    </button>
                </div>
            )}

            {/* Upload Progress Overlay */}
            {uploading && (
                <UploadProgressOverlay
                    fileName={uploadFileName}
                    stageIndex={uploadStage}
                    pct={uploadPct}
                    done={uploadDone}
                    error={uploadError}
                />
            )}

            {/* Sidebar + Main row */}
            <div className="chat-body">
                <Sidebar
                    conversations={conversations}
                    activeId={activeId}
                    onSelect={selectConversation}
                    onNewChat={handleNewChat}
                    onDelete={deleteConversation}
                    onRename={renameConversation}
                    isCollapsed={sidebarCollapsed}
                    onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
                />

                <main className={`chat-main ${sidebarCollapsed ? 'sidebar-collapsed' : ''}`}>
                    {activeConversation && activeConversation.messages.length > 0 ? (
                        <>
                            <ChatHeader
                                onToggleSidebar={() => setSidebarCollapsed(!sidebarCollapsed)}
                                sidebarCollapsed={sidebarCollapsed}
                            />
                            <MessageList
                                messages={activeConversation.messages}
                                isLoading={isLoading}
                                lastQuery={lastQuery}
                                onFollowUp={handleSend}
                            />
                            <ChatInput
                                onSend={handleSend}
                                onUpload={handleUpload}
                                isLoading={isLoading}
                                uploadedFile={activeSession?.fileName || null}
                                onClearUpload={handleClearUpload}
                            />
                        </>
                    ) : (
                        <div className="chat-welcome">
                            <div className="chat-welcome-content">
                                <div className="chat-welcome-icon">
                                    <TrendingUp size={38} />
                                </div>
                                <h1>Cognifin</h1>
                                <p>Your AI-powered financial analyst — trained on NIFTY 50 annual reports. Ask about financials, risk factors, shareholding, and market insights.</p>

                                <div className="chat-welcome-prompts">
                                    <h3>Suggested questions</h3>
                                    <div className="chat-prompt-cards">
                                        {[
                                            { icon: <BarChart2 size={15} />, text: 'Compare TCS and Infosys revenue and profit' },
                                            { icon: <TrendingUp size={15} />, text: 'What are the key risk factors?' },
                                            { icon: <Users size={15} />, text: 'Who are the promoters and their shareholding?' },
                                            { icon: <DollarSign size={15} />, text: 'Summarise the financial highlights' },
                                        ].map(({ icon, text }) => (
                                            <button
                                                key={text}
                                                className="chat-prompt-card"
                                                onClick={() => handleSend(text)}
                                            >
                                                {icon}
                                                <span>{text}</span>
                                            </button>
                                        ))}
                                    </div>
                                </div>

                                <div className="chat-shortcuts">
                                    <span>⌨️ Shortcuts:</span>
                                    <kbd>Ctrl+N</kbd> New Chat
                                    <kbd>Ctrl+B</kbd> Toggle Sidebar
                                </div>
                            </div>
                            <ChatInput
                                onSend={handleSend}
                                onUpload={handleUpload}
                                isLoading={isLoading}
                                uploadedFile={activeSession?.fileName || null}
                                onClearUpload={handleClearUpload}
                            />
                        </div>
                    )}
                </main>
            </div>
        </div>
    );
}
