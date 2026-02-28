/**
 * ChatPage
 * =========
 * Main chat interface with sidebar, header, message area, and keyboard shortcuts.
 * Uses the existing api.js askQuestion() to talk to the backend.
 */

import { useState, useCallback, useEffect } from 'react';
import { Toaster, toast } from 'react-hot-toast';
import { askQuestion, checkHealth } from '../api';
import useConversations from '../hooks/useConversations';
import Sidebar from '../components/chat/Sidebar';
import ChatHeader from '../components/chat/ChatHeader';
import MessageList from '../components/chat/MessageList';
import ChatInput from '../components/chat/ChatInput';
import { Sparkles, MessageSquare, TrendingUp, BarChart2, Users, DollarSign } from 'lucide-react';
import '../styles/chat.css';
import '../styles/api.css';

export default function ChatPage() {
    const {
        conversations,
        activeConversation,
        activeId,
        createConversation,
        addMessage,
        selectConversation,
        deleteConversation,
        renameConversation,
    } = useConversations();

    const [isLoading, setIsLoading] = useState(false);
    const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
    const [backendDown, setBackendDown] = useState(false);
    const [bannerDismissed, setBannerDismissed] = useState(false);

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
            // Ctrl/Cmd + N → New Chat
            if ((e.ctrlKey || e.metaKey) && e.key === 'n') {
                e.preventDefault();
                handleNewChat();
                toast('New chat created', { icon: '💬', duration: 1500 });
            }

            // Ctrl/Cmd + B → Toggle Sidebar
            if ((e.ctrlKey || e.metaKey) && e.key === 'b') {
                e.preventDefault();
                setSidebarCollapsed((prev) => !prev);
            }

            // Ctrl/Cmd + Backspace → Delete current chat
            if ((e.ctrlKey || e.metaKey) && e.key === 'Backspace' && activeId) {
                e.preventDefault();
                deleteConversation(activeId);
                toast('Chat deleted', { icon: '🗑️', duration: 1500 });
            }
        };

        window.addEventListener('keydown', handleKeyboard);
        return () => window.removeEventListener('keydown', handleKeyboard);
    }, [activeId, deleteConversation]);

    const handleSend = useCallback(
        async (question) => {
            // Create conversation if none active
            let convId = activeId;
            if (!convId) {
                convId = createConversation();
            }

            // Add user message
            addMessage(convId, 'user', question);

            // Call backend (same pattern as original App.jsx)
            setIsLoading(true);
            try {
                const data = await askQuestion(question);
                addMessage(convId, 'assistant', data.answer, {
                    citations: data.citations || [],
                    evidence: data.evidence || [],
                });
            } catch (err) {
                const errorMsg = err.message || 'Something went wrong. Is the backend running?';
                addMessage(convId, 'assistant', `⚠️ Error: ${errorMsg}`, {});
                toast.error(errorMsg, { duration: 4000 });
            } finally {
                setIsLoading(false);
            }
        },
        [activeId, createConversation, addMessage]
    );

    const handleNewChat = () => {
        selectConversation(null);   // just show the welcome screen — no conversation created yet
    };

    return (
        <div className="chat-page">
            {/* Toast Notifications */}
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

            {/* Sidebar + Main row */}
            <div className="chat-body">
                {/* Sidebar */}
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

                {/* Main Chat Area */}
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
                            />
                            <ChatInput onSend={handleSend} isLoading={isLoading} />
                        </>
                    ) : (
                        /* Empty State / Welcome */
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
                                            { icon: <BarChart2 size={15} />, text: 'What is the revenue and profit trend?' },
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

                                {/* Keyboard Shortcuts Hint */}
                                <div className="chat-shortcuts">
                                    <span>⌨️ Shortcuts:</span>
                                    <kbd>Ctrl+N</kbd> New Chat
                                    <kbd>Ctrl+B</kbd> Toggle Sidebar
                                </div>
                            </div>
                            <ChatInput onSend={handleSend} isLoading={isLoading} />
                        </div>
                    )}
                </main>
            </div>
        </div>
    );
}
