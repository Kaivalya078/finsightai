/**
 * ChatInput Component
 * ====================
 * Bottom input area for sending messages.
 * Uses the same API call pattern as the original QuestionBox.
 */

import { useState, useRef, useEffect } from 'react';
import { Send } from 'lucide-react';

export default function ChatInput({ onSend, isLoading }) {
    const [question, setQuestion] = useState('');
    const textareaRef = useRef(null);

    // Auto-resize textarea
    useEffect(() => {
        const el = textareaRef.current;
        if (el) {
            el.style.height = 'auto';
            el.style.height = Math.min(el.scrollHeight, 160) + 'px';
        }
    }, [question]);

    const handleSubmit = (e) => {
        e.preventDefault();
        const trimmed = question.trim();
        if (!trimmed || isLoading) return;
        onSend(trimmed);
        setQuestion('');
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit(e);
        }
    };

    return (
        <form className="chat-input-form" onSubmit={handleSubmit}>
            <div className="chat-input-inner">
                <div className="chat-input-wrapper">
                    <textarea
                        ref={textareaRef}
                        value={question}
                        onChange={(e) => setQuestion(e.target.value)}
                        onKeyDown={handleKeyDown}
                        placeholder="Ask about the financial document..."
                        disabled={isLoading}
                        rows={1}
                    />
                    <button
                        type="submit"
                        className="chat-send-btn"
                        disabled={isLoading || !question.trim()}
                        title="Send message"
                    >
                        <Send size={18} />
                    </button>
                </div>
                <p className="chat-input-hint">
                    Press Enter to send, Shift+Enter for new line
                </p>
            </div>
        </form>
    );
}
