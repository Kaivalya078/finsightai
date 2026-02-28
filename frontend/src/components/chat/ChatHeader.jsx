/**
 * ChatHeader — shows model info only, no title (title lives in sidebar)
 */
import { TrendingUp, PanelLeftOpen } from 'lucide-react';

export default function ChatHeader({ onToggleSidebar, sidebarCollapsed }) {
    return (
        <header className="chat-header">
            {sidebarCollapsed && (
                <button
                    className="chat-header-menu"
                    onClick={onToggleSidebar}
                    title="Open sidebar"
                >
                    <PanelLeftOpen size={20} />
                </button>
            )}

            <div className="chat-header-info">
                <div className="chat-header-model">
                    <TrendingUp size={11} />
                    <span>GPT-4o-mini · RAG Pipeline · NIFTY 50 Corpus</span>
                </div>
            </div>

            <div className="chat-header-status">
                <span className="status-dot" />
                <span>Analyst Ready</span>
            </div>
        </header>
    );
}
