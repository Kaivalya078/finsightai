/**
 * Sidebar — with right-click context menu for rename/delete
 */

import { useState, useEffect, useRef } from 'react';
import {
    Plus, MessageSquare, Trash2, LogOut, TrendingUp,
    PanelLeftClose, PanelLeftOpen, Pencil, Search, X, SquarePen,
} from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import { useNavigate } from 'react-router-dom';

export default function Sidebar({
    conversations,
    activeId,
    onSelect,
    onNewChat,
    onDelete,
    onRename,
    isCollapsed,
    onToggleCollapse,
}) {
    const { user, logout } = useAuth();
    const navigate = useNavigate();
    const [query, setQuery] = useState('');

    const handleLogout = () => { logout(); navigate('/'); };

    const today = new Date();
    const todayStr = today.toDateString();
    const yestStr = new Date(today - 86400000).toDateString();
    const grouped = { today: [], yesterday: [], older: [] };

    // Filter then group
    const filtered = query.trim()
        ? conversations.filter((c) =>
            c.title.toLowerCase().includes(query.toLowerCase())
        )
        : conversations;

    filtered.forEach((conv) => {
        const d = new Date(conv.updatedAt).toDateString();
        if (d === todayStr) grouped.today.push(conv);
        else if (d === yestStr) grouped.yesterday.push(conv);
        else grouped.older.push(conv);
    });

    const c = isCollapsed;

    return (
        <aside className={`sidebar ${c ? 'sidebar-collapsed' : ''}`}>

            {/* Header */}
            <div className="sidebar-header">
                {!c && (
                    <div className="sidebar-brand">
                        <TrendingUp size={18} className="sidebar-brand-icon" />
                        <span className="sidebar-brand-text">Cognifin</span>
                    </div>
                )}
                <button
                    className={`sidebar-toggle-btn ${c ? 'sidebar-toggle-collapsed' : ''}`}
                    onClick={onToggleCollapse}
                    title={c ? 'Expand sidebar' : 'Collapse sidebar'}
                    style={{ marginLeft: c ? 0 : 'auto' }}
                >
                    {c ? (
                        /* Cognifin monogram by default; arrow on hover via CSS */
                        <>
                            <span className="sidebar-toggle-logo">C</span>
                            <PanelLeftOpen size={19} className="sidebar-toggle-arrow" />
                        </>
                    ) : (
                        <PanelLeftClose size={19} />
                    )}
                </button>
            </div>

            {/* New Chat / New Analysis */}
            {c ? (
                <div className="sidebar-collapsed-actions">
                    <button className="sidebar-icon-btn" onClick={onNewChat} title="New Analysis">
                        <SquarePen size={19} />
                    </button>
                    <button
                        className="sidebar-icon-btn"
                        onClick={onToggleCollapse}
                        title="Search chats (open sidebar)">
                        <Search size={19} />
                    </button>
                </div>
            ) : (
                <button className="sidebar-new-chat" onClick={onNewChat}>
                    <Plus size={17} />
                    <span>New Analysis</span>
                </button>
            )}

            {/* Search */}
            {!c && (
                <div className="sidebar-search">
                    <Search size={13} className="sidebar-search-icon" />
                    <input
                        className="sidebar-search-input"
                        type="text"
                        placeholder="Search chats..."
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                    />
                    {query && (
                        <button className="sidebar-search-clear" onClick={() => setQuery('')}>
                            <X size={12} />
                        </button>
                    )}
                </div>
            )}

            {/* Conversation List */}
            {!c && (
                <div className="sidebar-conversations">
                    {filtered.length === 0 && (
                        <div className="sidebar-empty">
                            {query
                                ? <p>No chats match <strong>"{query}"</strong></p>
                                : <p>No conversations yet.<br />Start a new analysis above.</p>
                            }
                        </div>
                    )}
                    {[
                        { label: 'Today', items: grouped.today },
                        { label: 'Yesterday', items: grouped.yesterday },
                        { label: 'Previous', items: grouped.older },
                    ].map(({ label, items }) =>
                        items.length > 0 && (
                            <div className="sidebar-group" key={label}>
                                <span className="sidebar-group-label">{label}</span>
                                {items.map((conv) => (
                                    <ConversationItem
                                        key={conv.id}
                                        conv={conv}
                                        isActive={conv.id === activeId}
                                        onSelect={() => onSelect(conv.id)}
                                        onDelete={() => onDelete(conv.id)}
                                        onRename={(title) => onRename(conv.id, title)}
                                    />
                                ))}
                            </div>
                        )
                    )}
                </div>
            )}

            {c && <div className="sidebar-collapsed-spacer" />}

            {/* User / Logout */}
            {c ? (
                <button className="sidebar-icon-btn sidebar-logout-icon" onClick={handleLogout} title="Logout">
                    <LogOut size={19} />
                </button>
            ) : (
                <div className="sidebar-user">
                    <div className="sidebar-user-info">
                        <div className="sidebar-user-avatar">{user?.avatar || 'U'}</div>
                        <div className="sidebar-user-details">
                            <span className="sidebar-user-name">{user?.name || 'User'}</span>
                            <span className="sidebar-user-email">{user?.email || ''}</span>
                        </div>
                    </div>
                    <button className="sidebar-logout-btn" onClick={handleLogout} title="Logout">
                        <LogOut size={17} />
                    </button>
                </div>
            )}
        </aside>
    );
}

/* ── Conversation Item with right-click context menu ── */
function ConversationItem({ conv, isActive, onSelect, onDelete, onRename }) {
    const [menu, setMenu] = useState(null);   // {x, y}
    const [renaming, setRenaming] = useState(false);
    const [draft, setDraft] = useState(conv.title);
    const inputRef = useRef(null);
    const menuRef = useRef(null);

    // Focus input when rename starts
    useEffect(() => {
        if (renaming) setTimeout(() => inputRef.current?.select(), 0);
    }, [renaming]);

    // Close menu on outside click
    useEffect(() => {
        if (!menu) return;
        const close = (e) => {
            if (menuRef.current && !menuRef.current.contains(e.target)) setMenu(null);
        };
        document.addEventListener('mousedown', close);
        return () => document.removeEventListener('mousedown', close);
    }, [menu]);

    const handleContextMenu = (e) => {
        e.preventDefault();
        setMenu({ x: e.clientX, y: e.clientY });
    };

    const startRename = () => {
        setDraft(conv.title);
        setRenaming(true);
        setMenu(null);
    };

    const commitRename = () => {
        const t = draft.trim();
        if (t && t !== conv.title) onRename(t);
        setRenaming(false);
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter') { e.preventDefault(); commitRename(); }
        if (e.key === 'Escape') { setRenaming(false); setDraft(conv.title); }
    };

    return (
        <>
            <div
                className={`sidebar-conv-item ${isActive ? 'active' : ''}`}
                onClick={!renaming ? onSelect : undefined}
                onContextMenu={handleContextMenu}
                title="Right-click to rename or delete"
            >
                <MessageSquare size={15} className="sidebar-conv-icon" />

                {renaming ? (
                    <input
                        ref={inputRef}
                        className="sidebar-conv-rename-input"
                        value={draft}
                        onChange={(e) => setDraft(e.target.value)}
                        onKeyDown={handleKeyDown}
                        onBlur={commitRename}
                        onClick={(e) => e.stopPropagation()}
                    />
                ) : (
                    <span className="sidebar-conv-title">{conv.title}</span>
                )}

                {/* Hover delete button */}
                {!renaming && (
                    <button
                        className="sidebar-conv-delete"
                        onClick={(e) => { e.stopPropagation(); onDelete(); }}
                        title="Delete"
                    >
                        <Trash2 size={14} />
                    </button>
                )}
            </div>

            {/* Context Menu */}
            {menu && (
                <div
                    ref={menuRef}
                    className="sidebar-context-menu"
                    style={{ top: menu.y, left: menu.x }}
                >
                    <button className="sidebar-context-item" onClick={startRename}>
                        <Pencil size={13} />
                        <span>Rename</span>
                    </button>
                    <div className="sidebar-context-divider" />
                    <button
                        className="sidebar-context-item sidebar-context-danger"
                        onClick={() => { setMenu(null); onDelete(); }}
                    >
                        <Trash2 size={13} />
                        <span>Delete</span>
                    </button>
                </div>
            )}
        </>
    );
}
