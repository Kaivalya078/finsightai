/**
 * useConversations Hook
 * ======================
 * Manages conversation history with localStorage persistence.
 */

import { useState, useEffect, useCallback } from 'react';

const STORAGE_KEY = 'cognifin_conversations';
const ACTIVE_KEY = 'cognifin_activeConversation';

function generateId() {
    return 'conv_' + Date.now() + '_' + Math.random().toString(36).slice(2, 8);
}

function generateTitle(question) {
    return question.length > 40 ? question.slice(0, 40) + '...' : question;
}

function loadFromStorage() {
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        return raw ? JSON.parse(raw) : [];
    } catch {
        return [];
    }
}

function saveToStorage(conversations) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
}

function loadActiveId() {
    // Always start on New Analysis screen — never resume last session
    return null;
}

function saveActiveId(id) {
    if (id) localStorage.setItem(ACTIVE_KEY, id);
    else localStorage.removeItem(ACTIVE_KEY);
}

export default function useConversations() {
    const [conversations, setConversations] = useState([]);
    const [activeId, setActiveId] = useState(null);

    // Load on mount
    useEffect(() => {
        const loaded = loadFromStorage();
        setConversations(loaded);
        const savedActive = loadActiveId();
        if (savedActive && loaded.find((c) => c.id === savedActive)) {
            setActiveId(savedActive);
        }
    }, []);

    // Persist conversations
    useEffect(() => {
        if (conversations.length > 0) {
            saveToStorage(conversations);
        }
    }, [conversations]);

    // Persist active id
    useEffect(() => {
        saveActiveId(activeId);
    }, [activeId]);

    // Get active conversation
    const activeConversation = conversations.find((c) => c.id === activeId) || null;

    // Create new conversation
    const createConversation = useCallback(() => {
        const newConv = {
            id: generateId(),
            title: 'New Chat',
            messages: [],
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString(),
        };
        setConversations((prev) => [newConv, ...prev]);
        setActiveId(newConv.id);
        return newConv.id;
    }, []);

    // Add message to active conversation
    const addMessage = useCallback((convId, role, content, metadata = {}) => {
        setConversations((prev) =>
            prev.map((c) => {
                if (c.id !== convId) return c;
                const newMsg = {
                    id: 'msg_' + Date.now(),
                    role, // 'user' or 'assistant'
                    content,
                    metadata, // { citations, evidence }
                    timestamp: new Date().toISOString(),
                };
                const updated = {
                    ...c,
                    messages: [...c.messages, newMsg],
                    updatedAt: new Date().toISOString(),
                };
                // Update title from first user message
                if (role === 'user' && c.messages.length === 0) {
                    updated.title = generateTitle(content);
                }
                return updated;
            })
        );
    }, []);

    // Select a conversation
    const selectConversation = useCallback((id) => {
        setActiveId(id);
    }, []);

    // Delete a conversation
    const deleteConversation = useCallback((id) => {
        setConversations((prev) => prev.filter((c) => c.id !== id));
        if (activeId === id) {
            setActiveId(null);
        }
    }, [activeId]);

    // Rename a conversation
    const renameConversation = useCallback((id, title) => {
        setConversations((prev) =>
            prev.map((c) => (c.id === id ? { ...c, title } : c))
        );
    }, []);

    return {
        conversations,
        activeConversation,
        activeId,
        createConversation,
        addMessage,
        selectConversation,
        deleteConversation,
        renameConversation,
    };
}
