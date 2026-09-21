/**
 * useConversations Hook
 * ======================
 * Manages conversation history via the backend API (MongoDB).
 * Replaces the old localStorage-based persistence.
 */

import { useState, useEffect, useCallback } from 'react';
import { getConversations, getConversation, deleteConversationApi, renameConversationApi } from '../api';
import { getToken } from '../utils/auth';

export default function useConversations() {
    const [conversations, setConversations] = useState([]);
    const [activeId, setActiveId] = useState(null);
    const [activeConversation, setActiveConversation] = useState(null);
    const [isLoadingConversations, setIsLoadingConversations] = useState(false);

    // ── Load conversation list from backend on mount ──────────
    const refreshConversations = useCallback(async () => {
        const token = getToken();
        if (!token) return;           // skip if not logged in yet

        setIsLoadingConversations(true);
        try {
            const data = await getConversations(1, 50);
            setConversations(data.conversations || []);
        } catch (err) {
            console.error('Failed to load conversations:', err);
        } finally {
            setIsLoadingConversations(false);
        }
    }, []);

    useEffect(() => {
        refreshConversations();
    }, [refreshConversations]);

    // ── Select a conversation (fetches full messages) ─────────
    const selectConversation = useCallback(async (id) => {
        setActiveId(id);
        if (!id) {
            setActiveConversation(null);
            return;
        }
        try {
            const full = await getConversation(id);
            setActiveConversation({
                id: full.id,
                title: full.title,
                messages: (full.messages || []).map((m, i) => ({
                    id: `msg_${i}`,
                    role: m.role,
                    content: m.content,
                    metadata: m.metadata || {},
                    timestamp: m.timestamp || '',
                })),
                createdAt: full.created_at,
                updatedAt: full.updated_at,
            });
        } catch (err) {
            console.error('Failed to load conversation:', err);
            setActiveConversation(null);
        }
    }, []);

    // ── Start new analysis (deselect active) ─────────────────
    const createConversation = useCallback(() => {
        // Don't create in DB yet — we'll get a conversation_id back
        // from the first /chat call. For now, just go to welcome screen.
        setActiveId(null);
        setActiveConversation(null);
        return null;
    }, []);

    // ── Add message locally (after /chat returns) ────────────
    const addMessage = useCallback((convId, role, content, metadata = {}) => {
        setActiveConversation((prev) => {
            if (!prev) {
                // New conversation — create in-memory placeholder
                return {
                    id: convId || 'pending',
                    title: role === 'user' ? (content.length > 40 ? content.slice(0, 40) + '...' : content) : 'New Chat',
                    messages: [
                        {
                            id: 'msg_0',
                            role,
                            content,
                            metadata,
                            timestamp: new Date().toISOString(),
                        },
                    ],
                    createdAt: new Date().toISOString(),
                    updatedAt: new Date().toISOString(),
                };
            }
            return {
                ...prev,
                id: convId || prev.id,
                messages: [
                    ...prev.messages,
                    {
                        id: `msg_${prev.messages.length}`,
                        role,
                        content,
                        metadata,
                        timestamp: new Date().toISOString(),
                    },
                ],
                updatedAt: new Date().toISOString(),
            };
        });

        // Update activeId if we just got a real conversation_id
        if (convId && convId !== 'pending') {
            setActiveId(convId);
        }
    }, []);

    // ── Update conversation_id on active conversation ────────
    const setConversationId = useCallback((newId) => {
        setActiveId(newId);
        setActiveConversation((prev) =>
            prev ? { ...prev, id: newId } : prev
        );
        // Refresh the sidebar list
        refreshConversations();
    }, [refreshConversations]);

    // ── Delete a conversation ────────────────────────────────
    const deleteConversation = useCallback(async (id) => {
        try {
            await deleteConversationApi(id);
            setConversations((prev) => prev.filter((c) => c.id !== id));
            if (activeId === id) {
                setActiveId(null);
                setActiveConversation(null);
            }
        } catch (err) {
            console.error('Failed to delete conversation:', err);
        }
    }, [activeId]);

    // ── Rename a conversation (optimistic + persist) ─────────
    const renameConversation = useCallback(async (id, title) => {
        // Optimistic local update
        setConversations((prev) =>
            prev.map((c) => (c.id === id ? { ...c, title } : c))
        );
        setActiveConversation((prev) =>
            prev && prev.id === id ? { ...prev, title } : prev
        );
        // Persist to backend
        try {
            await renameConversationApi(id, title);
        } catch (err) {
            console.error('Failed to persist rename:', err);
        }
    }, []);

    return {
        conversations,
        activeConversation,
        activeId,
        isLoadingConversations,
        createConversation,
        addMessage,
        setConversationId,
        selectConversation,
        deleteConversation,
        renameConversation,
        refreshConversations,
    };
}
