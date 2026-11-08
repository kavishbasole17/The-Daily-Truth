// src/components/ChatHistorySidebar.jsx
//
// Previous chats on the Paper Trail page. Signed in, it lists the user's
// conversations grouped by recency; signed out, it explains that history
// needs an account.

"use client";

import { useState } from "react";
import Link from "next/link";

const VERDICT_DOT = {
    likely_real: "real",
    likely_fake: "fake",
    uncertain: "uncertain",
};

// SQLite timestamps come back without a timezone but are UTC.
export function parseServerTime(s) {
    if (!s) return 0;
    const hasZone = /[zZ]|[+-]\d\d:?\d\d$/.test(s);
    return Date.parse(hasZone ? s : `${s}Z`) || 0;
}

function groupByRecency(conversations) {
    const startOfToday = new Date();
    startOfToday.setHours(0, 0, 0, 0);
    const today = startOfToday.getTime();
    const day = 86400000;
    const groups = [
        ["Today", c => c >= today],
        ["Yesterday", c => c >= today - day],
        ["Previous 7 days", c => c >= today - 7 * day],
        ["Previous 30 days", c => c >= today - 30 * day],
        ["Older", () => true],
    ];
    const out = groups.map(([label]) => ({ label, items: [] }));
    for (const conv of conversations) {
        const t = parseServerTime(conv.updated_at || conv.created_at);
        const idx = groups.findIndex(([, test]) => test(t));
        out[idx].items.push(conv);
    }
    return out.filter(g => g.items.length);
}

function ChatRow({ conv, active, onSelect, onDelete, disabled }) {
    const [confirming, setConfirming] = useState(false);
    const dot = VERDICT_DOT[conv.last_verdict];

    return (
        <li className={`chat-history-row ${active ? "active" : ""}`}>
            <button
                type="button" className="chat-history-open"
                onClick={() => onSelect(conv.id)} disabled={disabled}
                title={conv.title}
            >
                {dot && <span className={`chat-history-dot ${dot}`} aria-hidden="true" />}
                <span className="chat-history-title">{conv.title}</span>
                <span className="chat-history-count">{conv.message_count}</span>
            </button>
            {confirming ? (
                <span className="chat-history-confirm">
                    <button type="button" className="danger" onClick={() => onDelete(conv.id)}>Delete</button>
                    <button type="button" onClick={() => setConfirming(false)}>Cancel</button>
                </span>
            ) : (
                <button
                    type="button" className="chat-history-delete"
                    onClick={() => setConfirming(true)} aria-label={`Delete chat: ${conv.title}`}
                    disabled={disabled}
                >
                    <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6" />
                    </svg>
                </button>
            )}
        </li>
    );
}

export default function ChatHistorySidebar({
    user, conversations, loading, activeId, open, busy,
    onSelect, onNew, onDelete, onClose,
}) {
    const groups = groupByRecency(conversations || []);

    return (
        <>
            {open && <div className="chat-history-backdrop" onClick={onClose} aria-hidden="true" />}
            <aside className={`chat-history ${open ? "open" : ""}`} aria-label="Previous chats">
                <div className="chat-history-head">
                    <h2>Previous chats</h2>
                    <button type="button" className="chat-history-close" onClick={onClose} aria-label="Close history">
                        <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                            <path d="M18 6L6 18M6 6l12 12" />
                        </svg>
                    </button>
                </div>

                <button type="button" className="chat-history-new" onClick={onNew} disabled={busy}>
                    <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                        <path d="M12 5v14M5 12h14" />
                    </svg>
                    New check
                </button>

                <div className="chat-history-body">
                    {!user ? (
                        <div className="chat-history-empty">
                            <p>Sign in to save every check and come back to it later.</p>
                            <Link href="/login?next=/verify" className="chat-history-signin">Sign in</Link>
                        </div>
                    ) : loading && !conversations?.length ? (
                        <p className="chat-history-empty muted">Loading your chats…</p>
                    ) : groups.length === 0 ? (
                        <p className="chat-history-empty muted">
                            No chats yet. Your checks will show up here.
                        </p>
                    ) : (
                        groups.map(g => (
                            <section key={g.label} className="chat-history-group">
                                <h3>{g.label}</h3>
                                <ul>
                                    {g.items.map(conv => (
                                        <ChatRow
                                            key={conv.id} conv={conv} active={conv.id === activeId}
                                            onSelect={onSelect} onDelete={onDelete} disabled={busy}
                                        />
                                    ))}
                                </ul>
                            </section>
                        ))
                    )}
                </div>
            </aside>
        </>
    );
}
