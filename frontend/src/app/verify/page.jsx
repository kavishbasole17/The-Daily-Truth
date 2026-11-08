// src/app/verify/page.jsx
//
// "Paper Trail" — the evidence-grounded verification pipeline, as opposed to
// the single-model / consensus pages which classify on text patterns alone.
// Extracts claims from the article, checks each against trusted news
// sources and Wikipedia, and shows the full reasoning trail.
//
// Presented as a chat conversation (Claude/Gemini-style): each submitted
// article is a "user" turn, and the verdict + evidence trail comes back as
// an "assistant" turn, so multiple articles can be checked back-to-back in
// one running thread.
//
// Signed out, one check is free; follow-ups need an account (SignInGate).
// Signed in, every check is saved to a conversation, listed in the
// "Previous chats" sidebar and reopenable via /verify?c=<id>.

"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { AnimatePresence, motion } from "framer-motion";

import Navbar from "@/components/Navbar";
import ChatComposer from "@/components/ChatComposer";
import ChatHistorySidebar, { parseServerTime } from "@/components/ChatHistorySidebar";
import { useAuth } from "@/context/AuthContext";

import {
    claimGuestChecks,
    deleteConversation,
    errorMessage,
    getConversation,
    getGuestStatus,
    isGuestLimitError,
    listConversations,
    pollVerification,
    verifyArticle,
    verifyImage,
} from "@/services/api";

import "@/styles/pages.css";
import "@/styles/chat.css";

function formatTime(ts) {
    if (!ts) return "";
    return new Date(ts).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
}

const VERDICT_META = {
    likely_real: { label: "LIKELY REAL", className: "real-card" },
    likely_fake: { label: "LIKELY MISINFORMATION", className: "fake-card" },
    uncertain: { label: "UNCERTAIN", className: "uncertain-card" },
};

const CLAIM_VERDICT_META = {
    entailment: { label: "SUPPORTED", className: "badge-real", trailClass: "verdict-entailment" },
    contradiction: { label: "CONTRADICTED", className: "badge-fake", trailClass: "verdict-contradiction" },
    neutral: { label: "NO CLEAR SIGNAL", className: "badge-neutral", trailClass: "verdict-neutral" },
    // Distinct from "neutral" on purpose. Neutral means we found evidence and
    // it was inconclusive; unverifiable means we found nothing to check
    // against. Collapsing the two is what previously let "no search results"
    // masquerade as a signal about the article.
    unverifiable: { label: "COULD NOT VERIFY", className: "badge-neutral", trailClass: "verdict-unverifiable" },
};

const SUGGESTIONS = [
    "Paste a breaking-news article",
    "Check a viral social media claim",
    "Paste (Ctrl+V) a copied screenshot",
];

const STAGE_COPY = {
    queued: "Queued for verification…",
    verifying: "Extracting claims and checking them against sources…",
    processing: "Extracting claims and checking them against sources…",
};

const OCR_STAGE_COPY = {
    queued: "Queued for verification…",
    verifying: "Reading the image and checking claims against sources…",
    processing: "Reading the image and checking claims against sources…",
};

// Mirrors api/routers/articles.py's _EXT_BY_CONTENT_TYPE / OCR_MAX_IMAGE_MB —
// checked client-side too so a bad file is rejected instantly instead of
// after a round trip.
const ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"];
const MAX_IMAGE_MB = 10;

function CheckIcon() {
    return (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
    );
}

function CopyTraceButton({ text }) {
    const [copied, setCopied] = useState(false);

    const handleCopy = async (e) => {
        // Lives inside a <summary> — stop the click from also toggling the
        // <details> open/closed, which is <summary>'s native click behavior.
        e.preventDefault();
        e.stopPropagation();
        try {
            await navigator.clipboard.writeText(text);
            setCopied(true);
            setTimeout(() => setCopied(false), 1500);
        } catch {
            // Clipboard API unavailable (e.g. insecure context) — nothing
            // useful to recover into; the trace is still visible to select.
        }
    };

    return (
        <button type="button" className={`chat-copy-btn ${copied ? "copied" : ""}`} onClick={handleCopy}>
            {copied ? (
                <>
                    <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M20 6L9 17l-5-5" />
                    </svg>
                    Copied
                </>
            ) : (
                <>
                    <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <rect x="9" y="9" width="13" height="13" rx="2" />
                        <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1" />
                    </svg>
                    Copy
                </>
            )}
        </button>
    );
}

function AssistantResult({ result, extractedText }) {
    const verdict = result?.verdict;
    if (!verdict) return null;
    const meta = VERDICT_META[verdict.verdict] || VERDICT_META.uncertain;

    return (
        <>
            {extractedText && (
                <div className="chat-extracted-text">
                    <div className="chat-section-title">
                        <CheckIcon />
                        Text extracted from image
                    </div>
                    <p>&ldquo;{extractedText}&rdquo;</p>
                </div>
            )}

            <div className={`chat-verdict ${meta.className}`}>
                <div>
                    <h2>{meta.label}</h2>
                    <p>
                        {verdict.verdict === "uncertain" && !verdict.confidence
                            ? "Not enough evidence to decide"
                            : `${Math.round(verdict.confidence * 100)}% confidence`}
                    </p>
                </div>
            </div>

            {verdict.degradations?.length > 0 && (
                <div className="chat-section">
                    <div className="chat-section-title">Limitations affecting this verdict</div>
                    <ul className="evidence-list">
                        {verdict.degradations.map((d, i) => (
                            <li key={i}>{d.detail}</li>
                        ))}
                    </ul>
                </div>
            )}

            <div className="chat-section">
                <div className="chat-section-title">
                    <CheckIcon />
                    Claim-by-claim trail
                </div>

                {verdict.claim_verdicts.length === 0 ? (
                    <p className="muted">
                        No individually verifiable claims were extracted from this article.
                    </p>
                ) : (
                    <div className="claim-trail-timeline">
                        {verdict.claim_verdicts.map((cv, i) => {
                            const cvMeta = CLAIM_VERDICT_META[cv.verdict] || CLAIM_VERDICT_META.neutral;
                            return (
                                <div className="claim-trail-row" key={i}>
                                    <span className={`claim-trail-dot ${cvMeta.trailClass}`} />
                                    <span className="claim-trail-line" />
                                    <div className={`claim-item ${cvMeta.trailClass}`}>
                                        <div className="claim-index">Claim {i + 1}</div>
                                        <div className="claim-item-header">
                                            <span className={cvMeta.className}>{cvMeta.label}</span>
                                            <span className="claim-confidence">
                                                {Math.round(cv.confidence * 100)}% confidence
                                            </span>
                                        </div>
                                        <p className="claim-text">&ldquo;{cv.claim}&rdquo;</p>
                                        {cv.location && (
                                            <p className="claim-location">
                                                <span className="evidence-source">
                                                    {cv.location.level || "place"}
                                                </span>
                                                {cv.location.label || cv.location.name}
                                                {cv.search_scope && (
                                                    <span className="muted">
                                                        {" "}· searched {cv.search_scope}
                                                        {cv.regional_evidence > 0
                                                            ? ` · ${cv.regional_evidence} local source${cv.regional_evidence === 1 ? "" : "s"}`
                                                            : " · no local sources found"}
                                                    </span>
                                                )}
                                            </p>
                                        )}
                                        {cv.event_corroborated && (
                                            <p className="claim-location">
                                                <span className="evidence-source">event</span>
                                                Reported by {cv.corroborating_sources.join(", ")}
                                                {cv.detail_mismatch && (
                                                    <span className="muted"> · figures differ from the reports</span>
                                                )}
                                            </p>
                                        )}
                                        {cv.evidence_snippets.length > 0 && (
                                            <ul className="evidence-list">
                                                {cv.evidence_snippets.map((s, j) => (
                                                    <li key={j}>
                                                        {cv.sources?.[j] && (
                                                            <span className="evidence-source">{cv.sources[j]}</span>
                                                        )}
                                                        {cv.in_region?.[j] && (
                                                            <span className="evidence-source evidence-local">local</span>
                                                        )}
                                                        {s}
                                                    </li>
                                                ))}
                                            </ul>
                                        )}
                                        {cv.verdict === "unverifiable" && (
                                            <p className="muted">
                                                No source covering this claim was found. That is not
                                                evidence against it — recent, regional and non-English
                                                topics are often missed.
                                            </p>
                                        )}
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                )}
            </div>

            <details className="model-card reasoning-trace">
                <summary>
                    Full reasoning trace
                    <CopyTraceButton text={verdict.reasoning_trace} />
                </summary>
                <pre className="model-content">{verdict.reasoning_trace}</pre>
            </details>
        </>
    );
}

function UserBubble({ message, expanded, onToggleExpand }) {
    if (message.kind === "image") {
        return (
            <div className="chat-message user">
                <div className="chat-avatar">YOU</div>
                <div className="chat-message-col">
                    <div className="chat-bubble">
                        {message.previewUrl && (
                            // eslint-disable-next-line @next/next/no-img-element
                            <img className="chat-image-preview" src={message.previewUrl} alt="Submitted for OCR" />
                        )}
                        <div className="chat-image-filename">
                            <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <rect x="3" y="3" width="18" height="18" rx="2" />
                                <circle cx="8.5" cy="8.5" r="1.5" />
                                <path d="M21 15l-5-5L5 21" />
                            </svg>
                            {message.fileName}
                        </div>
                    </div>
                    <span className="chat-timestamp">{formatTime(message.timestamp)}</span>
                </div>
            </div>
        );
    }

    const text = message.text;
    const isLong = text.length > 400;
    return (
        <div className="chat-message user">
            <div className="chat-avatar">YOU</div>
            <div className="chat-message-col">
                <div className="chat-bubble">
                    <div className={`chat-article-text ${expanded ? "expanded" : ""}`}>
                        {text}
                    </div>
                    {isLong && (
                        <button type="button" className="chat-expand-btn" onClick={onToggleExpand}>
                            {expanded ? "Show less" : "Show full article"}
                        </button>
                    )}
                </div>
                <span className="chat-timestamp">{formatTime(message.timestamp)}</span>
            </div>
        </div>
    );
}

function AssistantBubble({ message }) {
    const stageCopy = message.kind === "image" ? OCR_STAGE_COPY : STAGE_COPY;
    const extractedText = message.kind === "image" ? message.result?.text_preview : null;

    return (
        <div className="chat-message assistant">
            <div className="chat-avatar">
                <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M14 3v5a1 1 0 001 1h5" />
                    <path d="M17 21H7a2 2 0 01-2-2V5a2 2 0 012-2h7l5 5v11a2 2 0 01-2 2z" />
                    <path d="M9 13h6M9 17h4" />
                </svg>
            </div>
            <div className="chat-message-col">
                <div className="chat-bubble">
                    {message.status === "loading" && (
                        <div className="chat-typing">
                            {stageCopy[message.stage] || "Working…"}
                            <span className="chat-typing-dots">
                                <span /><span /><span />
                            </span>
                        </div>
                    )}

                    {message.status === "error" && (
                        <div className="chat-error-inline">{message.error}</div>
                    )}

                    {message.status === "done" && (
                        <AssistantResult result={message.result} extractedText={extractedText} />
                    )}
                </div>
                {message.status !== "loading" && (
                    <span className="chat-timestamp">{formatTime(message.timestamp)}</span>
                )}
            </div>
        </div>
    );
}

const GUEST_THREAD_KEY = "dt_guest_thread";

function readGuestThread() {
    try {
        return JSON.parse(sessionStorage.getItem(GUEST_THREAD_KEY) || "null")?.articleIds || [];
    } catch {
        return [];
    }
}

function rememberGuestArticle(id) {
    try {
        const ids = readGuestThread();
        sessionStorage.setItem(GUEST_THREAD_KEY, JSON.stringify({ articleIds: [...ids, id] }));
    } catch {
        // storage unavailable — the check just won't carry over after sign-in
    }
}

function forgetGuestThread() {
    try {
        sessionStorage.removeItem(GUEST_THREAD_KEY);
    } catch {
        // ignore
    }
}

// A stored check (from /conversations/{id}) as the two chat bubbles it was.
function turnToMessages(turn) {
    const kind = turn.kind === "image" ? "image" : "text";
    const submitted = parseServerTime(turn.submitted_at);
    return [
        {
            id: `a${turn.id}-user`, role: "user", kind,
            text: turn.text || turn.text_preview || "",
            fileName: "Image submission",
            expanded: false, timestamp: submitted,
        },
        {
            id: `a${turn.id}-assistant`, articleId: turn.id, role: "assistant", kind,
            status: turn.status === "completed" ? "done" : turn.status === "failed" ? "error" : "loading",
            stage: turn.stage || "queued",
            result: turn,
            error: turn.status === "failed" ? "Verification failed for this check." : undefined,
            timestamp: parseServerTime(turn.verdict?.created_at) || submitted,
        },
    ];
}

function SignInGate() {
    return (
        <div className="chat-composer">
            <div className="chat-composer-inset">
                <div className="chat-gate" role="status">
                    <div className="chat-gate-text">
                        <strong>You&rsquo;ve used your free check.</strong>
                        <span>Sign in to ask follow-ups. Your first check will be saved to your chat history.</span>
                    </div>
                    <div className="chat-gate-actions">
                        <Link href="/login?next=/verify&reason=limit" className="chat-gate-primary">Sign in</Link>
                        <Link href="/login?next=/verify&mode=register&reason=limit" className="chat-gate-secondary">
                            Create account
                        </Link>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default function VerifyPage() {
    const { user, ready } = useAuth();

    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState("");
    const [sending, setSending] = useState(false);
    const [attachError, setAttachError] = useState("");
    const [showJump, setShowJump] = useState(false);

    const [conversationId, setConversationId] = useState(null);
    const [conversations, setConversations] = useState([]);
    const [historyLoading, setHistoryLoading] = useState(false);
    const [historyOpen, setHistoryOpen] = useState(false);
    const [loadingChat, setLoadingChat] = useState(false);
    const [guestRemaining, setGuestRemaining] = useState(null);

    const threadEndRef = useRef(null);
    const previewUrlsRef = useRef([]);
    const composerRef = useRef(null);
    const conversationIdRef = useRef(null);
    // Bumped whenever the visible chat changes, so a poll that finishes for a
    // chat you've navigated away from doesn't write into the new one.
    const viewTokenRef = useRef(0);

    const setActiveConversation = useCallback((id) => {
        conversationIdRef.current = id;
        setConversationId(id);
        const url = new URL(window.location.href);
        url.searchParams.delete("history");
        url.searchParams.delete("new");
        if (id) url.searchParams.set("c", String(id));
        else url.searchParams.delete("c");
        window.history.replaceState(null, "", url.pathname + url.search);
    }, []);

    const gated = ready && !user && guestRemaining === 0;

    useEffect(() => {
        threadEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
    }, [messages]);

    // Surfaces a "jump to latest" affordance once the thread has scrolled far
    // enough that the newest message (and, more importantly, a still-running
    // verification) is out of view.
    useEffect(() => {
        const onScroll = () => {
            const nearBottom =
                window.innerHeight + window.scrollY >= document.body.scrollHeight - 200;
            setShowJump(!nearBottom && messages.length > 0);
        };
        window.addEventListener("scroll", onScroll, { passive: true });
        onScroll();
        return () => window.removeEventListener("scroll", onScroll);
    }, [messages.length]);

    // Revoke every object URL created for an image preview once the page
    // unmounts (e.g. navigating away mid-conversation) — they otherwise leak
    // for the lifetime of the tab.
    useEffect(() => () => {
        previewUrlsRef.current.forEach(url => URL.revokeObjectURL(url));
    }, []);

    const updateAssistant = useCallback((id, patch) => {
        setMessages(prev => prev.map(m => (m.id === id ? { ...m, ...patch } : m)));
    }, []);

    const refreshConversations = useCallback(async () => {
        setHistoryLoading(true);
        try {
            const { data } = await listConversations();
            setConversations(data);
        } catch {
            // keep whatever list we had; the sidebar is a convenience
        } finally {
            setHistoryLoading(false);
        }
    }, []);

    // Show a stored chat, resuming the poll for any check still running.
    const showConversation = useCallback((detail) => {
        const token = ++viewTokenRef.current;
        setActiveConversation(detail.id);
        const msgs = detail.turns.flatMap(turnToMessages);
        setMessages(msgs);
        msgs.filter(m => m.role === "assistant" && m.status === "loading").forEach(m => {
            pollVerification(m.articleId, {
                onStage: stage => viewTokenRef.current === token && updateAssistant(m.id, { stage }),
            })
                .then(data => viewTokenRef.current === token && updateAssistant(m.id, { status: "done", result: data }))
                .catch(err => viewTokenRef.current === token &&
                    updateAssistant(m.id, { status: "error", error: errorMessage(err, "Failed to verify.") }));
        });
    }, [setActiveConversation, updateAssistant]);

    const openConversation = useCallback(async (id) => {
        if (sending) return;
        setHistoryOpen(false);
        setAttachError("");
        setLoadingChat(true);
        try {
            const { data } = await getConversation(id);
            showConversation(data);
        } catch (err) {
            setAttachError(errorMessage(err, "Couldn't open that chat."));
            if (err?.response?.status === 404) {
                setActiveConversation(null);
                refreshConversations();
            }
        } finally {
            setLoadingChat(false);
        }
    }, [sending, showConversation, setActiveConversation, refreshConversations]);

    const startNewConversation = useCallback(() => {
        if (sending) return;
        viewTokenRef.current++;
        setActiveConversation(null);
        setMessages([]);
        setInput("");
        setAttachError("");
        setHistoryOpen(false);
    }, [sending, setActiveConversation]);

    // Once auth is known: signed out → ask how many free checks are left;
    // signed in → load history, adopt the guest check made before signing in,
    // and honour ?c= / ?history= / ?new= from the URL.
    useEffect(() => {
        if (!ready) return;
        let cancelled = false;
        const params = new URLSearchParams(window.location.search);

        if (!user) {
            // Signed out (or just signed out): drop any saved chat from view.
            viewTokenRef.current++;
            setConversations([]);
            setActiveConversation(null);
            setMessages(prev => (prev.some(m => m.role === "user" && !m.guest) ? [] : prev));
            getGuestStatus()
                .then(({ data }) => !cancelled && setGuestRemaining(data.remaining))
                .catch(() => !cancelled && setGuestRemaining(null));
            return () => { cancelled = true; };
        }

        setGuestRemaining(null);
        if (params.get("history") === "1") setHistoryOpen(true);

        (async () => {
            const guestIds = readGuestThread();
            if (guestIds.length) {
                forgetGuestThread();
                try {
                    const { data } = await claimGuestChecks(guestIds);
                    if (!cancelled) showConversation(data);
                } catch {
                    // claimed already, or from another network — nothing to carry over
                }
            } else if (params.get("new") === "1") {
                startNewConversation();
            } else if (params.get("c")) {
                openConversation(Number(params.get("c")));
            }
            if (!cancelled) refreshConversations();
        })();
        return () => { cancelled = true; };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [ready, user?.id]);

    const jumpToLatest = () => {
        threadEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
    };

    const focusComposer = () => composerRef.current?.focus();

    const toggleExpand = (id) => {
        setMessages(prev => prev.map(m => (m.id === id ? { ...m, expanded: !m.expanded } : m)));
    };

    // Shared by text and image submissions. Returns false when the server
    // refused because the free signed-out check is already used.
    const runCheck = async ({ userMsg, kind, submit, fallbackError }) => {
        const turnId = `${Date.now()}`;
        const userId = `${turnId}-user`;
        const assistantId = `${turnId}-assistant`;
        const timestamp = Date.now();
        const signedIn = !!user;
        const token = viewTokenRef.current;

        setAttachError("");
        setMessages(prev => [
            ...prev,
            { id: userId, role: "user", kind, expanded: false, timestamp, guest: !signedIn, ...userMsg },
            { id: assistantId, role: "assistant", kind, status: "loading", stage: "queued", timestamp, guest: !signedIn },
        ]);
        setSending(true);

        try {
            const data = await submit({
                conversationId: signedIn ? conversationIdRef.current : null,
                onStage: stage => updateAssistant(assistantId, { stage }),
                onSubmitted: article => {
                    updateAssistant(assistantId, { articleId: article.id });
                    if (!signedIn) {
                        rememberGuestArticle(article.id);
                        setGuestRemaining(r => (r == null ? r : Math.max(0, r - 1)));
                    } else if (article.conversation_id && viewTokenRef.current === token) {
                        if (!conversationIdRef.current) setActiveConversation(article.conversation_id);
                        refreshConversations();
                    }
                },
            });
            updateAssistant(assistantId, { status: "done", result: data });
            if (signedIn) refreshConversations();
            return true;
        } catch (err) {
            if (isGuestLimitError(err)) {
                // Not a failed check — the free one is used. Undo the optimistic
                // turn and show the sign-in prompt instead.
                setMessages(prev => prev.filter(m => m.id !== userId && m.id !== assistantId));
                setGuestRemaining(0);
                return false;
            }
            updateAssistant(assistantId, { status: "error", error: errorMessage(err, fallbackError) });
            return true;
        } finally {
            setSending(false);
        }
    };

    const handleSend = async () => {
        const text = input.trim();
        if (!text || sending || gated) return;
        setInput("");
        const accepted = await runCheck({
            userMsg: { text },
            kind: "text",
            submit: opts => verifyArticle(text, opts),
            fallbackError: "Failed to verify article.",
        });
        if (!accepted) setInput(text);
    };

    const handleImageSelect = async (file) => {
        if (sending || gated) return;

        if (!ALLOWED_IMAGE_TYPES.includes(file.type)) {
            setAttachError("Unsupported image type. Use JPEG, PNG, WEBP, BMP or TIFF.");
            return;
        }
        if (file.size > MAX_IMAGE_MB * 1024 * 1024) {
            setAttachError(`Image exceeds the ${MAX_IMAGE_MB}MB limit.`);
            return;
        }

        const previewUrl = URL.createObjectURL(file);
        previewUrlsRef.current.push(previewUrl);

        await runCheck({
            userMsg: { previewUrl, fileName: file.name || "Pasted image" },
            kind: "image",
            submit: opts => verifyImage(file, opts),
            fallbackError: "Failed to verify image.",
        });
    };

    const handleDeleteConversation = async (id) => {
        try {
            await deleteConversation(id);
            setConversations(prev => prev.filter(c => c.id !== id));
            if (conversationIdRef.current === id) startNewConversation();
        } catch (err) {
            setAttachError(errorMessage(err, "Couldn't delete that chat."));
        }
    };

    const activeTitle = conversations.find(c => c.id === conversationId)?.title;

    return (
        <>
            <Navbar />

            <section className="chat-page has-history">
                <ChatHistorySidebar
                    user={user}
                    conversations={conversations}
                    loading={historyLoading}
                    activeId={conversationId}
                    open={historyOpen}
                    busy={sending}
                    onSelect={openConversation}
                    onNew={startNewConversation}
                    onDelete={handleDeleteConversation}
                    onClose={() => setHistoryOpen(false)}
                />

                <div className="chat-shell">
                    <div className="chat-header">
                        <div className="chat-header-title">
                            <button
                                type="button" className="chat-history-toggle"
                                onClick={() => setHistoryOpen(true)} aria-label="Show previous chats"
                            >
                                <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                                    <path d="M4 6h16M4 12h16M4 18h10" />
                                </svg>
                            </button>
                            <div className="chat-header-icon">
                                <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                    <path d="M14 3v5a1 1 0 001 1h5" />
                                    <path d="M17 21H7a2 2 0 01-2-2V5a2 2 0 012-2h7l5 5v11a2 2 0 01-2 2z" />
                                    <path d="M9 13h6M9 17h4" />
                                </svg>
                            </div>
                            <div>
                                <h1>Paper Trail</h1>
                                <p>{activeTitle || "Evidence-grounded verification, claim by claim."}</p>
                            </div>
                        </div>

                        {messages.length > 0 && (
                            <button type="button" className="chat-new-btn" onClick={startNewConversation} disabled={sending}>
                                <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                    <path d="M12 5v14M5 12h14" />
                                </svg>
                                New check
                            </button>
                        )}
                    </div>

                    {loadingChat ? (
                        <div className="chat-welcome">
                            <p className="muted">Opening chat…</p>
                        </div>
                    ) : messages.length === 0 ? (
                        <div className="chat-welcome">
                            <div className="chat-welcome-icon">
                                <svg viewBox="0 0 24 24" width="30" height="30" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                                    <path d="M14 3v5a1 1 0 001 1h5" />
                                    <path d="M17 21H7a2 2 0 01-2-2V5a2 2 0 012-2h7l5 5v11a2 2 0 01-2 2z" />
                                    <path d="M9 13h6M9 17h4" />
                                </svg>
                            </div>
                            <h2>What are we fact-checking today?</h2>
                            <p>
                                Paste an article below, or attach a photo or screenshot of one.
                                Paper Trail extracts its factual claims, checks each against
                                trusted news outlets and reference sources, and shows you the
                                receipts.
                            </p>
                            {ready && !user && guestRemaining > 0 && (
                                <p className="chat-guest-note">
                                    You can run {guestRemaining === 1 ? "one check" : `${guestRemaining} checks`} without
                                    an account. <Link href="/login?next=/verify">Sign in</Link> to ask follow-ups and
                                    keep your history.
                                </p>
                            )}
                            {!gated && (
                                <div className="chat-suggestions">
                                    {SUGGESTIONS.map(s => (
                                        <button
                                            type="button"
                                            className="chat-suggestion"
                                            key={s}
                                            onClick={focusComposer}
                                        >
                                            {s}
                                        </button>
                                    ))}
                                </div>
                            )}
                        </div>
                    ) : (
                        <div className="chat-thread">
                            <AnimatePresence initial={false}>
                                {messages.map(m => (
                                    <motion.div
                                        key={m.id}
                                        initial={{ opacity: 0, y: 12 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        transition={{ duration: 0.3, ease: "easeOut" }}
                                    >
                                        {m.role === "user" ? (
                                            <UserBubble
                                                message={m}
                                                expanded={m.expanded}
                                                onToggleExpand={() => toggleExpand(m.id)}
                                            />
                                        ) : (
                                            <AssistantBubble message={m} />
                                        )}
                                    </motion.div>
                                ))}
                            </AnimatePresence>
                            <div ref={threadEndRef} />
                        </div>
                    )}

                    {attachError && <div className="chat-error-inline chat-attach-error">{attachError}</div>}

                    {gated ? (
                        <SignInGate />
                    ) : (
                        <ChatComposer
                            ref={composerRef}
                            value={input}
                            onChange={setInput}
                            onSend={handleSend}
                            onImageSelect={handleImageSelect}
                            disabled={sending || loadingChat}
                            placeholder={
                                messages.length > 0
                                    ? "Ask a follow-up, or paste another article..."
                                    : "Paste article content, or attach a photo of one..."
                            }
                        />
                    )}
                </div>

                {showJump && (
                    <button type="button" className="chat-jump-btn" onClick={jumpToLatest}>
                        <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M12 5v14M5 12l7 7 7-7" />
                        </svg>
                        Jump to latest
                    </button>
                )}
            </section>
        </>
    );
}
