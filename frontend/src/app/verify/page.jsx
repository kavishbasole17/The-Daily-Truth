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
//
// Inside a saved chat the conversation continues: short messages go to the
// assistant, which answers from every check and earlier exchange in the chat
// and can run new checks on statements you make ("but I read it was 1887").
// "Discuss" on any claim or verdict focuses the next message on that finding.
// Articles, links and images still start a new check; the hint above the
// composer says which will happen and lets you switch.

"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { AnimatePresence, motion } from "framer-motion";

import Navbar from "@/components/Navbar";
import ChatComposer from "@/components/ChatComposer";
import ChatHistorySidebar, { parseServerTime } from "@/components/ChatHistorySidebar";
import VerdictReport, { HighlightedText, VERDICT_META } from "@/components/VerdictReport";
import CheckActions from "@/components/CheckActions";
import { useAuth } from "@/context/AuthContext";

import {
    claimGuestChecks,
    deleteConversation,
    errorMessage,
    getConversation,
    getGuestStatus,
    isGuestLimitError,
    isLink,
    listConversations,
    pollChatMessage,
    pollVerification,
    sendChatMessage,
    verifyArticle,
    verifyImage,
} from "@/services/api";

import "@/styles/pages.css";
import "@/styles/chat.css";

function formatTime(ts) {
    if (!ts) return "";
    return new Date(ts).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
}

const SUGGESTIONS = [
    "Paste a breaking-news article",
    "Paste a link to a news story",
    "Paste (Ctrl+V) a copied screenshot",
];

const STAGE_COPY = {
    text: {
        queued: "Queued for verification…",
        verifying: "Extracting claims and checking them against sources…",
        processing: "Extracting claims and checking them against sources…",
    },
    image: {
        queued: "Queued for verification…",
        verifying: "Reading the image and checking claims against sources…",
        processing: "Reading the image and checking claims against sources…",
    },
    url: {
        queued: "Queued for verification…",
        fetching: "Fetching the article from the link…",
        verifying: "Checking the article's claims against sources…",
        processing: "Checking the article's claims against sources…",
    },
};

// Mirrors api/routers/articles.py's _EXT_BY_CONTENT_TYPE / OCR_MAX_IMAGE_MB —
// checked client-side too so a bad file is rejected instantly instead of
// after a round trip.
const ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"];
const MAX_IMAGE_MB = 10;

function AssistantAvatar() {
    return (
        <div className="chat-avatar">
            <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M14 3v5a1 1 0 001 1h5" />
                <path d="M17 21H7a2 2 0 01-2-2V5a2 2 0 012-2h7l5 5v11a2 2 0 01-2 2z" />
                <path d="M9 13h6M9 17h4" />
            </svg>
        </div>
    );
}

function safeHost(url) {
    try {
        return new URL(url).hostname.replace(/^www\./, "");
    } catch {
        return url;
    }
}

function UserBubble({ message, expanded, onToggleExpand, claimVerdicts, anchorPrefix }) {
    let body;
    if (message.kind === "image") {
        body = (
            <>
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
            </>
        );
    } else if (message.kind === "url") {
        body = (
            <a className="chat-link-card" href={message.text} target="_blank" rel="noopener noreferrer nofollow">
                <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M10 13a5 5 0 007.5.5l3-3a5 5 0 00-7-7l-1.7 1.7M14 11a5 5 0 00-7.5-.5l-3 3a5 5 0 007 7l1.7-1.7" />
                </svg>
                <span>
                    <strong>{message.title || safeHost(message.text)}</strong>
                    <small>{message.text}</small>
                </span>
            </a>
        );
    } else if (message.kind === "question") {
        body = (
            <>
                {message.focusLabel && <div className="chat-focus-tag">Re: {message.focusLabel}</div>}
                <div className="chat-question-text">{message.text}</div>
            </>
        );
    } else {
        const text = message.text;
        const isLong = text.length > 400;
        body = (
            <>
                <div className={`chat-article-text ${expanded ? "expanded" : ""}`}>
                    {claimVerdicts?.length ? (
                        <HighlightedText text={text} claimVerdicts={claimVerdicts} anchorPrefix={anchorPrefix} />
                    ) : text}
                </div>
                {isLong && (
                    <button type="button" className="chat-expand-btn" onClick={onToggleExpand}>
                        {expanded ? "Show less" : "Show full article"}
                    </button>
                )}
            </>
        );
    }

    return (
        <div className="chat-message user">
            <div className="chat-avatar">YOU</div>
            <div className="chat-message-col">
                <div className={`chat-bubble ${message.kind === "question" ? "question" : ""}`}>{body}</div>
                <span className="chat-timestamp">{formatTime(message.timestamp)}</span>
            </div>
        </div>
    );
}

const CHAT_STAGE_COPY = {
    thinking: "Thinking it through…",
    verifying: "Running a new check on that…",
    writing: "Writing a reply…",
};

// "[2]" in an answer becomes a link to citation 2 below it; **x** is bold.
function AnswerText({ text, citations, anchorPrefix }) {
    const known = new Set((citations || []).map(c => c.n));
    const parts = text.split(/(\[\d{1,3}\]|\*\*[^*\n]+\*\*)/g);
    return (
        <div className="chat-answer-text">
            {parts.map((p, i) => {
                const m = p.match(/^\[(\d{1,3})\]$/);
                if (m && known.has(Number(m[1]))) {
                    return (
                        <a key={i} className="cite-ref" href={`#${anchorPrefix}-cite-${m[1]}`}>{m[1]}</a>
                    );
                }
                const b = p.match(/^\*\*([^*\n]+)\*\*$/);
                if (b) return <strong key={i}>{b[1]}</strong>;
                return <span key={i}>{p}</span>;
            })}
        </div>
    );
}

// A check the assistant ran mid-conversation: compact, expandable to the full report.
function InlineCheck({ check, signedIn, onDiscuss }) {
    const [open, setOpen] = useState(false);
    const v = check.verdict;
    if (check.status !== "completed" || !v) {
        return (
            <div className="inline-check pending">
                <span className="inline-check-label">
                    {check.status === "failed" ? "Check failed" : "Checking…"}
                </span>
                <span className="inline-check-claim">&ldquo;{check.full_text || check.text_preview}&rdquo;</span>
                {check.status === "failed" && check.error && <span className="muted">{check.error}</span>}
            </div>
        );
    }
    const meta = VERDICT_META[v.verdict] || VERDICT_META.uncertain;
    return (
        <div className={`inline-check ${meta.className}`}>
            <div className="inline-check-head">
                <span className="inline-check-label">New check · {meta.label}</span>
                <span className="muted">{Math.round((v.confidence || 0) * 100)}%</span>
            </div>
            <span className="inline-check-claim">&ldquo;{check.full_text || check.text_preview}&rdquo;</span>
            <button type="button" className="check-link-btn" onClick={() => setOpen(o => !o)}>
                {open ? "Hide full report" : "Show full report"}
            </button>
            {open && (
                <div className="inline-check-report">
                    <VerdictReport
                        result={check}
                        anchorPrefix={`c${check.id}`}
                        onDiscuss={onDiscuss ? idx => onDiscuss(check, idx) : undefined}
                    />
                    <CheckActions result={check} signedIn={signedIn} />
                </div>
            )}
        </div>
    );
}

function AnswerView({ answer, anchorPrefix, signedIn, onDiscuss }) {
    return (
        <>
            {answer.checks?.length > 0 && (
                <div className="inline-checks">
                    {answer.checks.map(c => (
                        <InlineCheck key={c.id} check={c} signedIn={signedIn} onDiscuss={onDiscuss} />
                    ))}
                </div>
            )}
            <AnswerText text={answer.answer} citations={answer.citations} anchorPrefix={anchorPrefix} />
            {answer.citations?.length > 0 && (
                <ol className="chat-citations">
                    {answer.citations.map(c => (
                        <li key={c.n} id={`${anchorPrefix}-cite-${c.n}`}>
                            <span className="cite-num">{c.n}</span>
                            <div>
                                {c.url ? (
                                    <a className="evidence-source evidence-link" href={c.url} target="_blank" rel="noopener noreferrer nofollow">
                                        {c.source || safeHost(c.url)}
                                    </a>
                                ) : (
                                    c.source && <span className="evidence-source">{c.source}</span>
                                )}
                                <span>{c.snippet}</span>
                            </div>
                        </li>
                    ))}
                </ol>
            )}
            {(answer.mode === "extractive" || answer.mode === "heuristic") && (
                <p className="muted chat-answer-note">
                    Basic reply mode: no AI model is configured, so replies are assembled from this chat&rsquo;s evidence.
                </p>
            )}
        </>
    );
}

function AssistantBubble({ message, signedIn, onDiscuss }) {
    const stageCopy = STAGE_COPY[message.kind] || STAGE_COPY.text;
    const anchorPrefix = `m${message.id}`;
    const isAnswer = message.kind === "answer";

    return (
        <div className="chat-message assistant">
            <AssistantAvatar />
            <div className="chat-message-col">
                <div className="chat-bubble">
                    {message.status === "loading" && (
                        <>
                            {isAnswer && message.answer?.checks?.length > 0 && (
                                <div className="inline-checks">
                                    {message.answer.checks.map(c => (
                                        <InlineCheck key={c.id} check={c} signedIn={signedIn} />
                                    ))}
                                </div>
                            )}
                            <div className="chat-typing">
                                {isAnswer
                                    ? CHAT_STAGE_COPY[message.stage] || CHAT_STAGE_COPY.thinking
                                    : stageCopy[message.stage] || "Working…"}
                                <span className="chat-typing-dots">
                                    <span /><span /><span />
                                </span>
                            </div>
                        </>
                    )}

                    {message.status === "error" && (
                        <div className="chat-error-inline">{message.error}</div>
                    )}

                    {message.status === "done" && isAnswer && (
                        <AnswerView answer={message.answer} anchorPrefix={anchorPrefix} signedIn={signedIn} onDiscuss={onDiscuss} />
                    )}

                    {message.status === "done" && !isAnswer && (
                        <>
                            <VerdictReport
                                result={message.result}
                                anchorPrefix={anchorPrefix}
                                showText={message.kind === "image" || message.kind === "url"}
                                textLabel={
                                    message.kind === "image" ? "Text extracted from image"
                                        : message.result?.source_title
                                            ? `Article: ${message.result.source_title}`
                                            : "Article text from the link"
                                }
                                onDiscuss={onDiscuss ? idx => onDiscuss(message.result, idx) : undefined}
                            />
                            <CheckActions result={message.result} signedIn={signedIn} />
                        </>
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
    const kind = turn.kind === "image" || turn.kind === "url" ? turn.kind : "text";
    const submitted = parseServerTime(turn.submitted_at);
    return [
        {
            id: `a${turn.id}-user`, role: "user", kind,
            text: kind === "url" ? turn.url || "" : turn.text || turn.text_preview || "",
            title: turn.source_title || null,
            fileName: "Image submission",
            expanded: false, timestamp: submitted,
        },
        {
            id: `a${turn.id}-assistant`, articleId: turn.id, role: "assistant", kind,
            status: turn.status === "completed" ? "done" : turn.status === "failed" ? "error" : "loading",
            stage: turn.stage || "queued",
            result: turn,
            error: turn.status === "failed" ? turn.error || "Verification failed for this check." : undefined,
            timestamp: parseServerTime(turn.verdict?.created_at) || submitted,
        },
    ];
}

export function focusLabel(result, claimIndex) {
    const v = result?.verdict;
    if (!v) return "this check";
    if (claimIndex == null) {
        return `the verdict (${(VERDICT_META[v.verdict] || VERDICT_META.uncertain).label.toLowerCase()})`;
    }
    const claim = v.claim_verdicts?.[claimIndex]?.claim || "";
    const short = claim.length > 70 ? `${claim.slice(0, 67)}…` : claim;
    return `claim ${claimIndex + 1}, “${short}”`;
}

function answerFromServer(q) {
    return { answer: q.answer, citations: q.citations, mode: q.mode, checks: q.checks || [] };
}

// A stored chat message as its two bubbles.
function questionToMessages(q, resultsById) {
    const t = parseServerTime(q.created_at);
    const status = q.status === "done" || !q.status ? "done" : q.status === "failed" ? "error" : "loading";
    return [
        {
            id: `q${q.id}-user`, role: "user", kind: "question", text: q.question, timestamp: t,
            focusLabel: q.focus ? focusLabel(resultsById[q.focus.article_id], q.focus.claim_index) : null,
        },
        {
            id: `q${q.id}-assistant`, chatMessageId: q.id, role: "assistant", kind: "answer",
            status, stage: q.stage, error: q.error || "Couldn't reply. Please try again.",
            answer: answerFromServer(q), timestamp: t,
        },
    ];
}

function detailToMessages(detail) {
    const resultsById = {};
    detail.turns.forEach(t => { resultsById[t.id] = t; });
    (detail.questions || []).forEach(q => (q.checks || []).forEach(c => { resultsById[c.id] = c; }));
    const pairs = [
        ...detail.turns.map(turnToMessages),
        ...(detail.questions || []).map(q => questionToMessages(q, resultsById)),
    ];
    pairs.sort((a, b) => a[0].timestamp - b[0].timestamp);
    return pairs.flat();
}

// Long text, links and images start a new check; short messages talk to the assistant.
function routeFor(text, { canChat, focus, forced }) {
    if (!canChat) return "check";
    if (forced) return forced;
    if (focus) return "chat";
    if (isLink(text)) return "check";
    const sentences = (text.match(/[.!?।](\s|$)/g) || []).length;
    if (text.length > 280 || sentences >= 3) return "check";
    return "chat";
}

const FOLLOW_UPS = [
    "What's your verdict, and why?",
    "Which source is the most reliable here?",
    "What would change this verdict?",
];

function ComposerHelper({ focus, onClearFocus, route, hasInput, onFlipRoute, showFollowUps, onFollowUp }) {
    return (
        <div className="composer-helper">
            {focus && (
                <div className="chat-focus-chip">
                    <span>Discussing {focus.label}</span>
                    <button type="button" onClick={onClearFocus} aria-label="Stop discussing this finding">×</button>
                </div>
            )}
            {!focus && showFollowUps && !hasInput && (
                <div className="chat-follow-ups">
                    {FOLLOW_UPS.map(f => (
                        <button key={f} type="button" className="chat-suggestion" onClick={() => onFollowUp(f)}>{f}</button>
                    ))}
                </div>
            )}
            {route && hasInput && (
                <div className="chat-route-hint">
                    {route === "chat" ? "Replying in this chat" : "Running a new check"}
                    <button type="button" className="check-link-btn" onClick={onFlipRoute}>
                        {route === "chat" ? "check it as a new article instead" : "reply in chat instead"}
                    </button>
                </div>
            )}
        </div>
    );
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
    // The finding the next message is about: {article_id, claim_index, label}
    const [focus, setFocus] = useState(null);
    // Manual override of automatic routing for the current draft: "chat" | "check" | null
    const [forcedRoute, setForcedRoute] = useState(null);

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
        const msgs = detailToMessages(detail);
        setMessages(msgs);
        setFocus(null);
        // Resume replies still being written (e.g. after a refresh).
        msgs.filter(m => m.kind === "answer" && m.status === "loading").forEach(m => {
            pollChatMessage(detail.id, m.chatMessageId, {
                onUpdate: d => viewTokenRef.current === token &&
                    updateAssistant(m.id, { stage: d.stage, answer: answerFromServer(d) }),
            })
                .then(d => viewTokenRef.current === token &&
                    updateAssistant(m.id, { status: "done", answer: answerFromServer(d) }))
                .catch(err => viewTokenRef.current === token &&
                    updateAssistant(m.id, { status: "error", error: errorMessage(err, "Couldn't reply.") }));
        });
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
        setFocus(null);
        setForcedRoute(null);
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
            if (kind === "url" && data.source_title) updateAssistant(userId, { title: data.source_title });
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

    const handleChat = async (text) => {
        const convId = conversationIdRef.current;
        if (!convId) return;
        const sentFocus = focus;
        const turnId = `${Date.now()}`;
        const userId = `${turnId}-user`;
        const assistantId = `${turnId}-assistant`;
        const token = viewTokenRef.current;
        const timestamp = Date.now();
        setAttachError("");
        setFocus(null);
        setMessages(prev => [
            ...prev,
            { id: userId, role: "user", kind: "question", text, timestamp, focusLabel: sentFocus?.label || null },
            { id: assistantId, role: "assistant", kind: "answer", status: "loading", stage: "thinking", timestamp },
        ]);
        setSending(true);
        try {
            const { data } = await sendChatMessage(
                convId, text,
                sentFocus ? { article_id: sentFocus.article_id, claim_index: sentFocus.claim_index } : null
            );
            const done = await pollChatMessage(convId, data.id, {
                onUpdate: d => viewTokenRef.current === token &&
                    updateAssistant(assistantId, { stage: d.stage, chatMessageId: d.id, answer: answerFromServer(d) }),
            });
            if (viewTokenRef.current === token) {
                updateAssistant(assistantId, { status: "done", answer: answerFromServer(done), timestamp: Date.now() });
            }
            refreshConversations();
            return true;
        } catch (err) {
            if (err?.response?.status === 409 || err?.response?.status === 422) {
                // Not sent: put the draft back.
                setMessages(prev => prev.filter(m => m.id !== userId && m.id !== assistantId));
                setAttachError(errorMessage(err, "Couldn't send that."));
                setFocus(sentFocus);
                return false;
            }
            updateAssistant(assistantId, {
                status: "error",
                error: errorMessage(err, "Couldn't reply. Please try again."),
            });
            return true;
        } finally {
            setSending(false);
        }
    };

    const handleSend = async () => {
        const text = input.trim();
        if (!text || sending || gated) return;
        const route = routeFor(text, { canChat, focus, forced: forcedRoute });
        setInput("");
        setForcedRoute(null);
        if (route === "chat") {
            if (!(await handleChat(text))) setInput(text);
            return;
        }
        setFocus(null);
        const link = isLink(text);
        const accepted = await runCheck({
            userMsg: { text },
            kind: link ? "url" : "text",
            submit: opts => verifyArticle(text, opts),
            fallbackError: link ? "Failed to check that link." : "Failed to verify article.",
        });
        if (!accepted) setInput(text);
    };

    // "Discuss" on a claim (index) or a whole verdict (null).
    const handleDiscuss = (result, claimIndex) => {
        if (!result?.id) return;
        setFocus({ article_id: result.id, claim_index: claimIndex, label: focusLabel(result, claimIndex) });
        setForcedRoute(null);
        composerRef.current?.focus();
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
    // Talking to the assistant needs a saved chat with at least one finished check.
    const canChat = !!user && !!conversationId &&
        messages.some(m => m.role === "assistant" && m.kind !== "answer" && m.status === "done");
    const draft = input.trim();
    const route = draft ? routeFor(draft, { canChat, focus, forced: forcedRoute }) : null;
    const lastMsg = messages[messages.length - 1];
    const showFollowUps = canChat && lastMsg?.role === "assistant" && lastMsg.status === "done";

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
                                {messages.map((m, idx) => {
                                    const reply = m.role === "user" ? messages[idx + 1] : null;
                                    return (
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
                                                    claimVerdicts={reply?.result?.verdict?.claim_verdicts}
                                                    anchorPrefix={reply ? `m${reply.id}` : undefined}
                                                />
                                            ) : (
                                                <AssistantBubble
                                                    message={m}
                                                    signedIn={!!user}
                                                    onDiscuss={canChat ? handleDiscuss : undefined}
                                                />
                                            )}
                                        </motion.div>
                                    );
                                })}
                            </AnimatePresence>
                            <div ref={threadEndRef} />
                        </div>
                    )}

                    {attachError && <div className="chat-error-inline chat-attach-error">{attachError}</div>}

                    {gated ? (
                        <SignInGate />
                    ) : (
                        <ChatComposer
                            topSlot={
                                canChat && (
                                    <ComposerHelper
                                        focus={focus}
                                        onClearFocus={() => setFocus(null)}
                                        route={route}
                                        hasInput={!!draft}
                                        onFlipRoute={() => setForcedRoute(route === "chat" ? "check" : "chat")}
                                        showFollowUps={showFollowUps}
                                        onFollowUp={f => { setInput(f); composerRef.current?.focus(); }}
                                    />
                                )
                            }
                            ref={composerRef}
                            value={input}
                            onChange={setInput}
                            onSend={handleSend}
                            onImageSelect={handleImageSelect}
                            disabled={sending || loadingChat}
                            placeholder={
                                focus
                                    ? "Ask about it, push back, or ask for my verdict…"
                                    : canChat
                                        ? "Ask about these findings, push back on one, or paste something new to check…"
                                        : messages.length > 0
                                            ? "Paste another article or link to check..."
                                            : "Paste an article or a link, or attach a photo of one..."
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
