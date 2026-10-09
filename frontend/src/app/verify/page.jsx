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

"use client";

import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";

import Navbar from "@/components/Navbar";
import ChatComposer from "@/components/ChatComposer";

import { verifyArticle, verifyImage } from "@/services/api";

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

export default function VerifyPage() {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState("");
    const [sending, setSending] = useState(false);
    const [attachError, setAttachError] = useState("");
    const [showJump, setShowJump] = useState(false);
    const threadEndRef = useRef(null);
    const previewUrlsRef = useRef([]);
    const composerRef = useRef(null);

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

    const jumpToLatest = () => {
        threadEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
    };

    const focusComposer = () => composerRef.current?.focus();

    const toggleExpand = (id) => {
        setMessages(prev => prev.map(m => (m.id === id ? { ...m, expanded: !m.expanded } : m)));
    };

    const updateAssistant = (id, patch) => {
        setMessages(prev => prev.map(m => (m.id === id ? { ...m, ...patch } : m)));
    };

    const handleSend = async () => {
        const text = input.trim();
        if (!text || sending) return;

        const turnId = `${Date.now()}`;
        const userId = `${turnId}-user`;
        const assistantId = `${turnId}-assistant`;

        const timestamp = Date.now();
        setAttachError("");
        setMessages(prev => [
            ...prev,
            { id: userId, role: "user", kind: "text", text, expanded: false, timestamp },
            { id: assistantId, role: "assistant", kind: "text", status: "loading", stage: "queued", timestamp },
        ]);
        setInput("");
        setSending(true);

        try {
            const data = await verifyArticle(text, {
                onStage: stage => updateAssistant(assistantId, { stage }),
            });
            updateAssistant(assistantId, { status: "done", result: data });
        } catch (err) {
            updateAssistant(assistantId, {
                status: "error",
                error:
                    err?.response?.data?.detail ||
                    err?.response?.data?.message ||
                    err?.message ||
                    "Failed to verify article.",
            });
        } finally {
            setSending(false);
        }
    };

    const handleImageSelect = async (file) => {
        if (sending) return;

        if (!ALLOWED_IMAGE_TYPES.includes(file.type)) {
            setAttachError("Unsupported image type. Use JPEG, PNG, WEBP, BMP or TIFF.");
            return;
        }
        if (file.size > MAX_IMAGE_MB * 1024 * 1024) {
            setAttachError(`Image exceeds the ${MAX_IMAGE_MB}MB limit.`);
            return;
        }
        setAttachError("");

        const turnId = `${Date.now()}`;
        const userId = `${turnId}-user`;
        const assistantId = `${turnId}-assistant`;
        const previewUrl = URL.createObjectURL(file);
        previewUrlsRef.current.push(previewUrl);
        const timestamp = Date.now();

        setMessages(prev => [
            ...prev,
            { id: userId, role: "user", kind: "image", previewUrl, fileName: file.name || "Pasted image", timestamp },
            { id: assistantId, role: "assistant", kind: "image", status: "loading", stage: "queued", timestamp },
        ]);
        setSending(true);

        try {
            const data = await verifyImage(file, {
                onStage: stage => updateAssistant(assistantId, { stage }),
            });
            updateAssistant(assistantId, { status: "done", result: data });
        } catch (err) {
            updateAssistant(assistantId, {
                status: "error",
                error:
                    err?.response?.data?.detail ||
                    err?.response?.data?.message ||
                    err?.message ||
                    "Failed to verify image.",
            });
        } finally {
            setSending(false);
        }
    };

    const handleNewConversation = () => {
        if (sending) return;
        setMessages([]);
        setInput("");
        setAttachError("");
    };

    return (
        <>
            <Navbar />

            <section className="chat-page">
                <div className="chat-shell">
                    <div className="chat-header">
                        <div className="chat-header-title">
                            <div className="chat-header-icon">
                                <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                    <path d="M14 3v5a1 1 0 001 1h5" />
                                    <path d="M17 21H7a2 2 0 01-2-2V5a2 2 0 012-2h7l5 5v11a2 2 0 01-2 2z" />
                                    <path d="M9 13h6M9 17h4" />
                                </svg>
                            </div>
                            <div>
                                <h1>Paper Trail</h1>
                                <p>Evidence-grounded verification, claim by claim.</p>
                            </div>
                        </div>

                        {messages.length > 0 && (
                            <button type="button" className="chat-new-btn" onClick={handleNewConversation}>
                                <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                    <path d="M12 5v14M5 12h14" />
                                </svg>
                                New check
                            </button>
                        )}
                    </div>

                    {messages.length === 0 ? (
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

                    <ChatComposer
                        ref={composerRef}
                        value={input}
                        onChange={setInput}
                        onSend={handleSend}
                        onImageSelect={handleImageSelect}
                        disabled={sending}
                        placeholder="Paste article content, or attach a photo of one..."
                    />
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
