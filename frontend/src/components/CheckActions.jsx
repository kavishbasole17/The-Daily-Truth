// src/components/CheckActions.jsx
//
// The action row under a finished verdict: share a public link, and rate the
// verdict (signed-in only). A thumbs-down asks what was wrong — that reason is
// stored as a labelled example for retraining the fusion model.

"use client";

import { useState } from "react";

import { errorMessage, rateArticle, shareArticle, unshareArticle } from "@/services/api";

const REASONS = [
    ["wrong_verdict", "Wrong verdict"],
    ["bad_sources", "Bad or outdated sources"],
    ["missed_claim", "Missed a claim"],
    ["other", "Something else"],
];

function ShareButton({ articleId, initialSlug }) {
    const [slug, setSlug] = useState(initialSlug || null);
    const [state, setState] = useState("idle"); // idle | busy | copied | error
    const [msg, setMsg] = useState("");

    const link = slug && typeof window !== "undefined" ? `${window.location.origin}/v/${slug}` : "";

    const share = async () => {
        setState("busy");
        try {
            const { data } = await shareArticle(articleId);
            setSlug(data.slug);
            const url = `${window.location.origin}${data.path}`;
            try {
                if (navigator.share && window.matchMedia("(pointer: coarse)").matches) {
                    await navigator.share({ title: "Fact-check from The Daily Truth", url });
                } else {
                    await navigator.clipboard.writeText(url);
                }
                setState("copied");
            } catch {
                setState("idle"); // dismissed the share sheet / clipboard blocked; link is shown anyway
            }
        } catch (err) {
            setMsg(errorMessage(err, "Couldn't create a link."));
            setState("error");
        }
    };

    const stop = async () => {
        try {
            await unshareArticle(articleId);
            setSlug(null);
            setState("idle");
        } catch (err) {
            setMsg(errorMessage(err, "Couldn't turn sharing off."));
            setState("error");
        }
    };

    return (
        <div className="check-share">
            <button type="button" className="check-action-btn" onClick={share} disabled={state === "busy"}>
                <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M4 12v7a2 2 0 002 2h12a2 2 0 002-2v-7M16 6l-4-4-4 4M12 2v13" />
                </svg>
                {state === "copied" ? "Link copied" : slug ? "Copy link" : "Share"}
            </button>
            {slug && (
                <span className="check-share-link">
                    <a href={link} target="_blank" rel="noopener noreferrer">{link.replace(/^https?:\/\//, "")}</a>
                    <button type="button" className="check-link-btn" onClick={stop}>Stop sharing</button>
                </span>
            )}
            {state === "error" && <span className="check-action-error">{msg}</span>}
        </div>
    );
}

function FeedbackBar({ articleId, initial }) {
    const [rating, setRating] = useState(initial?.rating ?? 0);
    const [reason, setReason] = useState(initial?.reason || "");
    const [comment, setComment] = useState(initial?.comment || "");
    const [asking, setAsking] = useState(false);
    const [saved, setSaved] = useState(false);
    const [error, setError] = useState("");

    const save = async (r, why = null, note = "") => {
        setError("");
        try {
            await rateArticle(articleId, { rating: r, reason: why, comment: note || null });
            setRating(r);
            setSaved(true);
            setTimeout(() => setSaved(false), 2000);
        } catch (err) {
            setError(errorMessage(err, "Couldn't save your feedback."));
        }
    };

    return (
        <div className="check-feedback">
            <span className="check-feedback-label">{saved ? "Thanks for the feedback" : "Was this right?"}</span>
            <button
                type="button" aria-pressed={rating === 1} aria-label="Yes, this verdict looks right"
                className={`check-thumb ${rating === 1 ? "on" : ""}`}
                onClick={() => { setAsking(false); save(1); }}
            >
                <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M7 10v11H3V10h4zm0 0l4-8a3 3 0 013 3v4h5.5a2 2 0 012 2.3l-1.4 8A2 2 0 0118.1 21H7" />
                </svg>
            </button>
            <button
                type="button" aria-pressed={rating === -1} aria-label="No, something is wrong"
                className={`check-thumb down ${rating === -1 ? "on" : ""}`}
                onClick={() => setAsking(a => !a)}
            >
                <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M17 14V3h4v11h-4zm0 0l-4 8a3 3 0 01-3-3v-4H4.5a2 2 0 01-2-2.3l1.4-8A2 2 0 015.9 3H17" />
                </svg>
            </button>
            {error && <span className="check-action-error">{error}</span>}

            {asking && (
                <form
                    className="check-feedback-form"
                    onSubmit={e => {
                        e.preventDefault();
                        save(-1, reason || "other", comment.trim());
                        setAsking(false);
                    }}
                >
                    <div className="check-feedback-reasons" role="radiogroup" aria-label="What was wrong?">
                        {REASONS.map(([key, label]) => (
                            <button
                                key={key} type="button" role="radio" aria-checked={reason === key}
                                className={`check-chip ${reason === key ? "on" : ""}`}
                                onClick={() => setReason(key)}
                            >
                                {label}
                            </button>
                        ))}
                    </div>
                    <textarea
                        rows={2} maxLength={1000} placeholder="Optional: what should it have said?"
                        value={comment} onChange={e => setComment(e.target.value)}
                    />
                    <div className="check-feedback-actions">
                        <button type="button" className="check-link-btn" onClick={() => setAsking(false)}>Cancel</button>
                        <button type="submit" className="check-action-btn primary" disabled={!reason}>Send</button>
                    </div>
                </form>
            )}
        </div>
    );
}

export default function CheckActions({ result, signedIn }) {
    if (!result?.id || result.status !== "completed") return null;
    return (
        <div className="check-actions">
            <ShareButton articleId={result.id} initialSlug={result.share_slug} />
            {signedIn && <FeedbackBar articleId={result.id} initial={result.feedback} />}
        </div>
    );
}
