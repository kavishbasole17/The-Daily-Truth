// src/components/VerdictReport.jsx
//
// The full result of one Paper Trail check: verdict, limitations, the
// claim-by-claim trail with linked sources, the checked text with each claim
// highlighted, and the raw reasoning trace. Used in the chat and on the public
// share page (/v/<slug>) so both always look the same.

"use client";

import { useState } from "react";

import { highlightSegments } from "@/lib/highlight";

export const VERDICT_META = {
    likely_real: { label: "LIKELY REAL", className: "real-card" },
    likely_fake: { label: "LIKELY MISINFORMATION", className: "fake-card" },
    uncertain: { label: "UNCERTAIN", className: "uncertain-card" },
};

export const CLAIM_VERDICT_META = {
    entailment: { label: "SUPPORTED", className: "badge-real", trailClass: "verdict-entailment" },
    contradiction: { label: "CONTRADICTED", className: "badge-fake", trailClass: "verdict-contradiction" },
    neutral: { label: "NO CLEAR SIGNAL", className: "badge-neutral", trailClass: "verdict-neutral" },
    // Distinct from "neutral" on purpose. Neutral means we found evidence and
    // it was inconclusive; unverifiable means we found nothing to check
    // against. Collapsing the two is what previously let "no search results"
    // masquerade as a signal about the article.
    unverifiable: { label: "COULD NOT VERIFY", className: "badge-neutral", trailClass: "verdict-unverifiable" },
};

export function CheckIcon() {
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

function formatPublished(raw) {
    if (!raw) return "";
    const t = Date.parse(raw);
    if (Number.isNaN(t)) return "";
    return new Date(t).toLocaleDateString([], { day: "numeric", month: "short", year: "numeric" });
}

function trustTier(trust) {
    if (trust == null) return null;
    if (trust >= 0.75) return { key: "high", label: "Trusted outlet" };
    if (trust >= 0.5) return { key: "mid", label: "Established source" };
    return { key: "low", label: "Lower-trust source" };
}

function SourceLine({ cv, j }) {
    const source = cv.sources?.[j];
    const url = cv.urls?.[j];
    const date = formatPublished(cv.published_at?.[j]);
    const tier = trustTier(cv.trust?.[j]);
    if (!source && !url) return null;
    let label = source;
    if (!label && url) {
        try { label = new URL(url).hostname.replace(/^www\./, ""); } catch { label = url; }
    }
    return (
        <span className="evidence-meta">
            {tier && <span className={`trust-dot trust-${tier.key}`} title={tier.label} aria-label={tier.label} />}
            {url ? (
                <a className="evidence-source evidence-link" href={url} target="_blank" rel="noopener noreferrer nofollow">
                    {label}
                    <svg viewBox="0 0 24 24" width="10" height="10" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                        <path d="M7 17L17 7M9 7h8v8" />
                    </svg>
                </a>
            ) : (
                <span className="evidence-source">{label}</span>
            )}
            {date && <span className="evidence-date">{date}</span>}
            {cv.in_region?.[j] && <span className="evidence-source evidence-local">local</span>}
        </span>
    );
}

/** The checked text with every located claim underlined in its verdict colour. */
export function HighlightedText({ text, claimVerdicts, anchorPrefix, className = "" }) {
    const segments = highlightSegments(text, (claimVerdicts || []).map(cv => cv.claim));
    const jump = (i) => {
        if (!anchorPrefix) return;
        document.getElementById(`${anchorPrefix}-claim-${i}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
    };
    return (
        <div className={`highlighted-text ${className}`}>
            {segments.map((seg, k) => {
                if (seg.claimIndex == null) return <span key={k}>{seg.text}</span>;
                const cv = claimVerdicts[seg.claimIndex];
                const meta = CLAIM_VERDICT_META[cv.verdict] || CLAIM_VERDICT_META.neutral;
                return (
                    <mark
                        key={k}
                        className={`claim-mark ${meta.trailClass}`}
                        title={`Claim ${seg.claimIndex + 1}: ${meta.label.toLowerCase()}`}
                        onClick={() => jump(seg.claimIndex)}
                        role={anchorPrefix ? "button" : undefined}
                        tabIndex={anchorPrefix ? 0 : undefined}
                        onKeyDown={e => { if (e.key === "Enter") jump(seg.claimIndex); }}
                    >
                        {seg.text}
                        <sup>{seg.claimIndex + 1}</sup>
                    </mark>
                );
            })}
        </div>
    );
}

export function ClaimLegend() {
    return (
        <div className="claim-legend" aria-hidden="true">
            <span><i className="verdict-entailment" /> supported</span>
            <span><i className="verdict-contradiction" /> contradicted</span>
            <span><i className="verdict-neutral" /> unclear</span>
        </div>
    );
}

/**
 * @param result        ArticleOut (or a shared check) with .verdict
 * @param anchorPrefix  unique per report, for claim jump links
 * @param showText      render the checked text (for link/image checks, whose
 *                      text isn't in the user's bubble)
 */
function DiscussButton({ onClick, label = "Discuss" }) {
    return (
        <button type="button" className="discuss-btn" onClick={onClick}>
            <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <path d="M21 12a8 8 0 01-11.6 7.1L4 20.5l1.4-4.9A8 8 0 1121 12z" />
            </svg>
            {label}
        </button>
    );
}

/**
 * `onDiscuss(claimIndex | null)` — when given (signed-in chat), each claim and
 * the overall verdict get a "Discuss" button that focuses the next message on
 * that finding.
 */
export default function VerdictReport({ result, anchorPrefix, showText = false, textLabel, onDiscuss }) {
    const verdict = result?.verdict;
    if (!verdict) return null;
    const meta = VERDICT_META[verdict.verdict] || VERDICT_META.uncertain;
    const fullText = result.full_text || result.text || "";

    return (
        <>
            {showText && fullText && (
                <details className="chat-extracted-text" open={fullText.length < 700}>
                    <summary className="chat-section-title">
                        <CheckIcon />
                        {textLabel || "Text checked"}
                        {verdict.claim_verdicts.length > 0 && <span className="muted"> · claims highlighted</span>}
                    </summary>
                    <HighlightedText text={fullText} claimVerdicts={verdict.claim_verdicts} anchorPrefix={anchorPrefix} />
                    {verdict.claim_verdicts.length > 0 && <ClaimLegend />}
                </details>
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
                {onDiscuss && <DiscussButton label="Discuss verdict" onClick={() => onDiscuss(null)} />}
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
                                <div className="claim-trail-row" key={i} id={anchorPrefix ? `${anchorPrefix}-claim-${i}` : undefined}>
                                    <span className={`claim-trail-dot ${cvMeta.trailClass}`} />
                                    <span className="claim-trail-line" />
                                    <div className={`claim-item ${cvMeta.trailClass}`}>
                                        <div className="claim-index">Claim {i + 1}</div>
                                        <div className="claim-item-header">
                                            <span className={cvMeta.className}>{cvMeta.label}</span>
                                            <span className="claim-item-header-right">
                                                <span className="claim-confidence">
                                                    {Math.round(cv.confidence * 100)}% confidence
                                                </span>
                                                {onDiscuss && <DiscussButton onClick={() => onDiscuss(i)} />}
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
                                                        <SourceLine cv={cv} j={j} />
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
