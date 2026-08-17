// src/app/verify/page.jsx
//
// "Paper Trail" — the evidence-grounded verification pipeline, as opposed to
// the single-model / consensus pages which classify on text patterns alone.
// Extracts claims from the article, checks each against trusted news
// sources and Wikipedia, and shows the full reasoning trail.

"use client";

import { useState } from "react";

import Navbar from "@/components/Navbar";
import LoadingOverlay from "@/components/LoadingOverlay";
import NewspaperButton from "@/components/NewspaperButton";
import ErrorState from "@/components/ErrorState";
import EmptyState from "@/components/EmptyState";
import Footer from "@/components/Footer";

import { verifyArticle } from "@/services/api";

import "@/styles/pages.css";

const VERDICT_META = {
    likely_real: { label: "LIKELY REAL", className: "real-card" },
    likely_fake: { label: "LIKELY MISINFORMATION", className: "fake-card" },
    uncertain: { label: "UNCERTAIN — NEEDS HUMAN REVIEW", className: "uncertain-card" },
};

const CLAIM_VERDICT_META = {
    entailment: { label: "SUPPORTED", className: "badge-real" },
    contradiction: { label: "CONTRADICTED", className: "badge-fake" },
    neutral: { label: "NO CLEAR SIGNAL", className: "badge-neutral" },
    // Distinct from "neutral" on purpose. Neutral means we found evidence and
    // it was inconclusive; unverifiable means we found nothing to check
    // against. Collapsing the two is what previously let "no search results"
    // masquerade as a signal about the article.
    unverifiable: { label: "COULD NOT VERIFY", className: "badge-neutral" },
};

export default function VerifyPage() {
    const [article, setArticle] = useState("");
    const [loading, setLoading] = useState(false);
    const [stage, setStage] = useState("");
    const [result, setResult] = useState(null);
    const [error, setError] = useState("");

    const handleSubmit = async () => {
        if (!article.trim()) return;

        try {
            setLoading(true);
            setError("");
            setResult(null);
            setStage("queued");

            const data = await verifyArticle(article, { onStage: setStage });
            setResult(data);
        } catch (err) {
            setError(
                err?.response?.data?.detail ||
                err?.response?.data?.message ||
                err?.message ||
                "Failed to verify article."
            );
        } finally {
            setLoading(false);
            setStage("");
        }
    };

    const verdict = result?.verdict;
    const meta = VERDICT_META[verdict?.verdict] || VERDICT_META.uncertain;

    return (
        <>
            <Navbar />

            <LoadingOverlay
                show={loading}
                message={
                    stage === "queued"
                        ? "Queued for verification..."
                        : "Extracting claims and checking them against sources..."
                }
            />

            <section className="prediction-page section">
                <div className="container">
                    <div className="page-header">
                        <h1>PAPER TRAIL</h1>
                        <p>
                            Extracts factual claims, checks each against trusted news outlets
                            and reference sources, and shows you the receipts.
                        </p>
                    </div>

                    <div className="prediction-card glass">
                        <textarea
                            className="article-input"
                            placeholder="Paste article content..."
                            value={article}
                            onChange={e => setArticle(e.target.value)}
                        />

                        <div className="prediction-controls">
                            <div />
                            <NewspaperButton onClick={handleSubmit} disabled={loading}>
                                VERIFY WITH EVIDENCE
                            </NewspaperButton>
                        </div>
                    </div>

                    {error && <ErrorState message={error} />}

                    {!error && !result && <EmptyState />}

                    {verdict && (
                        <>
                            <div className={`verdict-card glass ${meta.className}`}>
                                <h2>{meta.label}</h2>
                                <p>{Math.round(verdict.confidence * 100)}% Confidence</p>
                                {verdict.needs_human_review && (
                                    <p className="review-flag">
                                        Flagged for analyst review — added to the review queue.
                                    </p>
                                )}
                            </div>

                            {/* The pipeline reports every point at which it ran below full
                                capability. Hiding these would make a degraded verdict look
                                identical to a healthy one. */}
                            {verdict.degradations?.length > 0 && (
                                <div className="claim-trail">
                                    <h3>Limitations affecting this verdict</h3>
                                    <ul className="evidence-list">
                                        {verdict.degradations.map((d, i) => (
                                            <li key={i}>{d.detail}</li>
                                        ))}
                                    </ul>
                                </div>
                            )}

                            <div className="claim-trail">
                                <h3>Claim-by-claim trail</h3>
                                {verdict.claim_verdicts.length === 0 && (
                                    <p className="muted">
                                        No individually verifiable claims were extracted from
                                        this article.
                                    </p>
                                )}
                                {verdict.claim_verdicts.map((cv, i) => {
                                    const cvMeta = CLAIM_VERDICT_META[cv.verdict] || CLAIM_VERDICT_META.neutral;
                                    return (
                                        <div className="claim-item" key={i}>
                                            <div className="claim-item-header">
                                                <span className={cvMeta.className}>{cvMeta.label}</span>
                                                <span className="claim-confidence">
                                                    {Math.round(cv.confidence * 100)}% confidence
                                                </span>
                                            </div>
                                            <p className="claim-text">&ldquo;{cv.claim}&rdquo;</p>
                                            {cv.evidence_snippets.length > 0 && (
                                                <ul className="evidence-list">
                                                    {cv.evidence_snippets.map((s, j) => (
                                                        <li key={j}>
                                                            {/* `sources` is positionally parallel to
                                                                `evidence_snippets` — showing where a
                                                                snippet came from is the whole point
                                                                of an evidence trail. */}
                                                            {cv.sources?.[j] && (
                                                                <span className="evidence-source">
                                                                    {cv.sources[j]}
                                                                </span>
                                                            )}
                                                            {s}
                                                        </li>
                                                    ))}
                                                </ul>
                                            )}
                                            {cv.verdict === "unverifiable" && (
                                                <p className="muted">
                                                    No source covering this claim was found. That is
                                                    not evidence against it — recent, regional and
                                                    non-English topics are often missed.
                                                </p>
                                            )}
                                        </div>
                                    );
                                })}
                            </div>

                            <details className="model-card reasoning-trace">
                                <summary>Full reasoning trace</summary>
                                <pre className="model-content">{verdict.reasoning_trace}</pre>
                            </details>
                        </>
                    )}
                </div>
            </section>

            <Footer />
        </>
    );
}
