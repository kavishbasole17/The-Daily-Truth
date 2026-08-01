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

import { submitForVerification } from "@/services/api";

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
};

export default function VerifyPage() {
    const [article, setArticle] = useState("");
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState("");

    const handleSubmit = async () => {
        if (!article.trim()) return;

        try {
            setLoading(true);
            setError("");
            setResult(null);

            const { data } = await submitForVerification(article);
            setResult(data);
        } catch (err) {
            setError(
                err?.response?.data?.detail ||
                err?.response?.data?.message ||
                "Failed to verify article."
            );
        } finally {
            setLoading(false);
        }
    };

    const verdict = result?.verdict;
    const meta = VERDICT_META[verdict?.verdict] || VERDICT_META.uncertain;

    return (
        <>
            <Navbar />

            <LoadingOverlay show={loading} />

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
                            <NewspaperButton onClick={handleSubmit}>
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
                                                        <li key={j}>{s}</li>
                                                    ))}
                                                </ul>
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
