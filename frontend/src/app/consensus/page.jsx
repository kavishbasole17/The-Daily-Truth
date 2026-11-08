// src/app/consensus/page.jsx

"use client";

import { useState } from "react";
import { motion } from "framer-motion";

import Navbar from "@/components/Navbar";
import LoadingOverlay from "@/components/LoadingOverlay";
import NewspaperButton from "@/components/NewspaperButton";
import ErrorState from "@/components/ErrorState";
import EmptyState from "@/components/EmptyState";
import Footer from "@/components/Footer";

import { consensusPredict, errorMessage } from "@/services/api";

import "@/styles/pages.css";

export default function ConsensusPage() {
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

            const { data } = await consensusPredict({
                text: article,
                threshold: 0.5,
            });

            setResult(data);
        } catch (err) {
            // detail can be a list of validation errors, not a string
            setError(errorMessage(err, "Failed to analyze article."));
        } finally {
            setLoading(false);
        }
    };

    const verdict = result?.consensus || "";
    const agreement = Math.round((result?.consensus_confidence ?? 0) * 100);
    const models = result?.results || [];

    const realCount = models.filter(m => (m.label || "").toLowerCase() !== "fake").length;
    const fakeCount = models.length - realCount;
    const realShare = models.length ? Math.round((realCount / models.length) * 100) : 0;

    return (
        <>
            <Navbar />

            <LoadingOverlay show={loading} message="Analyzing consensus..." />

            <section className="prediction-page section">
                <div className="container">
                    <div className="page-header">
                        <span className="page-eyebrow">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <circle cx="6" cy="12" r="3" />
                                <circle cx="18" cy="7" r="3" />
                                <circle cx="18" cy="17" r="3" />
                                <path d="M8.6 10.5l6.8-3M8.6 13.5l6.8 3" />
                            </svg>
                            Multi-Model Panel
                        </span>
                        <h1>EDITORIAL CONSENSUS</h1>
                        <p>Compare multiple model opinions.</p>
                    </div>

                    <div className="prediction-card glass">
            <textarea
                className="article-input"
                placeholder="Paste article content..."
                value={article}
                onChange={e => setArticle(e.target.value)}
            />

                        <div className="prediction-controls">
                            <NewspaperButton onClick={handleSubmit} disabled={loading}>
                                ANALYZE CONSENSUS
                            </NewspaperButton>
                        </div>
                    </div>

                    {error && (
                        <ErrorState message={error} />
                    )}

                    {!error && !result && (
                        <EmptyState
                            title="Awaiting Consensus"
                            message="Submit an article to compare all models."
                        />
                    )}

                    {result && (
                        <motion.div
                            initial={{ opacity: 0, y: 16 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.4, ease: "easeOut" }}
                        >
                            <div className={`consensus-banner glass ${verdict === "fake" ? "fake-card" : "real-card"}`}>
                                <h2>{verdict.toUpperCase()}</h2>
                                <p>{agreement}% Agreement</p>
                            </div>

                            {models.length > 0 && (
                                <div className="agreement-split">
                                    <div className="agreement-split-track">
                                        <div className="agreement-split-real" style={{ width: `${realShare}%` }} />
                                        <div className="agreement-split-fake" style={{ width: `${100 - realShare}%` }} />
                                    </div>
                                    <div className="agreement-split-labels">
                                        <span className="real">{realCount} models say REAL</span>
                                        <span className="fake">{fakeCount} models say FAKE</span>
                                    </div>
                                </div>
                            )}

                            <div className="consensus-grid">
                                {models.map((m, i) => {
                                    const isFake = (m.label || "").toLowerCase() === "fake";
                                    const conf = Math.round((m.confidence ?? 0) * 100);
                                    return (
                                        <div
                                            key={i}
                                            className={`model-card glass ${isFake ? "result-fake" : "result-real"}`}
                                        >
                                            <div className="model-card-header">
                                                <span className="model-card-name">
                                                    {m.display_name || m.model_name || `Model ${i + 1}`}
                                                </span>
                                                <span className={isFake ? "badge-fake" : "badge-real"}>
                                                    {m.label || "Unknown"}
                                                </span>
                                            </div>

                                            <div className="model-card-bar">
                                                <div
                                                    className={`model-card-bar-fill ${isFake ? "result-fake" : "result-real"}`}
                                                    style={{ width: `${conf}%` }}
                                                />
                                            </div>

                                            <div className="model-card-meta">
                                                <span>{conf}% confidence</span>
                                                <span>{m.category || "N/A"}</span>
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                        </motion.div>
                    )}
                </div>
            </section>

            <Footer />
        </>
    );
}
