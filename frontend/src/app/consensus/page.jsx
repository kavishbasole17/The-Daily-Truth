// src/app/consensus/page.jsx

"use client";

import { useState } from "react";

import Navbar from "@/components/Navbar";
import LoadingOverlay from "@/components/LoadingOverlay";
import NewspaperButton from "@/components/NewspaperButton";
import ErrorState from "@/components/ErrorState";
import EmptyState from "@/components/EmptyState";
import Footer from "@/components/Footer";

import { consensusPredict } from "@/services/api";

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
                article
            });

            setResult(data);
        } catch (err) {
            setError(
                err?.response?.data?.message ||
                "Failed to analyze article."
            );
        } finally {
            setLoading(false);
        }
    };

    const verdict =
        result?.consensus ||
        result?.prediction ||
        result?.verdict ||
        "";

    const agreement =
        result?.agreement ||
        result?.agreement_percentage ||
        result?.consensus_score ||
        0;

    const models =
        result?.models ||
        result?.results ||
        result?.predictions ||
        [];

    return (
        <>
            <Navbar />

            <LoadingOverlay show={loading} message="Analyzing consensus..." />

            <section className="prediction-page section">
                <div className="container">
                    <div className="page-header">
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
                            <NewspaperButton onClick={handleSubmit}>
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
                        <>
                            <div className="consensus-banner glass">
                                <h2>{verdict}</h2>
                                <p>{agreement}% Agreement</p>
                            </div>

                            <div className="consensus-grid">
                                {models.map((model, i) => (
                                    <details
                                        key={i}
                                        className="model-card glass"
                                    >
                                        <summary>
                                            {model.name ||
                                                model.model ||
                                                `Model ${i + 1}`}
                                        </summary>

                                        <div className="model-content">
                                            <div>
                                                Prediction:
                                                {" "}
                                                {model.prediction ||
                                                    model.label ||
                                                    model.verdict}
                                            </div>

                                            <div>
                                                Confidence:
                                                {" "}
                                                {Math.round(
                                                    model.confidence ||
                                                    model.score ||
                                                    model.probability ||
                                                    0
                                                )}
                                                %
                                            </div>

                                            <div>
                                                Time:
                                                {" "}
                                                {model.inference_time ||
                                                    model.time ||
                                                    "N/A"}
                                            </div>
                                        </div>
                                    </details>
                                ))}
                            </div>
                        </>
                    )}
                </div>
            </section>

            <Footer />
        </>
    );
}