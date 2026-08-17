// src/app/single/page.jsx

"use client";

import { useState, useEffect } from "react";

import Navbar from "@/components/Navbar";
import LoadingOverlay from "@/components/LoadingOverlay";
import ModelSelector from "@/components/ModelSelector";
import NewspaperButton from "@/components/NewspaperButton";
import VerdictCard from "@/components/VerdictCard";
import ConfidenceGauge from "@/components/ConfidenceGauge";
import ProbabilityBar from "@/components/ProbabilityBar";
import ErrorState from "@/components/ErrorState";
import EmptyState from "@/components/EmptyState";
import Footer from "@/components/Footer";

import { predictArticle, getModels } from "@/services/api";

import "@/styles/pages.css";

export default function SinglePredictionPage() {
    const [article, setArticle] = useState("");
    const [model, setModel] = useState("");
    const [availableModels, setAvailableModels] = useState([]);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState("");

    useEffect(() => {
        getModels()
            .then(({ data }) => {
                const list = (data.models || [])
                    .filter(m => m.loaded)
                    .map(m => m.name);
                setAvailableModels(list);
                if (list.length > 0) setModel(list[0]);
            })
            .catch(() => {
                // fallback — user will see error when they submit
            });
    }, []);

    const handleSubmit = async () => {
        if (!article.trim() || !model) return;

        try {
            setLoading(true);
            setError("");
            setResult(null);

            const { data } = await predictArticle({
                text: article,
                model_name: model,
                threshold: 0.5,
            });

            setResult(data.result);
        } catch (err) {
            setError(
                err?.response?.data?.detail ||
                err?.response?.data?.message ||
                "Failed to analyze article."
            );
        } finally {
            setLoading(false);
        }
    };

    const prediction = result?.label || "";
    const confidence = Math.round((result?.confidence ?? 0) * 100);
    const realProbability = Math.round((result?.prob_real ?? 0) * 100);
    const fakeProbability = Math.round((result?.prob_fake ?? 0) * 100);

    return (
        <>
            <Navbar />

            <LoadingOverlay show={loading} />

            <section className="prediction-page section">
                <div className="container">
                    <div className="page-header">
                        <h1>SINGLE ARTICLE VERIFICATION</h1>
                        <p>Submit an article for newsroom analysis.</p>
                    </div>

                    <div className="prediction-card glass">
            <textarea
                className="article-input"
                placeholder="Paste article content..."
                value={article}
                onChange={e => setArticle(e.target.value)}
            />

                        <div className="prediction-controls">
                            {availableModels.length > 0 && (
                                <ModelSelector
                                    models={availableModels}
                                    value={model}
                                    onChange={setModel}
                                />
                            )}

                            <NewspaperButton onClick={handleSubmit} disabled={loading}>
                                VERIFY ARTICLE
                            </NewspaperButton>
                        </div>
                    </div>

                    {error && (
                        <ErrorState message={error} />
                    )}

                    {!error && !result && (
                        <EmptyState />
                    )}

                    {result && (
                        <>
                            <VerdictCard
                                prediction={prediction}
                                confidence={confidence}
                            />

                            <ConfidenceGauge
                                value={confidence}
                            />

                            <ProbabilityBar
                                real={realProbability}
                                fake={fakeProbability}
                            />
                        </>
                    )}
                </div>
            </section>

            <Footer />
        </>
    );
}
