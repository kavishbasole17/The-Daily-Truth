// src/app/single/page.jsx

"use client";

import { useState } from "react";

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

import { predictArticle } from "@/services/api";

import "@/styles/pages.css";

export default function SinglePredictionPage() {
    const [article, setArticle] = useState("");
    const [model, setModel] = useState("distilbert");
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState("");

    const models = [
        "distilbert",
        "bert",
        "roberta",
        "deberta",
        "xlnet"
    ];

    const handleSubmit = async () => {
        if (!article.trim()) return;

        try {
            setLoading(true);
            setError("");
            setResult(null);

            const { data } = await predictArticle({
                article,
                model
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

    const prediction =
        result?.prediction ||
        result?.label ||
        result?.verdict ||
        "";

    const confidence = Math.round(
        result?.confidence ||
        result?.score ||
        result?.probability ||
        0
    );

    const realProbability = Math.round(
        result?.real_probability ||
        result?.real ||
        0
    );

    const fakeProbability = Math.round(
        result?.fake_probability ||
        result?.fake ||
        0
    );

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
                            <ModelSelector
                                models={models}
                                value={model}
                                onChange={setModel}
                            />

                            <NewspaperButton onClick={handleSubmit}>
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