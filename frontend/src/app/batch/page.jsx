// src/app/batch/page.jsx

"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";

import Navbar from "@/components/Navbar";
import LoadingOverlay from "@/components/LoadingOverlay";
import ModelSelector from "@/components/ModelSelector";
import NewspaperButton from "@/components/NewspaperButton";
import ErrorState from "@/components/ErrorState";
import EmptyState from "@/components/EmptyState";
import Footer from "@/components/Footer";

import { batchPredict, getModels } from "@/services/api";

import "@/styles/pages.css";

export default function BatchPage() {
    const [articles, setArticles] = useState("");
    const [model, setModel] = useState("");
    const [availableModels, setAvailableModels] = useState([]);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState([]);
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
            .catch(() => {});
    }, []);

    const handleSubmit = async () => {
        const lines = articles
            .split("\n")
            .map(x => x.trim())
            .filter(Boolean);

        if (!lines.length || !model) return;

        if (lines.length > 100) {
            setError("Maximum 100 articles allowed.");
            return;
        }

        try {
            setLoading(true);
            setError("");
            setResult([]);

            const { data } = await batchPredict({
                texts: lines,
                model_name: model,
                threshold: 0.5,
            });

            setResult(data.results || []);
        } catch (err) {
            setError(
                err?.response?.data?.detail ||
                err?.response?.data?.message ||
                "Batch analysis failed."
            );
        } finally {
            setLoading(false);
        }
    };

    return (
        <>
            <Navbar />

            <LoadingOverlay show={loading} message="Analyzing batch articles..." />

            <section className="prediction-page section">
                <div className="container">
                    <div className="page-header">
                        <span className="page-eyebrow">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M4 6h16M4 12h16M4 18h10" />
                            </svg>
                            Bulk Processing
                        </span>
                        <h1>BATCH ANALYSIS</h1>
                        <p>One article per line.</p>
                    </div>

                    <div className="prediction-card glass">
            <textarea
                className="article-input"
                placeholder="Article 1&#10;Article 2&#10;Article 3"
                value={articles}
                onChange={e => setArticles(e.target.value)}
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
                                ANALYZE BATCH
                            </NewspaperButton>
                        </div>
                    </div>

                    {error && (
                        <ErrorState message={error} />
                    )}

                    {!error && !result.length && (
                        <EmptyState
                            title="Awaiting Batch Analysis"
                            message="Enter multiple articles to begin."
                        />
                    )}

                    {!!result.length && (
                        <motion.div
                            initial={{ opacity: 0, y: 16 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.4, ease: "easeOut" }}
                        >
                            <div className="batch-stats">
                                <div className="batch-stat glass">
                                    <span className="batch-stat-label">Analyzed</span>
                                    <span className="batch-stat-value">{result.length}</span>
                                </div>
                                <div className="batch-stat glass">
                                    <span className="batch-stat-label">Likely Real</span>
                                    <span className="batch-stat-value real">
                                        {result.filter(r => (r.label || "").toLowerCase() !== "fake").length}
                                    </span>
                                </div>
                                <div className="batch-stat glass">
                                    <span className="batch-stat-label">Likely Fake</span>
                                    <span className="batch-stat-value fake">
                                        {result.filter(r => (r.label || "").toLowerCase() === "fake").length}
                                    </span>
                                </div>
                                <div className="batch-stat glass">
                                    <span className="batch-stat-label">Avg. Confidence</span>
                                    <span className="batch-stat-value">
                                        {Math.round(
                                            (result.reduce((sum, r) => sum + (r.confidence ?? 0), 0) / result.length) * 100
                                        )}%
                                    </span>
                                </div>
                            </div>

                            <div className="batch-table glass">
                                <table>
                                    <thead>
                                    <tr>
                                        <th>#</th>
                                        <th>Preview</th>
                                        <th>Prediction</th>
                                        <th>Confidence</th>
                                    </tr>
                                    </thead>

                                    <tbody>
                                    {result.map((item, i) => {
                                        const isFake = (item.label || "").toLowerCase() === "fake";
                                        const conf = item.confidence != null ? Math.round(item.confidence * 100) : null;
                                        return (
                                            <tr key={i}>
                                                <td>{i + 1}</td>

                                                <td className="text-preview">
                                                    {item.text_preview || `Article ${i + 1}`}
                                                </td>

                                                <td>
                                                    <span className={isFake ? "badge-fake" : "badge-real"}>
                                                        {item.label || "Unknown"}
                                                    </span>
                                                </td>

                                                <td>
                                                    {conf != null ? (
                                                        <div className="batch-confidence-cell">
                                                            <div className="batch-confidence-track">
                                                                <div
                                                                    className={`batch-confidence-fill ${isFake ? "result-fake" : "result-real"}`}
                                                                    style={{ width: `${conf}%` }}
                                                                />
                                                            </div>
                                                            <span>{conf}%</span>
                                                        </div>
                                                    ) : "N/A"}
                                                </td>
                                            </tr>
                                        );
                                    })}
                                    </tbody>
                                </table>
                            </div>
                        </motion.div>
                    )}
                </div>
            </section>

            <Footer />
        </>
    );
}
