// src/app/batch/page.jsx

"use client";

import { useState, useEffect } from "react";

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
                                {result.map((item, i) => (
                                    <tr key={i}>
                                        <td>{i + 1}</td>

                                        <td className="text-preview">
                                            {item.text_preview || `Article ${i + 1}`}
                                        </td>

                                        <td>
                        <span
                            className={
                                (item.label || "").toLowerCase() === "fake"
                                    ? "badge-fake"
                                    : "badge-real"
                            }
                        >
                          {item.label || "Unknown"}
                        </span>
                                        </td>

                                        <td>
                                            {item.confidence != null
                                                ? `${Math.round(item.confidence * 100)}%`
                                                : "N/A"}
                                        </td>
                                    </tr>
                                ))}
                                </tbody>
                            </table>
                        </div>
                    )}
                </div>
            </section>

            <Footer />
        </>
    );
}
