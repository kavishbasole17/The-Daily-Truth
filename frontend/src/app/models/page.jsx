// src/app/models/page.jsx

"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";

import Navbar from "@/components/Navbar";
import LoadingOverlay from "@/components/LoadingOverlay";
import ErrorState from "@/components/ErrorState";
import EmptyState from "@/components/EmptyState";
import Footer from "@/components/Footer";

import { getModels, errorMessage } from "@/services/api";

import "@/styles/pages.css";

export default function ModelsPage() {
    const [models, setModels] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState("");

    useEffect(() => {
        const loadModels = async () => {
            try {
                const { data } = await getModels();

                const list =
                    data.models ||
                    data.data ||
                    data ||
                    [];

                setModels(list);
            } catch (err) {
                setError(errorMessage(err, "Failed to load models."));
            } finally {
                setLoading(false);
            }
        };

        loadModels();
    }, []);

    // Metrics live under model.metrics.{accuracy,f1,precision,recall,auc} per
    // the /models response schema (ModelInfo) — not top-level fields.
    const sortedModels = [...models].sort(
        (a, b) =>
            (b.metrics?.accuracy ?? b.metrics?.f1 ?? 0) -
            (a.metrics?.accuracy ?? a.metrics?.f1 ?? 0)
    );

    const pct = (v) => (typeof v === "number" ? `${(v * 100).toFixed(1)}%` : "N/A");

    const hasMetrics = (model) =>
        model.metrics && Object.values(model.metrics).some(v => typeof v === "number");

    return (
        <>
            <Navbar />

            <LoadingOverlay show={loading} />

            <section className="prediction-page section">
                <div className="container">
                    <div className="page-header">
                        <span className="page-eyebrow">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M4 20V10m6 10V4m6 16v-7" />
                            </svg>
                            Benchmark Rankings
                        </span>
                        <h1>MODEL LEADERBOARD</h1>
                        <p>Performance rankings and metrics.</p>
                    </div>

                    {error && (
                        <ErrorState message={error} />
                    )}

                    {!error && !sortedModels.length && !loading && (
                        <EmptyState
                            title="No Models Found"
                            message="Backend returned no model information."
                        />
                    )}

                    {!!sortedModels.length && (
                        <div className="models-grid">
                            {sortedModels.map((model, index) => {
                                const metricEntries = [
                                    ["Accuracy", model.metrics?.accuracy],
                                    ["F1 Score", model.metrics?.f1],
                                    ["Precision", model.metrics?.precision],
                                    ["Recall", model.metrics?.recall],
                                ];

                                return (
                                    <motion.div
                                        key={model.name || index}
                                        initial={{ opacity: 0, y: 16 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        transition={{ duration: 0.35, delay: index * 0.05, ease: "easeOut" }}
                                        className={`model-ranking-card glass ${
                                            index === 0 ? "leader-card" : ""
                                        }`}
                                    >
                                        <div className="model-rank">
                                            #{index + 1}
                                        </div>

                                        <h2>
                                            {model.name ||
                                                model.model ||
                                                `Model ${index + 1}`}
                                        </h2>

                                        {!model.loaded && (
                                            <p className="model-not-loaded">Not loaded on this server</p>
                                        )}

                                        {hasMetrics(model) ? (
                                            <div className="model-metrics">
                                                {metricEntries.map(([label, value]) => (
                                                    <div className="metric-row" key={label}>
                                                        <span>{label}</span>
                                                        <div className="metric-track">
                                                            <div
                                                                className="metric-fill"
                                                                style={{ width: typeof value === "number" ? `${value * 100}%` : "0%" }}
                                                            />
                                                        </div>
                                                        <strong>{pct(value)}</strong>
                                                    </div>
                                                ))}

                                                <div className="metric-row">
                                                    <span>AUC</span>
                                                    <div className="metric-track">
                                                        <div
                                                            className="metric-fill"
                                                            style={{ width: typeof model.metrics?.auc === "number" ? `${model.metrics.auc * 100}%` : "0%" }}
                                                        />
                                                    </div>
                                                    <strong>
                                                        {typeof model.metrics?.auc === "number"
                                                            ? model.metrics.auc.toFixed(4)
                                                            : "N/A"}
                                                    </strong>
                                                </div>
                                            </div>
                                        ) : (
                                            <p className="model-no-benchmark">
                                                {model.n_base_models
                                                    ? `Meta-ensemble over ${model.n_base_models} base models — not yet independently benchmarked.`
                                                    : "Not yet independently benchmarked."}
                                            </p>
                                        )}
                                    </motion.div>
                                );
                            })}
                        </div>
                    )}
                </div>
            </section>

            <Footer />
        </>
    );
}