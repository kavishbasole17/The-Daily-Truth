// src/app/models/page.jsx

"use client";

import { useEffect, useState } from "react";

import Navbar from "@/components/Navbar";
import LoadingOverlay from "@/components/LoadingOverlay";
import ErrorState from "@/components/ErrorState";
import EmptyState from "@/components/EmptyState";
import Footer from "@/components/Footer";

import { getModels } from "@/services/api";

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
                setError(
                    err?.response?.data?.message ||
                    "Failed to load models."
                );
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

    return (
        <>
            <Navbar />

            <LoadingOverlay show={loading} />

            <section className="prediction-page section">
                <div className="container">
                    <div className="page-header">
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
                            {sortedModels.map((model, index) => (
                                <div
                                    key={model.name || index}
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

                                    <div className="model-metrics">
                                        <div>
                                            <span>Accuracy</span>
                                            <strong>{pct(model.metrics?.accuracy)}</strong>
                                        </div>

                                        <div>
                                            <span>F1 Score</span>
                                            <strong>{pct(model.metrics?.f1)}</strong>
                                        </div>

                                        <div>
                                            <span>Precision</span>
                                            <strong>{pct(model.metrics?.precision)}</strong>
                                        </div>

                                        <div>
                                            <span>Recall</span>
                                            <strong>{pct(model.metrics?.recall)}</strong>
                                        </div>

                                        <div>
                                            <span>AUC</span>
                                            <strong>
                                                {typeof model.metrics?.auc === "number"
                                                    ? model.metrics.auc.toFixed(4)
                                                    : "N/A"}
                                            </strong>
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </section>

            <Footer />
        </>
    );
}