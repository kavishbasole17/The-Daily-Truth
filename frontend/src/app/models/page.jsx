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

    const sortedModels = [...models].sort(
        (a, b) =>
            (b.accuracy || b.f1_score || 0) -
            (a.accuracy || a.f1_score || 0)
    );

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

                                    <div className="model-metrics">
                                        <div>
                                            <span>Accuracy</span>
                                            <strong>
                                                {model.accuracy ??
                                                    model.acc ??
                                                    "N/A"}
                                            </strong>
                                        </div>

                                        <div>
                                            <span>F1 Score</span>
                                            <strong>
                                                {model.f1_score ??
                                                    model.f1 ??
                                                    "N/A"}
                                            </strong>
                                        </div>

                                        <div>
                                            <span>Precision</span>
                                            <strong>
                                                {model.precision ??
                                                    "N/A"}
                                            </strong>
                                        </div>

                                        <div>
                                            <span>Recall</span>
                                            <strong>
                                                {model.recall ??
                                                    "N/A"}
                                            </strong>
                                        </div>

                                        <div>
                                            <span>Inference</span>
                                            <strong>
                                                {model.inference_time ??
                                                    model.speed ??
                                                    "N/A"}
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