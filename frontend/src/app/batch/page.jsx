// src/app/batch/page.jsx

"use client";

import { useState } from "react";

import Navbar from "@/components/Navbar";
import LoadingOverlay from "@/components/LoadingOverlay";
import NewspaperButton from "@/components/NewspaperButton";
import ErrorState from "@/components/ErrorState";
import EmptyState from "@/components/EmptyState";
import Footer from "@/components/Footer";

import { batchPredict } from "@/services/api";

import "@/styles/pages.css";

export default function BatchPage() {
    const [articles, setArticles] = useState("");
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState([]);
    const [error, setError] = useState("");

    const handleSubmit = async () => {
        const lines = articles
            .split("\n")
            .map(x => x.trim())
            .filter(Boolean);

        if (!lines.length) return;

        if (lines.length > 100) {
            setError("Maximum 100 articles allowed.");
            return;
        }

        try {
            setLoading(true);
            setError("");
            setResult([]);

            const { data } = await batchPredict({
                articles: lines
            });

            setResult(
                data.results ||
                data.predictions ||
                data.data ||
                data ||
                []
            );
        } catch (err) {
            setError(
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
                            <NewspaperButton onClick={handleSubmit}>
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
                                    <th>Prediction</th>
                                    <th>Confidence</th>
                                </tr>
                                </thead>

                                <tbody>
                                {result.map((item, i) => (
                                    <tr key={`${i}-${item.id || item.article || i}`}>
                                        <td>{i + 1}</td>

                                        <td>
                        <span
                            className={
                                (
                                    item.prediction ||
                                    item.label ||
                                    item.verdict ||
                                    ""
                                )
                                    .toLowerCase()
                                    .includes("fake")
                                    ? "badge-fake"
                                    : "badge-real"
                            }
                        >
                          {item.prediction ||
                              item.label ||
                              item.verdict ||
                              "Unknown"}
                        </span>
                                        </td>

                                        <td>
                                            {Math.round(
                                                item.confidence ||
                                                item.score ||
                                                item.probability ||
                                                0
                                            )}
                                            %
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