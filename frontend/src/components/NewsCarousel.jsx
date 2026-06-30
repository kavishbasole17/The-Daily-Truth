// src/components/NewsCarousel.jsx

"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import "@/styles/carousel.css";

export default function NewsCarousel() {
    const [articles, setArticles] = useState([]);
    const [currentIndex, setCurrentIndex] = useState(0);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        fetchHottestNews();
    }, []);

    const fetchHottestNews = async () => {
        try {
            setLoading(true);
            const response = await fetch("/api/news");

            if (!response.ok) {
                const data = await response.json();
                throw new Error(data.error || "Failed to fetch news");
            }

            const data = await response.json();

            if (data.articles && data.articles.length > 0) {
                setArticles(data.articles);
                setError(null);
            } else {
                throw new Error("No articles found");
            }
        } catch (err) {
            console.error("Fetch error:", err);
            setError(err.message);
            setArticles([]);
        } finally {
            setLoading(false);
        }
    };

    const nextSlide = () => {
        setCurrentIndex((prev) => (prev + 1) % (articles.length || 1));
    };

    const prevSlide = () => {
        setCurrentIndex((prev) =>
            prev === 0 ? (articles.length || 1) - 1 : prev - 1
        );
    };

    const goToSlide = (index) => {
        setCurrentIndex(index);
    };

    if (loading) {
        return (
            <section className="news-carousel-section">
                <div className="container">
                    <div className="carousel-header">
                        <h2>Hottest News Around the World</h2>
                    </div>
                    <div className="carousel-container loading">
                        <div className="carousel-loader" />
                    </div>
                </div>
            </section>
        );
    }

    if (error || articles.length === 0) {
        return (
            <section className="news-carousel-section">
                <div className="container">
                    <div className="carousel-header">
                        <h2>Hottest News Around the World</h2>
                    </div>
                    <div className="carousel-container error">
                        <p>{error || "No articles available"}</p>
                    </div>
                </div>
            </section>
        );
    }

    const currentArticle = articles[currentIndex];

    return (
        <section className="news-carousel-section">
            <div className="container">
                <div className="carousel-header">
                    <h2>Hottest News Around the World</h2>
                    <p>Stay updated with trending stories globally</p>
                </div>

                <div className="carousel-container">
                    <motion.div
                        className="carousel-main"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ duration: 0.5 }}
                        key={currentIndex}
                    >
                        <div className="carousel-image-wrapper">
                            {currentArticle.urlToImage ? (
                                <img
                                    src={currentArticle.urlToImage}
                                    alt={currentArticle.title}
                                    className="carousel-image"
                                />
                            ) : (
                                <div className="carousel-image-placeholder">
                                    <span>No image</span>
                                </div>
                            )}
                            <div className="carousel-overlay" />
                        </div>

                        <div className="carousel-content glass">
                            <div>
                                <span className="carousel-source">
                                    {currentArticle.source?.name ||
                                        "News Source"}
                                </span>

                                <h3 className="carousel-title">
                                    {currentArticle.title}
                                </h3>

                                <p className="carousel-description">
                                    {currentArticle.description ||
                                        "Read the full article for more details"}
                                </p>
                            </div>

                            <div className="carousel-meta">
                                <span className="carousel-date">
                                    {new Date(
                                        currentArticle.publishedAt
                                    ).toLocaleDateString()}
                                </span>
                                <a
                                    href={currentArticle.url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="carousel-link"
                                >
                                    Read More →
                                </a>
                            </div>
                        </div>
                    </motion.div>

                    <div className="carousel-controls">
                        <button
                            className="carousel-button prev"
                            onClick={prevSlide}
                            aria-label="Previous article"
                        >
                            ← Prev
                        </button>

                        <div className="carousel-indicators">
                            {articles.map((_, index) => (
                                <button
                                    key={index}
                                    className={`indicator ${
                                        index === currentIndex ? "active" : ""
                                    }`}
                                    onClick={() => goToSlide(index)}
                                    aria-label={`Go to article ${index + 1}`}
                                />
                            ))}
                        </div>

                        <button
                            className="carousel-button next"
                            onClick={nextSlide}
                            aria-label="Next article"
                        >
                            Next →
                        </button>

                        <span className="carousel-counter">
                            {currentIndex + 1}/{articles.length}
                        </span>
                    </div>
                </div>
            </div>
        </section>
    );
}