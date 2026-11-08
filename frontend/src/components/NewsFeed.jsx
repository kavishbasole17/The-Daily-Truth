// src/components/NewsFeed.jsx

"use client";

import { useEffect, useState } from "react";
import Image from "next/image";
import "@/styles/feed.css";

// Thumbnails come from whichever publisher wrote the story, so the hosts
// can't be allow-listed for Next's optimizer; `unoptimized` serves them
// directly while next/image still gives lazy loading and a reserved box
// (no layout shift). `fill` sizes them to their positioned container.
function Thumb({ src, priority = false }) {
    return <Image src={src} alt="" fill unoptimized priority={priority} sizes="(max-width: 900px) 100vw, 50vw" />;
}

const CATEGORIES = [
    ["all", "All"],
    ["india", "India"],
    ["world", "World"],
    ["sports", "Sports"],
    ["business", "Business"],
    ["entertainment", "Entertainment"],
    ["technology", "Technology"]
];

function timeAgo(iso) {
    if (!iso) return "";
    const diffMs = Date.now() - new Date(iso).getTime();
    const mins = Math.floor(diffMs / 60000);
    if (mins < 1) return "just now";
    if (mins < 60) return `${mins}m ago`;
    const hours = Math.floor(mins / 60);
    if (hours < 24) return `${hours}h ago`;
    return `${Math.floor(hours / 24)}d ago`;
}

function StoryRow({ article }) {
    return (
        <a
            className="story-row"
            href={article.url}
            target="_blank"
            rel="noopener noreferrer"
        >
            <div className="story-row-thumb">
                {article.image ? (
                    <Thumb src={article.image} />
                ) : (
                    <div className="story-row-thumb placeholder" />
                )}
            </div>
            <div className="story-row-body">
                <span className="story-row-title">{article.title}</span>
                <span className="story-row-meta">
                    {article.source} · {timeAgo(article.publishedAt)}
                </span>
            </div>
        </a>
    );
}

export default function NewsFeed() {
    const [category, setCategory] = useState("all");
    const [articles, setArticles] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        let cancelled = false;
        setLoading(true);
        setError(null);

        fetch(`/api/news?category=${category}`)
            .then(async (res) => {
                const data = await res.json();
                if (!res.ok) throw new Error(data.error || "Failed to load news");
                return data;
            })
            .then((data) => {
                if (!cancelled) setArticles(data.articles || []);
            })
            .catch((err) => {
                if (!cancelled) {
                    setError(err.message);
                    setArticles([]);
                }
            })
            .finally(() => {
                if (!cancelled) setLoading(false);
            });

        return () => {
            cancelled = true;
        };
    }, [category]);

    const [hero, ...rest] = articles;
    const secondary = rest.slice(0, 2);
    const exploreItems = rest.slice(2, 8);

    return (
        <section className="feed-section">
            <div className="container">
                <div className="feed-tabs">
                    {CATEGORIES.map(([value, label]) => (
                        <button
                            key={value}
                            className={`feed-tab ${category === value ? "active" : ""}`}
                            onClick={() => setCategory(value)}
                        >
                            {label}
                        </button>
                    ))}
                </div>

                {loading && (
                    <div className="feed-loading">
                        <div className="feed-loader" />
                    </div>
                )}

                {!loading && error && (
                    <div className="feed-error">
                        <p>{error}</p>
                    </div>
                )}

                {!loading && !error && hero && (
                    <div className="feed-grid">
                        <a
                            className="feed-hero"
                            href={hero.url}
                            target="_blank"
                            rel="noopener noreferrer"
                        >
                            <div className="feed-hero-media">
                                {hero.image ? (
                                    <Thumb src={hero.image} priority />
                                ) : (
                                    <div className="feed-hero-media placeholder" />
                                )}
                                <div className="feed-hero-scrim" />
                                <span className="feed-badge live">Live</span>
                            </div>

                            <div className="feed-hero-content">
                                <span className="feed-hero-source">
                                    {hero.source} · {timeAgo(hero.publishedAt)}
                                </span>
                                <h2 className="feed-hero-title">{hero.title}</h2>
                                {hero.description && (
                                    <p className="feed-hero-desc">{hero.description}</p>
                                )}
                                <span className="feed-hero-cta">Read Article →</span>
                            </div>
                        </a>

                        <div className="feed-secondary">
                            {secondary.map((article) => (
                                <StoryRow key={article.url} article={article} />
                            ))}
                        </div>
                    </div>
                )}

                {!loading && !error && exploreItems.length > 0 && (
                    <div className="feed-explore">
                        <h3 className="feed-explore-title">Explore More</h3>
                        <div className="feed-explore-grid">
                            {exploreItems.map((article) => (
                                <a
                                    key={article.url}
                                    className="explore-card"
                                    href={article.url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                >
                                    <div className="explore-card-media">
                                        {article.image ? (
                                            <Thumb src={article.image} />
                                        ) : (
                                            <div className="explore-card-media placeholder" />
                                        )}
                                        <span className="feed-badge">{category === "all" ? "News" : category}</span>
                                    </div>
                                    <span className="explore-card-title">{article.title}</span>
                                    <span className="explore-card-meta">
                                        {article.source} · {timeAgo(article.publishedAt)}
                                    </span>
                                </a>
                            ))}
                        </div>
                    </div>
                )}
            </div>
        </section>
    );
}
