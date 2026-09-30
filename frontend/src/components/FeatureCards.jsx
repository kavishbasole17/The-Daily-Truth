// src/components/FeatureCards.jsx

import "@/styles/landing.css";

const features = [
    {
        title: "Single Verification",
        text: "Analyze one article instantly using a selected model.",
        href: "/single",
        icon: (
            <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
        )
    },
    {
        title: "Consensus Engine",
        text: "Compare predictions from all available models.",
        href: "/consensus",
        icon: (
            <>
                <circle cx="6" cy="12" r="3" />
                <circle cx="18" cy="7" r="3" />
                <circle cx="18" cy="17" r="3" />
                <path d="M8.6 10.5l6.8-3M8.6 13.5l6.8 3" />
            </>
        )
    },
    {
        title: "Batch Analysis",
        text: "Verify hundreds of articles in one request.",
        href: "/batch",
        icon: (
            <>
                <path d="M4 6h16M4 12h16M4 18h10" />
            </>
        )
    },
    {
        title: "Model Leaderboard",
        text: "Track performance, confidence and rankings.",
        href: "/models",
        icon: (
            <path d="M4 20V10m6 10V4m6 16v-7" />
        )
    }
];

export default function FeatureCards() {
    return (
        <section className="features section">
            <div className="container">
                <div className="section-title">
                    <span className="section-eyebrow">What we offer</span>
                    <h2>NEWSROOM TOOLS</h2>
                </div>

                <div className="feature-grid">
                    {features.map((feature, i) => (
                        <a key={feature.title} href={feature.href} className="feature-card glass">
                            <span className="feature-index">{String(i + 1).padStart(2, "0")}</span>
                            <div className="feature-icon">
                                <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                                    {feature.icon}
                                </svg>
                            </div>
                            <h3>{feature.title}</h3>
                            <p>{feature.text}</p>
                            <span className="feature-arrow">Explore →</span>
                        </a>
                    ))}
                </div>
            </div>
        </section>
    );
}
