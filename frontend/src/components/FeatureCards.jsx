// src/components/FeatureCards.jsx

import "@/styles/landing.css";

const features = [
    {
        title: "Single Verification",
        text: "Analyze one article instantly using a selected model."
    },
    {
        title: "Consensus Engine",
        text: "Compare predictions from all available models."
    },
    {
        title: "Batch Analysis",
        text: "Verify hundreds of articles in one request."
    },
    {
        title: "Model Leaderboard",
        text: "Track performance, confidence and rankings."
    }
];

export default function FeatureCards() {
    return (
        <section className="features section">
            <div className="container">
                <div className="section-title">
                    <h2>NEWSROOM TOOLS</h2>
                </div>

                <div className="feature-grid">
                    {features.map(feature => (
                        <div key={feature.title} className="feature-card glass">
                            <h3>{feature.title}</h3>
                            <p>{feature.text}</p>
                        </div>
                    ))}
                </div>
            </div>
        </section>
    );
}