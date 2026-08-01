// src/components/BreakingTicker.jsx

import "@/styles/ticker.css";

export default function BreakingTicker() {
    const items = [
        "AI Credibility Engine Online",
        "Consensus Analysis Available",
        "22 ML Models Benchmarked Live",
        "Batch Verification Enabled",
        "Evidence-Grounded Verification Active"
    ];

    return (
        <section className="ticker">
            <div className="ticker-tag">LATEST</div>

            <div className="ticker-wrapper">
                <div className="ticker-track">
                    {[...items, ...items].map((item, i) => (
                        <span key={i}>{item}</span>
                    ))}
                </div>
            </div>
        </section>
    );
}