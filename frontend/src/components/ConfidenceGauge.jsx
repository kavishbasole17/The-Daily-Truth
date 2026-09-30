// src/components/ConfidenceGauge.jsx

export default function ConfidenceGauge({ value = 0 }) {
    const color =
        value >= 70 ? "var(--success)" :
        value >= 40 ? "var(--yellow)" :
        "var(--red)";

    return (
        <div className="gauge-card glass">
            <div
                className="gauge"
                style={{
                    background: `conic-gradient(${color} ${value * 3.6}deg, rgba(255,255,255,.08) 0)`
                }}
            >
                <div className="gauge-inner">
                    <span>{value}%</span>
                </div>
            </div>
            <span className="gauge-card-label">Confidence Score</span>
        </div>
    );
}
