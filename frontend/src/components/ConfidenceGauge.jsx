// src/components/ConfidenceGauge.jsx

export default function ConfidenceGauge({ value = 0 }) {
    return (
        <div
            className="gauge"
            style={{
                background: `conic-gradient(var(--yellow) ${value * 3.6}deg, rgba(255,255,255,.08) 0)`
            }}
        >
            <div className="gauge-inner">
                <span>{value}%</span>
            </div>
        </div>
    );
}