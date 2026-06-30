// src/components/VerdictCard.jsx

export default function VerdictCard({ prediction, confidence }) {
    const fake = prediction?.toLowerCase() === "fake";

    return (
        <div className={`verdict-card glass ${fake ? "fake-card" : "real-card"}`}>
            <h2>{fake ? "LIKELY MISINFORMATION" : "VERIFIED"}</h2>
            <p>{confidence}% Confidence</p>
        </div>
    );
}