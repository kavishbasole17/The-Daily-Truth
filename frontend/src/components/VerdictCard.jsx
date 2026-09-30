// src/components/VerdictCard.jsx

export default function VerdictCard({ prediction, confidence }) {
    const fake = prediction?.toLowerCase() === "fake";

    return (
        <div className={`verdict-card glass ${fake ? "fake-card" : "real-card"}`}>
            <div className="verdict-icon">
                {fake ? (
                    <svg viewBox="0 0 24 24" width="26" height="26" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <circle cx="12" cy="12" r="9" />
                        <path d="M9.5 9.5l5 5M14.5 9.5l-5 5" />
                    </svg>
                ) : (
                    <svg viewBox="0 0 24 24" width="26" height="26" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <circle cx="12" cy="12" r="9" />
                        <path d="M8.5 12.5l2.5 2.5 5-5" />
                    </svg>
                )}
            </div>
            <div>
                <h2>{fake ? "LIKELY MISINFORMATION" : "VERIFIED"}</h2>
                <p>{confidence}% Confidence</p>
            </div>
        </div>
    );
}
