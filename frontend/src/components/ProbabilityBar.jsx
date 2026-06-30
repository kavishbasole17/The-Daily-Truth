// src/components/ProbabilityBar.jsx

export default function ProbabilityBar({ real = 0, fake = 0 }) {
    return (
        <div className="probability-card glass">
            <div className="prob-row">
                <span>REAL</span>
                <span>{real}%</span>
            </div>

            <div className="prob-track">
                <div
                    className="prob-real"
                    style={{ width: `${real}%` }}
                />
            </div>

            <div className="prob-row">
                <span>FAKE</span>
                <span>{fake}%</span>
            </div>

            <div className="prob-track">
                <div
                    className="prob-fake"
                    style={{ width: `${fake}%` }}
                />
            </div>
        </div>
    );
}