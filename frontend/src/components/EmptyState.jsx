// src/components/EmptyState.jsx

export default function EmptyState({
                                       title = "Awaiting Analysis",
                                       message = "Submit an article to begin verification."
                                   }) {
    return (
        <div className="empty-state glass">
            <div className="state-icon">
                <svg viewBox="0 0 24 24" width="26" height="26" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M14 3v5a1 1 0 001 1h5" />
                    <path d="M17 21H7a2 2 0 01-2-2V5a2 2 0 012-2h7l5 5v11a2 2 0 01-2 2z" />
                    <path d="M9 13h6M9 17h4" />
                </svg>
            </div>
            <h2>{title}</h2>
            <p>{message}</p>
        </div>
    );
}
