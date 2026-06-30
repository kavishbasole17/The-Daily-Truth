// src/components/EmptyState.jsx

export default function EmptyState({
                                       title = "Awaiting Analysis",
                                       message = "Submit an article to begin verification."
                                   }) {
    return (
        <div className="empty-state glass">
            <h2>{title}</h2>
            <p>{message}</p>
        </div>
    );
}