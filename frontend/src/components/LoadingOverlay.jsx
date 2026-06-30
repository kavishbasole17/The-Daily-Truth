// src/components/LoadingOverlay.jsx

// src/components/LoadingOverlay.jsx

export default function LoadingOverlay({ show, message = "Analyzing..." }) {
    if (!show) return null;

    return (
        <div className="loading-overlay">
            <div className="loader" />
            <p>{message}</p>
        </div>
    );
}