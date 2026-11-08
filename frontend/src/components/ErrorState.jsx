// src/components/ErrorState.jsx

export default function ErrorState({
                                       message = "Something went wrong."
                                   }) {
    return (
        <div className="error-state glass">
            <div className="state-icon">
                <svg viewBox="0 0 24 24" width="26" height="26" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M12 9v4M12 17h.01" />
                    <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
                </svg>
            </div>
            <h2>ERROR</h2>
            <p>{typeof message === "string" && message ? message : "Something went wrong."}</p>
        </div>
    );
}
