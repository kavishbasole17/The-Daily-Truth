// src/components/ErrorState.jsx

export default function ErrorState({
                                       message = "Something went wrong."
                                   }) {
    return (
        <div className="error-state glass">
            <h2>ERROR</h2>
            <p>{message}</p>
        </div>
    );
}