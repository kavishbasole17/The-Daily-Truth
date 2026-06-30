// src/components/NewspaperButton.jsx

export default function NewspaperButton({
                                            children,
                                            onClick,
                                            disabled,
                                            type = "button"
                                        }) {
    return (
        <button
            type={type}
            onClick={onClick}
            disabled={disabled}
            className="newspaper-btn"
        >
            {children}
        </button>
    );
}