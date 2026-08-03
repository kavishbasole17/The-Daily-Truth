// src/components/ThemeToggle.jsx

"use client";

import { useEffect, useState } from "react";

export default function ThemeToggle() {
    const [theme, setTheme] = useState(null);

    useEffect(() => {
        setTheme(document.documentElement.getAttribute("data-theme") || "dark");
    }, []);

    function toggle() {
        const next = theme === "light" ? "dark" : "light";
        document.documentElement.setAttribute("data-theme", next);
        localStorage.setItem("theme", next);
        setTheme(next);
    }

    return (
        <button
            className="theme-toggle"
            onClick={toggle}
            aria-label="Toggle light and dark mode"
            title="Toggle theme"
        >
            {theme === "light" ? "🌙" : "☀️"}
        </button>
    );
}
