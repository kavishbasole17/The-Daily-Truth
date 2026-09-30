// src/app/layout.jsx

import "@/styles/globals.css";

import "@/styles/navbar.css";
import "@/styles/landing.css";
import "@/styles/pages.css";
import "@/styles/chat.css";
import "@/styles/footer.css";

import ErrorBoundary from "@/components/ErrorBoundary";

export const metadata = {
    title: "The Daily Truth",
    description: "AI Powered Fake News Detection"
};

const THEME_INIT_SCRIPT = `
(function () {
    try {
        var stored = localStorage.getItem("theme");
        var theme = stored === "light" || stored === "dark" ? stored : "dark";
        document.documentElement.setAttribute("data-theme", theme);
    } catch (e) {
        document.documentElement.setAttribute("data-theme", "dark");
    }
})();
`;

export default function RootLayout({ children }) {
    return (
        <html lang="en">
        <head>
            <script dangerouslySetInnerHTML={{ __html: THEME_INIT_SCRIPT }} />
        </head>
        <body>
            <ErrorBoundary>
                {children}
            </ErrorBoundary>
        </body>
        </html>
    );
}