// src/app/layout.jsx

import "@/styles/globals.css";

import "@/styles/navbar.css";
import "@/styles/landing.css";
import "@/styles/pages.css";
import "@/styles/chat.css";
import "@/styles/footer.css";
import "@/styles/auth.css";
import "@/styles/features.css";

import ErrorBoundary from "@/components/ErrorBoundary";
import { AuthProvider } from "@/context/AuthContext";

export const metadata = {
    // Absolute base for OpenGraph image URLs on shared verdict pages.
    metadataBase: new URL(process.env.NEXT_PUBLIC_SITE_URL || "http://localhost:3000"),
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
                <AuthProvider>
                    {children}
                </AuthProvider>
            </ErrorBoundary>
        </body>
        </html>
    );
}