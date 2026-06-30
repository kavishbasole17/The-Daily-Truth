// src/app/layout.jsx

import "@/styles/globals.css";

import "@/styles/navbar.css";
import "@/styles/ticker.css";
import "@/styles/hero.css";
import "@/styles/landing.css";
import "@/styles/pages.css";
import "@/styles/footer.css";

import ErrorBoundary from "@/components/ErrorBoundary";

export const metadata = {
    title: "The Daily Truth",
    description: "AI Powered Fake News Detection"
};

export default function RootLayout({ children }) {
    return (
        <html lang="en">
        <body>
            <ErrorBoundary>
                {children}
            </ErrorBoundary>
        </body>
        </html>
    );
}