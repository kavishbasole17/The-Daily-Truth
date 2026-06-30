// src/components/Footer.jsx

import Link from "next/link";

export default function Footer() {
    return (
        <footer className="footer">
            <div className="container footer-content">
                <div>
                    <h2>THE DAILY TRUTH</h2>
                    <p>AI Powered Fake News Detection Platform</p>
                </div>

                <div className="footer-links">
                    <Link href="/">Home</Link>
                    <Link href="/single">Single Check</Link>
                    <Link href="/consensus">Consensus</Link>
                    <Link href="/batch">Batch Analysis</Link>
                    <Link href="/models">Models</Link>
                </div>
            </div>

            <div className="footer-bottom">
                © {new Date().getFullYear()} The Daily Truth
            </div>
        </footer>
    );
}