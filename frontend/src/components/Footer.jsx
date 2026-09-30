// src/components/Footer.jsx

import Link from "next/link";
import Logo from "@/components/Logo";

const TEAM = ["Kavish Basole", "Nishik Varma", "Sashank Solasa", "Virat Tallapally"];

export default function Footer() {
    return (
        <footer className="footer">
            <div className="container footer-content">
                <div>
                    <div className="footer-brand">
                        <Logo size={30} />
                        <h2>
                            The Daily Truth<span className="logo-dot">.</span>
                        </h2>
                    </div>
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

            <div className="container footer-credits">
                <span className="footer-credits-label">Project Team</span>
                <p>{TEAM.join(" · ")}</p>
                <p className="footer-credits-guide">
                    Under the guidance of <strong>Prof. Ganesh Reddy Karri</strong> — Fall Semester 2026–27
                </p>
            </div>

            <div className="footer-bottom">
                © {new Date().getFullYear()} The Daily Truth
            </div>
        </footer>
    );
}