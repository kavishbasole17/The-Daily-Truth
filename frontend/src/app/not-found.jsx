// src/app/not-found.jsx
//
// Next.js renders this automatically for any unmatched route, and whenever
// a page calls notFound(). Keeps the same newspaper/chat visual language as
// the rest of the site rather than falling back to the framework default.

import Link from "next/link";

import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";

import "@/styles/notfound.css";

export const metadata = {
    title: "404 — Page Not Found | The Daily Truth",
};

const QUICK_LINKS = [
    ["Paper Trail", "/verify"],
    ["Single Check", "/single"],
    ["Consensus", "/consensus"],
    ["Batch Analysis", "/batch"],
    ["Models", "/models"],
];

export default function NotFound() {
    return (
        <>
            <Navbar />

            <section className="notfound-page">
                <div className="container">
                    <div className="notfound-content">
                        <span className="page-eyebrow">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M12 9v4M12 17h.01" />
                                <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
                            </svg>
                            Breaking: Page Missing
                        </span>

                        <h1 className="notfound-numeral">4<span className="accent">0</span>4</h1>

                        <div className="notfound-rule" />

                        <h2>This story didn&rsquo;t run in today&rsquo;s edition</h2>
                        <p>
                            The page you&rsquo;re looking for doesn&rsquo;t exist, was moved, or the
                            link is out of date. Head back to the front page, or jump straight
                            into one of the verification tools below.
                        </p>

                        <div className="notfound-actions">
                            <Link href="/" className="newspaper-btn">
                                Back to Homepage
                            </Link>
                        </div>

                        <div className="notfound-links">
                            {QUICK_LINKS.map(([label, href]) => (
                                <Link key={href} href={href} className="chat-suggestion">
                                    {label}
                                </Link>
                            ))}
                        </div>
                    </div>
                </div>
            </section>

            <Footer />
        </>
    );
}
