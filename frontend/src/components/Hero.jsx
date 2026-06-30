// src/components/Hero.jsx

"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import "@/styles/hero.css";

export default function Hero() {
    return (
        <section className="hero">
            <div className="container hero-grid">
                <motion.div
                    initial={{ opacity: 0, x: -40 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: .6 }}
                    className="hero-left"
                >
                    <p className="hero-kicker">INVESTIGATIVE AI JOURNALISM</p>

                    <h1>
                        THE DAILY
                        <br />
                        TRUTH
                    </h1>

                    <p className="hero-desc">
                        Detect misinformation, verify credibility, and compare AI model
                        consensus before sharing any article.
                    </p>

                    <div className="hero-actions">
                        <Link href="/single" className="hero-btn primary">
                            Verify Article
                        </Link>

                        <Link href="/consensus" className="hero-btn secondary">
                            Consensus Check
                        </Link>
                    </div>

                    <div className="hero-stats">
                        <div>
                            <h3>5+</h3>
                            <span>Models</span>
                        </div>

                        <div>
                            <h3>100+</h3>
                            <span>Batch Size</span>
                        </div>

                        <div>
                            <h3>&lt;200ms</h3>
                            <span>Inference</span>
                        </div>
                    </div>
                </motion.div>

                <motion.div
                    initial={{ opacity: 0, x: 40 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: .6 }}
                    className="hero-right"
                >
                    <div className="newspaper-stack">
                        <div className="paper paper-1">
                            <span>HEADLINES</span>
                            <div />
                            <div />
                            <div />
                            <div />
                        </div>

                        <div className="paper paper-2">
                            <span>FACT CHECK</span>
                            <div />
                            <div />
                            <div />
                            <div />
                        </div>

                        <div className="paper paper-3">
                            <span>VERIFIED</span>
                            <div />
                            <div />
                            <div />
                            <div />
                        </div>
                    </div>
                </motion.div>
            </div>
        </section>
    );
}