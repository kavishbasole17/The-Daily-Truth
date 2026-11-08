// src/app/v/[slug]/SharedCheckView.jsx

"use client";

import Link from "next/link";

import Navbar from "@/components/Navbar";
import VerdictReport, { ClaimLegend, HighlightedText } from "@/components/VerdictReport";

import "@/styles/pages.css";
import "@/styles/chat.css";

function safeHost(url) {
    try {
        return new URL(url).hostname.replace(/^www\./, "");
    } catch {
        return url;
    }
}

export default function SharedCheckView({ data, error }) {
    return (
        <>
            <Navbar />
            <section className="shared-page">
                <div className="shared-shell">
                    <p className="shared-kicker">Shared fact-check</p>

                    {error ? (
                        <div className="chat-error-inline">{error} Try again in a moment.</div>
                    ) : (
                        <>
                            <div className="shared-source">
                                {data.kind === "url" && data.url ? (
                                    <>
                                        <h1>{data.source_title || safeHost(data.url)}</h1>
                                        <a href={data.url} target="_blank" rel="noopener noreferrer nofollow">
                                            {safeHost(data.url)} ↗
                                        </a>
                                    </>
                                ) : (
                                    <h1>{data.kind === "image" ? "Text from a screenshot" : "Text that was checked"}</h1>
                                )}
                                <span className="muted">
                                    Checked {new Date(data.submitted_at.endsWith("Z") ? data.submitted_at : `${data.submitted_at}Z`)
                                        .toLocaleDateString([], { day: "numeric", month: "long", year: "numeric" })}
                                </span>
                            </div>

                            {data.text && (
                                <div className="shared-text">
                                    <HighlightedText
                                        text={data.text}
                                        claimVerdicts={data.verdict.claim_verdicts}
                                        anchorPrefix="shared"
                                    />
                                    {data.verdict.claim_verdicts.length > 0 && <ClaimLegend />}
                                </div>
                            )}

                            <div className="shared-report">
                                <VerdictReport result={data} anchorPrefix="shared" />
                            </div>
                        </>
                    )}

                    <div className="shared-cta">
                        <div>
                            <strong>Got something you&rsquo;re not sure about?</strong>
                            <span>Paste an article, a link or a screenshot and see the receipts.</span>
                        </div>
                        <Link href="/verify" className="chat-gate-primary">Check it yourself</Link>
                    </div>

                    <p className="shared-disclaimer muted">
                        Verdicts are produced automatically from public sources and can be wrong.
                        &ldquo;Could not verify&rdquo; is not the same as false.
                    </p>
                </div>
            </section>
        </>
    );
}
