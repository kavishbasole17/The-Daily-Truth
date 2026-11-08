// src/app/v/[slug]/opengraph-image.jsx
//
// The image card shown when a shared verdict link is pasted into WhatsApp,
// X, Slack, etc. Generated on demand from the shared check.

import { ImageResponse } from "next/og";

import { loadSharedCheck, shareSummary } from "@/lib/shared";

// Edge runtime: generates fast near the user, and sidesteps a Next 14 bug where
// the Node runtime can't locate @vercel/og's bundled font on Windows.
export const runtime = "edge";
export const alt = "Fact-check verdict from The Daily Truth";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

const COLORS = {
    likely_real: "#22c55e",
    likely_fake: "#e10600",
    uncertain: "#eab308",
};

export default async function Image({ params }) {
    const { data } = await loadSharedCheck(params.slug);
    const verdict = data?.verdict?.verdict || "uncertain";
    const accent = COLORS[verdict] || COLORS.uncertain;
    const { label, subject, detail } = data
        ? shareSummary(data)
        : { label: "Fact-check", subject: "This shared check is no longer available.", detail: "" };

    return new ImageResponse(
        (
            <div
                style={{
                    width: "100%", height: "100%", display: "flex", flexDirection: "column",
                    justifyContent: "space-between", padding: "64px 72px",
                    background: "#0c0c0c", color: "#f2f2f2", borderLeft: `24px solid ${accent}`,
                    fontFamily: "sans-serif",
                }}
            >
                <div style={{ display: "flex", alignItems: "center", gap: 16, fontSize: 30, color: "#9a9a9a" }}>
                    <div
                        style={{
                            width: 52, height: 52, borderRadius: 12, background: "#e10600", color: "#fff",
                            display: "flex", alignItems: "center", justifyContent: "center",
                            fontSize: 34, fontWeight: 800,
                        }}
                    >
                        T
                    </div>
                    The Daily Truth · Paper Trail
                </div>

                <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
                    <div
                        style={{
                            display: "flex", alignSelf: "flex-start", padding: "10px 22px", borderRadius: 999,
                            background: accent, color: "#0c0c0c", fontSize: 34, fontWeight: 800,
                            textTransform: "uppercase", letterSpacing: 1,
                        }}
                    >
                        {label}
                    </div>
                    <div style={{ display: "flex", fontSize: 54, fontWeight: 700, lineHeight: 1.2 }}>
                        &ldquo;{subject}&rdquo;
                    </div>
                </div>

                <div style={{ display: "flex", fontSize: 28, color: "#9a9a9a" }}>{detail}</div>
            </div>
        ),
        size
    );
}
