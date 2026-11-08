// src/app/v/[slug]/page.jsx
//
// A shared fact-check, readable by anyone with the link. Rendered on the
// server so link previews (WhatsApp, X, Slack…) get a real title, description
// and the image card from ./opengraph-image.jsx.

import { notFound } from "next/navigation";

import SharedCheckView from "./SharedCheckView";
import { loadSharedCheck, shareSummary } from "@/lib/shared";

export const dynamic = "force-dynamic";

export async function generateMetadata({ params }) {
    const { data } = await loadSharedCheck(params.slug);
    if (!data) return { title: "Shared fact-check · The Daily Truth" };
    const { label, subject, detail } = shareSummary(data);
    const title = `${label}: ${subject}`;
    return {
        title: `${title} · The Daily Truth`,
        description: detail,
        openGraph: { title, description: detail, type: "article", siteName: "The Daily Truth" },
        twitter: { card: "summary_large_image", title, description: detail },
        robots: { index: false },
    };
}

export default async function SharedCheckPage({ params }) {
    const { data, notFound: missing, error } = await loadSharedCheck(params.slug);
    if (missing) notFound();
    return <SharedCheckView data={data} error={error} />;
}
