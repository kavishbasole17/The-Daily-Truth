// src/lib/shared.js — server-side loading for shared verdict pages (/v/<slug>).

// The Next.js server may reach the API on a different address than the
// browser does (e.g. a Docker service name), hence API_INTERNAL_URL.
export const API_INTERNAL =
    process.env.API_INTERNAL_URL || process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export const VERDICT_LABELS = {
    likely_real: "Likely real",
    likely_fake: "Likely misinformation",
    uncertain: "Uncertain",
};

/** @returns {Promise<{data?: object, notFound?: boolean, error?: string}>} */
export async function loadSharedCheck(slug) {
    try {
        const res = await fetch(`${API_INTERNAL}/share/${encodeURIComponent(slug)}`, {
            cache: "no-store",
        });
        if (res.status === 404) return { notFound: true };
        if (!res.ok) return { error: `The server returned ${res.status}.` };
        return { data: await res.json() };
    } catch {
        return { error: "The fact-checking server isn't reachable right now." };
    }
}

export function shareSummary(data) {
    const v = data.verdict;
    const label = VERDICT_LABELS[v.verdict] || "Fact-check";
    const subject = data.source_title || data.verdict.claim_verdicts?.[0]?.claim || data.text || "";
    const short = subject.length > 110 ? `${subject.slice(0, 107)}…` : subject;
    const counts = { entailment: 0, contradiction: 0 };
    (v.claim_verdicts || []).forEach(c => { if (c.verdict in counts) counts[c.verdict]++; });
    const n = v.claim_verdicts?.length || 0;
    const detail = n
        ? `${n} claim${n === 1 ? "" : "s"} checked: ${counts.entailment} supported, ${counts.contradiction} contradicted.`
        : "Checked against trusted news and reference sources.";
    return { label, subject: short, detail };
}
