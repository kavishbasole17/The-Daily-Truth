// src/lib/highlight.js
//
// Finds where each extracted claim sits in the text that was checked, so the
// UI can underline it. Claims are often lightly rewritten by the extractor
// (compound sentences split, whitespace normalised, translation), so an exact
// match is tried first and then the sentence with the best word overlap.

const STOP = new Set(
    "a an the is are was were be been of to in on for and or but with that this it its as at by from has have had will would".split(" ")
);

function words(s) {
    return (s.toLowerCase().match(/[\p{L}\p{N}]+/gu) || []).filter(w => w.length > 1 && !STOP.has(w));
}

function sentences(text) {
    const out = [];
    const re = /[^.!?।\n]+[.!?।]*/gu;
    let m;
    while ((m = re.exec(text))) {
        const raw = m[0];
        const lead = raw.length - raw.trimStart().length;
        const body = raw.trim();
        if (body) out.push({ start: m.index + lead, end: m.index + lead + body.length, text: body });
    }
    return out;
}

function findRange(text, lower, claim, sents) {
    const c = claim.trim().replace(/[.!?]+$/, "");
    if (c.length >= 8) {
        const idx = lower.indexOf(c.toLowerCase());
        if (idx >= 0) return { start: idx, end: idx + c.length };
    }
    const cw = new Set(words(claim));
    if (cw.size === 0) return null;
    let best = null;
    let bestScore = 0;
    for (const s of sents) {
        const sw = new Set(words(s.text));
        let hit = 0;
        cw.forEach(w => { if (sw.has(w)) hit++; });
        const score = hit / cw.size;
        if (score > bestScore) {
            bestScore = score;
            best = s;
        }
    }
    return bestScore >= 0.6 ? { start: best.start, end: best.end } : null;
}

/**
 * Split `text` into segments: [{ text, claimIndex | null }].
 * Claims that can't be located are simply not highlighted.
 */
export function highlightSegments(text, claims) {
    if (!text) return [];
    if (!claims?.length) return [{ text, claimIndex: null }];

    const lower = text.toLowerCase();
    const sents = sentences(text);
    const ranges = [];
    claims.forEach((claim, i) => {
        const r = claim && findRange(text, lower, claim, sents);
        if (!r) return;
        if (ranges.some(o => r.start < o.end && o.start < r.end)) return; // no overlaps
        ranges.push({ ...r, claimIndex: i });
    });
    ranges.sort((a, b) => a.start - b.start);

    const out = [];
    let pos = 0;
    for (const r of ranges) {
        if (r.start > pos) out.push({ text: text.slice(pos, r.start), claimIndex: null });
        out.push({ text: text.slice(r.start, r.end), claimIndex: r.claimIndex });
        pos = r.end;
    }
    if (pos < text.length) out.push({ text: text.slice(pos), claimIndex: null });
    return out;
}
