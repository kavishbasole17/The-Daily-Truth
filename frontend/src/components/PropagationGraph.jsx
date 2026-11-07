"use client";

// Propagation graph for one claim's origin trail.
//
// Each source is a dot placed by publish date, in a lane for what it did with the
// claim: spread it, check it, or merely mention it (stance unknown). Thin links
// join every later source to the claim's first known appearance — the claimant
// named by a fact-check, else the earliest source spreading it. They show order in
// time only: the trail does not know who copied whom, and the caption says so.
//
// Colors come from --pg-spread / --pg-check (pages.css), validated as a pair for
// both themes; lanes and marker shapes carry the same distinction, so identity is
// never color alone. The list rendered under the graph is its table view.

import { useEffect, useMemo, useRef, useState } from "react";

const LANE_DEFS = [
    { key: "origin", label: "Origin", kinds: [] },
    { key: "spread", label: "Spreading", kinds: ["repeats"] },
    { key: "check", label: "Checking", kinds: ["debunks", "fact_check"] },
    { key: "mention", label: "Mentions", kinds: ["mentions"] },
];

const KIND_TEXT = {
    repeats: "repeats claim",
    debunks: "contradicts claim",
    fact_check: "fact-check",
    mentions: "mentions claim",
};

const MARGIN = { left: 96, right: 20, top: 18, bottom: 30 };
const NARROW = 420;            // below this the lane labels get a tighter column
const CHAR_W = 6.2;            // approx. width of an 11px label character
const LANE_H = 46;
const DAY = 86400000;
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

function parseDay(s) {
    if (!/^\d{4}-\d{2}-\d{2}$/.test(s || "")) return null;
    const t = Date.UTC(+s.slice(0, 4), +s.slice(5, 7) - 1, +s.slice(8, 10));
    return Number.isFinite(t) ? t : null;
}

function formatDay(t) {
    const d = new Date(t);
    return `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]} ${d.getUTCFullYear()}`;
}

// Calendar-aligned ticks: the coarsest unit that still gives a few ticks.
function timeTicks(t0, t1, maxTicks) {
    const candidates = [
        ["year", 10], ["year", 5], ["year", 2], ["year", 1],
        ["month", 6], ["month", 3], ["month", 1],
        ["day", 14], ["day", 7], ["day", 3], ["day", 2], ["day", 1],
    ];
    let best = [];
    for (const [unit, step] of candidates) {
        const ticks = [];
        const start = new Date(t0);
        let y = start.getUTCFullYear();
        let m = unit === "year" ? 0 : start.getUTCMonth();
        let d = unit === "day" ? start.getUTCDate() : 1;
        if (unit === "year") y = Math.ceil(y / step) * step;
        for (let guard = 0; guard < 400; guard++) {
            const t = Date.UTC(y, m, d);
            if (t > t1) break;
            if (t >= t0) ticks.push({ t, unit });
            if (unit === "year") y += step;
            else if (unit === "month") m += step;
            else d += step;
        }
        if (ticks.length > maxTicks) break;
        if (ticks.length >= 2) best = ticks;
    }
    if (best.length) return best;
    // Nothing calendar-aligned fits: label the two ends, in a unit the span needs.
    const span = t1 - t0;
    const unit = span > 2 * 365 * DAY ? "year" : span > 60 * DAY ? "month" : "day";
    return [{ t: t0, unit, anchor: "start" }, { t: t1, unit, anchor: "end" }];
}

function tickLabel({ t, unit }) {
    const d = new Date(t);
    if (unit === "year") return String(d.getUTCFullYear());
    if (unit === "month") return `${MONTHS[d.getUTCMonth()]} ${d.getUTCFullYear()}`;
    return `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]}`;
}

function Marker({ lane, size }) {
    const r = size / 2;
    if (lane === "check") {
        return <rect x={-r} y={-r} width={size} height={size} rx={2}
                     transform="rotate(45)" className="pg-mark pg-mark-check" />;
    }
    if (lane === "mention") return <circle r={r} className="pg-mark pg-mark-mention" />;
    if (lane === "origin") return <circle r={r + 1} className="pg-mark pg-mark-origin" />;
    return <circle r={r} className="pg-mark pg-mark-spread" />;
}

function LaneKey({ lane }) {
    return (
        <svg width="12" height="12" viewBox="-6 -6 12 12" aria-hidden="true">
            <Marker lane={lane} size={lane === "check" ? 7 : 8.5} />
        </svg>
    );
}

export default function PropagationGraph({ origin }) {
    const wrapRef = useRef(null);
    const [width, setWidth] = useState(600);
    const [active, setActive] = useState(null);

    useEffect(() => {
        const el = wrapRef.current;
        if (!el || typeof ResizeObserver === "undefined") return undefined;
        const ro = new ResizeObserver(([entry]) => setWidth(Math.max(160, entry.contentRect.width)));
        ro.observe(el);
        return () => ro.disconnect();
    }, []);

    const model = useMemo(() => {
        const points = [];
        const claimDate = parseDay(origin.claim_date);
        if (origin.claimant && claimDate !== null) {
            points.push({
                lane: "origin", t: claimDate, domain: origin.claimant,
                kindText: "claimed by", trust: null, foundBy: "fact_check",
                via: origin.claimant_source,
            });
        }
        let undated = 0;
        for (const e of origin.timeline || []) {
            const t = parseDay(e.date);
            const lane = LANE_DEFS.find(l => l.kinds.includes(e.kind));
            if (!lane) continue;
            if (t === null) { undated += 1; continue; }
            points.push({
                lane: lane.key, t, domain: e.domain || "unknown source",
                kindText: e.kind === "fact_check" && e.label
                    ? `fact-check: ${e.label}` : KIND_TEXT[e.kind] || e.kind,
                trust: e.trust, foundBy: e.found_by, url: e.url,
            });
        }
        const lanes = LANE_DEFS.filter(l => points.some(p => p.lane === l.key));

        // The first known appearance: the claimant if a fact-check names one,
        // else the earliest dated source spreading the claim. None means no link.
        let anchor = points.find(p => p.lane === "origin") || null;
        if (!anchor) {
            const spread = points.filter(p => p.lane === "spread").sort((a, b) => a.t - b.t);
            anchor = spread[0] || null;
        }
        return { points, lanes, anchor, undated };
    }, [origin]);

    const { points, lanes, anchor, undated } = model;
    if (points.length < 2) return null;

    const left = width < NARROW ? 66 : MARGIN.left;
    const right = width < NARROW ? 10 : MARGIN.right;
    const plotW = width - left - right;
    const height = MARGIN.top + lanes.length * LANE_H + MARGIN.bottom;
    let t0 = Math.min(...points.map(p => p.t));
    let t1 = Math.max(...points.map(p => p.t));
    // Short trails keep their own scale (a week stays a week wide); only a
    // single-day trail is widened so its dots don't sit on the axis ends.
    if (t1 - t0 < 2 * DAY) { t0 -= DAY; t1 += DAY; }
    const pad = (t1 - t0) * 0.05;
    t0 -= pad; t1 += pad;
    const x = t => left + ((t - t0) / (t1 - t0)) * plotW;
    const laneY = key => MARGIN.top + lanes.findIndex(l => l.key === key) * LANE_H + LANE_H / 2;

    // Same-day sources in a lane would sit on top of each other: fan them out.
    const placed = [];
    const offsets = [0, -11, 11, -20, 20];
    for (const lane of lanes) {
        const last = offsets.map(() => -Infinity);
        points.filter(p => p.lane === lane.key).sort((a, b) => a.t - b.t).forEach(p => {
            const px = x(p.t);
            const slot = Math.max(0, offsets.findIndex((_, i) => px - last[i] >= 14));
            last[slot] = px;
            placed.push({ ...p, px, py: laneY(lane.key) + offsets[slot] });
        });
    }
    const anchorPlaced = anchor ? placed.find(p => p.t === anchor.t && p.domain === anchor.domain
                                                  && p.lane === anchor.lane) : null;

    const ticks = timeTicks(t0, t1, Math.max(2, Math.floor(plotW / 80)));
    const axisY = MARGIN.top + lanes.length * LANE_H;
    const hovered = active !== null ? placed[active] : null;

    const anchorLabel = anchorPlaced
        ? (anchorPlaced.lane === "origin" ? `claimed by ${anchorPlaced.domain}` : "earliest found")
        : null;
    // Above the anchor, starting at it, slid left only as far as needed to stay
    // inside the plot; very narrow plots shorten it.
    let shownLabel = anchorLabel || "";
    const room = width - left;
    if (shownLabel.length * CHAR_W > room) {
        shownLabel = shownLabel.slice(0, Math.max(4, Math.floor(room / CHAR_W) - 1)) + "…";
    }
    const labelX = anchorPlaced
        ? Math.max(left, Math.min(anchorPlaced.px - 6, width - shownLabel.length * CHAR_W - 2))
        : 0;

    const nSpread = placed.filter(p => p.lane === "spread").length;
    const nCheck = placed.filter(p => p.lane === "check").length;
    const summary = `Propagation graph: ${placed.length} dated sources from ${formatDay(
        Math.min(...points.map(p => p.t)))} to ${formatDay(Math.max(...points.map(p => p.t)))}; `
        + `${nSpread} spreading the claim, ${nCheck} checking it.`;

    return (
        <figure className="pg" ref={wrapRef}>
            <div className="pg-plot">
                <svg width={width} height={height} role="img" aria-label={summary}>
                    {lanes.map(l => (
                        <g key={l.key}>
                            <line x1={left} x2={width - right}
                                  y1={laneY(l.key)} y2={laneY(l.key)} className="pg-lane" />
                            <text x={width < NARROW ? 0 : 12} y={laneY(l.key)} dy="0.35em"
                                  className="pg-lane-label">{l.label}</text>
                        </g>
                    ))}

                    <line x1={left} x2={width - right} y1={axisY} y2={axisY} className="pg-axis" />
                    {ticks.map(tk => (
                        <g key={tk.t} transform={`translate(${x(tk.t)},${axisY})`}>
                            <line y2={4} className="pg-axis" />
                            <text y={17} textAnchor={tk.anchor || "middle"} className="pg-tick">{tickLabel(tk)}</text>
                        </g>
                    ))}

                    {anchorPlaced && placed.filter(p => p !== anchorPlaced && p.t >= anchorPlaced.t).map((p, i) => {
                        const mx = (anchorPlaced.px + p.px) / 2;
                        return (
                            <path key={i} className={`pg-link pg-link-${p.lane}`}
                                  d={`M${anchorPlaced.px},${anchorPlaced.py} C${mx},${anchorPlaced.py} ${mx},${p.py} ${p.px},${p.py}`} />
                        );
                    })}

                    {placed.map((p, i) => (
                        <g key={i} transform={`translate(${p.px},${p.py})`}
                           className={`pg-node${active === i ? " is-active" : ""}`}
                           tabIndex={0} role="img"
                           aria-label={`${formatDay(p.t)}, ${p.domain}, ${p.kindText}`}
                           onPointerEnter={() => setActive(i)} onPointerLeave={() => setActive(null)}
                           onFocus={() => setActive(i)} onBlur={() => setActive(null)}>
                            <circle r={12} className="pg-hit" />
                            <Marker lane={p.lane} size={p.lane === "check" ? 10 : 12} />
                        </g>
                    ))}

                    {anchorPlaced && (
                        <text x={labelX} y={anchorPlaced.py - 11} className="pg-anchor-label">
                            <title>{anchorLabel}</title>
                            {shownLabel}
                        </text>
                    )}
                </svg>

                {hovered && (
                    <div className="pg-tip" role="status"
                         style={{
                             left: Math.min(Math.max(hovered.px, 90), width - 90),
                             top: hovered.py - 12,
                         }}>
                        <strong>{formatDay(hovered.t)}</strong>
                        <span>{hovered.domain}</span>
                        <span className="muted">
                            {hovered.kindText}
                            {hovered.trust !== null && hovered.trust !== undefined
                                ? ` · trust ${Number(hovered.trust).toFixed(2)}` : ""}
                            {hovered.foundBy === "wider_search" ? " · wider search" : ""}
                            {hovered.lane === "origin" && hovered.via ? ` · per ${hovered.via}` : ""}
                        </span>
                    </div>
                )}
            </div>

            <figcaption className="pg-caption">
                <span className="pg-keys">
                    {lanes.map(l => (
                        <span key={l.key} className="pg-key"><LaneKey lane={l.key} />{l.label}</span>
                    ))}
                </span>
                <span className="muted">
                    {anchorPlaced
                        ? "Lines join each source to the first known appearance. They show order in time, not who copied whom."
                        : "No dated source spreading the claim was found, so there is no starting point to draw from."}
                    {undated > 0 && ` ${undated} undated source${undated === 1 ? " is" : "s are"} listed below but not plotted.`}
                </span>
            </figcaption>
        </figure>
    );
}
