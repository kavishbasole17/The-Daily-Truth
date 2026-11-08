// src/app/api/news/route.js
//
// Pulls live headlines directly from Times of India and Hindustan Times RSS
// feeds (no API key required) instead of NewsAPI, per category.

import { XMLParser } from "fast-xml-parser";

const FEEDS = {
    all: [
        { source: "Times of India", url: "https://timesofindia.indiatimes.com/rssfeedstopstories.cms" },
        { source: "Hindustan Times", url: "https://www.hindustantimes.com/feeds/rss/india-news/rssfeed.xml" }
    ],
    india: [
        { source: "Times of India", url: "https://timesofindia.indiatimes.com/rssfeeds/-2128936835.cms" },
        { source: "Hindustan Times", url: "https://www.hindustantimes.com/feeds/rss/india-news/rssfeed.xml" }
    ],
    world: [
        { source: "Times of India", url: "https://timesofindia.indiatimes.com/rssfeeds/296589292.cms" },
        { source: "Hindustan Times", url: "https://www.hindustantimes.com/feeds/rss/world-news/rssfeed.xml" }
    ],
    sports: [
        { source: "Times of India", url: "https://timesofindia.indiatimes.com/rssfeeds/4719148.cms" },
        { source: "Hindustan Times", url: "https://www.hindustantimes.com/feeds/rss/sports/rssfeed.xml" }
    ],
    business: [
        { source: "Times of India", url: "https://timesofindia.indiatimes.com/rssfeeds/1898055.cms" },
        { source: "Hindustan Times", url: "https://www.hindustantimes.com/feeds/rss/business/rssfeed.xml" }
    ],
    entertainment: [
        { source: "Times of India", url: "https://timesofindia.indiatimes.com/rssfeeds/1081479906.cms" },
        { source: "Hindustan Times", url: "https://www.hindustantimes.com/feeds/rss/entertainment/rssfeed.xml" }
    ],
    technology: [
        { source: "Times of India", url: "https://timesofindia.indiatimes.com/rssfeeds/66949542.cms" },
        { source: "Hindustan Times", url: "https://www.hindustantimes.com/feeds/rss/technology/rssfeed.xml" }
    ]
};

const parser = new XMLParser({ ignoreAttributes: false, attributeNamePrefix: "@_" });

const HTML_ENTITIES = {
    "&nbsp;": " ", "&amp;": "&", "&quot;": "\"", "&#39;": "'", "&apos;": "'",
    "&lt;": "<", "&gt;": ">", "&rsquo;": "’", "&lsquo;": "‘",
    "&rdquo;": "”", "&ldquo;": "“", "&mdash;": "—", "&ndash;": "–"
};

function cleanText(value) {
    if (typeof value !== "string") return "";
    const withoutTags = value.replace(/<[^>]*>/g, " ");
    const decoded = withoutTags.replace(/&[a-z#0-9]+;/gi, (m) => HTML_ENTITIES[m] ?? m);
    return decoded.replace(/\s+/g, " ").trim();
}

// Only http(s) links reach the page: a `javascript:` URL in a feed would
// otherwise run in the reader's browser when they click the story.
function safeUrl(value) {
    if (typeof value !== "string") return null;
    try {
        const u = new URL(value.trim());
        return u.protocol === "http:" || u.protocol === "https:" ? u.href : null;
    } catch {
        return null;
    }
}

// One malformed date must not throw (it would discard the whole feed).
function toIso(value) {
    if (!value) return null;
    const t = Date.parse(value);
    return Number.isNaN(t) ? null : new Date(t).toISOString();
}

function extractImage(item) {
    const media = item.enclosure ?? item["media:content"] ?? item["media:thumbnail"];
    if (!media) return null;
    const node = Array.isArray(media) ? media[0] : media;
    return safeUrl(node?.["@_url"]);
}

async function fetchFeed({ source, url }, category) {
    try {
        const res = await fetch(url, {
            headers: { "User-Agent": "Mozilla/5.0 (compatible; DailyTruthBot/1.0)" },
            next: { revalidate: 300 }
        });
        if (!res.ok) return [];

        const xml = await res.text();
        const items = parser.parse(xml)?.rss?.channel?.item;
        if (!items) return [];

        const list = Array.isArray(items) ? items : [items];
        return list
            .map((item) => {
                const title = cleanText(item.title);
                const link = safeUrl(cleanText(item.link));
                const pubDate = cleanText(item.pubDate);
                return {
                    title,
                    description: cleanText(item.description),
                    url: link,
                    image: extractImage(item),
                    source,
                    category,
                    publishedAt: toIso(pubDate)
                };
            })
            .filter((article) => article.title && article.url);
    } catch (error) {
        console.error(`Failed to fetch feed ${url}:`, error.message);
        return [];
    }
}

export async function GET(request) {
    const { searchParams } = new URL(request.url);
    const category = (searchParams.get("category") || "all").toLowerCase();
    const feeds = FEEDS[category] || FEEDS.all;

    const results = await Promise.all(feeds.map((feed) => fetchFeed(feed, category)));

    const seen = new Set();
    const articles = results
        .flat()
        .filter((article) => {
            const key = article.title.toLowerCase();
            if (seen.has(key)) return false;
            seen.add(key);
            return true;
        })
        .sort((a, b) => new Date(b.publishedAt || 0) - new Date(a.publishedAt || 0))
        .slice(0, 30);

    if (articles.length === 0) {
        return Response.json({ error: "No articles available", articles: [] }, { status: 502 });
    }

    return Response.json({ articles, category });
}
