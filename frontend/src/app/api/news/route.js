// src/app/api/news/route.js

// Some publishers in NewsAPI's aggregated feed serve their own content with
// mismatched encoding upstream (valid UTF-8 bytes that got decoded as
// Latin-1/Windows-1252 somewhere in their own pipeline before reaching
// NewsAPI), which shows up as mojibake like "donâ€™t" instead of "don't".
// Repair it only when the tell-tale byte pattern is present, and only keep
// the repair if reinterpreting the bytes as UTF-8 actually succeeds —
// otherwise leave the original text untouched rather than risk mangling it
// further.
function repairMojibake(text) {
    if (typeof text !== "string" || !text || !/[ÂÃâ€]/.test(text)) return text;
    if ([...text].some((ch) => ch.charCodeAt(0) > 255)) return text;
    try {
        const bytes = Uint8Array.from([...text].map((ch) => ch.charCodeAt(0)));
        return new TextDecoder("utf-8", { fatal: true }).decode(bytes);
    } catch {
        return text;
    }
}

function repairArticle(article) {
    return {
        ...article,
        title: repairMojibake(article.title),
        description: repairMojibake(article.description),
        author: repairMojibake(article.author),
        content: repairMojibake(article.content),
        source: article.source
            ? { ...article.source, name: repairMojibake(article.source.name) }
            : article.source,
    };
}

export async function GET(request) {
    try {
        const apiKey = process.env.NEXT_PUBLIC_NEWS_API_KEY;
        const apiUrl = process.env.NEXT_PUBLIC_NEWS_API_URL;

        if (!apiKey || apiKey === "your_newsapi_key_here") {
            return Response.json(
                { error: "News API key not configured" },
                { status: 400 }
            );
        }

        const response = await fetch(
            `${apiUrl}/top-headlines?category=general&language=en&apiKey=${apiKey}&pageSize=10`,
            {
                headers: {
                    "User-Agent": "Mozilla/5.0 (compatible; NewsCarousel/1.0)"
                }
            }
        );

        if (!response.ok) {
            const error = await response.json();
            return Response.json(
                {
                    error: error.message || "Failed to fetch news from API",
                    details: error
                },
                { status: response.status }
            );
        }

        const data = await response.json();
        if (Array.isArray(data.articles)) {
            data.articles = data.articles.map(repairArticle);
        }
        return Response.json(data);
    } catch (error) {
        console.error("News API Error:", error);
        return Response.json(
            { error: "Internal server error: " + error.message },
            { status: 500 }
        );
    }
}
