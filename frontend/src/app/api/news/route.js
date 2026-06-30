// src/app/api/news/route.js

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
        return Response.json(data);
    } catch (error) {
        console.error("News API Error:", error);
        return Response.json(
            { error: "Internal server error: " + error.message },
            { status: 500 }
        );
    }
}
