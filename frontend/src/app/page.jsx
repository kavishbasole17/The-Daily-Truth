// src/app/page.jsx

import Navbar from "@/components/Navbar";
import BreakingTicker from "@/components/BreakingTicker";
import NewsFeed from "@/components/NewsFeed";
import FeatureCards from "@/components/FeatureCards";
import Footer from "@/components/Footer";

export default function HomePage() {
    return (
        <>
            <Navbar />
            <BreakingTicker />
            <NewsFeed />
            <FeatureCards />
            <Footer />
        </>
    );
}