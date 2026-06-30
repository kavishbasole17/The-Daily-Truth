// src/app/page.jsx

import Navbar from "@/components/Navbar";
import BreakingTicker from "@/components/BreakingTicker";
import Hero from "@/components/Hero";
import NewsCarousel from "@/components/NewsCarousel";
import HealthBadge from "@/components/HealthBadge";
import FeatureCards from "@/components/FeatureCards";
import Footer from "@/components/Footer";

export default function HomePage() {
    return (
        <>
            <Navbar />
            <BreakingTicker />
            <Hero />

            <NewsCarousel />

            <section className="section">
                <div className="container">
                    <HealthBadge />
                </div>
            </section>

            <FeatureCards />
            <Footer />
        </>
    );
}