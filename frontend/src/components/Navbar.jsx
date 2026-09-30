// src/components/Navbar.jsx

"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import Logo from "@/components/Logo";
import ThemeToggle from "@/components/ThemeToggle";
import "@/styles/navbar.css";

export default function Navbar() {
    const path = usePathname();

    const links = [
        ["Home", "/"],
        ["Paper Trail", "/verify"],
        ["Single Check", "/single"],
        ["Consensus", "/consensus"],
        ["Batch Analysis", "/batch"],
        ["Models", "/models"]
    ];

    return (
        <nav className="navbar">
            <div className="container nav-content">
                <Link href="/" className="logo">
                    <Logo size={34} />
                    <span className="logo-wordmark">
                        The Daily Truth<span className="logo-dot">.</span>
                    </span>
                </Link>

                <div className="nav-links">
                    {links.map(([name, href]) => (
                        <Link
                            key={href}
                            href={href}
                            className={path === href ? "active" : ""}
                        >
                            {name}
                        </Link>
                    ))}
                </div>

                <ThemeToggle />
            </div>
        </nav>
    );
}