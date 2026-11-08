// src/components/Navbar.jsx

"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import Logo from "@/components/Logo";
import ThemeToggle from "@/components/ThemeToggle";
import { useAuth } from "@/context/AuthContext";
import "@/styles/navbar.css";
import "@/styles/auth.css";

function initials(user) {
    const name = user?.full_name || user?.username || "?";
    const parts = name.trim().split(/\s+/);
    return ((parts[0]?.[0] || "") + (parts[1]?.[0] || "")).toUpperCase() || "?";
}

function AccountMenu() {
    const { user, ready, logout } = useAuth();
    const router = useRouter();
    const path = usePathname();
    const [open, setOpen] = useState(false);
    const ref = useRef(null);

    useEffect(() => {
        if (!open) return;
        const close = e => { if (!ref.current?.contains(e.target)) setOpen(false); };
        const esc = e => { if (e.key === "Escape") setOpen(false); };
        document.addEventListener("mousedown", close);
        document.addEventListener("keydown", esc);
        return () => {
            document.removeEventListener("mousedown", close);
            document.removeEventListener("keydown", esc);
        };
    }, [open]);

    if (!ready) return null;

    if (!user) {
        if (path === "/login") return null;
        return (
            <Link href={`/login?next=${encodeURIComponent(path || "/verify")}`} className="nav-signin">
                Sign in
            </Link>
        );
    }

    return (
        <div className="nav-user" ref={ref}>
            <button
                type="button" className="nav-user-btn"
                onClick={() => setOpen(o => !o)} aria-haspopup="menu" aria-expanded={open}
            >
                <span className="nav-avatar">{initials(user)}</span>
                <span className="nav-user-name">{user.full_name || user.username}</span>
            </button>
            {open && (
                <div className="nav-menu" role="menu">
                    <div className="nav-menu-head">
                        <strong>{user.full_name || user.username}</strong>
                        <span>{user.email || `@${user.username}`}</span>
                    </div>
                    <Link href="/verify?history=1" role="menuitem" onClick={() => setOpen(false)}>
                        Previous chats
                    </Link>
                    <Link href="/verify?new=1" role="menuitem" onClick={() => setOpen(false)}>
                        New check
                    </Link>
                    <button
                        type="button" role="menuitem"
                        onClick={() => {
                            setOpen(false);
                            logout();
                            router.push("/");
                        }}
                    >
                        Sign out
                    </button>
                </div>
            )}
        </div>
    );
}

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

                <div className="nav-actions">
                    <ThemeToggle />
                    <AccountMenu />
                </div>
            </div>
        </nav>
    );
}
