// src/app/login/page.jsx
//
// Sign in / create account. Reached from the navbar, or from Paper Trail once
// the free signed-out check is used (?next=/verify brings you back there).

"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";

import Navbar from "@/components/Navbar";
import Logo from "@/components/Logo";
import GoogleSignInButton from "@/components/GoogleSignInButton";
import { useAuth } from "@/context/AuthContext";
import { errorMessage } from "@/services/api";

import "@/styles/auth.css";

const TEST_ACCOUNT = { identifier: "testuser", password: "Test@1234" };

// Only same-site paths, so ?next= can't bounce someone to another domain.
function safeNext(raw) {
    return raw && raw.startsWith("/") && !raw.startsWith("//") ? raw : "/verify";
}

export default function LoginPage() {
    const router = useRouter();
    const { login, register, loginWithGoogle, isAuthenticated, ready } = useAuth();

    const [mode, setMode] = useState("signin");
    const [next, setNext] = useState("/verify");
    const [reason, setReason] = useState("");
    const [form, setForm] = useState({
        identifier: "", password: "", username: "", email: "", fullName: "", confirm: "",
    });
    const [showPassword, setShowPassword] = useState(false);
    const [error, setError] = useState("");
    const [busy, setBusy] = useState(false);

    useEffect(() => {
        const params = new URLSearchParams(window.location.search);
        setNext(safeNext(params.get("next")));
        if (params.get("mode") === "register") setMode("register");
        if (params.get("reason") === "limit") {
            setReason("You've used your free check. Sign in to ask follow-ups — your first check will be saved to your history.");
        }
    }, []);

    useEffect(() => {
        if (ready && isAuthenticated && !busy) router.replace(next);
    }, [ready, isAuthenticated, busy, next, router]);

    const set = key => e => setForm(f => ({ ...f, [key]: e.target.value }));

    const switchMode = m => {
        setMode(m);
        setError("");
    };

    const run = async fn => {
        setError("");
        setBusy(true);
        try {
            await fn();
            router.replace(next);
        } catch (err) {
            setError(errorMessage(err, "Sign-in failed. Please try again."));
            setBusy(false);
        }
    };

    const handleSubmit = e => {
        e.preventDefault();
        if (busy) return;
        if (mode === "signin") {
            run(() => login(form.identifier.trim(), form.password));
            return;
        }
        if (form.password.length < 8) {
            setError("Password must be at least 8 characters.");
            return;
        }
        if (form.password !== form.confirm) {
            setError("Passwords don't match.");
            return;
        }
        run(() => register({
            username: form.username.trim(),
            email: form.email.trim(),
            password: form.password,
            full_name: form.fullName.trim() || undefined,
        }));
    };

    const useTestAccount = () => {
        setMode("signin");
        setForm(f => ({ ...f, identifier: TEST_ACCOUNT.identifier, password: TEST_ACCOUNT.password }));
        setError("");
    };

    const handleGoogle = useCallback(credential => run(() => loginWithGoogle(credential)),
        // eslint-disable-next-line react-hooks/exhaustive-deps
        [loginWithGoogle, next]);

    const isSignin = mode === "signin";

    return (
        <>
            <Navbar />
            <section className="auth-page">
                <div className="auth-card">
                    <div className="auth-brand">
                        <Logo size={40} />
                        <h1>{isSignin ? "Welcome back" : "Create your account"}</h1>
                        <p>
                            {isSignin
                                ? "Sign in to keep fact-checking and pick up your previous chats."
                                : "Save every check you run and come back to it any time."}
                        </p>
                    </div>

                    {reason && <div className="auth-reason">{reason}</div>}

                    <div className="auth-tabs" role="tablist">
                        <button
                            type="button" role="tab" aria-selected={isSignin}
                            className={isSignin ? "active" : ""} onClick={() => switchMode("signin")}
                        >
                            Sign in
                        </button>
                        <button
                            type="button" role="tab" aria-selected={!isSignin}
                            className={!isSignin ? "active" : ""} onClick={() => switchMode("register")}
                        >
                            Create account
                        </button>
                    </div>

                    <GoogleSignInButton onCredential={handleGoogle} disabled={busy} />

                    <div className="auth-divider"><span>or</span></div>

                    <form className="auth-form" onSubmit={handleSubmit} noValidate>
                        {isSignin ? (
                            <label className="auth-field">
                                <span>Username or email</span>
                                <input
                                    type="text" autoComplete="username" required autoFocus
                                    value={form.identifier} onChange={set("identifier")}
                                    placeholder="you@example.com"
                                />
                            </label>
                        ) : (
                            <>
                                <label className="auth-field">
                                    <span>Full name <em>(optional)</em></span>
                                    <input type="text" autoComplete="name"
                                           value={form.fullName} onChange={set("fullName")} />
                                </label>
                                <label className="auth-field">
                                    <span>Username</span>
                                    <input
                                        type="text" autoComplete="username" required
                                        value={form.username} onChange={set("username")}
                                        placeholder="3–32 letters, numbers, . _ -"
                                    />
                                </label>
                                <label className="auth-field">
                                    <span>Email</span>
                                    <input type="email" autoComplete="email" required
                                           value={form.email} onChange={set("email")}
                                           placeholder="you@example.com" />
                                </label>
                            </>
                        )}

                        <label className="auth-field">
                            <span>Password</span>
                            <div className="auth-password">
                                <input
                                    type={showPassword ? "text" : "password"}
                                    autoComplete={isSignin ? "current-password" : "new-password"}
                                    required value={form.password} onChange={set("password")}
                                    placeholder={isSignin ? "" : "At least 8 characters"}
                                />
                                <button
                                    type="button" className="auth-reveal"
                                    onClick={() => setShowPassword(s => !s)}
                                    aria-label={showPassword ? "Hide password" : "Show password"}
                                >
                                    {showPassword ? "Hide" : "Show"}
                                </button>
                            </div>
                        </label>

                        {!isSignin && (
                            <label className="auth-field">
                                <span>Confirm password</span>
                                <input
                                    type={showPassword ? "text" : "password"} autoComplete="new-password"
                                    required value={form.confirm} onChange={set("confirm")}
                                />
                            </label>
                        )}

                        {error && <div className="auth-error" role="alert">{error}</div>}

                        <button type="submit" className="auth-submit" disabled={busy}>
                            {busy ? "Please wait…" : isSignin ? "Sign in" : "Create account"}
                        </button>
                    </form>

                    <div className="auth-test">
                        <div>
                            <strong>Test account</strong>
                            <span><code>{TEST_ACCOUNT.identifier}</code> / <code>{TEST_ACCOUNT.password}</code></span>
                        </div>
                        <button type="button" onClick={useTestAccount}>Fill in</button>
                    </div>

                    <p className="auth-footnote">
                        Just looking? <Link href="/verify">Try one check without an account</Link>.
                    </p>
                </div>
            </section>
        </>
    );
}
