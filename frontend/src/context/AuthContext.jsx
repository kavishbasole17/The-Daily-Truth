// src/context/AuthContext.jsx
//
// Signed-in state for the whole app. The JWT and a copy of the user live in
// localStorage so a refresh keeps you signed in; services/api.js reads the
// token from there on every request.

"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

import {
    AUTH_CHANGED_EVENT,
    clearSession,
    fetchMe,
    loginRequest,
    googleLoginRequest,
    readSession,
    registerRequest,
    saveSession,
} from "@/services/api";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
    const [user, setUser] = useState(null);
    const [ready, setReady] = useState(false);

    // Restore the session, then confirm the token is still accepted.
    useEffect(() => {
        const { token, user: stored } = readSession();
        if (!token) {
            setReady(true);
            return;
        }
        setUser(stored);
        fetchMe()
            .then(({ data }) => {
                setUser(data);
                saveSession(token, data);
            })
            .catch(err => {
                if (err?.response?.status === 401) {
                    clearSession();
                    setUser(null);
                }
            })
            .finally(() => setReady(true));
    }, []);

    // The API client clears the session when the server rejects the token;
    // mirror that here (and across tabs via the storage event).
    useEffect(() => {
        const sync = () => setUser(readSession().user);
        window.addEventListener(AUTH_CHANGED_EVENT, sync);
        window.addEventListener("storage", sync);
        return () => {
            window.removeEventListener(AUTH_CHANGED_EVENT, sync);
            window.removeEventListener("storage", sync);
        };
    }, []);

    const finish = useCallback(({ data }) => {
        saveSession(data.access_token, data.user);
        setUser(data.user);
        return data.user;
    }, []);

    const login = useCallback(
        (identifier, password) => loginRequest(identifier, password).then(finish),
        [finish]
    );
    const register = useCallback(payload => registerRequest(payload).then(finish), [finish]);
    const loginWithGoogle = useCallback(
        credential => googleLoginRequest(credential).then(finish),
        [finish]
    );
    const logout = useCallback(() => {
        clearSession();
        setUser(null);
    }, []);

    const value = useMemo(
        () => ({ user, ready, isAuthenticated: !!user, login, register, loginWithGoogle, logout }),
        [user, ready, login, register, loginWithGoogle, logout]
    );

    return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
    const ctx = useContext(AuthContext);
    if (!ctx) throw new Error("useAuth must be used inside <AuthProvider>");
    return ctx;
}
