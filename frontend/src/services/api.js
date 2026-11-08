// src/services/api.js

import axios from "axios";

const api = axios.create({
    baseURL: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000",
    headers: { "Content-Type": "application/json" },
    // Each individual call is short (submit returns 202 immediately, polls are
    // cheap reads); the long wait is handled by pollVerification, not by a
    // single long-lived request.
    timeout: 30000
});

// Error interceptor
api.interceptors.response.use(
    response => response,
    error => {
        if (error.code === "ECONNABORTED") {
            return Promise.reject({
                ...error,
                message: "Request timeout. Please try again."
            });
        }
        return Promise.reject(error);
    }
);

export const healthCheck = () => api.get("/health");

export const getModels = () => api.get("/models");

export const predictArticle = payload =>
    api.post("/predict", payload);

export const consensusPredict = payload =>
    api.post("/predict/all", payload);

export const batchPredict = payload =>
    api.post("/predict/batch", payload);

// ── Session (JWT) ───────────────────────────────────────────────────────────
// Stored in localStorage so a refresh keeps you signed in. Every request
// attaches the token automatically (see the request interceptor below).
const TOKEN_KEY = "dt_token";
const USER_KEY = "dt_user";
export const AUTH_CHANGED_EVENT = "dt-auth-changed";
export const GUEST_LIMIT_CODE = "guest_limit_reached";

const storage = () => {
    try {
        return typeof window !== "undefined" ? window.localStorage : null;
    } catch {
        return null;
    }
};

export const readSession = () => {
    const ls = storage();
    if (!ls) return { token: null, user: null };
    let user = null;
    try {
        user = JSON.parse(ls.getItem(USER_KEY) || "null");
    } catch {
        user = null;
    }
    return { token: ls.getItem(TOKEN_KEY), user };
};

export const saveSession = (token, user) => {
    const ls = storage();
    if (!ls) return;
    ls.setItem(TOKEN_KEY, token);
    ls.setItem(USER_KEY, JSON.stringify(user || null));
};

export const clearSession = () => {
    const ls = storage();
    if (!ls) return;
    ls.removeItem(TOKEN_KEY);
    ls.removeItem(USER_KEY);
};

api.interceptors.request.use(config => {
    const { token } = readSession();
    if (token && !config.headers?.Authorization) {
        config.headers = config.headers || {};
        config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
});

// A stored token the server no longer accepts (expired, account deleted)
// signs you out rather than failing every call. The guest-limit 401 is a
// different thing: it means "sign in", not "your token is bad".
api.interceptors.response.use(
    response => response,
    error => {
        const status = error?.response?.status;
        const detail = error?.response?.data?.detail;
        const sentToken = !!error?.config?.headers?.Authorization;
        if (status === 401 && sentToken && detail?.code !== GUEST_LIMIT_CODE) {
            clearSession();
            if (typeof window !== "undefined") window.dispatchEvent(new Event(AUTH_CHANGED_EVENT));
        }
        return Promise.reject(error);
    }
);

/** True when the server refused a signed-out query because the free one is used. */
export const isGuestLimitError = err =>
    err?.response?.status === 401 && err?.response?.data?.detail?.code === GUEST_LIMIT_CODE;

/** Turns any API error into one readable sentence. */
export const errorMessage = (err, fallback = "Something went wrong.") => {
    const detail = err?.response?.data?.detail;
    if (typeof detail === "string") return detail;
    if (detail?.message) return detail.message;
    if (Array.isArray(detail) && detail.length) {
        // FastAPI validation errors: "Value error, Enter a valid email address."
        return detail.map(d => String(d.msg || "").replace(/^Value error,\s*/, "")).join(" ");
    }
    return err?.response?.data?.message || err?.message || fallback;
};

// ── Auth ────────────────────────────────────────────────────────────────────
export const loginRequest = (username, password) =>
    api.post("/auth/token", { username, password });

export const registerRequest = payload => api.post("/auth/register", payload);

export const googleLoginRequest = credential => api.post("/auth/google", { credential });

export const fetchMe = () => api.get("/auth/me");

export const getGuestStatus = () => api.get("/auth/guest");

// ── Chat history ────────────────────────────────────────────────────────────
export const listConversations = () => api.get("/conversations");
export const getConversation = id => api.get(`/conversations/${id}`);
export const renameConversation = (id, title) => api.patch(`/conversations/${id}`, { title });
export const deleteConversation = id => api.delete(`/conversations/${id}`);
export const claimGuestChecks = articleIds =>
    api.post("/conversations/claim", { article_ids: articleIds });

// ── Agentic verification pipeline (evidence-grounded) ──────────────────────
// Submission is asynchronous: POST returns 202 with status="processing" and the
// verdict arrives later. The pipeline retrieves live evidence and runs a
// transformer NLI model, which takes seconds — a synchronous request would sit
// under the axios timeout waiting on external search providers.
//
// Signed out, the server allows one free submission; after that it answers
// 401 guest_limit_reached. Signed in, each submission joins `conversationId`
// (or starts a new chat when it's omitted).
export const submitForVerification = (text, { conversationId } = {}) =>
    api.post("/articles/submit", {
        text,
        ...(conversationId ? { conversation_id: conversationId } : {})
    });

export const getArticle = id => api.get(`/articles/${id}`);

// Same async submit/poll shape as text, but the server runs OCR (Tesseract,
// Hindi+English) over the image first and verifies whatever text it recovers.
export const submitImageForVerification = (file, { conversationId } = {}) => {
    const form = new FormData();
    form.append("file", file);
    if (conversationId) form.append("conversation_id", String(conversationId));
    return api.post("/articles/submit-image", form, {
        headers: { "Content-Type": "multipart/form-data" }
    });
};

/**
 * Poll an article until the pipeline finishes.
 *
 * @param {number}   id            article id from submitForVerification
 * @param {object}   opts
 * @param {number}   opts.intervalMs   delay between polls
 * @param {number}   opts.timeoutMs    give up after this long
 * @param {function} opts.onStage      called with the current pipeline stage
 * @returns {Promise<object>} the completed ArticleOut body
 */
export const pollVerification = async (id, opts = {}) => {
    const { intervalMs = 1500, timeoutMs = 180000, onStage } = opts;
    const deadline = Date.now() + timeoutMs;

    // Poll rather than hold a long request open, so a slow or wedged pipeline
    // shows progress instead of a dead-looking spinner that eventually times out.
    while (Date.now() < deadline) {
        const { data } = await getArticle(id);

        if (onStage) onStage(data.stage || data.status);

        if (data.status === "completed") return data;
        if (data.status === "failed") {
            throw new Error("Verification failed. Please try again.");
        }
        await new Promise(r => setTimeout(r, intervalMs));
    }
    throw new Error("Verification timed out. The article may be unusually long.");
};

/**
 * Submit and wait for the verdict.
 * `opts.onSubmitted(article)` fires once the server accepts it — that is when
 * the article id and conversation id are known.
 */
export const verifyArticle = async (text, opts = {}) => {
    const { data } = await submitForVerification(text, opts);
    opts.onSubmitted?.(data);
    return pollVerification(data.id, opts);
};

/** Submit an image and wait for the verdict (OCR runs server-side first). */
export const verifyImage = async (file, opts = {}) => {
    const { data } = await submitImageForVerification(file, opts);
    opts.onSubmitted?.(data);
    return pollVerification(data.id, opts);
};

export default api;