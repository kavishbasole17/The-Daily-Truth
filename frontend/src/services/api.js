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

// ── Agentic verification pipeline (evidence-grounded) ──────────────────────
// Demo credentials (see README) — fine for a capstone demo; a real deployment
// would use a proper login flow instead of fetching a token per submission.
const getDemoToken = () =>
    api.post("/auth/token", { username: "admin", password: "admin123" })
        .then(({ data }) => data.access_token);

// Submission is asynchronous: POST returns 202 with status="processing" and the
// verdict arrives later. The pipeline retrieves live evidence and runs a
// transformer NLI model, which takes seconds — a synchronous request would sit
// under the axios timeout waiting on external search providers.
export const submitForVerification = async (text) => {
    const token = await getDemoToken();
    return api.post(
        "/articles/submit",
        { text },
        { headers: { Authorization: `Bearer ${token}` } }
    );
};

export const getArticle = async (id) => {
    const token = await getDemoToken();
    return api.get(`/articles/${id}`, {
        headers: { Authorization: `Bearer ${token}` }
    });
};

// Same async submit/poll shape as text, but the server runs OCR (Tesseract,
// Hindi+English) over the image first and verifies whatever text it recovers.
export const submitImageForVerification = async (file) => {
    const token = await getDemoToken();
    const form = new FormData();
    form.append("file", file);
    return api.post("/articles/submit-image", form, {
        headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "multipart/form-data"
        }
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

/** Submit and wait for the verdict. */
export const verifyArticle = async (text, opts = {}) => {
    const { data } = await submitForVerification(text);
    return pollVerification(data.id, opts);
};

/** Submit an image and wait for the verdict (OCR runs server-side first). */
export const verifyImage = async (file, opts = {}) => {
    const { data } = await submitImageForVerification(file);
    return pollVerification(data.id, opts);
};

export default api;