// src/services/api.js

import axios from "axios";

const api = axios.create({
    baseURL: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000",
    headers: { "Content-Type": "application/json" },
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

export const submitForVerification = async (text) => {
    const token = await getDemoToken();
    return api.post(
        "/articles/submit",
        { text },
        { headers: { Authorization: `Bearer ${token}` } }
    );
};

export default api;