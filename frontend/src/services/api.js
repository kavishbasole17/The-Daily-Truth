// src/services/api.js

import axios from "axios";

const api = axios.create({
    baseURL: process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000/api",
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

export default api;