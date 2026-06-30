// src/components/HealthBadge.jsx

"use client";

import { useEffect, useState } from "react";
import { healthCheck } from "@/services/api";

export default function HealthBadge() {
    const [status, setStatus] = useState("Checking");

    useEffect(() => {
        const checkHealth = async () => {
            try {
                await healthCheck();
                setStatus("Online");
            } catch {
                setStatus("Offline");
            }
        };
        checkHealth();
    }, []);

    return (
        <div className={`health-widget ${status.toLowerCase()}`}>
            <span />
            API {status}
        </div>
    );
}