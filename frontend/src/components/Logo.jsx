// src/components/Logo.jsx
//
// The "T." app-icon mark — sourced from creatives/Logo Options-selection (2).png.
// Dot is brand red today; the design spec calls for it to eventually track
// verdict color (red/false, green/true, amber/review) as a status system.

export default function Logo({ size = 32, className = "" }) {
    return (
        // eslint-disable-next-line @next/next/no-img-element
        <img
            className={`logo-mark ${className}`}
            src="/logo/icon-128.png"
            alt="The Daily Truth"
            width={size}
            height={size}
            style={{ width: size, height: size, borderRadius: size * 0.28, flexShrink: 0 }}
        />
    );
}
