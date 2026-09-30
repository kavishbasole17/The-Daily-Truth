// src/components/ChatComposer.jsx

import { forwardRef, useEffect, useImperativeHandle, useRef } from "react";

const MAX_CHARS = 50000; // mirrors api/routers/articles.py's MAX_ARTICLE_TEXT_LEN

const ChatComposer = forwardRef(function ChatComposer({
                                          value,
                                          onChange,
                                          onSend,
                                          onImageSelect,
                                          disabled,
                                          placeholder
                                      }, ref) {
    const textareaRef = useRef(null);
    const fileInputRef = useRef(null);

    useImperativeHandle(ref, () => ({
        focus: () => textareaRef.current?.focus(),
    }));

    useEffect(() => {
        const el = textareaRef.current;
        if (!el) return;
        el.style.height = "auto";
        el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
    }, [value]);

    const handleKeyDown = (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            if (!disabled && value.trim()) onSend();
        }
    };

    const handleFileChange = (e) => {
        const file = e.target.files?.[0];
        e.target.value = "";
        if (file) onImageSelect(file);
    };

    // Catches Ctrl+V of an image copied from elsewhere (a screenshot tool, a
    // browser, another app) — clipboard image data arrives as a File-like
    // blob on clipboardData.items, never as text, so it never collides with
    // pasting article text into the field below.
    const handlePaste = (e) => {
        if (disabled || !onImageSelect) return;
        const items = e.clipboardData?.items;
        if (!items) return;

        for (const item of items) {
            if (item.kind === "file" && item.type.startsWith("image/")) {
                const file = item.getAsFile();
                if (file) {
                    e.preventDefault();
                    onImageSelect(file);
                }
                return;
            }
        }
    };

    const nearLimit = value.length > MAX_CHARS * 0.9;
    const overLimit = value.length > MAX_CHARS;

    return (
        <div className="chat-composer">
            <div className="chat-composer-inset">
                <div className={`chat-composer-inner ${overLimit ? "over-limit" : ""}`} onPaste={handlePaste}>
                    {onImageSelect && (
                        <>
                            <input
                                ref={fileInputRef}
                                type="file"
                                accept="image/jpeg,image/png,image/webp,image/bmp,image/tiff"
                                hidden
                                onChange={handleFileChange}
                            />
                            <button
                                type="button"
                                className="chat-attach-btn"
                                onClick={() => fileInputRef.current?.click()}
                                disabled={disabled}
                                aria-label="Attach an image to verify"
                                title="Attach an image (OCR)"
                            >
                                <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                    <path d="M21.44 11.05l-9.19 9.19a6 6 0 01-8.49-8.49l9.19-9.19a4 4 0 015.66 5.66l-9.2 9.19a2 2 0 01-2.83-2.83l8.49-8.48" />
                                </svg>
                            </button>
                        </>
                    )}

                    <textarea
                        ref={textareaRef}
                        className="chat-composer-textarea"
                        rows={1}
                        placeholder={placeholder}
                        value={value}
                        onChange={e => onChange(e.target.value)}
                        onKeyDown={handleKeyDown}
                    />
                    <button
                        type="button"
                        className="chat-send-btn"
                        onClick={onSend}
                        disabled={disabled || !value.trim() || overLimit}
                        aria-label="Send for verification"
                    >
                        <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M5 12h14M13 6l6 6-6 6" />
                        </svg>
                    </button>
                </div>

                <div className="chat-composer-footer">
                    <p className="chat-composer-hint">
                        Enter to send · Shift + Enter for a new line
                        {onImageSelect && " · Attach, or paste (Ctrl+V) an image to OCR it"}
                    </p>
                    {value.length > 200 && (
                        <span className={`chat-char-count ${nearLimit ? "warn" : ""}`}>
                            {value.length.toLocaleString()} / {MAX_CHARS.toLocaleString()}
                        </span>
                    )}
                </div>
            </div>
        </div>
    );
});

export default ChatComposer;
