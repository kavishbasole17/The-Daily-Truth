from __future__ import annotations

from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Development flags
    # ------------------------------------------------------------------
    # Global default for stages that need a paid/heavy dependency (LLM claim
    # extraction, transformer-based NLI). Evidence retrieval is controlled
    # separately below since it has a free, keyless real mode (Wikipedia)
    # and doesn't need to wait on LLM/GPU setup.
    MOCK_MODE: bool = True  # use canned responses instead of real API calls

    # Evidence retrieval defaults to real: Wikipedia needs no key at all, and
    # NewsAPI is used automatically when NEWS_API_KEY is set. Set to True to
    # force canned snippets (e.g. offline dev, rate-limit avoidance).
    EVIDENCE_MOCK_MODE: bool = False

    # Ensemble scoring also defaults to real: the classical/hybrid models run
    # fully locally (no API key, no network), so there's no reason to mock
    # this stage once export_models.py has been run.
    ENSEMBLE_MOCK_MODE: bool = False

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------
    ANTHROPIC_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    LLM_PROVIDER: str = "anthropic"           # "anthropic" | "openai"
    LLM_MODEL: str = "claude-haiku-4-5-20251001"

    # ------------------------------------------------------------------
    # Search APIs
    # ------------------------------------------------------------------
    NEWS_API_KEY: Optional[str] = None        # newsapi.org — trusted-outlet headline search
    TAVILY_API_KEY: Optional[str] = None
    SERPAPI_KEY: Optional[str] = None
    SEARCH_PROVIDER: str = "newsapi"          # "newsapi" | "wikipedia" | "tavily" | "serpapi"

    # ------------------------------------------------------------------
    # JWT / auth
    # ------------------------------------------------------------------
    JWT_SECRET_KEY: str = "dev-secret-key-CHANGE-IN-PRODUCTION"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 60

    # ------------------------------------------------------------------
    # Databases
    # ------------------------------------------------------------------
    DATABASE_URL: str = "sqlite:///./paper_trail.db"

    # ------------------------------------------------------------------
    # Pipeline controls
    # ------------------------------------------------------------------
    MAX_EVIDENCE_CALLS: int = 5
    MAX_CLAIMS: int = 5
    ESCALATION_LOW: float = 0.4
    ESCALATION_HIGH: float = 0.6


settings = Settings()
