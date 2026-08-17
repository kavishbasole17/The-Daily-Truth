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
    # Per-stage backends
    # ------------------------------------------------------------------
    # A single MOCK_MODE flag used to gate two unrelated stages (LLM claim
    # extraction and transformer NLI) at once. Because it defaulted to True and
    # the LLM path was never reachable, the pipeline silently ran a crude
    # lexical heuristic in place of NLI for its entire life. Each stage now
    # names its own backend, so "which algorithm actually ran" is answerable
    # from config alone.
    CLAIM_BACKEND: str = "heuristic"    # heuristic | llm | naive(regression-test only)
    NLI_BACKEND: str = "transformer"    # transformer | lexical | off
    EVIDENCE_BACKEND: str = "live"      # live | cache_only | mock
    ENSEMBLE_BACKEND: str = "real"      # real | mock

    # Back-compat: honoured only if explicitly set in .env. MOCK_MODE=true
    # downgrades the two stages it used to gate.
    MOCK_MODE: Optional[bool] = None
    EVIDENCE_MOCK_MODE: Optional[bool] = None
    ENSEMBLE_MOCK_MODE: Optional[bool] = None

    # ------------------------------------------------------------------
    # NLI (claim verification)
    # ------------------------------------------------------------------
    # DeBERTa-v3-base-mnli-fever-anli over bart-large-mnli: it is trained on
    # FEVER (evidence-vs-claim fact verification — our exact task) plus
    # adversarial ANLI, while being ~4x smaller. Note the two checkpoints have
    # *inverted* label orders, which is why verifier code reads id2label from
    # the loaded config instead of assuming an index order.
    NLI_MODEL: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    NLI_DEVICE: str = "auto"            # auto | cuda | cpu
    NLI_FP16: bool = True
    NLI_MAX_PAIRS: int = 200            # per article, highest-relevance pairs kept
    NLI_BATCH_SIZE: int = 32
    NLI_MAX_LENGTH: int = 256

    # ------------------------------------------------------------------
    # Embeddings (evidence relevance gating + claim dedup)
    # ------------------------------------------------------------------
    EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Cosine floor for a snippet to count as evidence. At 0.35 the gate let in
    # topically-adjacent pages that the NLI model then confidently mislabelled
    # as contradictions of true claims; 0.45 keeps recall usable while cutting
    # that failure mode.
    RELEVANCE_MIN: float = 0.45
    CLAIM_DEDUP_COSINE: float = 0.92

    # ------------------------------------------------------------------
    # Evidence cache
    # ------------------------------------------------------------------
    EVIDENCE_CACHE_DIR: str = "cache/evidence"
    EVIDENCE_CACHE_TTL_DAYS: int = 7    # 0 = never expire (use for eval replays)

    # ------------------------------------------------------------------
    # Fusion adjudicator
    # ------------------------------------------------------------------
    FUSION_MODEL_PATH: str = "models/fusion/adjudicator_v1.joblib"
    TARGET_PRECISION: float = 0.90      # precision to hold on committed verdicts

    # Warm NLI + embedding singletons during FastAPI startup so the first
    # request doesn't pay model-load latency. Tests set this False.
    WARM_MODELS: bool = True

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
    MAX_EVIDENCE_CALLS: int = 12
    # 5 rather than 8: retrieval is serialised behind Wikipedia's rate limit, so
    # each extra claim costs ~1-2s of wall time for diminishing evidence value.
    MAX_CLAIMS: int = 5
    CHECKWORTHY_MIN: float = 0.35       # floor on claim check-worthiness score
    ESCALATION_LOW: float = 0.4
    ESCALATION_HIGH: float = 0.6
    STAGE_TIMEOUT_S: float = 90.0
    # Kept low deliberately. Retrieval is network-bound so threads help, but
    # Wikipedia rate-limits the extracts API aggressively — at 6 workers the
    # backoff pinned itself at its ceiling and the stage timed out, which
    # downstream is indistinguishable from "no evidence exists".
    RETRIEVAL_WORKERS: int = 3

    def apply_legacy_mock_flags(self) -> None:
        """
        Map the deprecated MOCK_MODE family onto the per-stage backends.

        Only applied when a legacy flag is explicitly present in the
        environment, so a fresh install gets the real backends and an existing
        .env keeps behaving as its author intended until they migrate.
        """
        if self.MOCK_MODE:
            self.CLAIM_BACKEND = "naive"
            self.NLI_BACKEND = "lexical"
        if self.EVIDENCE_MOCK_MODE:
            self.EVIDENCE_BACKEND = "mock"
        if self.ENSEMBLE_MOCK_MODE:
            self.ENSEMBLE_BACKEND = "mock"


settings = Settings()
settings.apply_legacy_mock_flags()
