"""
Single source of configuration. Every env var the backend reads is a field
here, with its one default. Read `settings.X`; never call os.getenv.

Precedence: real environment > backend/.env > defaults below.
.env keys that aren't fields are rejected at startup, so config can't drift.
See .env.example for what each key does.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent / ".env",
        env_file_encoding="utf-8",
        extra="forbid",
    )

    # Secrets (JWT_SECRET is enforced in auth.py, so ingestion runs without it)
    JWT_SECRET: str = ""
    OPENAI_API_KEY: str = ""
    MONGO_URI: str = "mongodb://localhost:27017"
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""
    GOOGLE_REDIRECT_URI: str = "http://localhost:8000/auth/callback"

    # Paths
    PDF_PATH: str = "data"
    INDEX_CACHE_DIR: str = "index_cache"

    # Chunking + embedding [RE-INGEST] — must match index_cache/index_manifest.json
    CHUNK_SIZE: int = 900
    CHUNK_OVERLAP: int = 150
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    QUERY_INSTRUCTION: str = ""

    # Retrieval
    RETRIEVAL_K: int = 60
    FINAL_K: int = 8
    TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.30
    CONTEXT_WINDOW: int = 1

    # Reranking
    RERANKER_ENABLED: bool = True
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"
    RERANKER_MAX_CANDIDATES: int = 60
    SKIP_RERANKER_THRESHOLD: float = 0.85

    # Hybrid search
    BM25_ENABLED: bool = True
    RRF_K: int = 60

    # Query understanding (each adds an LLM round-trip)
    MULTI_QUERY_ENABLED: bool = False
    MULTI_QUERY_COUNT: int = 3
    INTELLIGENT_PARSING_ENABLED: bool = False
    MAX_RETRIEVAL_STEPS: int = 3
    MULTI_STEP_RETRIEVAL_K: int = 10

    # Ranking adjustments — boosts are ADDED to the score, not multiplied
    BOOST_COMPANY: float = 0.08
    BOOST_YEAR: float = 0.04
    BOOST_DOCTYPE: float = 0.03
    DEDUP_THRESHOLD: float = 0.85
    MAX_FROM_ONE_DOC: float = 0.6
    MAX_DOC_CONCENTRATION: float = 0.6
    DEFAULT_COMPANY: str = ""
    DEFAULT_DOC_TYPE: str = ""
    DEFAULT_YEAR: str = ""

    # Generation — OPENAI_BASE_URL swaps in any OpenAI-compatible provider
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_BASE_URL: str = ""

    # Database
    MONGO_DB_NAME: str = "Cognifin"

    # CORS / frontend
    ALLOWED_ORIGINS: str = "http://localhost:5173"
    FRONTEND_URL: str = "http://localhost:5173"

    # Response cache
    CACHE_ENABLED: bool = True
    CACHE_MAX_SIZE: int = 500
    CACHE_TTL_SECONDS: int = 3600

    # Assets / deployment
    ASSET_MODE: str = "local"
    HF_CACHE_URL: str = ""
    HF_PDF_BASE_URL: str = ""

    # Uploads
    UPLOAD_MAX_MB: int = 25

    # Logging
    LOG_LEVEL: str = "INFO"
    QUERY_LOG_ENABLED: bool = False


settings = Settings()
