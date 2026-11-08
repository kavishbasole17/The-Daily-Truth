"""SQLAlchemy engine, session factory, and Base for the application database."""
from __future__ import annotations

import os
import sys

# Ensure project root is importable regardless of working directory
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

try:
    from config import settings
    _DATABASE_URL = settings.DATABASE_URL
except ImportError:
    _DATABASE_URL = "sqlite:///./paper_trail.db"

_connect_args = {"check_same_thread": False} if "sqlite" in _DATABASE_URL else {}
engine = create_engine(_DATABASE_URL, connect_args=_connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    """FastAPI dependency — yields a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables() -> None:
    """Create all tables defined on Base. Call once at startup."""
    Base.metadata.create_all(bind=engine)


# Columns added after the initial schema shipped. `create_all` only creates
# missing *tables*, never missing columns, and paper_trail.db already holds
# real rows — so without this, adding a field to the ORM produces
# "no such column" at query time on any existing install.
_ADDED_COLUMNS: dict[str, dict[str, str]] = {
    "articles": {
        "stage": "VARCHAR(64)",
        "user_id": "INTEGER",
        "conversation_id": "INTEGER",
        "guest_key": "VARCHAR(128)",
        "kind": "VARCHAR(16)",
    },
    "verdicts": {"degradations": "TEXT", "fused_score": "FLOAT"},
}


def ensure_columns() -> None:
    """
    Idempotently add any missing columns to existing tables.

    Deliberately minimal rather than pulling in Alembic: these are additive,
    nullable columns on a SQLite file, and a migration framework would be more
    machinery than the change warrants.
    """
    import logging

    from sqlalchemy import inspect, text

    log = logging.getLogger(__name__)
    if not engine.url.get_backend_name().startswith("sqlite"):
        return  # other backends should use a real migration tool

    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())

    with engine.begin() as conn:
        for table, columns in _ADDED_COLUMNS.items():
            if table not in existing_tables:
                continue
            present = {c["name"] for c in inspector.get_columns(table)}
            for name, ddl_type in columns.items():
                if name in present:
                    continue
                log.info("database: adding column %s.%s", table, name)
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl_type}"))
