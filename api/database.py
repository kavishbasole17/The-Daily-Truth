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
