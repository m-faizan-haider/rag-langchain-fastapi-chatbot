# Backend/logging_config.py
"""
Centralized logging configuration.
Call setup_logging() once at application startup.
All modules use: logger = logging.getLogger(__name__)
"""
import logging
import logging.handlers
import sys
import os
from pathlib import Path

# Log directory — same as RAG_LOG but for structured app logs
LOG_DIR = Path(__file__).resolve().parent.parent / "results"
LOG_FILE = LOG_DIR / "app.log"


def setup_logging(log_level: str | None = None) -> None:
    """
    Configure root logger with:
      - Console handler (human-readable in dev, JSON-like in prod)
      - Rotating file handler (10MB max, 5 backups)
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    level_str = log_level or os.getenv("LOG_LEVEL", "INFO")
    level     = getattr(logging, level_str.upper(), logging.INFO)

    # ── Formatter ────────────────────────────────────────────────────────────
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler (force UTF-8 so emoji don't crash on Windows cp1252) ──
    import io
    console = logging.StreamHandler(
        stream=io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        if hasattr(sys.stdout, "buffer") else sys.stdout
    )
    console.setFormatter(fmt)
    console.setLevel(level)

    # ── Rotating file handler ─────────────────────────────────────────────────
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.DEBUG)   # always capture DEBUG to file

    # ── Root logger ───────────────────────────────────────────────────────────
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)   # root captures everything; handlers filter
    root.handlers.clear()
    root.addHandler(console)
    root.addHandler(file_handler)

    # ── Silence noisy third-party loggers ─────────────────────────────────────
    for noisy in ("httpx", "httpcore", "urllib3", "sentence_transformers", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        "Logging initialized | level=%s | file=%s", level_str, LOG_FILE
    )
