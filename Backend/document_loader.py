# Backend/document_loader.py
"""
Multi-format document loader.
Supported: PDF, TXT, DOCX, Markdown, HTML, CSV
All loaders enrich metadata with file_type and ingested_at timestamp.
"""
import logging
import time
from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    DirectoryLoader,
    CSVLoader,
)
from dotenv import load_dotenv

load_dotenv()
from Backend.config import DATA_DIR

logger = logging.getLogger(__name__)


def _enrich_metadata(docs: list, file_type: str) -> list:
    """Add file_type and ingested_at to every document's metadata."""
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    for doc in docs:
        doc.metadata["file_type"]   = file_type
        doc.metadata["ingested_at"] = ts
    return docs


def load_documents(data_dir: Path | None = None) -> List:
    """
    Load all supported documents from data_dir (defaults to DATA_DIR from config).
    Returns a flat list of LangChain Document objects.
    """
    source_dir = data_dir or DATA_DIR

    if not source_dir.exists():
        raise RuntimeError(f"Data directory not found: {source_dir}")

    all_docs = []

    # ── PDF ──────────────────────────────────────────────────────────────────
    try:
        loader = DirectoryLoader(str(source_dir), glob="**/*.pdf", loader_cls=PyPDFLoader)
        docs   = loader.load()
        all_docs.extend(_enrich_metadata(docs, "pdf"))
        logger.info("Loaded %d PDF pages", len(docs))
    except Exception as e:
        logger.warning("PDF loader error (skipping): %s", e)

    # ── TXT ──────────────────────────────────────────────────────────────────
    try:
        loader = DirectoryLoader(
            str(source_dir),
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8", "autodetect_encoding": True},
        )
        docs   = loader.load()
        all_docs.extend(_enrich_metadata(docs, "txt"))
        logger.info("Loaded %d TXT documents", len(docs))
    except Exception as e:
        logger.warning("TXT loader error (skipping): %s", e)

    # ── DOCX ──────────────────────────────────────────────────────────────────
    try:
        from langchain_community.document_loaders import Docx2txtLoader
        for fpath in source_dir.rglob("*.docx"):
            try:
                docs = Docx2txtLoader(str(fpath)).load()
                all_docs.extend(_enrich_metadata(docs, "docx"))
            except Exception as e:
                logger.warning("Skipped %s: %s", fpath.name, e)
        logger.info("DOCX loading complete")
    except ImportError:
        logger.debug("docx2txt not installed — skipping .docx files")
    except Exception as e:
        logger.warning("DOCX loader error (skipping): %s", e)

    # ── Markdown ─────────────────────────────────────────────────────────────
    try:
        loader = DirectoryLoader(
            str(source_dir),
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        docs   = loader.load()
        all_docs.extend(_enrich_metadata(docs, "markdown"))
        logger.info("Loaded %d Markdown files", len(docs))
    except Exception as e:
        logger.warning("Markdown loader error (skipping): %s", e)

    # ── HTML ──────────────────────────────────────────────────────────────────
    try:
        from langchain_community.document_loaders import BSHTMLLoader
        for fpath in source_dir.rglob("*.html"):
            try:
                docs = BSHTMLLoader(str(fpath), open_encoding="utf-8").load()
                all_docs.extend(_enrich_metadata(docs, "html"))
            except Exception as e:
                logger.warning("Skipped %s: %s", fpath.name, e)
        logger.info("HTML loading complete")
    except ImportError:
        logger.debug("beautifulsoup4 not installed — skipping .html files")
    except Exception as e:
        logger.warning("HTML loader error (skipping): %s", e)

    # ── CSV ───────────────────────────────────────────────────────────────────
    try:
        for fpath in source_dir.rglob("*.csv"):
            try:
                docs = CSVLoader(str(fpath)).load()
                all_docs.extend(_enrich_metadata(docs, "csv"))
            except Exception as e:
                logger.warning("Skipped %s: %s", fpath.name, e)
        logger.info("CSV loading complete")
    except Exception as e:
        logger.warning("CSV loader error (skipping): %s", e)

    logger.info("Total documents loaded: %d", len(all_docs))
    return all_docs
