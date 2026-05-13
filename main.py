# main.py
"""
Entrypoint for interactive CLI mode.
For the HTTP API server, run:
    uvicorn Backend.api:app --reload --port 8000

Structure:
  Backend/          - Python package with all modules
  data/sample_docs/ - Source documents (PDF, TXT, DOCX, MD, HTML, CSV)
  results/          - Query logs and app logs
  venv/             - Virtual environment (not tracked by git)
"""
import argparse
import logging

from Backend.logging_config import setup_logging
from Backend.faiss_manager import check_index, build_or_update_faiss_index
from Backend.retriever import load_vectorstore_and_check
from Backend.rag_query_rerank import ask_question

setup_logging(log_level="INFO")
logger = logging.getLogger(__name__)


def run_interactive(show_sources: bool = False):
    """Ensure index exists, load vectorstore, run interactive Q&A loop."""
    if not check_index():
        logger.warning("FAISS index missing or out-of-date. Building now...")
        build_or_update_faiss_index()

    vectorstore, metadata = load_vectorstore_and_check()
    print("\n🟢 RAG Interactive QA (type 'exit' to quit)\n")

    while True:
        try:
            user_query = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting. Goodbye!")
            break

        if user_query.lower() in ("exit", "quit", "q"):
            print("Exiting. Goodbye!")
            break

        if not user_query:
            continue

        result = ask_question(vectorstore, user_query, show_sources=show_sources)
        final_text, picked = result

        print("\n=== ANSWER ===")
        print(final_text.strip())
        print()

        if show_sources:
            from pathlib import Path
            print("=== SOURCES ===")
            for score, doc in picked:
                print(f"  [{round(score,1)}] {Path(doc.metadata.get('source', '?')).name} | page {doc.metadata.get('page','N/A')}")
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG LangChain FastAPI Chatbot — CLI mode")
    parser.add_argument("--show-sources", action="store_true", help="Display document sources in terminal")
    args = parser.parse_args()
    run_interactive(show_sources=args.show_sources)
