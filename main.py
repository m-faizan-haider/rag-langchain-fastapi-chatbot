# main.py
"""
Entrypoint for your backend RAG system.

Structure expected:
- Backend/          (package with modules)
- data/sample_docs/ (documents)
- results/          (will be used for results/results.txt)
- venv/             (your virtualenv; not created by this script)
"""

import argparse
from Backend.faiss_manager import check_index, build_or_update_faiss_index
from Backend.retriever import load_vectorstore_and_check
from Backend.rag_query_rerank import ask_question

def run_interactive(show_sources: bool = False):
    """
    Ensure index exists, load vectorstore, run interactive Q&A loop.
    """
    if not check_index():
        print("⚠️ FAISS index missing or out-of-date. Building now...")
        build_or_update_faiss_index()

    vectorstore, metadata = load_vectorstore_and_check()
    print("🟢 RAG Interactive QA (type 'exit' to quit)\n")
    while True:
        user_query = input("What is your question? ").strip()
        if user_query.lower() == "exit":
            print("Exiting. Goodbye!")
            break
        ask_question(vectorstore, user_query, show_sources=show_sources)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--show-sources", action="store_true", help="Display document sources in terminal")
    args = parser.parse_args()
    run_interactive(show_sources=args.show_sources)
