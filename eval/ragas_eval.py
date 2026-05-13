# eval/ragas_eval.py
"""
RAGAS evaluation script for the RAG pipeline.
Measures: Faithfulness, Answer Relevancy, Context Recall, Context Precision.

Usage:
    python eval/ragas_eval.py --questions eval/questions.json

Install:
    pip install ragas datasets

questions.json format:
[
  {
    "question": "What is cloud computing?",
    "ground_truth": "Cloud computing is the delivery of computing services..."
  },
  ...
]
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_questions(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_rag_for_question(question: str, vectorstore, debug: bool = True) -> dict:
    """Run the RAG pipeline and return structured results."""
    from Backend.rag_query_rerank import ask_question

    result = ask_question(vectorstore, question, show_sources=False, debug=True)
    final_text, picked, facts_text, verification = result

    contexts = [doc.page_content for _, doc in picked]

    return {
        "question":  question,
        "answer":    final_text,
        "contexts":  contexts,
        "facts":     facts_text,
    }


def evaluate(questions_path: str, output_path: str = "eval/results.json"):
    """Run RAGAS evaluation over a question set."""

    # ── Load questions ─────────────────────────────────────────────────────────
    questions_data = load_questions(questions_path)
    logger.info("Loaded %d questions from %s", len(questions_data), questions_path)

    # ── Load vectorstore ───────────────────────────────────────────────────────
    from Backend.faiss_manager import check_index, build_or_update_faiss_index
    from Backend.retriever import load_vectorstore_and_check

    if not check_index():
        logger.warning("FAISS index missing — building...")
        build_or_update_faiss_index()

    vectorstore, _ = load_vectorstore_and_check()
    logger.info("Vectorstore loaded.")

    # ── Run pipeline for each question ─────────────────────────────────────────
    results = []
    for i, item in enumerate(questions_data):
        question     = item["question"]
        ground_truth = item.get("ground_truth", "")
        logger.info("[%d/%d] %s", i + 1, len(questions_data), question[:80])

        t0 = time.time()
        res = run_rag_for_question(question, vectorstore)
        res["ground_truth"] = ground_truth
        res["elapsed_s"]    = round(time.time() - t0, 2)
        results.append(res)

    # ── RAGAS evaluation ──────────────────────────────────────────────────────
    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        )
        from datasets import Dataset

        logger.info("Running RAGAS evaluation...")
        dataset = Dataset.from_list([
            {
                "question":     r["question"],
                "answer":       r["answer"],
                "contexts":     r["contexts"],
                "ground_truth": r["ground_truth"],
            }
            for r in results
        ])

        ragas_result = ragas_evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        )

        scores = ragas_result.to_pandas().to_dict(orient="list")
        logger.info("RAGAS Scores:")
        for metric, vals in scores.items():
            if metric not in ("question", "answer", "contexts", "ground_truth"):
                avg = sum(v for v in vals if v is not None) / max(len(vals), 1)
                logger.info("  %-30s %.3f", metric, avg)

    except ImportError:
        logger.warning("ragas not installed — skipping metric computation. Run: pip install ragas datasets")
        scores = {}

    # ── Save results ──────────────────────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output = {
        "questions":  len(questions_data),
        "results":    results,
        "ragas_scores": scores,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info("Results saved to %s", output_path)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAGAS evaluation for the RAG pipeline")
    parser.add_argument(
        "--questions",
        default="eval/questions.json",
        help="Path to questions JSON file",
    )
    parser.add_argument(
        "--output",
        default="eval/results.json",
        help="Path to save evaluation results",
    )
    args = parser.parse_args()
    evaluate(args.questions, args.output)
