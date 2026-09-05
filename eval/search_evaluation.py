"""
eval/search_evaluation.py

Retrieval evaluation for knowledge_base.py, following
07-project-example/lessons/02-evaluating-retrieval.md exactly: generate
ground-truth questions per document with an LLM, then measure Hit Rate
and MRR for each retrieval method.

This is the piece that lets the project rubric's "Retrieval evaluation"
criterion say "multiple retrieval approaches are evaluated, and the best
one is used" (2/2) instead of evaluating only one approach: we compare
keyword search, vector search, and hybrid (RRF) search against each
other and print which wins.

Run with: python eval/search_evaluation.py
Requires documents already ingested (run ingest_kb.py first).
Writes: eval/search_ground_truth.csv, eval/search_evaluation_results.csv
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from db_documents import get_all_documents  # noqa: E402
from evaluation_utils import calc_total_price, get_client, llm_structured  # noqa: E402
from knowledge_base import KnowledgeBase  # noqa: E402

client = get_client()

GROUND_TRUTH_PATH = Path(__file__).resolve().parent / "search_ground_truth.csv"
RESULTS_PATH = Path(__file__).resolve().parent / "search_evaluation_results.csv"

data_gen_instructions = """
You emulate a retail investor researching stocks. Given a document from
a company-research knowledge base (a company overview or a news
article), formulate 2 questions this document would answer. Use
different wording from the document itself -- write the way people
actually search, not too formal, not too short, not too long.
""".strip()


class Questions(BaseModel):
    questions: list[str]


# ----------------------------------------------------------------------
# Step 1: ground truth generation (mirrors 02-ground-truth.md)
# ----------------------------------------------------------------------

def generate_ground_truth(documents: list[dict]) -> tuple[list[dict], list]:
    records = []
    usages = []

    for doc in tqdm(documents, desc="Generating ground truth"):
        user_prompt = f"Title: {doc['title']}\n\nText: {doc['text']}"
        result, usage = llm_structured(client, data_gen_instructions, user_prompt, Questions)
        usages.append(usage)
        for q in result.questions:
            records.append({"question": q, "document": doc["id"]})

    return records, usages


# ----------------------------------------------------------------------
# Step 2: Hit Rate / MRR (mirrors 07-project-example/lessons/02-evaluating-retrieval.md)
# ----------------------------------------------------------------------

def hit_rate(relevance_total):
    return sum(True in line for line in relevance_total) / len(relevance_total)


def mrr(relevance_total):
    total_score = 0.0
    for line in relevance_total:
        for rank in range(len(line)):
            if line[rank]:
                total_score += 1 / (rank + 1)
                break
    return total_score / len(relevance_total)


def evaluate(ground_truth, search_function):
    relevance_total = []
    for q in tqdm(ground_truth, desc="Evaluating"):
        doc_id = q["document"]
        results = search_function(q["question"])
        relevance = [d["id"] == doc_id for d in results]
        relevance_total.append(relevance)
    return {"hit_rate": hit_rate(relevance_total), "mrr": mrr(relevance_total)}


def main():
    documents = get_all_documents()
    if not documents:
        raise SystemExit("No documents in the knowledge base. Run ingest_kb.py first.")

    kb = KnowledgeBase(documents)

    if GROUND_TRUTH_PATH.exists():
        print(f"Reusing existing ground truth at {GROUND_TRUTH_PATH}")
        ground_truth = pd.read_csv(GROUND_TRUTH_PATH).to_dict(orient="records")
    else:
        ground_truth, usages = generate_ground_truth(documents)
        pd.DataFrame(ground_truth).to_csv(GROUND_TRUTH_PATH, index=False)
        print(f"Wrote {len(ground_truth)} ground-truth questions to {GROUND_TRUTH_PATH}")
        print(f"Generation cost: ${calc_total_price(usages):.5f}")

    methods = {
        "keyword": lambda q: kb.keyword_search(q, num_results=5),
        "vector": lambda q: kb.vector_search(q, num_results=5),
        "hybrid": lambda q: kb.hybrid_search(q, num_results=5),
    }

    results = {}
    for name, fn in methods.items():
        print(f"\nEvaluating {name} search...")
        results[name] = evaluate(ground_truth, fn)

    df_results = pd.DataFrame(results).T
    df_results.to_csv(RESULTS_PATH)

    print("\n=== Summary ===")
    print(df_results.to_string())

    best = df_results["hit_rate"].idxmax()
    print(f"\nBest method by Hit Rate: {best}")
    print(
        "Update knowledge_base.KnowledgeBase.hybrid_search / the tool's default method "
        "in tools.py if a single method (not hybrid) wins clearly -- same manual "
        "step the course takes after 06-search-tuning.md."
    )


if __name__ == "__main__":
    main()
