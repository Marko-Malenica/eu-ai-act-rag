"""
Usage:
    docker compose exec backend python scripts/run_eval.py
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from backend.confidence import compute_confidence, llm
from backend.rag import query_eu_ai_act

DATASET_PATH = PROJECT_ROOT / "eval" / "dataset.json"
RESULTS_PATH = PROJECT_ROOT / "eval" / "results.json"

JUDGE_PROMPT = PromptTemplate.from_template("""You are grading a RAG system's answer against a ground-truth summary.

Question: {question}

Ground truth answer: {ground_truth}

System's answer: {predicted}

Does the system's answer correctly convey the key facts of the ground truth answer?
Minor wording differences are fine. Missing or contradicting the key facts is a fail.

Respond with exactly one word on the first line: PASS or FAIL.""")

judge_chain = JUDGE_PROMPT | llm | StrOutputParser()

RECALL_PROMPT = PromptTemplate.from_template("""You are checking whether retrieved context is sufficient to answer a question about the EU AI Act.

Question: {question}

Expected answer: {ground_truth}

Retrieved context:
{context}

Does the retrieved context above contain the information needed to construct the expected
answer? Answer YES if the key facts from the expected answer are present (even if phrased
differently, or spread across the chunks). Answer NO if the context is missing the key facts
or discusses a different provision entirely.

Respond with exactly one word on the first line: YES or NO.""")

recall_chain = RECALL_PROMPT | llm | StrOutputParser()


def recall_at_5(question: str, ground_truth: str, source_docs) -> bool:
    context = "\n\n".join(doc.page_content for doc in source_docs)
    verdict = recall_chain.invoke({
        "question": question,
        "ground_truth": ground_truth,
        "context": context,
    })
    first_line = verdict.strip().splitlines()[0].strip().upper()
    return first_line.startswith("YES")


def grade_answer(question: str, ground_truth: str, predicted: str) -> bool:
    verdict = judge_chain.invoke({
        "question": question,
        "ground_truth": ground_truth,
        "predicted": predicted,
    })
    first_line = verdict.strip().splitlines()[0].strip().upper()
    return first_line.startswith("PASS")


def bucket_for(confidence: float) -> str:
    if confidence < 0.6:
        return "0.0-0.6"
    if confidence < 0.8:
        return "0.6-0.8"
    return "0.8-1.0"


def run() -> list[dict]:
    dataset = json.loads(DATASET_PATH.read_text())
    results = []

    for i, item in enumerate(dataset, 1):
        question = item["question"]
        print(f"[{i}/{len(dataset)}] {question[:70]}")

        rag_result = query_eu_ai_act(question)
        confidence, out_of_scope_flag = compute_confidence(
            question=question,
            answer=rag_result["answer"],
            source_docs=rag_result["source_docs"],
            similarities=rag_result["similarities"],
        )

        record = {
            "question": question,
            "in_scope": item["in_scope"],
            "reference": item.get("reference"),
            "ground_truth_answer": item.get("ground_truth_answer"),
            "predicted_answer": rag_result["answer"],
            "confidence": float(confidence),
            "flagged_out_of_scope": out_of_scope_flag is not None,
            "sources": rag_result["sources"],
        }

        if item["in_scope"]:
            record["recall_at_5"] = recall_at_5(
                question, item["ground_truth_answer"], rag_result["source_docs"]
            )
            record["answer_correct"] = grade_answer(
                question, item["ground_truth_answer"], rag_result["answer"]
            )
        else:
            record["recall_at_5"] = None
            record["answer_correct"] = None

        print(f"    confidence={record['confidence']}  flagged_oos={record['flagged_out_of_scope']}"
              f"  recall@5={record['recall_at_5']}  correct={record['answer_correct']}")

        results.append(record)

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nRaw results written to {RESULTS_PATH}")
    return results


def summarize(results: list[dict]) -> None:
    in_scope = [r for r in results if r["in_scope"]]
    out_of_scope = [r for r in results if not r["in_scope"]]

    oos_correct = sum(1 for r in out_of_scope if r["flagged_out_of_scope"])
    is_false_positive = sum(1 for r in in_scope if r["flagged_out_of_scope"])

    recall_evaluable = [r for r in in_scope if r.get("recall_at_5") is not None]
    recall_hits = sum(1 for r in recall_evaluable if r["recall_at_5"])

    correct = sum(1 for r in in_scope if r["answer_correct"])

    buckets: dict[str, list[dict]] = {"0.0-0.6": [], "0.6-0.8": [], "0.8-1.0": []}
    for r in in_scope:
        buckets[bucket_for(r["confidence"])].append(r)

    print("\n=== Scope detection ===")
    print(f"Out-of-scope correctly flagged: {oos_correct}/{len(out_of_scope)}")
    print(f"In-scope false-positive flags:  {is_false_positive}/{len(in_scope)}")

    print("\n=== Retrieval Recall@5 ===")
    if recall_evaluable:
        print(f"{recall_hits}/{len(recall_evaluable)} ({recall_hits / len(recall_evaluable):.1%})")
    else:
        print("n/a - no references had extractable labels")

    print("\n=== Answer accuracy (LLM-judged) ===")
    print(f"{correct}/{len(in_scope)} ({correct / len(in_scope):.1%})")

    print("\n=== Calibration ===")
    ece = 0.0
    for label, items in buckets.items():
        if not items:
            print(f"{label}: n=0")
            continue
        avg_conf = sum(r["confidence"] for r in items) / len(items)
        acc = sum(1 for r in items if r["answer_correct"]) / len(items)
        gap = abs(avg_conf - acc)
        ece += (len(items) / len(in_scope)) * gap
        print(f"{label}: n={len(items)}  avg_confidence={avg_conf:.2f}  accuracy={acc:.2f}  gap={gap:.2f}")
    print(f"\nExpected Calibration Error (ECE): {ece:.3f}")


if __name__ == "__main__":
    summarize(run())
