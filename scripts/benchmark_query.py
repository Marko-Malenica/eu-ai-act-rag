"""
Phase 1 latency benchmark. Hits the running /query endpoint N times and
reports timing stats. Run once before the Phase 1 fix and once after,
against the same question set, to get an honest before/after comparison.

Usage:
    python scripts/benchmark_query.py --n 20
    python scripts/benchmark_query.py --n 20 --out results/before.json
"""
import argparse
import json
import statistics
import time
import uuid

import requests

QUESTIONS = [
    "What is a high-risk AI system under the EU AI Act?",
    "What are the transparency obligations for AI systems?",
    "What is the definition of an AI system in the regulation?",
    "What are the penalties for non-compliance with the EU AI Act?",
    "Which AI practices are prohibited under the Act?",
    "What obligations do providers of high-risk AI systems have?",
    "What is a notified body?",
    "How does the Act define a deployer?",
    "What are the requirements for conformity assessment?",
    "What is the role of the AI Office?",
]


def run(n: int, base_url: str) -> list[float]:
    session_id = str(uuid.uuid4())
    timings = []
    for i in range(n):
        question = QUESTIONS[i % len(QUESTIONS)]
        start = time.perf_counter()
        resp = requests.post(
            f"{base_url}/query",
            json={"question": question, "session_id": session_id},
            timeout=120,
        )
        elapsed = time.perf_counter() - start
        resp.raise_for_status()
        timings.append(elapsed)
        print(f"[{i + 1}/{n}] {elapsed:.2f}s  confidence={resp.json().get('confidence')}")
    return timings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--out", default=None, help="optional path to dump raw timings as JSON")
    args = parser.parse_args()

    timings = run(args.n, args.base_url)

    print("\n--- results ---")
    print(f"n:      {len(timings)}")
    print(f"mean:   {statistics.mean(timings):.2f}s")
    print(f"median: {statistics.median(timings):.2f}s")
    print(f"stdev:  {statistics.stdev(timings):.2f}s")
    print(f"min:    {min(timings):.2f}s")
    print(f"max:    {max(timings):.2f}s")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"timings": timings}, f, indent=2)
        print(f"\nraw timings written to {args.out}")


if __name__ == "__main__":
    main()
