
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import perf_counter

from ollama import chat

MODEL = os.getenv("PROMPTTREMOR_MODEL", "gemma4:e4b")
BENCHMARK_FILE = Path("benchmark.json")
RESULTS_DIR = Path("results")
RESULTS_FILE = RESULTS_DIR / "latest_run.json"


def normalize(text):
    """Normalize whitespace and surrounding punctuation for strict scoring."""
    text = text.strip().casefold()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\n\r.,!?;:\"'`")


def main():
    if not BENCHMARK_FILE.exists():
        raise FileNotFoundError(
            f"Benchmark file not found: {BENCHMARK_FILE}"
        )

    with BENCHMARK_FILE.open(encoding="utf-8") as file:
        benchmark = json.load(file)

    results = []
    fully_correct_tasks = 0
    task_count = len(benchmark)

    for task in benchmark:
        task_passed = True

        print(f"\nTesting {task['id']} ({task['category']})")

        for index, prompt in enumerate(task["prompts"], start=1):
            start = perf_counter()
            response_text = ""
            error = None

            try:
                response = chat(
                    model=MODEL,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    options={"temperature": 0},
                )
                response_text = response.message.content.strip()

            except Exception as exc:
                error = str(exc)

            latency = perf_counter() - start
            correct = (
                error is None
                and normalize(response_text)
                == normalize(task["expected"])
            )

            if not correct:
                task_passed = False

            result = {
                "task_id": task["id"],
                "category": task["category"],
                "variant": index,
                "prompt": prompt,
                "expected": task["expected"],
                "response": response_text,
                "correct": correct,
                "latency_seconds": round(latency, 3),
                "error": error,
            }
            results.append(result)

            status = "PASS" if correct else "FAIL"
            print(
                f"  Variant {index}: {status} | "
                f"{latency:.2f}s | Response: {response_text!r}"
            )

            if error:
                print(f"    Error: {error}")

        if task_passed:
            fully_correct_tasks += 1

    total = len(results)
    correct_count = sum(row["correct"] for row in results)
    valid_latencies = [
        row["latency_seconds"]
        for row in results
        if row["error"] is None
    ]

    summary = {
        "model": MODEL,
        "total_tasks": task_count,
        "total_variations": total,
        "correct_responses": correct_count,
        "accuracy_percent": round(
            100 * correct_count / total, 2
        ) if total else 0,
        "fully_robust_tasks": fully_correct_tasks,
        "strict_robustness_percent": round(
            100 * fully_correct_tasks / task_count, 2
        ) if task_count else 0,
        "average_latency_seconds": round(
            mean(valid_latencies), 3
        ) if valid_latencies else None,
    }

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "results": results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== PromptTremor Summary ===")
    print(f"Model: {MODEL}")
    print(
        f"Accuracy: {summary['accuracy_percent']}% "
        f"({correct_count}/{total})"
    )
    print(
        f"Strict robustness: "
        f"{summary['strict_robustness_percent']}%"
    )
    print(
        f"Average latency: "
        f"{summary['average_latency_seconds']} seconds"
    )
    print(f"Report saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()

