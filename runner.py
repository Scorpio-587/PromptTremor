
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import perf_counter

from ollama import chat

MODEL = os.getenv("PROMPTTREMOR_MODEL", "gemma4:e4b")
BASE_DIR = Path(__file__).resolve().parent
BENCHMARK_FILE = BASE_DIR / "benchmark.json"
RESULTS_FILE = BASE_DIR / "results" / "latest_run.json"
MODEL_RUNS_DIR = BASE_DIR / "results" / "model_runs"


def normalize(text):
    text = text.strip().casefold()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\n\r.,!?;:\"'`")


def main():
    with BENCHMARK_FILE.open(encoding="utf-8") as file:
        benchmark = json.load(file)

    results = []

    for task in benchmark:
        print(f"\nTesting {task['id']} ({task['category']})")

        for index, variant in enumerate(task["variants"], start=1):
            prompt = variant["prompt"]
            response_text = ""
            error = None
            start = perf_counter()

            try:
                response = chat(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
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

            result = {
                "task_id": task["id"],
                "category": task["category"],
                "variant": index,
                "variant_type": variant["type"],
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
                f"  {variant['type']}: {status} | "
                f"{latency:.2f}s | {response_text!r}"
            )

            if error:
                print(f"    Error: {error}")

    total = len(results)
    correct_count = sum(r["correct"] for r in results)

    task_passes = []
    for task in benchmark:
        task_results = [
            r for r in results if r["task_id"] == task["id"]
        ]
        task_passes.append(
            bool(task_results) and all(r["correct"] for r in task_results)
        )

    valid_latencies = [
        r["latency_seconds"] for r in results if r["error"] is None
    ]

    summary = {
        "model": MODEL,
        "total_tasks": len(benchmark),
        "total_variations": total,
        "correct_responses": correct_count,
        "accuracy_percent": round(
            100 * correct_count / total, 2
        ) if total else 0,
        "fully_robust_tasks": sum(task_passes),
        "strict_robustness_percent": round(
            100 * sum(task_passes) / len(benchmark), 2
        ) if benchmark else 0,
        "average_latency_seconds": round(
            mean(valid_latencies), 3
        ) if valid_latencies else None,
    }

    # Accuracy by perturbation type
    type_accuracy = {}
    for kind in sorted({r["variant_type"] for r in results}):
        rows = [r for r in results if r["variant_type"] == kind]
        type_accuracy[kind] = round(
            100 * sum(r["correct"] for r in rows) / len(rows), 2
        )

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "accuracy_by_variant_type": type_accuracy,
        "results": results,
    }

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    MODEL_RUNS_DIR.mkdir(parents=True, exist_ok=True)

    model_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", MODEL)
    model_report_file = MODEL_RUNS_DIR / f"{model_slug}.json"

    model_report_file.write_text(
    json.dumps(report, indent=2, ensure_ascii=False),
    encoding="utf-8",
    )

    print(f"Model-specific report saved to: {model_report_file}")

    print("\n=== PromptTremor Summary ===")
    print(f"Model: {MODEL}")
    print(f"Accuracy: {summary['accuracy_percent']}%")
    print(f"Strict robustness: {summary['strict_robustness_percent']}%")
    print("Accuracy by perturbation type:")
    for kind, accuracy in type_accuracy.items():
        print(f"  {kind}: {accuracy}%")

    print(f"Average latency: {summary['average_latency_seconds']}s")
    print(f"Report saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()

