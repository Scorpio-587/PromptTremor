import json
import os
import subprocess
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
RUNNER_FILE = BASE_DIR / "runner.py"
RESULTS_FILE = BASE_DIR / "results" / "latest_run.json"
MODEL_RUNS_DIR = BASE_DIR / "results" / "model_runs"
AVAILABLE_MODELS = ["gemma4:e4b", "gemma4:e2b", "phi4-mini"]

st.set_page_config(
    page_title="PromptTremor",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 PromptTremor")
st.caption("Explore how prompt variations affect local LLM reliability.")

# Sidebar controls
with st.sidebar:
    st.header("Experiment")
    selected_model = st.selectbox(
        "Model to run",
        options=AVAILABLE_MODELS,
        index=0,
    )
    st.write("Runtime: Ollama")
    st.write("Evaluation: expected-answer matching")

    run_button = st.button(
        "▶ Run benchmark",
        type="primary",
        use_container_width=True,
    )

# Run the benchmark with the selected model.
if run_button:
    if not RUNNER_FILE.exists():
        st.error("runner.py was not found in the project directory.")
    else:
        try:
            run_env = os.environ.copy()
            run_env["PROMPTTREMOR_MODEL"] = selected_model

            with st.spinner(f"Running benchmark against {selected_model}..."):
                completed = subprocess.run(
                    [sys.executable, str(RUNNER_FILE)],
                    cwd=BASE_DIR,
                    env=run_env,
                    capture_output=True,
                    text=True,
                    timeout=900,
                    check=False,
                )

            output = completed.stdout or ""
            if completed.stderr:
                output += "\n" + completed.stderr
            st.session_state["runner_output"] = output

            if completed.returncode != 0:
                st.error("The benchmark failed. See the output below for details.")
            elif not RESULTS_FILE.exists():
                st.error("The benchmark finished without creating a report.")
            else:
                st.success(f"Benchmark completed for {selected_model}!")

        except subprocess.TimeoutExpired:
            st.error("The benchmark timed out after 15 minutes.")
        except OSError as exc:
            st.error(f"Could not start the benchmark: {exc}")

if "runner_output" in st.session_state:
    with st.expander("Benchmark terminal output", expanded=False):
        st.code(st.session_state["runner_output"])

# Load the most recent run. Continue rendering the comparison section even if
# this file is absent or invalid, because saved per-model reports may still exist.
report = None
summary = None
df = pd.DataFrame()

if RESULTS_FILE.exists():
    try:
        with RESULTS_FILE.open(encoding="utf-8") as file:
            candidate_report = json.load(file)

        if not isinstance(candidate_report, dict):
            raise ValueError("The report must contain a JSON object.")
        if not isinstance(candidate_report.get("summary"), dict):
            raise ValueError("The report is missing a valid 'summary' object.")
        if not isinstance(candidate_report.get("results"), list):
            raise ValueError("The report is missing a valid 'results' list.")

        report = candidate_report
        summary = report["summary"]
        df = pd.DataFrame(report["results"])

    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        st.error(f"Could not read the latest benchmark report: {exc}")
else:
    st.info("No latest benchmark report yet. Run a benchmark or view saved model runs below.")

# Latest run overview and charts
if report is not None:
    st.caption(f"Model: {summary.get('model', 'Unknown model')}")
    st.caption(f"Run timestamp: {report.get('timestamp', 'Unknown')}")

    st.subheader("Experiment overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Accuracy",
        f"{summary.get('accuracy_percent', 0):.1f}%",
        f"{summary.get('correct_responses', 0)} / "
        f"{summary.get('total_variations', len(df))} responses",
    )
    col2.metric(
        "Strict robustness",
        f"{summary.get('strict_robustness_percent', 0):.1f}%",
        f"{summary.get('fully_robust_tasks', 0)} / "
        f"{summary.get('total_tasks', 0)} tasks",
    )

    latency = summary.get("average_latency_seconds")
    col3.metric(
        "Average latency",
        f"{latency:.2f} s" if isinstance(latency, (int, float)) else "N/A",
    )
    col4.metric("Prompt variations", summary.get("total_variations", len(df)))

    if df.empty:
        st.warning("The latest report contains no prompt results.")
    else:
        if "correct" in df.columns:
            df["correct"] = df["correct"].fillna(False).astype(bool)
        else:
            st.warning("The latest report has no 'correct' field; accuracy charts are unavailable.")

        st.divider()

        if {"category", "correct"}.issubset(df.columns):
            st.subheader("Performance by category")
            category_stats = (
                df.groupby("category")["correct"]
                .mean()
                .mul(100)
                .rename("Accuracy (%)")
            )
            st.bar_chart(category_stats)

        if {"variant_type", "correct"}.issubset(df.columns):
            st.subheader("Accuracy by perturbation type")
            variant_stats = (
                df.groupby("variant_type")["correct"]
                .mean()
                .mul(100)
                .rename("Accuracy (%)")
            )
            variant_chart_data = variant_stats.rename_axis("variant_type").reset_index()
            variant_chart_data.columns = ["variant_type", "accuracy"]
            variant_chart = (
                alt.Chart(variant_chart_data)
                .mark_bar()
                .encode(
                    x=alt.X("variant_type:N", title="Perturbation type", sort=None),
                    y=alt.Y(
                        "accuracy:Q",
                        title="Accuracy (%)",
                        scale=alt.Scale(domain=[0, 100]),
                        axis=alt.Axis(values=[0, 20, 40, 60, 80, 100]),
                    ),
                    tooltip=[
                        "variant_type:N",
                        alt.Tooltip("accuracy:Q", format=".1f"),
                    ],
                )
            )
            st.altair_chart(variant_chart, use_container_width=True)
        else:
            st.info("This report does not contain perturbation labels.")

# Load saved per-model reports for comparison.
st.divider()
st.subheader("Compare saved model runs")
saved_reports = []

if MODEL_RUNS_DIR.exists():
    for report_path in sorted(MODEL_RUNS_DIR.glob("*.json")):
        try:
            with report_path.open(encoding="utf-8") as file:
                saved_report = json.load(file)

            if (
                isinstance(saved_report, dict)
                and isinstance(saved_report.get("summary"), dict)
                and isinstance(saved_report.get("results"), list)
            ):
                saved_reports.append(saved_report)
            else:
                st.warning(f"Skipping invalid report format: {report_path.name}")

        except (OSError, json.JSONDecodeError):
            st.warning(f"Could not read {report_path.name}")

if not saved_reports:
    st.info(
        "No saved model reports yet. Run the benchmark for each model to "
        "populate this comparison. Per-model reports are saved by runner.py "
        "when its model-specific report-saving code is enabled."
    )
else:
    def benchmark_signature(saved_report):
        return tuple(sorted(
            (
                str(row.get("task_id", "")),
                str(row.get("variant_type", "")),
                str(row.get("prompt", "")),
                str(row.get("expected", "")),
            )
            for row in saved_report["results"]
        ))

    signatures = {benchmark_signature(saved_report) for saved_report in saved_reports}
    if len(signatures) == 1:
        st.caption("All saved reports use the same test cases.")
    else:
        st.warning(
            "These reports use different test cases. Compare their scores cautiously; "
            "rerun all models on the same benchmark for a fair comparison."
        )

    comparison_rows = []
    for saved_report in saved_reports:
        saved_summary = saved_report["summary"]
        comparison_rows.append({
            "Model": saved_summary.get("model", "Unknown"),
            "Accuracy (%)": saved_summary.get("accuracy_percent", 0),
            "Strict robustness (%)": saved_summary.get("strict_robustness_percent", 0),
            "Average latency (s)": saved_summary.get("average_latency_seconds"),
            "Responses": saved_summary.get("total_variations", 0),
        })

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df = (
        comparison_df.drop_duplicates(subset=["Model"], keep="last")
        .sort_values("Model")
        .reset_index(drop=True)
    )

    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    st.markdown("**Accuracy and strict robustness**")
    scores = comparison_df.melt(
        id_vars=["Model"],
        value_vars=["Accuracy (%)", "Strict robustness (%)"],
        var_name="Metric",
        value_name="Score",
    )
    scores["Score"] = pd.to_numeric(scores["Score"], errors="coerce")

    score_chart = (
        alt.Chart(scores.dropna(subset=["Score"]))
        .mark_bar()
        .encode(
            x=alt.X("Model:N", title="Model"),
            xOffset=alt.XOffset("Metric:N"),
            y=alt.Y(
                "Score:Q",
                title="Percentage (%)",
                scale=alt.Scale(domain=[0, 100]),
                axis=alt.Axis(values=[0, 20, 40, 60, 80, 100]),
            ),
            color=alt.Color("Metric:N", title="Metric"),
            tooltip=[
                "Model:N",
                "Metric:N",
                alt.Tooltip("Score:Q", format=".1f"),
            ],
        )
    )
    st.altair_chart(score_chart, use_container_width=True)

    st.markdown("**Average inference latency**")
    latency_df = comparison_df.dropna(subset=["Average latency (s)"])
    if not latency_df.empty:
        latency_chart = (
            alt.Chart(latency_df)
            .mark_bar()
            .encode(
                x=alt.X("Model:N", title="Model"),
                y=alt.Y("Average latency (s):Q", title="Average latency (seconds)"),
                tooltip=["Model:N", alt.Tooltip("Average latency (s):Q", format=".3f")],
            )
        )
        st.altair_chart(latency_chart, use_container_width=True)
    else:
        st.info("No latency measurements are available in the saved reports.")

    detail_rows = []
    for saved_report in saved_reports:
        model_name = saved_report["summary"].get("model", "Unknown")
        for result in saved_report["results"]:
            if result.get("variant_type"):
                detail_rows.append({
                    "Model": model_name,
                    "Perturbation": result["variant_type"],
                    "Correct": bool(result.get("correct", False)),
                })

    if detail_rows:
        detail_df = pd.DataFrame(detail_rows)
        perturbation_comparison = (
            detail_df.groupby(["Perturbation", "Model"], as_index=False)["Correct"]
            .mean()
            .assign(Accuracy=lambda data: data["Correct"] * 100)
        )

        st.markdown("**Accuracy by perturbation type (%)**")
        pivot = perturbation_comparison.pivot(
            index="Perturbation", columns="Model", values="Accuracy"
        ).round(1)
        st.dataframe(pivot, use_container_width=True)

        perturbation_chart = (
            alt.Chart(perturbation_comparison)
            .mark_bar()
            .encode(
                x=alt.X("Perturbation:N", title="Perturbation type"),
                xOffset=alt.XOffset("Model:N"),
                y=alt.Y(
                    "Accuracy:Q",
                    title="Accuracy (%)",
                    scale=alt.Scale(domain=[0, 100]),
                    axis=alt.Axis(values=[0, 20, 40, 60, 80, 100]),
                ),
                color=alt.Color("Model:N", title="Model"),
                tooltip=[
                    "Perturbation:N",
                    "Model:N",
                    alt.Tooltip("Accuracy:Q", format=".1f"),
                ],
            )
        )
        st.altair_chart(perturbation_chart, use_container_width=True)

# Inspect individual responses from the latest run.
if report is not None and not df.empty:
    st.divider()
    st.subheader("Inspect individual responses")

    if "category" in df.columns:
        categories = sorted(df["category"].dropna().unique().tolist())
        selected_categories = st.multiselect(
            "Filter categories",
            options=categories,
            default=categories,
        )
        filtered = df[df["category"].isin(selected_categories)].copy()
    else:
        filtered = df.copy()

    if "correct" in filtered.columns:
        show_failures_only = st.checkbox("Show failed responses only")
        if show_failures_only:
            filtered = filtered[~filtered["correct"]]

    if filtered.empty:
        st.info("No responses match the selected filters.")
    else:
        st_columns = [
            column for column in [
                "task_id", "category", "variant", "variant_type",
                "expected", "response", "correct", "latency_seconds",
            ]
            if column in filtered.columns
        ]
        st.dataframe(
            filtered[st_columns],
            use_container_width=True,
            hide_index=True,
        )

        if "prompt" in filtered.columns:
            st.subheader("Prompt details")
            selected_index = st.selectbox(
                "Choose a prompt result",
                options=filtered.index.tolist(),
                format_func=lambda idx: (
                    f"{filtered.loc[idx].get('task_id', 'Task')} — "
                    f"Variant {filtered.loc[idx].get('variant', '?')}"
                ),
            )
            selected = filtered.loc[selected_index]
            st.markdown("**Prompt**")
            st.code(str(selected.get("prompt", "")), language="text")
            st.markdown("**Expected answer**")
            st.write(selected.get("expected", "Not available"))
            st.markdown("**Actual answer**")
            st.write(selected.get("response", "Not available"))
            if selected.get("error"):
                st.error(f"Model error: {selected['error']}")

        st.divider()
        st.download_button(
            "Download results as CSV",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="prompttremor-results.csv",
            mime="text/csv",
        )

st.caption(
    "PromptTremor uses expected-answer matching. Results are limited by the "
    "benchmark size, test design, scoring rules, hardware, and run conditions."
)
