
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
RUNNER_FILE = BASE_DIR / "runner.py"
RESULTS_FILE = BASE_DIR / "results" / "latest_run.json"

st.set_page_config(
    page_title="PromptTremor",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 PromptTremor")
st.caption("Explore how prompt variations affect local LLM reliability.")

with st.sidebar:
    st.header("Experiment")
    st.write("Model: `gemma4:e4b`")
    st.write("Runtime: Ollama")
    st.write("Evaluation: expected-answer matching")

    run_button = st.button(
        "▶ Run benchmark",
        type="primary",
        use_container_width=True,
    )

if run_button:
    if not RUNNER_FILE.exists():
        st.error("runner.py was not found in the project directory.")
    else:
        try:
            with st.spinner(
                "Running benchmark against your local model..."
            ):
                completed = subprocess.run(
                    [sys.executable, str(RUNNER_FILE)],
                    cwd=BASE_DIR,
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
                st.error(
                    "The benchmark failed. Inspect the output below."
                )
            elif not RESULTS_FILE.exists():
                st.error("The benchmark finished without a report.")
            else:
                st.success("Benchmark completed successfully!")

        except subprocess.TimeoutExpired:
            st.error("The benchmark timed out after 15 minutes.")
        except OSError as exc:
            st.error(f"Could not start the benchmark: {exc}")

if "runner_output" in st.session_state:
    with st.expander("Benchmark terminal output", expanded=False):
        st.code(st.session_state["runner_output"])

if not RESULTS_FILE.exists():
    st.info(
        "No benchmark report yet. Click 'Run benchmark' "
        "to generate your first report."
    )
    st.stop()

try:
    with RESULTS_FILE.open(encoding="utf-8") as file:
        report = json.load(file)

    summary = report["summary"]
    results = report["results"]

except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
    st.error(f"Could not read the benchmark report: {exc}")
    st.stop()

st.caption(f"Model: {summary.get('model', 'Unknown model')}")
st.caption(f"Run timestamp: {report.get('timestamp', 'Unknown')}")

st.subheader("Experiment overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Accuracy",
    f"{summary.get('accuracy_percent', 0):.1f}%",
    f"{summary.get('correct_responses', 0)} / "
    f"{summary.get('total_variations', 0)} responses",
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
    f"{latency:.2f} s" if latency is not None else "N/A",
)

col4.metric(
    "Prompt variations",
    summary.get("total_variations", 0),
)

if not results:
    st.warning("The report contains no prompt results.")
    st.stop()

df = pd.DataFrame(results)
df["correct"] = df["correct"].fillna(False).astype(bool)

st.divider()
st.subheader("Performance by category")

category_stats = (
    df.groupby("category")["correct"]
    .mean()
    .mul(100)
    .rename("Accuracy (%)")
)

st.bar_chart(category_stats)

st.subheader("Accuracy by perturbation type")

if "variant_type" in df.columns:
    variant_stats = (
        df.groupby("variant_type")["correct"]
        .mean()
        .mul(100)
        .rename("Accuracy (%)")
    )

    st.bar_chart(variant_stats)
else:
    st.info("This report does not contain perturbation labels.")

st.divider()
st.subheader("Inspect individual responses")

categories = sorted(df["category"].dropna().unique().tolist())

selected_categories = st.multiselect(
    "Filter categories",
    options=categories,
    default=categories,
)

show_failures_only = st.checkbox("Show failed responses only")

filtered = df[df["category"].isin(selected_categories)].copy()

if show_failures_only:
    filtered = filtered[~filtered["correct"]]

columns_to_show = [
    "task_id",
    "category",
    "variant",
    "variant_type",
    "expected",
    "response",
    "correct",
    "latency_seconds",
]

st.dataframe(
    filtered[columns_to_show],
    use_container_width=True,
    hide_index=True,
)

st.subheader("Prompt details")

if not filtered.empty:
    selected_index = st.selectbox(
        "Choose a prompt result",
        options=filtered.index.tolist(),
        format_func=lambda idx: (
            f"{filtered.loc[idx, 'task_id']} — "
            f"Variant {filtered.loc[idx, 'variant']}"
        ),
    )

    selected = filtered.loc[selected_index]

    st.markdown("**Prompt**")
    st.code(str(selected["prompt"]), language="text")

    st.markdown("**Expected answer**")
    st.write(selected["expected"])

    st.markdown("**Actual answer**")
    st.write(selected["response"])

    if selected.get("error"):
        st.error(f"Model error: {selected['error']}")

st.divider()

csv_data = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download results as CSV",
    data=csv_data,
    file_name="prompttremor-results.csv",
    mime="text/csv",
)

st.caption(
    "PromptTremor uses strict expected-answer matching. "
    "Benchmark accuracy is limited by the task set and scoring rules."
)

