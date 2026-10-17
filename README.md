# PromptTremor

**A local-first LLM prompt robustness testing harness.**

PromptTremor evaluates how language models respond to different prompt variations. It measures answer accuracy, strict robustness, and inference latency, and provides a Streamlit dashboard to inspect results and compare models.

## Features

- Test prompt variations across multiple tasks.
- Evaluate responses against expected answers.
- Measure accuracy, strict robustness, and latency.
- Compare models using saved benchmark reports.
- Run models locally through Ollama.
- Export benchmark results as CSV.

## Tech Stack

Python · Ollama · Streamlit · Pandas · Altair

## Getting Started

### 1. Prerequisites

Install [Python](https://www.python.org/), [Git](https://git-scm.com/), and [Ollama](https://ollama.com/).

### 2. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd PromptTremor
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Linux — Bash**
```bash
source .venv/bin/activate
```

**Linux — Fish**
```fish
source .venv/bin/activate.fish
```

### 4. Install dependencies

```bash
pip install ollama streamlit pandas altair
```

### 5. Download a model

For example, using Gemma 4 E4B:

```bash
ollama pull gemma4:e4b
```

Other supported models in the project include `gemma4:e2b` and `phi4-mini`.

### 6. Run the benchmark

```bash
python runner.py
```

To select a different model:

```bash
env PROMPTTREMOR_MODEL=phi4-mini python runner.py
```

The benchmark uses `benchmark.json` and saves the latest report to `results/latest_run.json`. Model-specific reports are saved under `results/model_runs/`.

### 7. Launch the dashboard

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal. The dashboard displays benchmark metrics, individual responses, and comparisons from saved model reports.

## Project Structure

```text
PromptTremor/
├── app.py
├── runner.py
├── benchmark.json
├── results/
├── README.md
├── LICENSE
└── .gitignore
```

## Evaluation Metrics

- **Accuracy:** Percentage of responses matching expected answers.
- **Strict robustness:** Percentage of tasks for which every prompt variation passes.
- **Average latency:** Mean response time across successful model calls.

## Limitations

The benchmark is experimental and currently contains a small set of test cases. Exact-answer matching can mark semantically correct responses as incorrect when their formatting differs. Results should not be interpreted as a definitive ranking of models.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

Gemma models are distributed under their own applicable terms. Refer to the [Gemma model documentation](https://ai.google.dev/gemma/docs) for details.
