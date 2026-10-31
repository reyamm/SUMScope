# SUMScore — Retrieval Evaluation Dashboard

A Streamlit interface for examining recorded relevance scores from the SUMScore experiment.

## Run locally

Place this folder at `~/Desktop/Humain/SUMScore_GitHub_Ready`.
Keep the original result files in `~/Desktop/Humain/q50`.

```bash
cd ~/Desktop/Humain/SUMScore_GitHub_Ready
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Data

The app reads `probes_<domain>.json` and `table_cost.csv` from `~/Desktop/Humain/q50`.
It uses `table_accuracy.csv`, `embed_<domain>.json`, and cached
`summaries_<domain>.json` when available. An optional `queries_<domain>.json`
can map saved query IDs to their exact original benchmark text.

## Interpretation

All question-specific model scores were generated in an earlier GPU experiment.
The dashboard performs a CPU-only replay, not live language-model inference.
It supports inspecting rankings, document-summary caches, and measured online
LLM call/token counts. New questions cannot be scored with the saved files alone.

The local result files are deliberately excluded from the Git repository.
