# SUMScope ( SUMScore ) the name is TBD haha

**Cache understanding. Cut compute.**

A compute-efficient long-document reranking experiment built for **HUMAIN Challenge 2 — Deliver AI with Less Compute**.

**Author:** Reyam Albalihi

SUMScope explores a simple question: *Can we preserve Periscope-style relevance judgments while avoiding repeated global document probing across queries?* It keeps Periscope's query-specific local probes and replaces its per-query strided probes with one **cached, query-independent document summary** that is scored once per query.

## Overview

```text
OFFLINE (per document)
Long document -> 100-token chunks -> hierarchical summary -> persisted summary cache

ONLINE (per query)
Query -> Qwen3 embedding retrieval -> top-100 candidate documents
      -> K local relevance probes + 1 cached summary relevance probe
      -> score = max(local probes) + summary probe -> reranked documents
```

The evaluation compares six methods on the same query-specific top-100 candidate pools: embedding-only, local rows, original Periscope, summary prepended to local rows, SUMScope's separate summary vote, and summary-only. The reference Periscope modules are imported at runtime; they are credited in [CREDITS.md](CREDITS.md).

## Reported experiment results

The accompanying project presentation reports the following results for **200 BRIGHT queries** (50 from each of Earth Science, Biology, Pony, Psychology; top-100 candidates per query). Values are the presentation's macro-averaged experiment summaries, not numbers from the Periscope paper.

| Method | NDCG@10 |
|---|---:|
| Embedding only | 0.462 |
| Local rows only | 0.553 |
| Periscope | 0.558 |
| **SUMScope (local + cached summary)** | **0.564** |

| Online reranking workload, per query | Periscope | SUMScope |
|---|---:|---:|
| Prompt tokens | 2.952M | 1.505M |
| LLM calls | 1,728 | 935 |

**Online prompt tokens decreased 49%, and online LLM calls decreased 46%.** The NDCG@10 difference against Periscope was +0.006 (95% confidence interval −0.023 to +0.035), so the quality difference is **not statistically significant**. Offline preparation was reported separately: 1,762 documents summarized across 3,458 LLM calls. The online savings figures exclude that offline preprocessing.

See [the presentation](presentation/SUMScope_Reyam_Albalihi_Presentation.pptx) for the study narrative.

## Repository structure

```text
SUMScope/
├── README.md
├── CREDITS.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── SUMScope_experiment.ipynb
└── presentation/
    └── SUMScope_Reyam_Albalihi_Presentation.pdf
```

## Run the experiment

**Environment:** Python on a CUDA-capable NVIDIA GPU with compatible PyTorch, Transformers and vLLM. Models are 4B parameters each; running the full experiment requires substantial GPU memory, storage, and compute. The notebook is a research experiment, not a one-click lightweight demo. On constrained hardware, adjust only the serving configuration and record changes when comparing runs.

1. Clone the repository and open `notebooks/SUMScope_experiment.ipynb` in Jupyter, Colab or a suitably provisioned GPU notebook environment.
2. Prepare a compatible Python environment with `torch`, `vllm`, `transformers`, `datasets`, `numpy`, and `pandas`. **Avoid blindly upgrading preinstalled GPU stacks**; use an isolated environment if needed. The `requirements.txt` file lists packages but intentionally does not claim a tested universal version combination.
3. Run the notebook sequentially. Its setup cell clones the original Periscope source and checks out the pinned reference commit, making the dependency explicit.
4. The notebook fetches BRIGHT from Hugging Face and model checkpoints from the model providers. Network access, access permissions and enough disk space are required.
5. It stores intermediate vectors, summaries, probe scores and result tables under `outputs/periscope_hackathon/` by default (or under Google Drive when run on Colab). These are ignored by Git.

**Configuration:** Edit `DOMAINS`, `N_QUERIES`, `TOPK`, `WINDOW`, `GPU_UTIL`, `GROUP_PROMPTS` in the notebook. Its original inference settings use BF16 and a large prompt batch; change these only to suit actual supported GPU hardware. Full run setting: `N_QUERIES = 50`.

**Reproducibility:** Periscope is pinned to commit `60da00aa7f3399c9a63f43792b9369b4ed140f0e`; the notebook specifies Qwen checkpoints, chunk size, pool size, evaluation metrics and summary policy. Dependency resolution and CUDA behavior vary by hardware.

## How SUMScope differs from Periscope

Periscope is the foundational method and is **not** authored by this project. It combines K local and K strided query-dependent probes for each document. SUMScope retains the local probes and uses an offline cached summary as a reusable global view, then scores that summary against each incoming query. The experimental variants, caching strategy, and benchmark comparison in this project explore that modification.

## Attribution

This project uses research and code from **Eltahir et al., *Periscope: Extending Frozen Language Models Beyond Their Context Window* (2026)** and evaluates on **Su et al., *BRIGHT* (ICLR 2025)**. See [CREDITS.md](CREDITS.md) for all authors and relevant upstream links. Third-party components retain their own licenses.
