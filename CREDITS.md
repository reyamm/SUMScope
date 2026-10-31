# Credits and research attribution

## SUMScope

**Project design, summary caching adaptation, experiments, analysis, and presentation:** Reyam Albalihi. Developed for the HUMAIN hackathon, Challenge 2: *Deliver AI with Less Compute*.

## Periscope — foundational research and code

SUMScope builds upon **Periscope: Extending Frozen Language Models Beyond Their Context Window** (2026), by Mohamed Eltahir, Anas Obayd, Raed Rashid, Abdulrahman Alghamdi, Abdulrahman Mousa, Abdallah Ahmed, Tanveer Hussain, and Naeemullah Khan.

- Paper: https://arxiv.org/abs/2610.04047
- Original source code: https://github.com/mohammad2012191/Periscope
- Reference revision used by the notebook: `60da00aa7f3399c9a63f43792b9369b4ed140f0e`
- The notebook imports the Periscope authors' `core` and `metrics` modules. These are *not* claimed as SUMScope-original code.
- Periscope developed the K×K local/strided probing method, the Yes/No relevance log-odds scoring scheme, and its reference evaluation functions.

## BRIGHT — evaluation benchmark

**BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval**, Hongjin Su et al., ICLR 2025.

- Dataset and paper: https://brightbenchmark.github.io/
- Repository: https://github.com/xlang-ai/BRIGHT
- Hugging Face: https://huggingface.co/datasets/xlangai/BRIGHT
- BRIGHT dataset/repository license and attribution: consult the upstream repository (CC BY 4.0). Dataset is loaded during execution and is not redistributed here.

## Models and infrastructure

- Qwen3-Embedding-4B: https://huggingface.co/Qwen/Qwen3-Embedding-4B
- Qwen3-4B-Instruct-2507: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
- vLLM: https://github.com/vllm-project/vllm
- Hugging Face Datasets/Transformers, PyTorch, NumPy and pandas.

All third-party software, models, and datasets remain subject to their respective licenses and usage terms. This repository does not bundle upstream code or model weights.
