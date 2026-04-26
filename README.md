# LLM-Based-Agent-Systems

Study of architectural patterns and framework usage in LLM-based agent systems.

## Overview

This repository supports research on classifying GitHub repositories as either:

- `llm-based agentic system`
- `other`

The main workflow uses GitHub repository metadata and README content with a local LLM to filter and label candidate repositories.

## Key scripts

### `sample.py`

Creates a reproducible random sample of repository rows from the full CSV dataset.

- Input: `notebooks/csv/github_agent_repos_python_20251225.csv`
- Output: `notebooks/csv/sample_for_llm_1.csv`

Usage:

```bash
python sample.py
```

### `sample_fullReadme.py`

Joins the short-readme sample dataset with the main repository CSV to create a sample containing the full README content.

- Input:
  - `notebooks/csv/github_agent_repos_python_final.csv`
  - `notebooks/csv/clean_sample_for_llm_shortReadme.csv`
- Output: `notebooks/csv/clean_sample_for_llm_completeReadme.csv`

Usage:

```bash
python sample_fullReadme.py
```

### `generate_fewshot_examples_fullReadme.py`

Enriches few-shot examples by matching entries from a short-readme few-shot JSON file with the main CSV dataset.

- Input:
  - `notebooks/data/few_shot_examples_shortReadme_binaryClass.json`
  - `notebooks/csv/github_agent_repos_python_final.csv`
- Output: `notebooks/data/few_shot_examples_fullReadme_binaryClass.json`

Usage:

```bash
python generate_fewshot_examples_fullReadme.py
```

### `llm_filtering.py`

Runs a local HuggingFace transformer model to classify repositories based on metadata and README content.

- Input: `notebooks/csv/clean_sample_for_llm_completeReadme.csv`
- Output: `notebooks/data/sample_agent_repos_llm_filtered_withFullReadme_DeepSeekR1.csv`

Loads `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` via `transformers`, auto-detects CUDA/MPS/CPU, builds few-shot prompts with token budgeting, and writes streamed classification results.

Usage:

```bash
python llm_filtering.py
```

### `llm_filtering_mlx.py`

Apple Silicon-optimized version of the classifier using `mlx_lm` instead of `transformers`. Significantly faster on M-series chips due to MLX's native support for Apple's unified memory architecture.

- Input: `notebooks/csv/clean_sample_for_llm_completeReadme.csv`
- Output: `notebooks/data/sample_agent_repos_llm_filtered_withFullReadme_DeepSeekR1_mlx.csv`

Loads `mlx-community/DeepSeek-R1-0528-Qwen3-8B-4bit` via `mlx_lm`. Same prompt logic and few-shot token budgeting as `llm_filtering.py`.

Usage:

```bash
python llm_filtering_mlx.py
```

> **Note:** Requires `mlx-lm` to be installed (`pip install mlx-lm`). If the MLX model is not available on HuggingFace, convert it locally:
> ```bash
> python -m mlx_lm.convert --hf-path deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
>     --mlx-path ~/mlx_models/DeepSeek-R1-0528-Qwen3-8B-4bit -q
> ```
> Then update the `MODEL` path in the script accordingly.

## Requirements

- Python 3.9+
- For `llm_filtering.py`: any platform with CUDA, MPS (Apple Silicon), or CPU
- For `llm_filtering_mlx.py`: Apple Silicon Mac (M1/M2/M3/M4) only

## Setup

**1. Create and activate a virtual environment:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**2. Install core dependencies:**

```bash
pip install -r requirements.txt
```

**3. (Apple Silicon only) Install mlx-lm for the faster MLX-based classifier:**

```bash
pip install mlx-lm
```

## How to run

Run the scripts in order:

```bash
# Step 1 — create a random sample from the full dataset
python sample.py

# Step 2 — enrich the sample with full README content
python sample_fullReadme.py

# Step 3 — generate few-shot examples with full READMEs
python generate_fewshot_examples_fullReadme.py

# Step 4a — classify repositories (HuggingFace transformers, all platforms)
python llm_filtering.py

# Step 4b — classify repositories (MLX, Apple Silicon only — faster)
python llm_filtering_mlx.py
```

Results are written incrementally to `notebooks/data/` so progress is preserved if a run is interrupted.

## Notes

- `llm_filtering.py` auto-detects available compute devices (CUDA, MPS, CPU).
- `llm_filtering_mlx.py` is recommended on Apple Silicon (M1/M2/M3/M4) for best performance.
- The few-shot example builder truncates example README content if the combined prompt exceeds the model's context window.
- Classification results include category, confidence score, key evidence, and a reasoning summary.

## File structure

- `README.md`
- `requirements.txt`
- `sample.py`
- `sample_fullReadme.py`
- `generate_fewshot_examples_fullReadme.py`
- `llm_filtering.py`
- `llm_filtering_mlx.py`
- `notebooks/csv/`
- `notebooks/data/`

