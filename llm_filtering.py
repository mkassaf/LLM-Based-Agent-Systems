import json
import os
import re
from tqdm import tqdm
import csv

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# CONFIG
SAMPLE_CSV_PATH = "notebooks/csv/clean_sample_for_llm_completeReadme.csv"
OUTPUT_CSV = "notebooks/data/sample_agent_repos_llm_filtered_withFullReadme_DeepSeekR1.csv"
FEW_SHOT_JSON = "notebooks/data/few_shot_examples_fullReadme_binaryClass.json"

MODEL = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
CACHE_DIR = os.path.expanduser("~/hf_cache")
MAX_NEW_TOKENS = 1500
SAFETY_MARGIN = 200

# Device
if torch.cuda.is_available():
    device_map = "auto"
    torch_dtype = torch.float16
elif torch.backends.mps.is_available():
    device_map = {"": "mps"}
    torch_dtype = torch.float16
else:
    device_map = {"": "cpu"}
    torch_dtype = torch.float32

CATEGORIES = [
    "llm-based agentic system",
    "other"
]

# LOAD MODEL
print("Loading tokenizer and model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    cache_dir=CACHE_DIR,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    cache_dir=CACHE_DIR,
    device_map=device_map,
    dtype=torch_dtype,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.eval()
print("Model loaded successfully.")

# Auto-detect model context length
MODEL_MAX_CONTEXT = None
for attr in ("max_position_embeddings", "seq_length", "model_max_length"):
    val = getattr(model.config, attr, None)
    if val is not None and isinstance(val, int):
        MODEL_MAX_CONTEXT = val
        break
if MODEL_MAX_CONTEXT is None:
    MODEL_MAX_CONTEXT = 32768
    print("WARNING: Could not detect context length. Using fallback:", MODEL_MAX_CONTEXT)
else:
    print("Detected model context length:", MODEL_MAX_CONTEXT)

MAX_INPUT_TOKENS = MODEL_MAX_CONTEXT - MAX_NEW_TOKENS - SAFETY_MARGIN
print("Max input tokens:", MAX_INPUT_TOKENS)


# FEW-SHOT EXAMPLES
def build_few_shot(path, max_readme_tokens=None):
    with open(path, "r") as f:
        examples = json.load(f)

    messages = []

    for i, ex in enumerate(examples, 1):
        readme = ex.get("readme_content", "") or ""
        print(f"Example {i} - {ex['full_name']} - readme length: {len(readme)} chars")

        if max_readme_tokens is not None and readme:
            tokens = tokenizer.encode(readme, add_special_tokens=False)
            if len(tokens) > max_readme_tokens:
                readme = tokenizer.decode(tokens[:max_readme_tokens], skip_special_tokens=True)
                readme += "\n... [README truncated]"

        example_text = f"""
### Example {i}
Input:
name: "{ex['full_name']}"
description: "{ex['description']}"
topics: {ex['topics']}
readme:
\"\"\"
{readme}
\"\"\"

Output:
{{"category":"{ex['category']}"}}"""

        messages.append({"role": "user", "content": example_text})

    return messages


# HELPERS
def count_message_tokens(messages):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    return len(tokenizer.encode(text, add_special_tokens=False))


def build_messages_with_token_check(main_prompt):
    main_msg = [{"role": "user", "content": main_prompt}]

    # Try with full readmes first
    few_shot_full = build_few_shot(FEW_SHOT_JSON)
    all_messages = few_shot_full + main_msg
    total_tokens = count_message_tokens(all_messages)

    if total_tokens <= MAX_INPUT_TOKENS:
        print(f"All content fits: {total_tokens}/{MAX_INPUT_TOKENS} tokens")
        return all_messages

    # Doesn't fit — calculate budget per example readme
    few_shot_no_readme = build_few_shot(FEW_SHOT_JSON, max_readme_tokens=0)
    tokens_without_readmes = count_message_tokens(few_shot_no_readme + main_msg)
    tokens_available = MAX_INPUT_TOKENS - tokens_without_readmes
    num_examples = len(few_shot_no_readme)

    if tokens_available <= 0 or num_examples == 0:
        print("WARNING: No room for example readmes. Using examples without readmes.")
        return few_shot_no_readme + main_msg

    token_per_example = tokens_available // num_examples
    print(f"Truncating each example readme to ~{token_per_example} tokens")

    few_shot_trimmed = build_few_shot(FEW_SHOT_JSON, max_readme_tokens=token_per_example)
    return few_shot_trimmed + main_msg


# CLASSIFICATION
def classify_row(row, retries=2):
    name = row.get("full_name", "")
    desc = row.get("description", "")
    topics = row.get("topics", "")
    readme = row.get("readme_content", "")

    main_prompt = f"""
You are an expert classifier of GitHub repositories related to LLM-based AI agents.
IMPORTANT: Your task is to classify the repository into ONE of the following categories:
1. llm-based agentic system
2. other

Instructions: 
- A repository is "llm-based agentic system" ONLY if the LLM actively reasons, 
  plans, and makes decisions to accomplish a task — not just generates text as 
  part of a fixed pipeline.
  - The LLM must be the core decision-making engine, not just a text generator 
    in a sequence of steps.
  - It MUST show agent behavior: tool use, multi-step reasoning, planning, 
    or autonomous action toward a goal.
  - It is STILL "llm-based agentic system" even if it uses frameworks like 
    LangChain or LangGraph to build the agent — what matters is that the repo 
    itself is a working agent application doing a specific task.
- A repository is "other" if ANY of the following apply, but not limited to:
  - It is a framework, template, or library intended for others to build agents
  - A benchmark or evaluation tool for LLM agents
  - A foundation or base model that could be used in agents but is not itself an agent application
  - A dataset for training or evaluating agents
  - A tutorial, guide, or educational resource about LLM agents
  - A curated list or collection of papers, tools, or agent projects
  - An infrastructure or DevOps tool that manages or scales LLMs
- Analyze the repository using **name**, **description**, **topics** and **readme**.
- Do NOT create new categories.
- Do NOT write any explanations, reasoning, or extra text.

Return only valid single JSON object in this format:
{{"category": "<category-name>"}}

### Now classify this repository:
name: "{name}"
description: "{desc}"
topics: {topics}
readme:
\"\"\"
{readme}
\"\"\"

Output format:
{{"category": "<category-name>"}}
"""

    messages = build_messages_with_token_check(main_prompt)

    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        decoded = re.sub(r"<think>.*?</think>", "", decoded, flags=re.DOTALL).strip()

        print("Decoded LLM output:", decoded)

    except Exception as e:
        print("Generation error:", e)
        if retries > 0:
            return classify_row(row, retries - 1)
        return "other"

    # Parse JSON safely
    try:
        match = re.search(r"\{.*\}", decoded, re.DOTALL)
        if not match:
            print("No JSON found in LLM output.")
            return "other"

        parsed = json.loads(match.group())
        cat = parsed.get("category", "").strip().lower()

        if cat in CATEGORIES:
            return cat
        return "other"

    except Exception as e:
        print("Error parsing JSON:", e)
        return "other"


if __name__ == "__main__":
    df = pd.read_csv(SAMPLE_CSV_PATH)
    print(f"Classifying {len(df)} repositories...")

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    fieldnames = list(df.columns) + ["category"]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for _, row in tqdm(df.iterrows(), total=len(df)):
            row_dict = row.to_dict()
            category = classify_row(row_dict)
            row_dict["category"] = category
            writer.writerow(row_dict)
            csvfile.flush()

    print("Saved streamed results to:", OUTPUT_CSV)