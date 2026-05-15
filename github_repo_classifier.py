"""
github_repo_classifier.py

Classifies GitHub repos from a CSV into binary categories using DeepSeek-R1
via mlx-lm. Metadata is fetched live from the GitHub GraphQL API and used
both for pre-filtering and for enriching the LLM prompt.

Categories:
  llm-based agentic system — executable autonomous multi-step LLM agent
  other                    —  tutorial, demo, framework, or non-agentic

Setup:
  1. Copy .env.example to .env and set GITHUB_TOKEN.
  2. python github_repo_classifier.py

Key settings are at the top of the CONFIG block below.
"""

import csv
import json
import os
import re

import pandas as pd
from dotenv import load_dotenv
from mlx_lm import load, generate
from tqdm import tqdm

from github_repo_fetcher import fetch_repo_metadata

load_dotenv()

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_CSV  = "notebooks/data/clean_sample_agent_repos_withShortReadme_manual.csv"
OUTPUT_CSV = "notebooks/data/github_repo_classified.csv"

MODEL           = "mlx-community/DeepSeek-R1-0528-Qwen3-8B-4bit"
MAX_NEW_TOKENS  = 3000
ENABLE_THINKING = True

MAX_ROWS = None  # int to test on first N rows; None = run all

CATEGORIES = ["llm-based agentic system", "other"]

# ── SYSTEM PROMPT ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
Classify a GitHub repository as exactly one of:
- "llm-based agentic system"
- "other"

Task:
Decide whether the repository is a real, LLM-based agentic system built for an end user task, or whether it is not.

Definition of "llm-based agentic system":
A repo is "llm-based agentic system" when it IS a  autonomous agent performing real-world tasks for end users, with the LLM as the core decision engine with The system must do more than generate text. It must use tools, APIs, files, browsers, databases, environments, or other external actions as part of the agent loop. 
The repo MUST have evidance that is agentic in nature  or it explicitly claims to be an agentic system.


EXCLUSIONS — require explicit textual evidence to apply:

R1  RL/non-LLM core     Core decision engine is RL (PPO, DQN, Q-learning) or rule-based;
                         LLM only assists. Includes source code for a paper studying RL agents.

R2  Pure infrastructure  The repo is a framework, library, or tool for building agents, but does not itself implement a specific agent. 
                         Examples: Template for building agents, framework to build agents, base classes for agents.
                         R2 does NOT apply if the repo implements agentic system. For example, agentic system to build agents.
                        

R3  Tutorial/course      Primary purpose is teaching: "tutorial", "learn how to", "course",
                         "workshop", "Hackathon". Readme is instructional prose, not operational docs.
                         For example, step-by-step instructions to build an agentic system do, how to build an agentic system, or a bootcamp assignment to build an agentic system would all trigger R3. 
                         NOT trigger R3 if the repo itself is a agentic system.

R4  RAG-only chatbot     Sole capability is corpus retrieval + answer. Do NOT apply on the
                         mere presence of "RAG" — absence of other tool evidence ≠ RAG-only.
                         Any mention of tool calling, web search, file ops, email, or routing
                         means R4 does NOT apply.

R5  Collection/showcase  Multiple unrelated agents or examples in one repo.

R6  Low-quality program  Explicitly a bootcamp, assignment, hackathon or course project with no meaningful readme or description.
                         Does NOT exclude general hackathon projects or thesis work.

R7  Simulation/env       Simulates fictional/social scenarios instead of performing real tasks:
                         RL gym, social simulation, debate simulator, "agent-based modeling", "experiment", "demo".

R8  Trivial/minimal      Apply only when BOTH are true:
                         (a) description ≤5 words or purely generic ("my agent", "testing", "simple") or
                         the readme without meaningful content ("lorem ipsum", "test", "demo", "example", "installation steps") AND
                         (b) readme has <3 sentences of actual feature description.

POSITIVE signals (support agentic, but don't override explicit exclusion evidence):
  - Uses LangChain/LangGraph/CrewAI/AutoGen to build a specific task agent
  - Routes between 2+ tools, APIs, or data sources
  - Deployable: pip install, Docker, Streamlit, FastAPI, CLI
  - Concrete task: travel booking, code generation, web research, data analysis
  - "multi-agent", "agentic", "autonomous agent" in description (verify not paper/sim/collection)
  - README or description MUST describe a business use case or real-world task.
  - A support section is a strong signal it is real, runnable system, not a tutorial or framework.

EXAMPLES:

Example 1 — using a framework ≠ being a framework (R2 does not apply)
Repo: "AI Research Assistant built with CrewAI. Searches arXiv, summarizes papers, drafts reviews. Run: python main.py"
Output: {"reasoning": "End-user research agent; uses CrewAI as dependency, not the product", "key_evidence": ["concrete task: research assistance", "runnable CLI"], "confidence": 0.9, "category": "llm-based agentic system"}

Example 2 — "test" in name is irrelevant; task is concrete (R8 does not apply)
Repo: "Test_Voice_Agent — handles customer support calls via Twilio + GPT-4. Deploy with Docker."
Output: {"reasoning": "Runnable customer-support voice agent with deployment artifact", "key_evidence": ["concrete task: customer support", "Docker deployment"], "confidence": 0.85, "category": "llm-based agentic system"}

Example 3 — genuine framework (R2 applies)
Repo: "AgentForge — Python framework for building agents. Base classes, memory backends. pip install agentforge; subclass BaseAgent."
Output: {"reasoning": "R2 — developer-facing framework; audience is developers, not end users", "key_evidence": ["framework for building agents", "subclass BaseAgent"], "confidence": 0.95, "category": "other"}

Example 4 — ambiguous "agent" with no description or readme (should NOT apply R8)
Repo: "AgentSmith — building agentic systems with LangGraph"
Output: {"reasoning": "Ambiguous name; no description or readme to determine intent", "key_evidence": ["ambiguous name: AgentSmith"], "confidence": 0.8, "category": "other"}

Example 5: Agentic system without mentioning agentic or autonomous in name or description
Repo: "WorkflowJobber - A sophisticated workflow with LLM-based decision making, tool use, and API calls. Run with Streamlit."
Output: {"reasoning": "Runnable workflow agent with concrete task and deployment, despite no mention of 'agent' or 'autonomous'", "key_evidence": ["concrete task: workflow management", "LLM-based decision making", "tool use and API calls", "Streamlit deployment"], "confidence": 0.95, "category": "llm-based agentic system"}

OUTPUT: Respond with exactly one JSON object. No markdown fences, no preamble, no trailing text.
{"reasoning": "<one sentence>", "key_evidence": ["<sig1>", "<sig2>"], "confidence": <0.0-1.0>, "category": "<llm-based agentic system or other>"}"""


# ── PRE-FILTER ────────────────────────────────────────────────────────────────

def _clean(val) -> str:
    s = str(val) if val is not None else ""
    return "" if s.lower() in ("nan", "none", "") else s.strip()


_AGENTIC_NAME_WORDS = {"agent", "agents", "agentic", "autonomous"}


def _metadata_skip(meta: dict) -> str | None:
    """Return a skip reason if GitHub metadata fails quality thresholds, else None."""
    if meta.get("stars", 0) < 10:
        return f"stars={meta['stars']} < 10"
    if meta.get("commit_count", 0) < 2:
        return f"commits={meta['commit_count']} < 2"
    if meta.get("number_of_contributors", 0) < 2:
        return f"contributors={meta['number_of_contributors']} < 2"
    if not meta.get("license"):
        return "no license"
    last_updated = str(meta.get("last_updated", ""))
    if not last_updated.startswith("2026"):
        return f"last updated {last_updated[:10]!r} — not in 2026"
    return None


def should_skip(row) -> str | None:
    desc    = _clean(row.get("description"))
    snippet = _clean(row.get("readme_snippet"))
    name    = _clean(row.get("full_name", "")).lower()
    # If the repo name contains agentic keywords, let the LLM decide even with sparse content
    if any(w in name.split("/")[-1].replace("-", " ").replace("_", " ").split()
           for w in _AGENTIC_NAME_WORDS):
        return None
    if not desc and not snippet:
        return "no description and no readme"
    if not desc and len(snippet.split()) < 5:
        return "no description and trivial readme"
    return None


# ── LLM HELPERS ───────────────────────────────────────────────────────────────

def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_json(text: str) -> dict | None:
    """Extract the first valid JSON object using brace balancing (handles nested braces)."""
    for i, ch in enumerate(text):
        if ch != '{':
            continue
        depth = 0
        for j, c in enumerate(text[i:], i):
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[i:j + 1])
                    except json.JSONDecodeError:
                        break
    return None


def extract_category_fallback(text: str) -> str | None:
    """Last-resort scan for the category string when JSON parsing fails entirely."""
    lower = text.lower()
    if "llm-based agentic system" in lower:
        return "llm-based agentic system"
    if '"other"' in lower or "'other'" in lower or ": other" in lower or ":other" in lower:
        return "other"
    return None


def build_user_content(row, meta: dict | None = None) -> str:
    snippet = str(row.get("readme_snippet", "") or "").strip()
    # Prefer live metadata over CSV columns when available
    m = meta or {}
    lines = [
        f"repo:         {row['full_name']}",
        f"description:  {row.get('description', '') or '(none)'}",
        f"topics:       {row.get('topics', '') or '(none)'}",
        f"language:     {m.get('primary_language') or row.get('primary_language', '') or row.get('language', '') or '(none)'}",
        f"license:      {m.get('license') or row.get('license', '') or '(none)'}",
        f"stars: {m.get('stars', row.get('stars', 0))}  forks: {m.get('forks', row.get('forks', 0))}",
        f"commits: {m.get('commit_count', '?')}  contributors: {m.get('number_of_contributors', '?')}",
        f"created: {str(m.get('created_at', row.get('created_at', '')))[:10] or '?'}"
        f"  last_updated: {str(m.get('last_updated', row.get('last_updated', '')))[:10] or '?'}",
    ]
    if snippet:
        lines.append(f'readme_snippet:\n"""\n{snippet[:800]}\n"""')
    return "\n".join(lines)


def classify(row, retries: int = 3) -> dict:
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_content(row)},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=ENABLE_THINKING,
    )

    try:
        raw     = generate(model, tokenizer, prompt=prompt,
                           max_tokens=MAX_NEW_TOKENS, verbose=False)
        decoded = strip_thinking(raw)
        print(f"  raw: {decoded[:200]}{'...' if len(decoded) > 200 else ''}")
    except Exception as e:
        print(f"  Generation error: {e}")
        if retries > 0:
            return classify(row, retries - 1)
        return _error_result("generation failed")

    # Try stripped output first; fall back to raw in case JSON is inside thinking block
    parsed = extract_json(decoded) or extract_json(raw)
    if parsed is None:
        if retries > 0:
            print("  No JSON in output — retrying.")
            return classify(row, retries - 1)
        # Last resort: scan raw text for the category string
        cat = extract_category_fallback(decoded) or extract_category_fallback(raw)
        if cat:
            print(f"  No JSON but found category via fallback: {cat}")
            return {"category": cat, "confidence": 0.4,
                    "key_evidence": "[]", "reasoning": "extracted via fallback scan"}
        print("  No JSON in output — giving up.")
        return _error_result("no json in output")

    try:
        cat        = parsed.get("category", "").strip().lower()
        confidence = float(parsed.get("confidence", 0.5))
        evidence   = parsed.get("key_evidence", [])
        reasoning  = parsed.get("reasoning", "")

        if cat not in CATEGORIES:
            print(f"  Unknown category '{cat}' — defaulting to other")
            cat = "other"

        print(f"  → {cat} ({confidence:.2f})")
        return {
            "category":     cat,
            "confidence":   confidence,
            "key_evidence": json.dumps(evidence),
            "reasoning":    reasoning,
        }

    except (json.JSONDecodeError, ValueError) as e:
        print(f"  Parse error: {e}")
        if retries > 0:
            return classify(row, retries - 1)
        return _error_result("parse error")


def _error_result(reason: str) -> dict:
    return {"category": "error", "confidence": 0.0,
            "key_evidence": "[]", "reasoning": reason}


# ── MAIN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Loading model: {MODEL}")
    model, tokenizer = load(MODEL)
    print("Model loaded.\n")

    df = pd.read_csv(INPUT_CSV)
    if MAX_ROWS is not None:
        df = df.head(MAX_ROWS)
    print(f"Total repos to process: {len(df)}")

    # Resume: skip rows already written to the output CSV
    done: set[str] = set()
    if os.path.exists(OUTPUT_CSV) and os.path.getsize(OUTPUT_CSV) > 0:
        try:
            done = set(pd.read_csv(OUTPUT_CSV)["full_name"].tolist())
            print(f"Resuming — {len(done)} done, {len(df) - len(done)} remaining.")
        except Exception:
            print("Output CSV unreadable — starting fresh.")
            os.remove(OUTPUT_CSV)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    fieldnames = ["full_name", "category", "confidence", "key_evidence", "reasoning"]

    write_header = not os.path.exists(OUTPUT_CSV) or len(done) == 0

    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()

        remaining = df[~df["full_name"].isin(done)]

        for _, row in tqdm(remaining.iterrows(), total=len(remaining)):
            full_name = row["full_name"]
            print(f"\n[{full_name}]")

            try:
                # ── 1. Fetch live metadata from GitHub ────────────────────────
                print("  Fetching GitHub metadata...")
                meta = fetch_repo_metadata(full_name)
                if meta is None:
                    print("  Could not fetch metadata — skipping.")
                    writer.writerow({"full_name": full_name,
                                     **_error_result("github fetch failed")})
                    csvfile.flush()
                    continue

                # ── 2. Quality pre-filter (metadata thresholds) ───────────────
                meta_skip = _metadata_skip(meta)
                if meta_skip:
                    print(f"  Skipped (quality): {meta_skip}")
                    writer.writerow({"full_name": full_name,
                                     "category": "other", "confidence": 0.0,
                                     "key_evidence": "[]",
                                     "reasoning": f"skipped: {meta_skip}"})
                    csvfile.flush()
                    continue

                # ── 3. Content pre-filter (description / readme) ──────────────
                skip_reason = should_skip(row)
                if skip_reason:
                    print(f"  Skipped (content): {skip_reason}")
                    result = {"category": "other", "confidence": 0.0,
                              "key_evidence": "[]",
                              "reasoning": f"skipped: {skip_reason}"}
                else:
                    print("  Classifying...")
                    result = classify(row, meta)

            except Exception as e:
                print(f"  Unexpected error: {e}")
                result = _error_result(f"unexpected error: {e}")

            writer.writerow({"full_name": full_name, **result})
            csvfile.flush()

    print(f"\nDone. Results saved to: {OUTPUT_CSV}")
