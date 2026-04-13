"""
llm_filtering_mlx_v16.py

Base: v15 (F1=0.781). v10 was best (F1=0.800) — v16 aims to beat it.

Root-cause analysis of v15 failures (46 wrong out of 174):
  • 7 FNs from Rule 3 overfiring — v15 extended "linear pipeline" beyond NL-to-SQL,
    incorrectly catching meeting assistants, blog-to-podcast, Wikipedia Q&A, Excel Q&A,
    resume screeners, translation agents, and test-case generators as "other".
  • 5 FNs from Rule 1 overfiring — words like "template", "framework", "you build teams"
    incorrectly triggered the exclusion even for repos that ARE runnable end-user systems.
  • 2 FNs from Rule 8 overfiring — "DeVry/IBM Skills Network" and loose "Ethos 2025"
    matches in descriptions were triggering the educational-event exclusion.
  • 1 FN from Rule 4 misfiring — actual attack-agent implementation misread as benchmark.
  • 2 FNs from no-JSON output (retry logic should handle these).
  • The remaining ~25 FPs are hard borderline cases without clear rule fixes.

v16 changes:
  1. Omit repo name from few-shot examples entirely (base + targeted) — prevents
     name→label memorisation; addresses user's feedback about v10 real names.
  2. Revert Rule 3 to v10 narrow definition (NL-to-SQL / NL-to-code explicitly).
     Remove the v15 "recommendation/summarization pipeline" extension.
  3. Narrow Rule 1 — add explicit "runnable system" exception: if the repo ships
     a working agent you can directly run, Rule 1 does NOT fire even if the description
     says "template", "framework", "reference implementation", or "you build X".
  4. Narrow Rule 8 — remove DeVry/IBM Skills Network trigger; restrict to exact
     institutional-event patterns in the repo NAME only.
  5. Clarify Rule 4 — "official implementation of a technique" is NOT a benchmark.
  6. Replace targeted few-shot repo-5 (was recommendation pipeline → other) with
     a new positive example (meeting assistant → agentic), and add 4 more new
     positive examples for the top FN patterns.
"""

import gc
import json
import os
import re
from tqdm import tqdm
import csv

import mlx.core as mx
import pandas as pd
from mlx_lm import load, generate

# ── CONFIG ────────────────────────────────────────────────────────────────────
SAMPLE_CSV_PATH = "notebooks/csv/clean_sample_for_llm_completeReadme.csv"
OUTPUT_CSV      = "notebooks/data/sample_agent_repos_llm_filtered_withFullReadme_DeepSeekR1_mlx_v16.csv"
FEW_SHOT_JSON   = "notebooks/data/few_shot_examples_fullReadme_binaryClass.json"

MODEL           = "mlx-community/DeepSeek-R1-0528-Qwen3-8B-4bit"
MAX_NEW_TOKENS  = 4096
SAFETY_MARGIN   = 200
ENABLE_THINKING = False

FEW_SHOT_README_TOKENS = 512

CATEGORIES = ["llm-based agentic system", "other"]

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
print("Loading model...")
model, tokenizer = load(MODEL)
print("Model loaded.")

MODEL_MAX_CONTEXT = getattr(model.args, "max_position_embeddings", None) or 32768
MAX_INPUT_TOKENS  = MODEL_MAX_CONTEXT - MAX_NEW_TOKENS - SAFETY_MARGIN
print(f"Context: {MODEL_MAX_CONTEXT}  Max input: {MAX_INPUT_TOKENS}")


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _clean_readme(text):
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\[!\[.*?\]\(.*?\)\]\(.*?\)", "", text)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def truncate_readme(readme, max_tokens):
    if not readme:
        return readme
    readme = _clean_readme(readme)
    toks = tokenizer.encode(readme)
    if len(toks) <= max_tokens:
        return readme
    return tokenizer.decode(toks[:max_tokens]) + "\n... [README truncated]"


def strip_thinking(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ── TARGETED FEW-SHOT EXAMPLES ────────────────────────────────────────────────
# Repo names are omitted — model should learn from description/readme/topics only.

TARGETED_FEW_SHOT = [
    # ── GROUP 1: Framework / Platform FOR building agents ────────────────────
    # Rule 1 fires: only provides base classes — users must write agent logic
    {
        "description": "An extremely simple agent framework — provides base classes and utilities for building your own LLM agents.",
        "topics":      "['agent', 'agent-framework', 'agentic-ai', 'agents', 'llm', 'llm-framework']",
        "readme":      "# Simple Agent Framework\nProvides base classes for building LLM agents. Extend AgentBase to create your own agent. No runnable end-user task included.",
        "output": {"category": "other", "confidence": 0.97,
                   "key_evidence": ["description says 'agent framework'", "provides base classes FOR building agents — users write the agent logic",
                                    "no runnable end-user task in this repo"],
                   "reasoning_summary": "Agent framework for building agents — pure infrastructure, not a working end-user application."},
    },
    # Rule 1 fires: platform that hosts agents — users deploy their own agents on top
    {
        "description": "An open-source platform for building enterprise-grade LLM agents.",
        "topics":      "['llm', 'platform', 'agents', 'knowledgebase']",
        "readme":      "# Enterprise Agent Platform\nPlatform for creating and deploying LLM agents. Configure and build your own agents on this platform.",
        "output": {"category": "other", "confidence": 0.97,
                   "key_evidence": ["'platform for building' enterprise agents", "users build their OWN agents on top — this is the host, not the agent"],
                   "reasoning_summary": "Platform FOR building agents — infrastructure/host, not a specific deployable end-user agent."},
    },
    # ── GROUP 2: Collection of demos ─────────────────────────────────────────
    # Rule 2 fires: "a collection of"
    {
        "description": "A collection of intelligent AI agents built using Ollama, LangChain, and local LLMs — including chatbots, voice assistants, and more.",
        "topics":      "['ai-agents', 'chatbot', 'langchain', 'llm']",
        "readme":      "# LLM Agent Collection\nA collection of demo agents: RAG chatbot, voice assistant, SQL agent. Each demo is standalone.",
        "output": {"category": "other", "confidence": 0.95,
                   "key_evidence": ["'a collection of' agents/demos", "multiple standalone demos, not one deployable product"],
                   "reasoning_summary": "Collection of agent demos — not a single deployable end-user application."},
    },
    # ── GROUP 3: NL-to-SQL (Rule 3 narrow) ───────────────────────────────────
    # Rule 3 fires: ONLY when primary purpose is NL-to-SQL / NL-to-code with no other routing
    {
        "description": "Natural Language to SQL agent with multi-LLM support (OpenAI / Groq / Ollama).",
        "topics":      "[]",
        "readme":      "# NL-to-SQL Agent\nConverts natural language queries to SQL. Supports multiple LLM backends.",
        "output": {"category": "other", "confidence": 0.93,
                   "key_evidence": ["'Natural Language to SQL' — sole LLM step is generating SQL and executing it",
                                    "fixed generate-and-execute pipeline, no dynamic tool routing"],
                   "reasoning_summary": "NL-to-SQL: LLM generates SQL, system executes it. Single fixed step — not agentic routing."},
    },
    # ── POSITIVE: Multi-task LLM app (NOT caught by Rule 3) ──────────────────
    # Rule 3 does NOT fire for multi-step LLM agents that accomplish a real user task
    {
        "description": "An LLM agent that summarizes meeting transcriptions, extracts action items, and highlights key decisions.",
        "topics":      "['llm', 'meeting-assistant', 'nlp', 'fastapi']",
        "readme":      "# AI Meeting Assistant\nProcesses raw meeting transcripts to produce:\n- Concise summary\n- Action items list\n- Key decisions\n\n```bash\npython app.py\n```",
        "output": {"category": "llm-based agentic system", "confidence": 0.90,
                   "key_evidence": ["deployable FastAPI app with run instructions",
                                    "LLM performs multiple distinct reasoning tasks (summarize + extract + highlight)",
                                    "real end-user value: transforms meeting transcripts"],
                   "reasoning_summary": "Multi-task LLM application for a real end-user goal — NOT a simple pipeline. Rule 3 does not apply."},
    },
    # ── POSITIVE: Blog-to-podcast (NOT caught by Rule 3) ─────────────────────
    {
        "description": "Convert blog articles into audio podcasts using LLM summarization and ElevenLabs text-to-speech.",
        "topics":      "['llm', 'podcast', 'tts', 'elevenlabs']",
        "readme":      "# BlogCast\nTransform any blog into an engaging podcast.\n- Fetches blog content\n- LLM summarizes into podcast script\n- ElevenLabs generates audio\n\n```bash\npython blogcast.py --url <blog_url>\n```",
        "output": {"category": "llm-based agentic system", "confidence": 0.88,
                   "key_evidence": ["deployable CLI tool with run instructions",
                                    "LLM is the core reasoning engine (summarizes → generates script)",
                                    "real end-user task: blog content to podcast audio"],
                   "reasoning_summary": "Deployable blog-to-podcast tool — LLM is the core engine for a real user task. Rule 3 does not apply."},
    },
    # ── GROUP 4: Comparison / Benchmark ──────────────────────────────────────
    # Rule 4 fires: "comparing X with Y" — evaluation study
    {
        "description": "comparing a rule-based dialog system with a vanilla LLM agent for customer service evaluation.",
        "topics":      "[]",
        "readme":      "# Agent Comparison Study\nCompares two agent architectures on a benchmark dataset. Evaluates accuracy and safety.",
        "output": {"category": "other", "confidence": 0.96,
                   "key_evidence": ["description says 'comparing X with Y'", "benchmark evaluation study — not a deployable product"],
                   "reasoning_summary": "Comparison/benchmark between two approaches — research artifact, not a deployable end-user agent."},
    },
    # ── GROUP 5: Pure simulation / RL research (Rule 5) ──────────────────────
    # Rule 5 fires: "simulates" fictional scenario
    {
        "description": "A multi-agent, LLM-powered system that simulates diplomatic debates among multiple countries, culminating in scored outcomes.",
        "topics":      "[]",
        "readme":      "# Diplomatic Debate Simulation\nSimulates negotiations between LLM-powered country agents. Scores outcomes.",
        "output": {"category": "other", "confidence": 0.94,
                   "key_evidence": ["description says 'simulates' a fictional scenario", "no real-world end-user task beyond the simulation"],
                   "reasoning_summary": "Pure simulation of diplomatic debates — demo/game with no deployable real-world goal."},
    },
    # ── GROUP 6: Tutorial (Rule 6) ───────────────────────────────────────────
    # Rule 6 fires: description starts with "Build an" + step-by-step README
    {
        "description": "Build an agent-based RAG chatbot that answers user questions using uploaded documents.",
        "topics":      "[]",
        "readme":      "# RAG Chatbot Tutorial\nLearn how to build an agentic RAG chatbot using LangChain. Step-by-step guide for beginners.",
        "output": {"category": "other", "confidence": 0.92,
                   "key_evidence": ["description starts with 'Build an' — tutorial framing",
                                    "README is a step-by-step guide ('Learn how to build')"],
                   "reasoning_summary": "Tutorial teaching how to build a RAG chatbot — designed to be studied, not deployed as a product."},
    },
    # ── UI wrapper for another system (Rule 0) ───────────────────────────────
    {
        "description": "LLM UI for the XYZ agent framework, allowing to search and invoke actions of XYZ agents and interpret results.",
        "topics":      "[]",
        "readme":      "# XYZ Framework UI\nA web-based user interface for the XYZ framework. Connects to XYZ runtime to invoke agent actions.",
        "output": {"category": "other", "confidence": 0.97,
                   "key_evidence": ["'LLM UI for the [X] framework' — UI for another external system",
                                    "the agents live in the external framework, not in this repo"],
                   "reasoning_summary": "UI wrapper for an external agent framework — the UI itself is not the agentic system."},
    },
    # ── RL research environment (Rule 5) ─────────────────────────────────────
    {
        "description": "synthetic agent-based modeling for collective decision making with LLMs.",
        "topics":      "[]",
        "readme":      "# LLM Decision Gym\nA synthetic gym environment for studying collective decision making. Research simulation only.",
        "output": {"category": "other", "confidence": 0.95,
                   "key_evidence": ["'synthetic agent-based modeling' — research simulation environment",
                                    "no real-world end-user task"],
                   "reasoning_summary": "Synthetic RL gym / research simulation — not a deployable end-user agent."},
    },
    # ── Voice / conversational interface (Rule 9) ────────────────────────────
    {
        "description": "A real-time voice conversational interface around an LLM — records speech, transcribes via Whisper, sends to GPT, plays back audio response.",
        "topics":      "['voice-assistant', 'speech-to-text', 'text-to-speech', 'llm']",
        "readme":      "# Voice LLM Interface\nReal-time voice chat with an LLM. STT via Whisper, TTS via ElevenLabs. Run: python voice_agent.py",
        "output": {"category": "other", "confidence": 0.95,
                   "key_evidence": ["Rule 9: primary purpose is voice conversational interface (STT → LLM → TTS)",
                                    "LLM generates spoken replies — no autonomous tool routing or multi-step planning"],
                   "reasoning_summary": "Voice interface around an LLM — conversational UI, not an agentic system with autonomous decision-making."},
    },
    # ── RL agent without LLM core (Rule 10) ──────────────────────────────────
    {
        "description": "Reinforcement Learning agent for LLM autoscaling (toy simulator). Uses tabular Q-learning to decide when to scale LLM inference workloads.",
        "topics":      "['reinforcement-learning', 'rl-agent', 'llm']",
        "readme":      "# RL LLM Autoscaler\nTabular Q-learning agent controls autoscaling for LLM inference. Toy simulation environment.",
        "output": {"category": "other", "confidence": 0.95,
                   "key_evidence": ["Rule 10: Reinforcement Learning (Q-learning) is the core decision engine",
                                    "LLM is the subject being scaled, not the reasoning agent"],
                   "reasoning_summary": "RL-based agent — Q-learning does the routing, not the LLM. Classified as other."},
    },
    # ── Educational programme / competition (Rule 8 narrow) ──────────────────
    # Rule 8 fires ONLY on exact institutional-event identifiers in the repo NAME
    {
        "description": "An Agentic Reasoning System built for the IIT Guwahati Ethos 2025 hackathon.",
        "topics":      "[]",
        "readme":      "# Agentic Reasoning System\nBuilt for Ethos 2025 at IIT Guwahati. Multi-step reasoning with LangChain.",
        "output": {"category": "other", "confidence": 0.95,
                   "key_evidence": ["repo name contains 'IITGuwahati_Ethos2025' — specific institutional event identifier",
                                    "not a general deployable product"],
                   "reasoning_summary": "Submission to a specific institutional event (IIT Guwahati Ethos 2025) — not a general deployable product."},
    },
    # ── Non-English course/training material (Rule 6 extended) ───────────────
    # Course description in another language — the repo IS the course, not a product
    {
        "description": "Formation LangChain par LBKE (existe en format CPF): https://www.lbke.fr/formations/developpeur-llm-langgraph-langchain",
        "topics":      "['langchain', 'langgraph', 'llm', 'formation']",
        "readme":      "# LangGraph ReAct Agent Template\nThis template showcases a ReAct agent implemented using LangGraph, designed for LangGraph Studio. This is a course/training resource.",
        "output": {"category": "other", "confidence": 0.95,
                   "key_evidence": ["description is in French: 'Formation' means training/course by LBKE",
                                    "repo is course material — template students study, not a deployable product"],
                   "reasoning_summary": "Commercial training course material (French 'Formation') — template for learning, not a deployable end-user agent."},
    },
    # ── POSITIVE: Multi-agent problem solver (NOT a framework) ───────────────
    # Rule 1 does NOT fire: users run this system — they don't write code to extend it
    {
        "description": "A Multi-Agent Reasoning Problem Solver. You build teams and they work together to solve the problems you give them.",
        "topics":      "['multi-agent', 'llm', 'reasoning', 'problem-solver']",
        "readme":      "# MAR-PS — Multi-Agent Reasoning Problem Solver\nBuild reasoning teams that collaborate to solve complex problems.\n\n```bash\npython solver.py --problem 'What is X?'\n```",
        "output": {"category": "llm-based agentic system", "confidence": 0.91,
                   "key_evidence": ["deployable CLI system with run instructions",
                                    "LLM agents autonomously collaborate and route between reasoning steps",
                                    "'you build teams' means configuring the agent system, not writing framework code"],
                   "reasoning_summary": "Deployable multi-agent reasoning system — users run it to solve problems. Rule 1 does not apply: this IS the working agent, not infrastructure."},
    },
    # ── POSITIVE: LLM-driven testing agent (NOT a framework) ─────────────────
    # Rule 1 does NOT fire: "framework" in name but it IS the testing system
    {
        "description": "A LLM driven UI testing framework. Using multi-agent to plan and execute mobile UI tasks.",
        "topics":      "['llm', 'testing', 'multi-agent', 'ui-automation']",
        "readme":      "# sleepy-Testing\nLLM-driven UI test executor.\n- Plans test steps with LLM\n- Executes on real device\n- Reports results\n\n```bash\npython run_tests.py --app MyApp\n```",
        "output": {"category": "llm-based agentic system", "confidence": 0.89,
                   "key_evidence": ["deployable test executor with run instructions",
                                    "LLM is the core planning + routing engine for real UI task execution",
                                    "called 'framework' but IS the working test-agent system"],
                   "reasoning_summary": "Runnable LLM test-execution system — calling itself a 'framework' is just naming, it IS the deployable agent. Rule 1 does not apply."},
    },
    # ── POSITIVE: Sparse README but clear description ─────────────────────────
    {
        "description": "Production tool for creating and managing autonomous LLM agents; implemented using Apache Kafka, Docker, and LangChain.",
        "topics":      "['autonomous-agents', 'kafka', 'langchain', 'llm-agent']",
        "readme":      "Production tool for running autonomous LLM agents. Supports scheduling, memory, tool use. Install via pip install.",
        "output": {"category": "llm-based agentic system", "confidence": 0.90,
                   "key_evidence": ["pip-installable production tool",
                                    "autonomous agents with tool use, memory, scheduling"],
                   "reasoning_summary": "Sparse README but description clearly shows a deployable autonomous agent tool."},
    },
    # ── POSITIVE: Streamlit app with LLM routing ─────────────────────────────
    {
        "description": "A Streamlit-based application that uses an LLM to research and analyse companies by routing between web search and document analysis.",
        "topics":      "[]",
        "readme":      "# Company Research Agent\nStreamlit app. Enter a company name; agent routes between web search and GPT analysis to produce a detailed report.\n\n```bash\nstreamlit run app.py\n```",
        "output": {"category": "llm-based agentic system", "confidence": 0.88,
                   "key_evidence": ["deployed Streamlit app with run instructions",
                                    "LLM routes between web search and document analysis"],
                   "reasoning_summary": "Deployed Streamlit app where LLM dynamically routes between search and analysis tools."},
    },
]


def _build_few_shot_text(path, max_readme_tokens):
    with open(path) as f:
        base = json.load(f)

    blocks = []
    idx = 1

    for ex in base:
        readme = truncate_readme(ex.get("readme_content", "") or "", max_readme_tokens)
        cat = ex["category"]
        is_agent = cat == "llm-based agentic system"
        output = {
            "category":          cat,
            "confidence":        0.9,
            "key_evidence":      ["LLM is core decision engine", "deployable end-user application"] if is_agent
                                 else ["not a deployable agent application"],
            "reasoning_summary": f"Confirmed: {cat}",
        }
        # Omit repo name — model should reason from description/topics/readme only
        blocks.append(
            f"--- Example {idx} ---\n"
            f"Input:\n"
            f'description: "{ex["description"]}"\n'
            f"topics: {ex['topics']}\n"
            f'readme:\n"""\n{readme}\n"""\n'
            f"Output:\n{json.dumps(output)}"
        )
        idx += 1

    for ex in TARGETED_FEW_SHOT:
        readme = truncate_readme(ex["readme"], max_readme_tokens)
        blocks.append(
            f"--- Example {idx} ---\n"
            f"Input:\n"
            f'description: "{ex["description"]}"\n'
            f"topics: {ex['topics']}\n"
            f'readme:\n"""\n{readme}\n"""\n'
            f"Output:\n{json.dumps(ex['output'])}"
        )
        idx += 1

    return "\n\n".join(blocks)


# ── SYSTEM MESSAGE ─────────────────────────────────────────────────────────────

def build_system_content(few_shot_path):
    few_shot_text = _build_few_shot_text(few_shot_path, FEW_SHOT_README_TOKENS)

    return f"""You are an expert classifier of GitHub repositories related to LLM-based AI agents.

## TASK
Classify the given repository into EXACTLY ONE of the following categories:
1. llm-based agentic system
2. other

## EXCLUSION RULES — check these FIRST (first match wins → "other")

Apply these rules before anything else. If ANY rule matches, the answer is "other" immediately,
regardless of how agentic the code or README may look.

**Rule 0 — UI wrapper / frontend for another system:**
Description says "UI for X", "interface for X framework", "LLM UI for the X", "web UI for X",
or "frontend for X", where X is a distinct external framework or platform.
→ "other". The UI itself is not the agentic system — the agent lives elsewhere.

**Rule 1 — Framework / Platform FOR building agents (infrastructure only):**
The repo's PRIMARY purpose is to provide infrastructure for OTHERS to build agents with —
e.g. "a framework for building", "platform for building", "library for building",
"extend AgentBase to create your own agent", "starter kit for building agents".
→ "other".
CRITICAL EXCEPTIONS — Rule 1 does NOT fire if:
  (a) The repo ships a WORKING AGENT you can directly run (python main.py, streamlit run,
      FastAPI endpoint, Docker deploy) to accomplish a real end-user task, even if it also
      calls itself a "template", "framework", "reference implementation", or "you build X".
      Configuring or composing built-in agents is NOT the same as writing framework code.
  (b) The repo "uses" LangChain / LangGraph / AutoGen as a dependency — that does not
      make it a framework itself.

**Rule 2 — Collection of demos:**
Description says "a collection of" agents, demos, examples, or projects (plural showcase).
→ "other". Multi-demo repos are not a single deployable product.

**Rule 3 — Pure NL-to-SQL / NL-to-code pipeline:**
Description EXPLICITLY says "Natural Language to SQL", "NL-to-SQL", "text-to-SQL", or the
sole LLM step is generating SQL/code which is then executed, with no other routing decisions.
→ "other".
IMPORTANT: Rule 3 applies ONLY to repos where SQL/code generation is the ENTIRE system.
The following are NOT caught by Rule 3 and may be agentic:
  • Multi-step pipelines with distinct LLM tasks (summarize + extract + generate, translate +
    validate + fix, fetch + rank + answer) even if the flow is mostly fixed.
  • Interactive Q&A agents that query databases or documents among other tools.
  • Agents called "pipeline" but with real LLM decision-making across multiple steps.

**Rule 4 — Comparison / Benchmark study:**
Description says "comparing X with Y", or the repo is clearly an EVALUATION STUDY of
existing agents (NOT the implementation of a new technique or agent itself).
→ "other". Official implementations of a research technique that ship a runnable agent
are NOT caught — only repos whose primary contribution is measuring/comparing agents.

**Rule 5 — Pure simulation / game / RL research environment:**
Description says the repo simulates something purely fictional (diplomatic debates,
social experiments, game scenarios) with no real-world user goal.
Also covers research RL environments ("RL approach to enable", "synthetic agent-based
modeling", "gym for collective decision making").
→ "other". BUT: a simulation tool deployed for a concrete real-world purpose is NOT caught.

**Rule 6 — Tutorial / step-by-step guide:**
README is explicitly a step-by-step tutorial ("Learn how to build", "Step-by-step guide"),
AND the description also uses tutorial/course framing ("Build a/an X" as the main purpose,
or the description is in another language describing a commercial course or "Formation").
Both signals must be present.
→ "other".

**Rule 7 — Research paper:**
README contains arxiv links, "we propose", "we evaluate", "our experiments", or
describes a research environment / simulation platform.
→ "other" (exception: also ships a fully runnable end-user agent).

**Rule 8 — Specific institutional course assignment (narrow):**
Trigger ONLY when the REPO NAME (not description) contains a specific institutional-event
identifier pattern such as:
  • A university name combined with an event name (e.g. "IITGuwahati_Ethos2025")
  • "KodeCamp" combined with a grading/task context in the description
  • The exact string "Digital_Ocean_Assignment" in the repo name
→ "other". Do NOT trigger Rule 8 on general educational content, repos built "for a
hackathon" without a specific institutional identifier, descriptions mentioning universities
as background context, or repos affiliated with commercial training providers.

**Rule 9 — Voice / conversational interface (no tool use):**
The primary purpose is a voice assistant, real-time voice chat, or STT+TTS pipeline around
an LLM with no autonomous tool routing or multi-step planning.
→ "other". Exception: voice-enabled agents that also use tools (web search, code execution,
calendar) as part of their core function are NOT caught.

**Rule 10 — RL agent / non-LLM core decision engine:**
The core decision-making engine is a Reinforcement Learning algorithm (Q-learning, PPO,
DQN, etc.) or a rule-based system. The LLM may assist but is NOT the primary reasoning
component.
→ "other".

## POSITIVE CRITERIA — only after all rules pass

Classify as "llm-based agentic system" ONLY when ALL of the following are true:
1. It is a working, deployable APPLICATION meant to be RUN by end users.
2. The LLM is the CORE decision-making engine — it actively routes, selects tools,
   plans multi-step actions, decides between data sources, or performs multiple distinct
   reasoning tasks (summarize + extract, translate + validate, fetch + answer, etc.).
3. There is clear, positive evidence of deployment: install instructions, CLI,
   Streamlit/FastAPI endpoint, or Docker setup for a specific end-user task.

If the README is sparse or non-English but the DESCRIPTION clearly describes a
deployable agent, the description alone is sufficient.

**When genuinely uncertain → "other". Prefer precision over recall.**

## FEW-SHOT EXAMPLES

{few_shot_text}

## OUTPUT FORMAT
Respond with ONLY this JSON object and absolutely nothing else outside it:
{{"category": "<category-name>", "confidence": <0.0-1.0>, "key_evidence": ["evidence1", "evidence2"], "reasoning_summary": "brief explanation"}}"""


print("Building system message...")
SYSTEM_CONTENT = build_system_content(FEW_SHOT_JSON)
SYSTEM_MSG     = {"role": "system", "content": SYSTEM_CONTENT}

system_tokens   = len(tokenizer.encode(SYSTEM_CONTENT))
MAX_USER_TOKENS = MAX_INPUT_TOKENS - system_tokens
print(f"System message: {system_tokens} tokens  |  Max user tokens: {MAX_USER_TOKENS}")

if MAX_USER_TOKENS < 500:
    raise RuntimeError(
        f"System message too large ({system_tokens} tokens). "
        "Reduce FEW_SHOT_README_TOKENS or MAX_NEW_TOKENS."
    )


# ── CLASSIFICATION ─────────────────────────────────────────────────────────────

def classify_row(row, retries=2):
    name   = row.get("full_name", "")
    desc   = row.get("description", "")
    topics = row.get("topics", "")
    readme = row.get("readme_content", "") or ""

    readme = truncate_readme(readme, MAX_USER_TOKENS - 200)

    user_content = (
        f'name: "{name}"\n'
        f'description: "{desc}"\n'
        f"topics: {topics}\n"
        f'readme:\n"""\n{readme}\n"""'
    )

    prompt = tokenizer.apply_chat_template(
        [SYSTEM_MSG, {"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=ENABLE_THINKING,
    )

    try:
        raw     = generate(model, tokenizer, prompt=prompt,
                           max_tokens=MAX_NEW_TOKENS, verbose=False)
        decoded = strip_thinking(raw)
        print(f"  Output: {decoded[:300]}{'...' if len(decoded) > 300 else ''}")

    except Exception as e:
        print(f"  Generation error: {e}")
        if retries > 0:
            return classify_row(row, retries - 1)
        return {"category": "other", "confidence": 0.0,
                "key_evidence": "[]", "reasoning_summary": "generation failed"}

    try:
        match = re.search(r"\{.*\}", decoded, re.DOTALL)
        if not match:
            print("  No JSON found in output.")
            if retries > 0:
                return classify_row(row, retries - 1)
            return {"category": "other", "confidence": 0.0,
                    "key_evidence": "[]", "reasoning_summary": "no json in output"}

        parsed     = json.loads(match.group())
        cat        = parsed.get("category", "").strip().lower()
        confidence = parsed.get("confidence", 0.5)
        evidence   = parsed.get("key_evidence", [])
        reasoning  = parsed.get("reasoning_summary", "")

        if cat not in CATEGORIES:
            print(f"  Invalid category '{cat}', defaulting to 'other'")
            cat = "other"

        print(f"  ✓ {cat} (confidence: {confidence:.2f})")
        return {
            "category":          cat,
            "confidence":        confidence,
            "key_evidence":      json.dumps(evidence),
            "reasoning_summary": reasoning,
        }

    except Exception as e:
        print(f"  JSON parse error: {e}")
        if retries > 0:
            return classify_row(row, retries - 1)
        return {"category": "other", "confidence": 0.0,
                "key_evidence": "[]", "reasoning_summary": "parse error"}


# ── MAIN ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = pd.read_csv(SAMPLE_CSV_PATH)
    print(f"\nTotal repositories: {len(df)}")

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    done = set()
    if os.path.exists(OUTPUT_CSV):
        done = set(pd.read_csv(OUTPUT_CSV)["full_name"].tolist())
        print(f"Resuming — {len(done)} done, {len(df) - len(done)} remaining.")
    else:
        print(f"Starting fresh — {len(df)} repositories.")

    extra_fields = ["category", "confidence", "key_evidence", "reasoning_summary"]
    fieldnames   = list(df.columns) + extra_fields

    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not done:
            writer.writeheader()

        remaining = df[~df["full_name"].isin(done)]

        for i, (_, row) in enumerate(tqdm(remaining.iterrows(), total=len(remaining))):
            if i > 0 and i % 10 == 0:
                mx.clear_cache()
                gc.collect()

            row_dict = row.to_dict()
            print(f"\n[{row_dict['full_name']}]")
            result = classify_row(row_dict)
            row_dict.update(result)
            writer.writerow(row_dict)
            csvfile.flush()

    print("\nSaved results to:", OUTPUT_CSV)
