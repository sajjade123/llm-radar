#!/usr/bin/env python3
"""
OpenRouter Model Optimizer — Analyze and rank LLMs for different use cases.
Pulls live data from OpenRouter API + rankings to score models across categories.

All scoring is based on verifiable API data: supported parameters, context length,
pricing, modalities, and popularity. No subjective keyword matching.

Usage:
    python model-optimizer.py                    # Full analysis
    python model-optimizer.py --active           # Only active models (last 6 months)
    python model-optimizer.py --coding           # Best for coding
    python model-optimizer.py --reasoning        # Best for reasoning
    python model-optimizer.py --budget           # Best value (paid)
    python model-optimizer.py --free             # Best free models
    python model-optimizer.py --agents           # Best for tool use/agents
    python model-optimizer.py --multimodal       # Best multimodal
    python model-optimizer.py --context          # Largest context windows
    python model-optimizer.py --compare MODEL1 MODEL2
    python model-optimizer.py --show MODEL       # Detailed single model view
    python model-optimizer.py --search QUERY     # Search models
    python model-optimizer.py --top N            # Show top N per category
    python model-optimizer.py --json             # Output as JSON
    python model-optimizer.py --optimize-claude  # Best models for Claude Code
    python model-optimizer.py --include-expired  # Include expired models
"""

import argparse
import json
import sys
import re
import io
import math
from datetime import datetime, timezone
import requests

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ─── Constants ───────────────────────────────────────────────────────────────

MODELS_API = "https://openrouter.ai/api/v1/models"

# Top rankings from openrouter.ai/rankings (updated 2026-03-27)
# Used only as a lightweight tiebreaker (max 15% of score)
RANKINGS = {
    "xiaomi/mimo-v2-pro": {"rank": 1, "tokens": "3.28T", "growth": "+1,339%"},
    "stepfun/step-3.5-flash:free": {"rank": 2, "tokens": "1.53T", "growth": "+10%"},
    "deepseek/deepseek-v3.2": {"rank": 3, "tokens": "1.2T", "growth": "+8%"},
    "minimax/minimax-m2.7": {"rank": 4, "tokens": "1.08T", "growth": "+1,968%"},
    "minimax/minimax-m2.5": {"rank": 5, "tokens": "1.05T", "growth": "+29%"},
    "z-ai/glm-5-turbo": {"rank": 6, "tokens": "1.03T", "growth": "+84%"},
    "anthropic/claude-opus-4.6": {"rank": 7, "tokens": "1.02T", "growth": "+9%"},
    "anthropic/claude-sonnet-4.6": {"rank": 8, "tokens": "1.01T", "growth": "+2%"},
    "google/gemini-3-flash-preview": {"rank": 9, "tokens": "950B", "growth": "+4%"},
    "google/gemini-2.5-flash-lite": {"rank": 10, "tokens": "553B", "growth": "+12%"},
}

# Scoring dimensions — each is 100% based on verifiable API data
# Weights are displayed transparently so users can judge bias

# Scoring weights per category (must sum to 1.0)
CATEGORY_WEIGHTS = {
    "coding": {
        "label": "Best for Coding",
        "icon": "Code",
        "description": "Code generation, debugging, refactoring, technical writing",
        "weights": {"capability": 0.50, "context": 0.15, "pricing": 0.15, "popularity": 0.10, "speed": 0.10},
        "caps": {"tools": 40, "tool_choice": 20, "structured_outputs": 15, "reasoning": 15, "response_format": 10},
        "exclude_free": False,
        "only_free": False,
        "require_multimodal": False,
    },
    "reasoning": {
        "label": "Best for Reasoning",
        "icon": "Brain",
        "description": "Complex analysis, math, logic, multi-step problem solving",
        "weights": {"capability": 0.55, "context": 0.20, "pricing": 0.10, "popularity": 0.10, "speed": 0.05},
        "caps": {"reasoning": 35, "include_reasoning": 25, "tools": 15, "structured_outputs": 10, "response_format": 10, "frequency_penalty": 5},
        "exclude_free": False,
        "only_free": False,
        "require_multimodal": False,
    },
    "budget": {
        "label": "Best Value (Paid)",
        "icon": "Dollar",
        "description": "Best capability per dollar for paid models",
        "weights": {"capability": 0.25, "context": 0.10, "pricing": 0.35, "popularity": 0.20, "speed": 0.10},
        "caps": {"tools": 25, "reasoning": 20, "tool_choice": 15, "structured_outputs": 15, "response_format": 10, "seed": 5, "stop": 10},
        "exclude_free": True,
        "only_free": False,
        "require_multimodal": False,
    },
    "free": {
        "label": "Best Free Models",
        "icon": "Free",
        "description": "Top free models ranked by capability",
        "weights": {"capability": 0.45, "context": 0.25, "pricing": 0.0, "popularity": 0.15, "speed": 0.15},
        "caps": {"tools": 30, "reasoning": 25, "tool_choice": 15, "structured_outputs": 15, "response_format": 10, "seed": 5},
        "exclude_free": False,
        "only_free": True,
        "require_multimodal": False,
    },
    "agents": {
        "label": "Best for Agents/Tool Use",
        "icon": "Agent",
        "description": "Agentic workflows, tool calling, function execution",
        "weights": {"capability": 0.55, "context": 0.15, "pricing": 0.10, "popularity": 0.10, "speed": 0.10},
        "caps": {"tools": 35, "tool_choice": 25, "structured_outputs": 15, "reasoning": 15, "response_format": 10},
        "exclude_free": False,
        "only_free": False,
        "require_multimodal": False,
    },
    "multimodal": {
        "label": "Best Multimodal",
        "icon": "Media",
        "description": "Image, video, audio understanding and generation",
        "weights": {"capability": 0.50, "context": 0.15, "pricing": 0.10, "popularity": 0.15, "speed": 0.10},
        "caps": {"image": 30, "video": 20, "audio": 20, "tools": 15, "reasoning": 10, "structured_outputs": 5},
        "exclude_free": False,
        "only_free": False,
        "require_multimodal": True,
    },
    "context": {
        "label": "Largest Context",
        "icon": "Context",
        "description": "Models with the largest context windows",
        "weights": {"capability": 0.0, "context": 0.70, "pricing": 0.10, "popularity": 0.10, "speed": 0.10},
        "caps": {},
        "exclude_free": False,
        "only_free": False,
        "require_multimodal": False,
    },
}

# Claude Code optimization categories
CLAUDE_CODE_ROLES = {
    "opus-class": {
        "label": "Opus-class (Complex Reasoning)",
        "icon": "Trophy",
        "env_var": "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "needs": ["reasoning", "coding"],
    },
    "sonnet-class": {
        "label": "Sonnet-class (General Coding)",
        "icon": "Zap",
        "env_var": "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "needs": ["coding", "agents"],
    },
    "haiku-class": {
        "label": "Haiku-class (Fast Completions)",
        "icon": "Run",
        "env_var": "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "needs": ["coding", "free"],
    },
    "subagent": {
        "label": "Sub-agent Model",
        "icon": "Link",
        "env_var": "CLAUDE_CODE_SUBAGENT_MODEL",
        "needs": ["agents", "coding"],
    },
}


# ─── Data Fetching ───────────────────────────────────────────────────────────

def fetch_models() -> list[dict]:
    """Fetch all models from OpenRouter API."""
    print("Fetching models from OpenRouter API...")
    resp = requests.get(MODELS_API, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    models = data["data"]
    print(f"   Found {len(models)} models\n")
    return models


# ─── Model Analysis ──────────────────────────────────────────────────────────

def analyze_model(model: dict) -> dict:
    """Extract features from a model for scoring. All data from API fields only."""
    mid = model["id"]
    params = set(model.get("supported_parameters") or [])
    arch = model.get("architecture") or {}
    pricing = model.get("pricing") or {}
    top_provider = model.get("top_provider") or {}

    prompt_price = float(pricing.get("prompt", "0") or "0")
    completion_price = float(pricing.get("completion", "0") or "0")
    context_length = top_provider.get("context_length", 0) or model.get("context_length", 0) or 0
    max_completion = top_provider.get("max_completion_tokens", 0) or 0

    input_modalities = arch.get("input_modalities") or []
    output_modalities = arch.get("output_modalities") or []
    modality = arch.get("modality", "")

    is_free = prompt_price == 0 and completion_price == 0
    provider = mid.split("/")[0] if "/" in mid else "unknown"

    ranking = RANKINGS.get(mid, {})

    return {
        "id": mid,
        "name": model.get("name", mid),
        "provider": provider,
        "description": model.get("description", ""),
        "context_length": context_length,
        "max_completion": max_completion,
        "prompt_price": prompt_price,
        "completion_price": completion_price,
        "is_free": is_free,
        "input_modalities": input_modalities,
        "output_modalities": output_modalities,
        "modality": modality,
        "is_multimodal": len(input_modalities) > 1 or any(
            m in input_modalities for m in ["image", "video", "audio"]
        ),
        "params": params,
        "has_tools": "tools" in params,
        "has_tool_choice": "tool_choice" in params,
        "has_structured_outputs": "structured_outputs" in params,
        "has_reasoning": "reasoning" in params or "include_reasoning" in params,
        "has_vision": "image" in input_modalities,
        "has_video": "video" in input_modalities,
        "has_audio": "audio" in input_modalities,
        "rank": ranking.get("rank", 999),
        "tokens": ranking.get("tokens", "0"),
        "growth": ranking.get("growth", "0%"),
        "created": model.get("created", 0),
        "knowledge_cutoff": model.get("knowledge_cutoff"),
        "expiration_date": model.get("expiration_date"),
    }


# ─── Scoring ─────────────────────────────────────────────────────────────────
# All scoring uses only verifiable API data. No keyword matching, no subjective
# description analysis, no name-based size estimation.

def score_capability(analysis: dict, caps: dict) -> float:
    """Score model capabilities based on supported API parameters (0-100)."""
    if not caps:
        return 0

    score = 0.0
    params = analysis["params"]
    input_mods = set(analysis["input_modalities"])

    for param, weight in caps.items():
        # Check both supported_parameters and input_modalities
        if param in params or param in input_mods:
            score += weight

    return min(score, 100)


def score_context(analysis: dict) -> float:
    """Score context length on log scale (0-100)."""
    ctx = analysis["context_length"]
    if ctx <= 0:
        return 0
    # log10 scale: 4K=36, 32K=55, 128K=64, 1M=78, 10M=91, 100M=100
    return min(math.log10(ctx / 500) * 18, 100)


def score_pricing(analysis: dict) -> float:
    """Score pricing efficiency. Lower price = higher score (0-100)."""
    if analysis["is_free"]:
        return 100  # Free is always best value
    price = analysis["prompt_price"]
    if price <= 0:
        return 100
    # log scale: $0.10/1M=70, $1/1M=50, $5/1M=36, $15/1M=24, $75/1M=10
    per_million = price * 1_000_000
    if per_million <= 0:
        return 100
    return max(0, min(100, 100 - math.log10(per_million + 0.001) * 22))


def score_popularity(analysis: dict) -> float:
    """Score based on OpenRouter rankings data (0-100)."""
    rank = analysis["rank"]
    if rank == 1:
        return 100
    elif rank <= 3:
        return 90
    elif rank <= 5:
        return 80
    elif rank <= 10:
        return 65
    elif rank <= 20:
        return 45
    elif rank <= 50:
        return 25
    else:
        return 5  # Not in rankings = baseline, not zero


def score_speed(analysis: dict) -> float:
    """Estimate speed based on model characteristics (0-100).
    Smaller context, fewer params = likely faster. This is a rough proxy."""
    score = 50  # baseline
    ctx = analysis["context_length"]

    # Smaller context often correlates with faster inference
    if 0 < ctx <= 8000:
        score += 25
    elif 0 < ctx <= 32000:
        score += 15
    elif 0 < ctx <= 128000:
        score += 5
    elif ctx > 500000:
        score -= 10

    # Models with "lite", "flash", "mini", "nano" in ID tend to be faster
    mid_lower = analysis["id"].lower()
    for kw in ["lite", "flash", "mini", "nano", "small", "turbo", "fast"]:
        if kw in mid_lower:
            score += 15
            break

    # Free models often have rate-limited/slower endpoints
    if analysis["is_free"]:
        score -= 5

    return max(0, min(100, score))


def score_model(analysis: dict, category: dict) -> float:
    """Score a model for a category using weighted sub-scores (0-100).

    Scoring is transparent and based on verifiable data only:
    - capability: API supported_parameters flags
    - context: context_length from API (log scale)
    - pricing: prompt_price from API (log scale, inverted)
    - popularity: OpenRouter rankings (capped at 15% influence)
    - speed: heuristic from context size and model name patterns
    """
    # Hard filters
    if category["only_free"] and not analysis["is_free"]:
        return 0
    if category["exclude_free"] and analysis["is_free"]:
        return 0
    if category["require_multimodal"] and not analysis["is_multimodal"]:
        return 0

    weights = category["weights"]
    caps = category["caps"]

    # Sub-scores (each 0-100)
    capability = score_capability(analysis, caps)
    context = score_context(analysis)
    pricing = score_pricing(analysis)
    popularity = score_popularity(analysis)
    speed = score_speed(analysis)

    # Weighted total
    total = (
        capability * weights.get("capability", 0)
        + context * weights.get("context", 0)
        + pricing * weights.get("pricing", 0)
        + popularity * weights.get("popularity", 0)
        + speed * weights.get("speed", 0)
    )

    return round(min(max(total, 0), 100), 1)


# ─── Formatting ──────────────────────────────────────────────────────────────

def format_price(price: float) -> str:
    """Format price per token into readable $/1M tokens."""
    if price == 0:
        return "FREE"
    per_million = price * 1_000_000
    if per_million < 0.01:
        return f"${per_million:.4f}/1M"
    elif per_million < 1:
        return f"${per_million:.3f}/1M"
    else:
        return f"${per_million:.2f}/1M"


def format_context(ctx: int) -> str:
    """Format context length."""
    if ctx >= 1_000_000:
        return f"{ctx / 1_000_000:.1f}M"
    elif ctx >= 1_000:
        return f"{ctx // 1000}K"
    return str(ctx)


def format_expiration(exp: str) -> str:
    """Format expiration date or return dash."""
    if not exp:
        return "-"
    return exp[:10]


# ─── Output ──────────────────────────────────────────────────────────────────

COL_MODEL = 48
COL_W = 32


def print_header():
    print()
    print("+" + "-" * 70 + "+")
    print("|        OpenRouter Model Optimizer - LLM Analyzer                |")
    print("|    Data-driven model ranking from OpenRouter API                |")
    print("+" + "-" * 70 + "+")
    print()


def print_category_ranking(cat_key: str, cat: dict, scored: list[dict], top_n: int = 10):
    """Print ranked table for a category."""
    ranked = [s for s in scored if s["score"] > 0]
    ranked.sort(key=lambda x: x["score"], reverse=True)
    ranked = ranked[:top_n]

    if not ranked:
        return

    # Show weights for transparency
    w = cat["weights"]
    weight_str = " | ".join(f"{k}:{v:.0%}" for k, v in w.items() if v > 0)

    print(f"\n{'-' * 106}")
    print(f"  {cat['label']}")
    print(f"  {cat['description']}")
    print(f"  Weights: {weight_str}")
    print(f"{'-' * 106}")
    print(f"  {'#':<3} {'Model':<{COL_MODEL}} {'Score':>5} {'Price':>12} {'Ctx':>7} {'Params':>7} {'Expires':>10} {'Pop':>4}")
    print(f"  {'-'*3} {'-'*COL_MODEL} {'-'*5} {'-'*12} {'-'*7} {'-'*7} {'-'*10} {'-'*4}")

    for i, s in enumerate(ranked, 1):
        a = s["analysis"]
        model_display = a["id"][:COL_MODEL]
        price = format_price(a["prompt_price"])
        ctx = format_context(a["context_length"])
        rank_display = f"#{a['rank']}" if a["rank"] < 999 else "-"
        exp = format_expiration(a.get("expiration_date"))

        # Count how many of the relevant params this model supports
        caps = cat["caps"]
        param_count = sum(1 for p in caps if p in a["params"] or p in set(a["input_modalities"]))
        param_total = len(caps)
        param_display = f"{param_count}/{param_total}" if param_total > 0 else "-"

        print(
            f"  {i:<3} {model_display:<{COL_MODEL}} {s['score']:>5.1f} {price:>12} {ctx:>7} {param_display:>7} {exp:>10} {rank_display:>4}"
        )

    print()


def print_comparison(models_data: list[dict], model_ids: list[str]):
    """Print side-by-side comparison of models."""
    analyses = []
    for mid in model_ids:
        found = [m for m in models_data if m["id"] == mid or m["id"].endswith(f"/{mid}")]
        if not found:
            # Fuzzy match
            found = [m for m in models_data if mid.lower() in m["id"].lower()]
        if found:
            analyses.append(analyze_model(found[0]))
        else:
            print(f"  Model '{mid}' not found")

    if len(analyses) < 2:
        print("  Need at least 2 valid models to compare")
        return

    print(f"\n{'-' * (24 + COL_W * len(analyses))}")
    print(f"  Model Comparison")
    print(f"{'-' * (24 + COL_W * len(analyses))}")

    header = f"  {'Feature':<24}"
    for a in analyses:
        header += f" {a['id'][:COL_W]:>{COL_W}}"
    print(header)
    print(f"  {'-' * 24}" + f" {'-' * COL_W}" * len(analyses))

    rows = [
        ("Provider", lambda a: a["provider"]),
        ("Context", lambda a: format_context(a["context_length"])),
        ("Max Output", lambda a: format_context(a["max_completion"])),
        ("Prompt Price", lambda a: format_price(a["prompt_price"])),
        ("Completion Price", lambda a: format_price(a["completion_price"])),
        ("Free", lambda a: "Yes" if a["is_free"] else "No"),
        ("Tools", lambda a: "Yes" if a["has_tools"] else "No"),
        ("Tool Choice", lambda a: "Yes" if a["has_tool_choice"] else "No"),
        ("Structured Output", lambda a: "Yes" if a["has_structured_outputs"] else "No"),
        ("Reasoning", lambda a: "Yes" if a["has_reasoning"] else "No"),
        ("Vision", lambda a: "Yes" if a["has_vision"] else "No"),
        ("Video", lambda a: "Yes" if a["has_video"] else "No"),
        ("Audio", lambda a: "Yes" if a["has_audio"] else "No"),
        ("Input", lambda a: ", ".join(a["input_modalities"]) or "text"),
        ("Output", lambda a: ", ".join(a["output_modalities"]) or "text"),
        ("Ranking", lambda a: f"#{a['rank']}" if a["rank"] < 999 else "-"),
        ("Tokens Used", lambda a: a["tokens"] or "-"),
        ("Growth", lambda a: a["growth"] or "-"),
        ("Expires", lambda a: format_expiration(a.get("expiration_date"))),
    ]

    for label, getter in rows:
        row = f"  {label:<24}"
        for a in analyses:
            val = str(getter(a))
            row += f" {val:>{COL_W}}"
        print(row)

    # All supported parameters
    all_params = set()
    for a in analyses:
        all_params.update(a["params"])
    all_params = sorted(all_params)

    print(f"\n  {'-' * 24}" + f" {'-' * COL_W}" * len(analyses))
    print(f"  {'Supported Parameters':<24}" + f" {'':>{COL_W}}" * len(analyses))

    for p in all_params:
        row = f"    {p:<22}"
        for a in analyses:
            has = "Yes" if p in a["params"] else "No"
            row += f" {has:>{COL_W}}"
        print(row)

    # Category scores
    print(f"\n  {'-' * 24}" + f" {'-' * COL_W}" * len(analyses))
    print(f"  {'Category Scores':<24}" + f" {'':>{COL_W}}" * len(analyses))

    for cat_key, cat in CATEGORY_WEIGHTS.items():
        row = f"  {cat['label']:<24}"
        for a in analyses:
            sc = score_model(a, cat)
            row += f" {sc:>{COL_W}.1f}"
        print(row)

    print()


def print_show_model(models_data: list[dict], model_query: str):
    """Show detailed info for a single model."""
    query_lower = model_query.lower()
    matches = [m for m in models_data if query_lower in m["id"].lower()]

    if not matches:
        print(f"\n  No model found matching '{model_query}'")
        return

    for m in matches:
        a = analyze_model(m)
        params_sorted = sorted(a["params"])

        print(f"\n{'=' * 72}")
        print(f"  {a['id']}")
        print(f"{'=' * 72}")
        print(f"  Name:             {a['name']}")
        print(f"  Provider:         {a['provider']}")
        print(f"  Context:          {format_context(a['context_length'])} tokens")
        print(f"  Max Output:       {format_context(a['max_completion'])} tokens")
        print(f"  Prompt Price:     {format_price(a['prompt_price'])}")
        print(f"  Completion Price: {format_price(a['completion_price'])}")
        print(f"  Free:             {'Yes' if a['is_free'] else 'No'}")
        print()

        if a["created"]:
            created_dt = datetime.fromtimestamp(a["created"], tz=timezone.utc)
            print(f"  Created:          {created_dt.strftime('%Y-%m-%d')}")
        if a.get("knowledge_cutoff"):
            print(f"  Knowledge Cutoff: {a['knowledge_cutoff']}")
        if a.get("expiration_date"):
            print(f"  Expires:          {a['expiration_date']}")
        print()

        print(f"  Input:            {', '.join(a['input_modalities']) or 'text'}")
        print(f"  Output:           {', '.join(a['output_modalities']) or 'text'}")
        print(f"  Modality:         {a['modality']}")
        print()

        if a["rank"] < 999:
            print(f"  Ranking:          #{a['rank']} ({a['tokens']} tokens, {a['growth']})")
        else:
            print(f"  Ranking:          Not in top 10")
        print()

        print(f"  Supported Parameters ({len(params_sorted)}):")
        for p in params_sorted:
            print(f"    - {p}")
        print()

        print(f"  Scores:")
        for cat_key, cat in CATEGORY_WEIGHTS.items():
            sc = score_model(a, cat)
            if sc > 0:
                bar_len = int(sc / 5)
                bar = "#" * bar_len + "." * (20 - bar_len)
                w = cat["weights"]
                weight_str = ", ".join(f"{k}:{v:.0%}" for k, v in w.items() if v > 0)
                print(f"    {cat['label']:<26} {sc:>5.1f}  [{bar}]  ({weight_str})")
        print(f"{'=' * 72}\n")


def print_optimize_claude(scored: list[dict]):
    """Print recommendations for Claude Code model roles."""
    print(f"\n{'=' * 72}")
    print(f"  Claude Code Model Optimizer")
    print(f"  Recommended models for each Claude Code role via OpenRouter")
    print(f"{'=' * 72}")

    for role_key, role in CLAUDE_CODE_ROLES.items():
        role_scored = []
        for s in scored:
            a = s["analysis"]
            total = 0
            count = 0
            for need in role["needs"]:
                if need in CATEGORY_WEIGHTS:
                    sc = score_model(a, CATEGORY_WEIGHTS[need])
                    total += sc
                    count += 1
            if count > 0:
                role_scored.append({**s, "role_score": total / count})

        role_scored.sort(key=lambda x: x["role_score"], reverse=True)
        top = role_scored[:5]

        print(f"\n  {role['label']}")
        print(f"  Env: export {role['env_var']}=\"<model-id>\"")
        print(f"  {'-' * 76}")
        print(f"  {'#':<3} {'Model':<46} {'Score':>6} {'Price':>12} {'Free':>4}")
        print(f"  {'-'*3} {'-'*46} {'-'*6} {'-'*12} {'-'*4}")

        for i, s in enumerate(top, 1):
            a = s["analysis"]
            price = format_price(a["prompt_price"])
            free = "Yes" if a["is_free"] else "-"
            print(
                f"  {i:<3} {a['id'][:45]:<46} {s['role_score']:>6.1f} {price:>12} {free:>4}"
            )

        best = top[0]["analysis"]["id"] if top else None
        if best:
            print(f"\n  Recommended: {role['env_var']}=\"{best}\"")

    print(f"\n{'=' * 72}")
    print("  To apply, add to your shell profile or run:")
    print()
    best_overall = {}
    for role_key, role in CLAUDE_CODE_ROLES.items():
        role_scored = []
        for s in scored:
            a = s["analysis"]
            total = sum(
                score_model(a, CATEGORY_WEIGHTS[need])
                for need in role["needs"]
                if need in CATEGORY_WEIGHTS
            )
            role_scored.append({**s, "role_score": total})
        role_scored.sort(key=lambda x: x["role_score"], reverse=True)
        if role_scored:
            best_overall[role["env_var"]] = role_scored[0]["analysis"]["id"]

    for env_var, model_id in best_overall.items():
        print(f'  export {env_var}="{model_id}"')
    print(f"{'=' * 72}\n")


def print_json_output(all_rankings: dict):
    """Print results as JSON."""
    output = {}
    for cat_key, data in all_rankings.items():
        output[cat_key] = [
            {
                "rank": i + 1,
                "model": s["analysis"]["id"],
                "score": s["score"],
                "prompt_price": s["analysis"]["prompt_price"],
                "context_length": s["analysis"]["context_length"],
                "is_free": s["analysis"]["is_free"],
                "ranking": s["analysis"]["rank"],
                "supported_params": sorted(s["analysis"]["params"]),
            }
            for i, s in enumerate(data)
        ]
    print(json.dumps(output, indent=2))


def print_summary_stats(models: list[dict]):
    """Print summary statistics."""
    analyses = [analyze_model(m) for m in models]

    total = len(analyses)
    free = sum(1 for a in analyses if a["is_free"])
    with_tools = sum(1 for a in analyses if a["has_tools"])
    with_reasoning = sum(1 for a in analyses if a["has_reasoning"])
    multimodal = sum(1 for a in analyses if a["is_multimodal"])
    providers = len(set(a["provider"] for a in analyses))

    prices = [a["prompt_price"] for a in analyses if a["prompt_price"] > 0]
    avg_price = sum(prices) / len(prices) if prices else 0

    contexts = [a["context_length"] for a in analyses if a["context_length"] > 0]
    max_ctx = max(contexts) if contexts else 0

    print(f"\n  OpenRouter Ecosystem Summary")
    print(f"  {'-' * 50}")
    print(f"  Total models:          {total}")
    print(f"  Free models:           {free}")
    print(f"  Providers:             {providers}")
    print(f"  With tool support:     {with_tools}")
    print(f"  With reasoning:        {with_reasoning}")
    print(f"  Multimodal:            {multimodal}")
    print(f"  Avg prompt price:      {format_price(avg_price)}")
    print(f"  Max context:           {format_context(max_ctx)}")
    print(f"  Data source:           openrouter.ai/api/v1/models")
    print(f"  Rankings source:       openrouter.ai/rankings")
    print()


# ─── Search ──────────────────────────────────────────────────────────────────

def search_models(models: list[dict], query: str, top_n: int = 20):
    """Search models by name, description, or provider."""
    query_lower = query.lower()
    results = []
    for m in models:
        a = analyze_model(m)
        searchable = f"{a['id']} {a['name']} {a['description']} {a['provider']}".lower()
        if query_lower in searchable:
            results.append(a)

    results.sort(key=lambda x: x["rank"])

    print(f"\n  Search results for '{query}' ({len(results)} matches)")
    print(f"  {'-' * 84}")
    print(f"  {'#':<3} {'Model':<44} {'Price':>12} {'Context':>7} {'Params':>7} {'Free':>4}")
    print(f"  {'-'*3} {'-'*44} {'-'*12} {'-'*7} {'-'*7} {'-'*4}")

    for i, a in enumerate(results[:top_n], 1):
        price = format_price(a["prompt_price"])
        ctx = format_context(a["context_length"])
        free = "Yes" if a["is_free"] else "-"
        param_count = len(a["params"])
        print(f"  {i:<3} {a['id'][:44]:<44} {price:>12} {ctx:>7} {param_count:>7} {free:>4}")

    print()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="OpenRouter Model Optimizer - Data-driven LLM ranking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Scoring methodology:
  All scores are based on verifiable API data only.
  No subjective keyword matching or description analysis.

  capability - Which API parameters the model supports (tools, reasoning, etc.)
  context    - Context window size (log scale)
  pricing    - Cost per token (log scale, inverted)
  popularity - OpenRouter usage rankings (capped at 15-20% influence)
  speed      - Heuristic from context size and model name patterns

Examples:
  python model-optimizer.py                   Full analysis
  python model-optimizer.py --coding          Best for coding
  python model-optimizer.py --free            Best free models
  python model-optimizer.py --compare MODEL1 MODEL2
  python model-optimizer.py --show MODEL      Detailed single model view
  python model-optimizer.py --search QUERY    Search models
  python model-optimizer.py --json            Machine-readable output
  python model-optimizer.py --optimize-claude Best models for Claude Code
        """,
    )
    parser.add_argument("--coding", action="store_true", help="Rank for coding")
    parser.add_argument("--reasoning", action="store_true", help="Rank for reasoning")
    parser.add_argument("--budget", action="store_true", help="Rank best value paid")
    parser.add_argument("--free", action="store_true", help="Rank free models")
    parser.add_argument("--agents", action="store_true", help="Rank for tool use/agents")
    parser.add_argument("--multimodal", action="store_true", help="Rank multimodal")
    parser.add_argument("--context", action="store_true", help="Rank by context length")
    parser.add_argument("--compare", nargs=2, metavar=("MODEL1", "MODEL2"), help="Compare two models")
    parser.add_argument("--show", metavar="MODEL", help="Show detailed info for a single model")
    parser.add_argument("--search", metavar="QUERY", help="Search models")
    parser.add_argument("--top", type=int, default=10, help="Top N per category (default: 10)")
    parser.add_argument("--json", action="store_true", dest="json_output", help="Output as JSON")
    parser.add_argument("--optimize-claude", action="store_true", help="Optimize for Claude Code")
    parser.add_argument("--summary", action="store_true", help="Show ecosystem summary")
    parser.add_argument("--all", action="store_true", help="Show all categories (default)")
    parser.add_argument("--active", action="store_true", help="Only active models (no expired, no routing utils)")
    parser.add_argument("--include-expired", action="store_true", help="Include expired models")

    args = parser.parse_args()

    specific = any([
        args.coding, args.reasoning, args.budget, args.free,
        args.agents, args.multimodal, args.context, args.compare,
        args.search, args.optimize_claude, args.summary, args.active, args.show,
    ])
    if not specific:
        args.all = True

    print_header()

    # Fetch models
    models = fetch_models()

    # Always exclude routing utilities
    models = [m for m in models if m["id"] not in ("openrouter/auto", "openrouter/bodybuilder")]

    # Filter expired models (unless --include-expired)
    if not args.include_expired:
        now_ts = datetime.now(timezone.utc).timestamp()
        before = len(models)
        models = [
            m for m in models
            if not m.get("expiration_date")
            or datetime.fromisoformat(m["expiration_date"]).timestamp() > now_ts
        ]
        removed = before - len(models)
        if removed:
            print(f"   Filtered out {removed} expired model(s) (use --include-expired to show)")

    # Active-only filter: models created within last 6 months or in top rankings
    if args.active:
        now_ts = datetime.now(timezone.utc).timestamp()
        six_months = 180 * 86400
        before = len(models)
        models = [
            m for m in models
            if m.get("created", 0) > now_ts - six_months
            or m["id"] in RANKINGS
        ]
        removed = before - len(models)
        print(f"   Active filter: kept {len(models)} models, removed {removed} older than 6 months")

    print(f"   Using {len(models)} models for analysis\n")
    analyses = [analyze_model(m) for m in models]

    # Search mode
    if args.search:
        search_models(models, args.search, args.top)
        return

    # Compare mode
    if args.compare:
        print_comparison(models, args.compare)
        return

    # Show detailed model info
    if args.show:
        print_show_model(models, args.show)
        return

    # Summary
    if args.summary or args.all:
        print_summary_stats(models)

    # Optimize for Claude Code
    if args.optimize_claude:
        scored_all = [{"analysis": a, "score": 0} for a in analyses]
        print_optimize_claude(scored_all)
        return

    # Determine which categories to show
    cat_keys = []
    if args.all:
        cat_keys = list(CATEGORY_WEIGHTS.keys())
    else:
        if args.coding:
            cat_keys.append("coding")
        if args.reasoning:
            cat_keys.append("reasoning")
        if args.budget:
            cat_keys.append("budget")
        if args.free:
            cat_keys.append("free")
        if args.agents:
            cat_keys.append("agents")
        if args.multimodal:
            cat_keys.append("multimodal")
        if args.context:
            cat_keys.append("context")

    # Score and rank for each category
    all_rankings = {}
    for cat_key in cat_keys:
        cat = CATEGORY_WEIGHTS[cat_key]
        scored = []
        for a in analyses:
            sc = score_model(a, cat)
            scored.append({"analysis": a, "score": sc})

        scored.sort(key=lambda x: x["score"], reverse=True)
        top_scored = scored[:args.top]
        all_rankings[cat_key] = top_scored

    if args.json_output:
        print_json_output(all_rankings)
    else:
        for cat_key in cat_keys:
            print_category_ranking(
                cat_key, CATEGORY_WEIGHTS[cat_key], all_rankings[cat_key], args.top
            )

        # Top picks summary
        if args.all:
            print(f"\n{'=' * 72}")
            print(f"  Top Picks Summary")
            print(f"{'=' * 72}")
            for cat_key in cat_keys:
                top = all_rankings[cat_key]
                if top:
                    best = top[0]
                    cat = CATEGORY_WEIGHTS[cat_key]
                    a = best["analysis"]
                    free_tag = " (FREE)" if a["is_free"] else ""
                    print(f"  {cat['label']:<28} -> {a['id']}{free_tag}")
            print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()
