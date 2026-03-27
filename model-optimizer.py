#!/usr/bin/env python3
"""
llm-radar - Data-driven LLM ranking tool for OpenRouter.

Analyze, compare, and rank 340+ models using real API data.
No subjective scoring - every score based on verifiable capabilities.

Usage:
    python model-optimizer.py                           # Full analysis
    python model-optimizer.py --coding                  # Best for coding
    python model-optimizer.py --free                    # Best free models
    python model-optimizer.py --compare MODEL1 MODEL2   # Compare models
    python model-optimizer.py --show MODEL              # Detailed view
    python model-optimizer.py --search QUERY            # Search models
    python model-optimizer.py --cost 10000 5000         # Cost for 10K in / 5K out tokens
    python model-optimizer.py --provider anthropic      # Filter by provider
    python model-optimizer.py --max-price 2.00          # Max $2/1M tokens
    python model-optimizer.py --min-context 128000      # Min 128K context
    python model-optimizer.py --has-tools               # Only models with tools
    python model-optimizer.py --has-reasoning           # Only models with reasoning
    python model-optimizer.py --csv --coding            # Export as CSV
    python model-optimizer.py --markdown --coding       # Export as Markdown
    python model-optimizer.py --json                    # Export as JSON
    python model-optimizer.py --optimize-claude         # Best for Claude Code
    python model-optimizer.py --update-rankings         # Fetch live rankings
"""

import argparse
import csv
import io
import json
import math
import re
import sys
from datetime import datetime, timezone
import requests

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ─── Constants ───────────────────────────────────────────────────────────────

MODELS_API = "https://openrouter.ai/api/v1/models"
RANKINGS_URL = "https://openrouter.ai/rankings"
RANKINGS_CACHE = "rankings_cache.json"

# Fallback rankings (used if live fetch fails)
FALLBACK_RANKINGS = {
    "xiaomi/mimo-v2-pro": {"rank": 1, "total_tokens": 3280000000000},
    "stepfun/step-3.5-flash": {"rank": 2, "total_tokens": 1530000000000},
    "deepseek/deepseek-v3.2": {"rank": 3, "total_tokens": 1200000000000},
    "minimax/minimax-m2.7": {"rank": 4, "total_tokens": 1080000000000},
    "minimax/minimax-m2.5": {"rank": 5, "total_tokens": 1050000000000},
    "z-ai/glm-5-turbo": {"rank": 6, "total_tokens": 1030000000000},
    "anthropic/claude-opus-4.6": {"rank": 7, "total_tokens": 1020000000000},
    "anthropic/claude-sonnet-4.6": {"rank": 8, "total_tokens": 1010000000000},
    "google/gemini-3-flash-preview": {"rank": 9, "total_tokens": 950000000000},
    "google/gemini-2.5-flash-lite": {"rank": 10, "total_tokens": 553000000000},
}

# Category definitions with scoring weights
CATEGORIES = {
    "coding": {
        "label": "Best for Coding",
        "description": "Code generation, debugging, refactoring, technical writing",
        "weights": {"capability": 0.50, "context": 0.15, "pricing": 0.15, "popularity": 0.10, "speed": 0.10},
        "caps": {"tools": 40, "tool_choice": 20, "structured_outputs": 15, "reasoning": 15, "response_format": 10},
        "exclude_free": False, "only_free": False, "require_multimodal": False,
    },
    "reasoning": {
        "label": "Best for Reasoning",
        "description": "Complex analysis, math, logic, multi-step problem solving",
        "weights": {"capability": 0.55, "context": 0.20, "pricing": 0.10, "popularity": 0.10, "speed": 0.05},
        "caps": {"reasoning": 35, "include_reasoning": 25, "tools": 15, "structured_outputs": 10, "response_format": 10, "frequency_penalty": 5},
        "exclude_free": False, "only_free": False, "require_multimodal": False,
    },
    "budget": {
        "label": "Best Value (Paid)",
        "description": "Best capability per dollar for paid models",
        "weights": {"capability": 0.25, "context": 0.10, "pricing": 0.35, "popularity": 0.20, "speed": 0.10},
        "caps": {"tools": 25, "reasoning": 20, "tool_choice": 15, "structured_outputs": 15, "response_format": 10, "seed": 5, "stop": 10},
        "exclude_free": True, "only_free": False, "require_multimodal": False,
    },
    "free": {
        "label": "Best Free Models",
        "description": "Top free models ranked by capability",
        "weights": {"capability": 0.45, "context": 0.25, "pricing": 0.0, "popularity": 0.15, "speed": 0.15},
        "caps": {"tools": 30, "reasoning": 25, "tool_choice": 15, "structured_outputs": 15, "response_format": 10, "seed": 5},
        "exclude_free": False, "only_free": True, "require_multimodal": False,
    },
    "agents": {
        "label": "Best for Agents/Tool Use",
        "description": "Agentic workflows, tool calling, function execution",
        "weights": {"capability": 0.55, "context": 0.15, "pricing": 0.10, "popularity": 0.10, "speed": 0.10},
        "caps": {"tools": 35, "tool_choice": 25, "structured_outputs": 15, "reasoning": 15, "response_format": 10},
        "exclude_free": False, "only_free": False, "require_multimodal": False,
    },
    "multimodal": {
        "label": "Best Multimodal",
        "description": "Image, video, audio understanding and generation",
        "weights": {"capability": 0.50, "context": 0.15, "pricing": 0.10, "popularity": 0.15, "speed": 0.10},
        "caps": {"image": 30, "video": 20, "audio": 20, "tools": 15, "reasoning": 10, "structured_outputs": 5},
        "exclude_free": False, "only_free": False, "require_multimodal": True,
    },
    "context": {
        "label": "Largest Context",
        "description": "Models with the largest context windows",
        "weights": {"capability": 0.0, "context": 0.70, "pricing": 0.10, "popularity": 0.10, "speed": 0.10},
        "caps": {},
        "exclude_free": False, "only_free": False, "require_multimodal": False,
    },
}

CLAUDE_CODE_ROLES = {
    "opus-class": {
        "label": "Opus-class (Complex Reasoning)",
        "env_var": "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "needs": ["reasoning", "coding"],
    },
    "sonnet-class": {
        "label": "Sonnet-class (General Coding)",
        "env_var": "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "needs": ["coding", "agents"],
    },
    "haiku-class": {
        "label": "Haiku-class (Fast Completions)",
        "env_var": "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "needs": ["coding", "free"],
    },
    "subagent": {
        "label": "Sub-agent Model",
        "env_var": "CLAUDE_CODE_SUBAGENT_MODEL",
        "needs": ["agents", "coding"],
    },
}


# ─── Rankings Fetching ───────────────────────────────────────────────────────

def fetch_live_rankings() -> dict:
    """Fetch live rankings from OpenRouter rankings page."""
    print("Fetching live rankings from OpenRouter...")
    try:
        resp = requests.get(RANKINGS_URL, timeout=30)
        resp.raise_for_status()
        html = resp.text

        # Extract RSC payload containing rankingData
        pushes = re.findall(r'self\.__next_f\.push\(\[1,"(.*?)"\]\)', html, re.DOTALL)

        ranking_data = []
        for p in pushes:
            if 'rankingData' in p and 'model_permaslug' in p:
                unescaped = p.replace('\\"', '"').replace('\\\\', '\\')
                # Find the array bounds
                arr_start = unescaped.find('"rankingData":[') + len('"rankingData":')
                depth = 0
                arr_end = arr_start
                for ci, c in enumerate(unescaped[arr_start:]):
                    if c == '[':
                        depth += 1
                    elif c == ']':
                        depth -= 1
                    if depth == 0:
                        arr_end = arr_start + ci + 1
                        break
                try:
                    ranking_data = json.loads(unescaped[arr_start:arr_end])
                except json.JSONDecodeError:
                    pass
                break

        if not ranking_data:
            print("   Could not parse live rankings, using fallback")
            return FALLBACK_RANKINGS

        # Aggregate by base model ID (strip date suffixes)
        models = {}
        for entry in ranking_data:
            slug = entry.get("model_permaslug", "")
            if not slug:
                continue
            # Normalize: strip date suffix like -20260205
            base_slug = re.sub(r'-\d{8}$', '', slug)
            total_tokens = entry.get("total_prompt_tokens", 0) + entry.get("total_completion_tokens", 0)

            if base_slug not in models:
                models[base_slug] = {"total_tokens": 0, "requests": 0}
            models[base_slug]["total_tokens"] += total_tokens
            models[base_slug]["requests"] += entry.get("count", 0)

        # Sort and rank
        ranked_list = sorted(models.items(), key=lambda x: x[1]["total_tokens"], reverse=True)
        rankings = {}
        for i, (slug, data) in enumerate(ranked_list[:100], 1):
            rankings[slug] = {
                "rank": i,
                "total_tokens": data["total_tokens"],
                "requests": data["requests"],
            }

        # Cache to file
        try:
            with open(RANKINGS_CACHE, "w") as f:
                json.dump({
                    "updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                    "source": "openrouter.ai/rankings",
                    "entries": len(ranking_data),
                    "rankings": rankings,
                }, f, indent=2)
            print(f"   Fetched {len(ranking_data)} entries, ranked {len(rankings)} models (cached to {RANKINGS_CACHE})")
        except Exception:
            print(f"   Fetched {len(ranking_data)} entries, ranked {len(rankings)} models")

        return rankings

    except Exception as e:
        print(f"   Live fetch failed: {e}")
        print("   Using fallback rankings")
        return FALLBACK_RANKINGS


def load_cached_rankings() -> dict:
    """Load rankings from cache file."""
    try:
        with open(RANKINGS_CACHE, "r") as f:
            data = json.load(f)
        print(f"   Loaded cached rankings (updated: {data.get('updated', 'unknown')})")
        return data.get("rankings", {})
    except Exception:
        return {}


def get_rankings(live: bool = False) -> dict:
    """Get rankings - either live fetch, cached, or fallback."""
    if live:
        return fetch_live_rankings()

    # Try cache first
    cached = load_cached_rankings()
    if cached:
        return cached

    return FALLBACK_RANKINGS


# ─── Data Fetching ───────────────────────────────────────────────────────────

def fetch_models() -> list[dict]:
    """Fetch all models from OpenRouter API."""
    print("Fetching models from OpenRouter API...")
    resp = requests.get(MODELS_API, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    models = data["data"]
    print(f"   Found {len(models)} models")
    return models


# ─── Model Analysis ──────────────────────────────────────────────────────────

def analyze_model(model: dict, rankings: dict) -> dict:
    """Extract features from a model for scoring."""
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

    ranking = rankings.get(mid, {})
    # Also try matching without version suffixes
    if not ranking:
        for key in rankings:
            if mid.startswith(key) or key.startswith(mid):
                ranking = rankings[key]
                break

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
        "total_tokens": ranking.get("total_tokens", 0),
        "requests": ranking.get("requests", 0),
        "created": model.get("created", 0),
        "knowledge_cutoff": model.get("knowledge_cutoff"),
        "expiration_date": model.get("expiration_date"),
    }


# ─── Scoring ─────────────────────────────────────────────────────────────────

def score_capability(analysis: dict, caps: dict) -> float:
    """Score model capabilities based on supported API parameters (0-100)."""
    if not caps:
        return 0
    score = 0.0
    params = analysis["params"]
    input_mods = set(analysis["input_modalities"])
    for param, weight in caps.items():
        if param in params or param in input_mods:
            score += weight
    return min(score, 100)


def score_context(analysis: dict) -> float:
    """Score context length on log scale (0-100)."""
    ctx = analysis["context_length"]
    if ctx <= 0:
        return 0
    return min(math.log10(ctx / 500) * 18, 100)


def score_pricing(analysis: dict) -> float:
    """Score pricing efficiency (0-100)."""
    if analysis["is_free"]:
        return 100
    price = analysis["prompt_price"]
    if price <= 0:
        return 100
    per_million = price * 1_000_000
    if per_million <= 0:
        return 100
    return max(0, min(100, 100 - math.log10(per_million + 0.001) * 22))


def score_popularity(analysis: dict) -> float:
    """Score based on OpenRouter rankings (0-100)."""
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
    return 5


def score_speed(analysis: dict) -> float:
    """Estimate speed from context size and model name (0-100)."""
    score = 50
    ctx = analysis["context_length"]
    if 0 < ctx <= 8000:
        score += 25
    elif 0 < ctx <= 32000:
        score += 15
    elif 0 < ctx <= 128000:
        score += 5
    elif ctx > 500000:
        score -= 10
    mid_lower = analysis["id"].lower()
    for kw in ["lite", "flash", "mini", "nano", "small", "turbo", "fast"]:
        if kw in mid_lower:
            score += 15
            break
    if analysis["is_free"]:
        score -= 5
    return max(0, min(100, score))


def score_model(analysis: dict, category: dict) -> float:
    """Score a model for a category using weighted sub-scores (0-100)."""
    if category["only_free"] and not analysis["is_free"]:
        return 0
    if category["exclude_free"] and analysis["is_free"]:
        return 0
    if category["require_multimodal"] and not analysis["is_multimodal"]:
        return 0

    w = category["weights"]
    caps = category["caps"]

    capability = score_capability(analysis, caps)
    context = score_context(analysis)
    pricing = score_pricing(analysis)
    popularity = score_popularity(analysis)
    speed = score_speed(analysis)

    total = (
        capability * w.get("capability", 0)
        + context * w.get("context", 0)
        + pricing * w.get("pricing", 0)
        + popularity * w.get("popularity", 0)
        + speed * w.get("speed", 0)
    )
    return round(min(max(total, 0), 100), 1)


# ─── Cost Calculator ─────────────────────────────────────────────────────────

def calculate_cost(prompt_price: float, completion_price: float,
                   input_tokens: int, output_tokens: int) -> float:
    """Calculate cost for a given token usage."""
    return (prompt_price * input_tokens) + (completion_price * output_tokens)


def format_cost(cost: float) -> str:
    """Format cost as readable string."""
    if cost == 0:
        return "$0.00"
    elif cost < 0.01:
        return f"${cost:.6f}"
    elif cost < 1:
        return f"${cost:.4f}"
    elif cost < 100:
        return f"${cost:.2f}"
    else:
        return f"${cost:.2f}"


# ─── Formatting ──────────────────────────────────────────────────────────────

def format_price(price: float) -> str:
    """Format price per 1M tokens."""
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


def format_tokens(tokens: int) -> str:
    """Format token count."""
    if tokens >= 1e12:
        return f"{tokens/1e12:.2f}T"
    elif tokens >= 1e9:
        return f"{tokens/1e9:.2f}B"
    elif tokens >= 1e6:
        return f"{tokens/1e6:.1f}M"
    elif tokens >= 1e3:
        return f"{tokens/1e3:.1f}K"
    return str(tokens)


def format_expiration(exp: str) -> str:
    return exp[:10] if exp else "-"


# ─── Filtering ───────────────────────────────────────────────────────────────

def apply_filters(models: list[dict], args) -> list[dict]:
    """Apply CLI filters to model list."""
    # Exclude routing utilities
    models = [m for m in models if m["id"] not in ("openrouter/auto", "openrouter/bodybuilder")]

    # Expired filter
    if not args.include_expired:
        now_ts = datetime.now(timezone.utc).timestamp()
        models = [
            m for m in models
            if not m.get("expiration_date")
            or datetime.fromisoformat(m["expiration_date"]).timestamp() > now_ts
        ]

    # Active filter
    if args.active:
        now_ts = datetime.now(timezone.utc).timestamp()
        six_months = 180 * 86400
        rankings = get_rankings(live=False)
        models = [
            m for m in models
            if m.get("created", 0) > now_ts - six_months
            or m["id"] in rankings
        ]

    return models


def apply_post_filters(analyses: list[dict], args) -> list[dict]:
    """Apply post-analysis filters."""
    if args.provider:
        provider = args.provider.lower()
        analyses = [a for a in analyses if a["provider"] == provider]

    if args.max_price is not None:
        max_p = args.max_price / 1_000_000  # Convert from $/1M to per-token
        analyses = [a for a in analyses if a["is_free"] or a["prompt_price"] <= max_p]

    if args.min_context:
        analyses = [a for a in analyses if a["context_length"] >= args.min_context]

    if args.has_tools:
        analyses = [a for a in analyses if a["has_tools"]]

    if args.has_reasoning:
        analyses = [a for a in analyses if a["has_reasoning"]]

    if args.free_only:
        analyses = [a for a in analyses if a["is_free"]]

    if args.paid_only:
        analyses = [a for a in analyses if not a["is_free"]]

    if args.multimodal_only:
        analyses = [a for a in analyses if a["is_multimodal"]]

    return analyses


# ─── Output: Tables ──────────────────────────────────────────────────────────

COL_MODEL = 48

def print_header():
    print()
    print("+" + "-" * 70 + "+")
    print("|  llm-radar - Data-driven LLM Ranking Tool for OpenRouter          |")
    print("+" + "-" * 70 + "+")
    print()


def print_category(cat_key, cat, scored, top_n=10):
    """Print ranked table for a category."""
    ranked = [s for s in scored if s["score"] > 0]
    ranked.sort(key=lambda x: x["score"], reverse=True)
    ranked = ranked[:top_n]

    if not ranked:
        return

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
        price = format_price(a["prompt_price"])
        ctx = format_context(a["context_length"])
        rank_d = f"#{a['rank']}" if a["rank"] < 999 else "-"
        exp = format_expiration(a.get("expiration_date"))
        caps = cat["caps"]
        param_count = sum(1 for p in caps if p in a["params"] or p in set(a["input_modalities"]))
        param_total = len(caps)
        param_d = f"{param_count}/{param_total}" if param_total > 0 else "-"
        print(f"  {i:<3} {a['id'][:COL_MODEL]:<{COL_MODEL}} {s['score']:>5.1f} {price:>12} {ctx:>7} {param_d:>7} {exp:>10} {rank_d:>4}")
    print()


# ─── Output: Cost Table ──────────────────────────────────────────────────────

def print_cost_table(analyses, input_tokens, output_tokens, top_n=20):
    """Print cost comparison table."""
    print(f"\n{'-' * 106}")
    print(f"  Cost Calculator: {format_tokens(input_tokens)} input + {format_tokens(output_tokens)} output tokens")
    print(f"{'-' * 106}")
    print(f"  {'#':<3} {'Model':<{COL_MODEL}} {'Input Cost':>12} {'Output Cost':>13} {'Total Cost':>12} {'Free':>4}")
    print(f"  {'-'*3} {'-'*COL_MODEL} {'-'*12} {'-'*13} {'-'*12} {'-'*4}")

    costs = []
    for a in analyses:
        ic = calculate_cost(a["prompt_price"], 0, input_tokens, 0)
        oc = calculate_cost(0, a["completion_price"], 0, output_tokens)
        total = ic + oc
        costs.append((a, ic, oc, total))

    costs.sort(key=lambda x: x[3])
    costs = costs[:top_n]

    for i, (a, ic, oc, total) in enumerate(costs, 1):
        free = "Yes" if a["is_free"] else "-"
        print(f"  {i:<3} {a['id'][:COL_MODEL]:<{COL_MODEL}} {format_cost(ic):>12} {format_cost(oc):>13} {format_cost(total):>12} {free:>4}")
    print()


# ─── Output: Comparison ──────────────────────────────────────────────────────

def print_comparison(models_raw, model_ids, rankings):
    """Side-by-side model comparison."""
    COL_W = 32
    analyses = []
    for mid in model_ids:
        found = [m for m in models_raw if mid.lower() in m["id"].lower()]
        if found:
            analyses.append(analyze_model(found[0], rankings))
        else:
            print(f"  Model '{mid}' not found")

    if len(analyses) < 2:
        print("  Need at least 2 valid models to compare")
        return

    width = 24 + COL_W * len(analyses)
    print(f"\n{'-' * width}")
    print(f"  Model Comparison")
    print(f"{'-' * width}")

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
        ("Total Tokens", lambda a: format_tokens(a["total_tokens"]) if a["total_tokens"] else "-"),
        ("Requests", lambda a: f"{a['requests']:,}" if a["requests"] else "-"),
        ("Expires", lambda a: format_expiration(a.get("expiration_date"))),
    ]

    for label, getter in rows:
        row = f"  {label:<24}"
        for a in analyses:
            row += f" {str(getter(a)):>{COL_W}}"
        print(row)

    # Supported parameters
    all_params = set()
    for a in analyses:
        all_params.update(a["params"])

    print(f"\n  {'-' * 24}" + f" {'-' * COL_W}" * len(analyses))
    print(f"  {'Supported Parameters':<24}" + f" {'':>{COL_W}}" * len(analyses))
    for p in sorted(all_params):
        row = f"    {p:<22}"
        for a in analyses:
            row += f" {'Yes' if p in a['params'] else 'No':>{COL_W}}"
        print(row)

    # Category scores
    print(f"\n  {'-' * 24}" + f" {'-' * COL_W}" * len(analyses))
    print(f"  {'Category Scores':<24}" + f" {'':>{COL_W}}" * len(analyses))
    for cat_key, cat in CATEGORIES.items():
        row = f"  {cat['label']:<24}"
        for a in analyses:
            row += f" {score_model(a, cat):>{COL_W}.1f}"
        print(row)

    print()


# ─── Output: Show Model ──────────────────────────────────────────────────────

def print_show_model(models_raw, query, rankings):
    """Detailed single model view."""
    matches = [m for m in models_raw if query.lower() in m["id"].lower()]
    if not matches:
        print(f"\n  No model found matching '{query}'")
        return

    for m in matches:
        a = analyze_model(m, rankings)
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
            print(f"  Ranking:          #{a['rank']} ({format_tokens(a['total_tokens'])} tokens, {a['requests']:,} requests)")
        else:
            print(f"  Ranking:          Not in top 100")
        print()

        print(f"  Supported Parameters ({len(params_sorted)}):")
        for p in params_sorted:
            print(f"    - {p}")
        print()

        print(f"  Scores:")
        for cat_key, cat in CATEGORIES.items():
            sc = score_model(a, cat)
            if sc > 0:
                bar = "#" * int(sc / 5) + "." * (20 - int(sc / 5))
                w = cat["weights"]
                ws = ", ".join(f"{k}:{v:.0%}" for k, v in w.items() if v > 0)
                print(f"    {cat['label']:<26} {sc:>5.1f}  [{bar}]  ({ws})")
        print(f"{'=' * 72}\n")


# ─── Output: Optimize Claude ─────────────────────────────────────────────────

def print_optimize_claude(analyses, rankings):
    """Claude Code model role optimizer."""
    print(f"\n{'=' * 72}")
    print(f"  Claude Code Model Optimizer")
    print(f"  Recommended models for each Claude Code role via OpenRouter")
    print(f"{'=' * 72}")

    for role_key, role in CLAUDE_CODE_ROLES.items():
        scored = []
        for a in analyses:
            total = sum(
                score_model(a, CATEGORIES[n])
                for n in role["needs"] if n in CATEGORIES
            )
            count = sum(1 for n in role["needs"] if n in CATEGORIES)
            if count > 0:
                scored.append({"analysis": a, "score": total / count})

        scored.sort(key=lambda x: x["score"], reverse=True)
        top = scored[:5]

        print(f"\n  {role['label']}")
        print(f"  Env: export {role['env_var']}=\"<model-id>\"")
        print(f"  {'-' * 76}")
        print(f"  {'#':<3} {'Model':<46} {'Score':>6} {'Price':>12} {'Free':>4}")
        print(f"  {'-'*3} {'-'*46} {'-'*6} {'-'*12} {'-'*4}")

        for i, s in enumerate(top, 1):
            a = s["analysis"]
            price = format_price(a["prompt_price"])
            free = "Yes" if a["is_free"] else "-"
            print(f"  {i:<3} {a['id'][:45]:<46} {s['score']:>6.1f} {price:>12} {free:>4}")

        if top:
            print(f"\n  Recommended: {role['env_var']}=\"{top[0]['analysis']['id']}\"")

    print(f"\n{'=' * 72}")
    print("  To apply, add to your shell profile or run:")
    print()
    for role_key, role in CLAUDE_CODE_ROLES.items():
        scored = []
        for a in analyses:
            total = sum(score_model(a, CATEGORIES[n]) for n in role["needs"] if n in CATEGORIES)
            scored.append({"analysis": a, "score": total})
        scored.sort(key=lambda x: x["score"], reverse=True)
        if scored:
            print(f'  export {role["env_var"]}="{scored[0]["analysis"]["id"]}"')
    print(f"{'=' * 72}\n")


# ─── Output: Search ──────────────────────────────────────────────────────────

def print_search(analyses, query, top_n):
    """Search models."""
    q = query.lower()
    results = [a for a in analyses if q in f"{a['id']} {a['name']} {a['description']} {a['provider']}".lower()]
    results.sort(key=lambda x: x["rank"])

    print(f"\n  Search results for '{query}' ({len(results)} matches)")
    print(f"  {'-' * 90}")
    print(f"  {'#':<3} {'Model':<46} {'Price':>12} {'Context':>7} {'Free':>4} {'Rank':>5}")
    print(f"  {'-'*3} {'-'*46} {'-'*12} {'-'*7} {'-'*4} {'-'*5}")

    for i, a in enumerate(results[:top_n], 1):
        price = format_price(a["prompt_price"])
        ctx = format_context(a["context_length"])
        free = "Yes" if a["is_free"] else "-"
        rank_d = f"#{a['rank']}" if a["rank"] < 999 else "-"
        print(f"  {i:<3} {a['id'][:46]:<46} {price:>12} {ctx:>7} {free:>4} {rank_d:>5}")
    print()


# ─── Output: Summary ─────────────────────────────────────────────────────────

def print_summary(analyses):
    """Ecosystem summary."""
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
    print()


# ─── Output: Export ──────────────────────────────────────────────────────────

def export_csv(all_rankings, analyses):
    """Export rankings as CSV to stdout."""
    writer = csv.writer(sys.stdout)
    writer.writerow(["Category", "Rank", "Model", "Score", "Price/1M", "Context", "Free", "Provider"])
    for cat_key, scored in all_rankings.items():
        for i, s in enumerate(scored, 1):
            a = s["analysis"]
            writer.writerow([
                cat_key, i, a["id"], s["score"],
                format_price(a["prompt_price"]),
                a["context_length"],
                a["is_free"],
                a["provider"],
            ])


def export_markdown(all_rankings):
    """Export rankings as Markdown to stdout."""
    for cat_key, scored in all_rankings.items():
        if not scored:
            continue
        cat = CATEGORIES[cat_key]
        print(f"\n## {cat['label']}\n")
        print(f"*{cat['description']}*\n")
        print(f"| # | Model | Score | Price | Context | Free |")
        print(f"|---|---|---|---|---|---|")
        for i, s in enumerate(scored, 1):
            a = s["analysis"]
            price = format_price(a["prompt_price"])
            ctx = format_context(a["context_length"])
            free = "Yes" if a["is_free"] else "No"
            print(f"| {i} | `{a['id']}` | {s['score']:.1f} | {price} | {ctx} | {free} |")
    print()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="llm-radar - Data-driven LLM ranking for OpenRouter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Category flags
    parser.add_argument("--coding", action="store_true", help="Rank for coding")
    parser.add_argument("--reasoning", action="store_true", help="Rank for reasoning")
    parser.add_argument("--budget", action="store_true", help="Rank best value paid")
    parser.add_argument("--free", action="store_true", help="Rank free models")
    parser.add_argument("--agents", action="store_true", help="Rank for tool use/agents")
    parser.add_argument("--multimodal", action="store_true", help="Rank multimodal")
    parser.add_argument("--context", action="store_true", help="Rank by context length")

    # View flags
    parser.add_argument("--compare", nargs=2, metavar=("M1", "M2"), help="Compare two models")
    parser.add_argument("--show", metavar="MODEL", help="Show detailed model info")
    parser.add_argument("--search", metavar="QUERY", help="Search models")
    parser.add_argument("--optimize-claude", action="store_true", help="Optimize for Claude Code")
    parser.add_argument("--summary", action="store_true", help="Show ecosystem summary")
    parser.add_argument("--all", action="store_true", help="Show all categories (default)")
    parser.add_argument("--top", type=int, default=10, help="Top N per category (default: 10)")

    # Cost calculator
    parser.add_argument("--cost", nargs=2, type=int, metavar=("INPUT", "OUTPUT"),
                        help="Compare costs for N input + N output tokens")

    # Filters
    parser.add_argument("--provider", metavar="NAME", help="Filter by provider (e.g. anthropic)")
    parser.add_argument("--max-price", type=float, metavar="$/1M", help="Max prompt price per 1M tokens")
    parser.add_argument("--min-context", type=int, metavar="TOKENS", help="Min context length")
    parser.add_argument("--has-tools", action="store_true", help="Only models with tool support")
    parser.add_argument("--has-reasoning", action="store_true", help="Only models with reasoning")
    parser.add_argument("--free-only", action="store_true", help="Only free models")
    parser.add_argument("--paid-only", action="store_true", help="Only paid models")
    parser.add_argument("--multimodal-only", action="store_true", help="Only multimodal models")
    parser.add_argument("--active", action="store_true", help="Only active models (last 6 months)")
    parser.add_argument("--include-expired", action="store_true", help="Include expired models")

    # Export
    parser.add_argument("--json", action="store_true", dest="json_output", help="Export as JSON")
    parser.add_argument("--csv", action="store_true", dest="csv_output", help="Export as CSV")
    parser.add_argument("--markdown", action="store_true", dest="md_output", help="Export as Markdown")

    # Rankings
    parser.add_argument("--update-rankings", action="store_true", help="Fetch live rankings from OpenRouter")

    args = parser.parse_args()

    # Determine mode
    category_flags = [args.coding, args.reasoning, args.budget, args.free,
                      args.agents, args.multimodal, args.context]
    view_flags = [args.compare, args.show, args.search, args.optimize_claude,
                  args.summary, args.cost, args.update_rankings]
    filter_flags = [args.provider, args.max_price, args.min_context,
                    args.has_tools, args.has_reasoning, args.free_only,
                    args.paid_only, args.multimodal_only, args.active]

    specific = any(category_flags + view_flags + filter_flags)
    # Default to all categories if no specific category/view selected
    if not any(category_flags + view_flags):
        args.all = True

    print_header()

    # Rankings
    if args.update_rankings:
        rankings = fetch_live_rankings()
        if not args.all and not any([args.coding, args.reasoning, args.budget, args.free,
                                      args.agents, args.multimodal, args.context,
                                      args.compare, args.show, args.search,
                                      args.optimize_claude, args.summary, args.cost]):
            return
    else:
        rankings = get_rankings(live=False)

    # Fetch and filter models
    models_raw = fetch_models()
    models_raw = apply_filters(models_raw, args)
    print(f"   Using {len(models_raw)} models for analysis\n")

    analyses = [analyze_model(m, rankings) for m in models_raw]
    analyses = apply_post_filters(analyses, args)

    if not analyses:
        print("  No models match the given filters")
        return

    # ── Mode dispatch ──

    if args.search:
        print_search(analyses, args.search, args.top)
        return

    if args.compare:
        print_comparison(models_raw, args.compare, rankings)
        return

    if args.show:
        print_show_model(models_raw, args.show, rankings)
        return

    if args.cost:
        input_t, output_t = args.cost
        print_cost_table(analyses, input_t, output_t, args.top)
        return

    if args.summary or args.all:
        print_summary(analyses)

    if args.optimize_claude:
        print_optimize_claude(analyses, rankings)
        return

    # Category ranking
    cat_keys = []
    if args.all:
        cat_keys = list(CATEGORIES.keys())
    else:
        if args.coding: cat_keys.append("coding")
        if args.reasoning: cat_keys.append("reasoning")
        if args.budget: cat_keys.append("budget")
        if args.free: cat_keys.append("free")
        if args.agents: cat_keys.append("agents")
        if args.multimodal: cat_keys.append("multimodal")
        if args.context: cat_keys.append("context")

    all_rankings = {}
    for cat_key in cat_keys:
        cat = CATEGORIES[cat_key]
        scored = [{"analysis": a, "score": score_model(a, cat)} for a in analyses]
        scored.sort(key=lambda x: x["score"], reverse=True)
        all_rankings[cat_key] = scored[:args.top]

    # Export or display
    if args.csv_output:
        export_csv(all_rankings, analyses)
    elif args.md_output:
        export_markdown(all_rankings)
    elif args.json_output:
        output = {}
        for cat_key, data in all_rankings.items():
            output[cat_key] = [
                {"rank": i+1, "model": s["analysis"]["id"], "score": s["score"],
                 "prompt_price": s["analysis"]["prompt_price"],
                 "context_length": s["analysis"]["context_length"],
                 "is_free": s["analysis"]["is_free"],
                 "supported_params": sorted(s["analysis"]["params"])}
                for i, s in enumerate(data)
            ]
        print(json.dumps(output, indent=2))
    else:
        for cat_key in cat_keys:
            print_category(cat_key, CATEGORIES[cat_key], all_rankings[cat_key], args.top)

        if args.all:
            print(f"\n{'=' * 72}")
            print(f"  Top Picks Summary")
            print(f"{'=' * 72}")
            for cat_key in cat_keys:
                top = all_rankings.get(cat_key, [])
                if top:
                    best = top[0]["analysis"]
                    tag = " (FREE)" if best["is_free"] else ""
                    print(f"  {CATEGORIES[cat_key]['label']:<28} -> {best['id']}{tag}")
            print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()
