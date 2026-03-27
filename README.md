# OpenRouter Model Optimizer

Data-driven LLM ranking tool that analyzes and compares all models on [OpenRouter](https://openrouter.ai) using real API data. No subjective scoring — every score is based on verifiable model capabilities.

## How It Works

Scoring is **transparent and unbiased**. Each model is evaluated on 5 measurable dimensions:

| Dimension | Source | What It Measures |
|---|---|---|
| **Capability** | `supported_parameters` API field | Tool use, reasoning, structured output, etc. |
| **Context** | `context_length` API field | Context window size (log scale) |
| **Pricing** | `pricing.prompt` API field | Cost efficiency (log scale, inverted) |
| **Popularity** | openrouter.ai/rankings | Usage-based ranking (capped at 15-20% weight) |
| **Speed** | Heuristic from context size + model name | Estimated inference speed |

Each category defines **weights** for these dimensions (shown in output) and **capability caps** (which API params matter). No keyword matching on descriptions. No model-name-based size estimation.

## Install

```bash
pip install requests
```

## Usage

```bash
# Full analysis across all categories
python model-optimizer.py

# Specific category
python model-optimizer.py --coding
python model-optimizer.py --reasoning
python model-optimizer.py --free
python model-optimizer.py --budget
python model-optimizer.py --agents
python model-optimizer.py --multimodal
python model-optimizer.py --context

# Compare two models side-by-side
python model-optimizer.py --compare claude-opus-4.6 gpt-5-codex

# Detailed single model view
python model-optimizer.py --show claude-opus-4.6

# Search models
python model-optimizer.py --search gemini

# Best models for Claude Code roles
python model-optimizer.py --optimize-claude

# Filter active models only (last 6 months + top ranked)
python model-optimizer.py --active

# Include expired models
python model-optimizer.py --include-expired

# JSON output
python model-optimizer.py --json --coding --top 5

# Limit results
python model-optimizer.py --coding --top 20
```

## Output Fields

| Field | Description |
|---|---|
| `Score` | Weighted score (0-100) based on category weights |
| `Price` | Prompt price per 1M tokens |
| `Ctx` | Context window size |
| `Params` | Supported API parameters / total relevant for category |
| `Expires` | Model expiration date (if set by provider) |
| `Pop` | OpenRouter popularity ranking (#N or -) |

## Categories

| Category | Capability | Context | Pricing | Popularity | Speed |
|---|---|---|---|---|---|
| Coding | 50% | 15% | 15% | 10% | 10% |
| Reasoning | 55% | 20% | 10% | 10% | 5% |
| Budget | 25% | 10% | 35% | 20% | 10% |
| Free | 45% | 25% | 0% | 15% | 15% |
| Agents | 55% | 15% | 10% | 10% | 10% |
| Multimodal | 50% | 15% | 10% | 15% | 10% |
| Context | 0% | 70% | 10% | 10% | 10% |

## Data Sources

- **Models API**: `openrouter.ai/api/v1/models` (346 models, live)
- **Rankings**: `openrouter.ai/rankings` (top 10 by weekly usage, hardcoded and updated manually)

## Requirements

- Python 3.8+
- `requests` library

## License

MIT
