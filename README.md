# llm-radar

Data-driven LLM ranking tool for [OpenRouter](https://openrouter.ai). Analyze, compare, and rank 340+ models using real API data. No subjective scoring.

## Features

- **7 category rankings** — coding, reasoning, budget, free, agents, multimodal, context
- **Live rankings** — fetches usage data directly from OpenRouter rankings page
- **Cost calculator** — compare costs across all models for any token count
- **Model comparison** — side-by-side with full parameter breakdown
- **Detailed model view** — all capabilities, pricing, ranking, scores
- **Filters** — by provider, price, context length, capabilities
- **Export** — CSV, Markdown, JSON
- **Claude Code optimizer** — best models for each role with export commands

## Install

```bash
pip install requests
```

## Usage

```bash
# Full analysis
python model-optimizer.py

# Category rankings
python model-optimizer.py --coding
python model-optimizer.py --free
python model-optimizer.py --budget

# Compare two models
python model-optimizer.py --compare claude-opus-4.6 gemini-3-flash-preview

# Detailed model view
python model-optimizer.py --show gemini-3-flash-preview

# Search
python model-optimizer.py --search claude

# Cost calculator (10K input + 5K output tokens)
python model-optimizer.py --cost 10000 5000

# Filters (combine any)
python model-optimizer.py --provider anthropic
python model-optimizer.py --max-price 1.00 --has-tools
python model-optimizer.py --min-context 128000 --free-only

# Export
python model-optimizer.py --csv --coding --top 20
python model-optimizer.py --markdown --free --top 10
python model-optimizer.py --json --coding --top 5

# Update live rankings
python model-optimizer.py --update-rankings

# Claude Code optimizer
python model-optimizer.py --optimize-claude
```

## Scoring

Every score is based on verifiable API data. No keyword matching.

| Dimension | Source | Weight varies by category |
|---|---|---|
| Capability | `supported_parameters` API field | 25-55% |
| Context | `context_length` API field (log scale) | 10-70% |
| Pricing | `pricing.prompt` API field (log scale) | 0-35% |
| Popularity | OpenRouter usage rankings (capped) | 10-20% |
| Speed | Heuristic from context + model name | 5-15% |

## All Flags

| Flag | Description |
|---|---|
| `--coding` | Rank for coding |
| `--reasoning` | Rank for reasoning |
| `--budget` | Best value paid models |
| `--free` | Best free models |
| `--agents` | Best for tool use/agents |
| `--multimodal` | Best multimodal |
| `--context` | Largest context windows |
| `--compare M1 M2` | Compare two models |
| `--show MODEL` | Detailed single model view |
| `--search QUERY` | Search models |
| `--cost IN OUT` | Cost calculator |
| `--provider NAME` | Filter by provider |
| `--max-price $/1M` | Max prompt price |
| `--min-context N` | Min context length |
| `--has-tools` | Only with tool support |
| `--has-reasoning` | Only with reasoning |
| `--free-only` | Only free models |
| `--paid-only` | Only paid models |
| `--multimodal-only` | Only multimodal |
| `--active` | Only active (last 6 months) |
| `--include-expired` | Include expired models |
| `--top N` | Top N per category |
| `--json` | Export as JSON |
| `--csv` | Export as CSV |
| `--markdown` | Export as Markdown |
| `--update-rankings` | Fetch live rankings |
| `--optimize-claude` | Best for Claude Code |

## Data Sources

- **Models**: `openrouter.ai/api/v1/models` (346 models, live)
- **Rankings**: `openrouter.ai/rankings` (live scraped, 100 models, cached)

## Requirements

- Python 3.8+
- `requests`

## License

MIT
