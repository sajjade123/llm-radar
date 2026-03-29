# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands
- `python model-optimizer.py` - Full analysis of all models
- `python model-optimizer.py --coding` - Top coding models
- `python model-optimizer.py --free` - Best free models
- `python model-optimizer.py --compare MODEL1 MODEL2` - Side-by-side comparison
- `python model-optimizer.py --show MODEL` - Detailed model view
- `python model-optimizer.py --cost IN OUT` - Cost calculator
- `python model-optimizer.py --update-rankings` - Fetch live rankings
- `python model-optimizer.py --optimize-claude` - Best models for Claude Code

## Code Architecture
The project is a Python CLI tool that:
1. Parses command-line arguments via `argparse`
2. Fetches live model data from OpenRouter API
3. Implements ranking algorithms based on capability, context, pricing, and popularity
4. Exports results in CSV/Markdown/JSON formats

## Code Structure
- Main file: `model-optimizer.py` (command parser and API integration)
- Requirements: `requests` library
- Data storage: `rankings_cache.json` for cached results

## Security Notice
No API keys or tokens are stored in this repository. All API requests go through OpenRouter's authenticated endpoints.
