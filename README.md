# Project Alpha-Discovery

**Event Study Framework for Energy Markets & News-Driven Strategies**

> Targeting: JERA Global Markets — Graduate Quant Researcher Position

## Overview

This framework quantifies the causal impact of specific news keywords (e.g., "Strike", "Cold Snap", "Nuclear Restart") on energy assets (UNG, TTF, TEPCO) using an **Event Study** methodology.

The architecture draws inspiration from Bluesky's [custom feed generator pattern](https://docs.bsky.app/docs/starter-templates/custom-feeds):
- A "firehose" of news events is filtered by keyword rules
- Each qualifying event triggers a forward-return calculation
- Results are aggregated into statistical summaries and professional heatmaps

## Modules

| Module | Description |
|--------|-------------|
| **Module 1** — Golden Dataset Generator | Synthetic Bloomberg/WSJ-style news + GBM price data with injected bias |
| **Module 2** — Event Processor | `EventAnalyzer` class: keyword filtering, time-snapping, forward-return computation |
| **Module 3** — Statistical Verification | Mean return, win rate, signal strength, t-test (scipy) |
| **Module 4** — Visualization | Seaborn heatmap (keyword × asset) + box/strip distribution plots |

## Quick Start

```bash
pip install -r requirements.txt
python alpha_discovery.py
```

## Output

- `heatmap_alpha_discovery.png` — Keyword × Asset correlation heatmap
- `distribution_alpha_discovery.png` — Forward return distributions
- Console statistical summary table

## Tech Stack

- Python 3.10+
- pandas (vectorized operations)
- numpy (GBM simulation)
- scipy (t-test)
- matplotlib + seaborn (visualization)
