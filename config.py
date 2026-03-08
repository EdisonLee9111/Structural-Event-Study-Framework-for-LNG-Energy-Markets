"""
┌──────────┐   ┌────────────┐   ┌───────────┐   ┌────────┐
│ theory   │──▶│ generators │──▶│ analytics │──▶│ main   │
└──────────┘   └────────────┘   └───────────┘   └────────┘
   ▲ config.py is imported by ALL modules above

config.py — Global constants & market parameters
=================================================
Single source of truth for:
  • Simulation settings (seed, event count, time window)
  • Keyword definitions and directional priors
  • Asset universe (spot, futures, options)
  • Futures curve tenors and option strike grid
  • Market-state thresholds used in state-conditional analysis

Reading order: config → theory → generators → analytics → main
"""

from __future__ import annotations
from typing import Dict, List

# ─────────────────────────────────────────────────────────────────────────────
# 1. Simulation Controls
# ─────────────────────────────────────────────────────────────────────────────

RANDOM_SEED: int = 42
"""Numpy/random seed for fully reproducible synthetic data."""

N_EVENTS: int = 150
"""
Total real news events across all keywords (30 per keyword × 5 keywords).

Rationale: Hotelling's T² requires n > p (sample size > response dimensions).
With 7 response dimensions, 12 events per keyword (n=60) yields a dangerously
thin n/p ratio ≈ 1.7.  At n=150 each keyword gets ~30 events → n/p ≈ 4.3,
providing adequate statistical power while remaining computationally trivial
for synthetic data generation.
"""

N_PLACEBO_EVENTS: int = 150
"""
Number of placebo (non-event) timestamps for Prediction 0 falsification test.
Placebo events must be separated from real events by at least 2 × FORWARD_WINDOW_MINUTES.
"""

FORWARD_WINDOW_MINUTES: int = 60
"""Minutes after a news event used to measure the forward price impact."""

SIM_START_DATE: str = "2025-02-11"
SIM_END_DATE:   str = "2026-02-11"
PRICE_FREQ:     str = "5min"          # intraday bar frequency

# ─────────────────────────────────────────────────────────────────────────────
# 2. Keyword Universe & Directional Priors
# ─────────────────────────────────────────────────────────────────────────────

KEYWORD_BIAS: Dict[str, float] = {
    "Strike":          +0.012,   # supply disruption  → bullish LNG
    "Outage":          +0.008,   # unplanned outage   → bullish LNG
    "Cold Snap":       +0.010,   # demand surge       → bullish LNG
    "Nuclear Restart": -0.010,   # new baseload power → bearish gas demand
    "Tariff":          +0.005,   # trade friction     → mild bullish LNG
}
"""
Directional bias injected into synthetic price data during event windows.
Positive = spot price tends to rise; negative = tends to fall.
Used ONLY in generators.py to make synthetic data statistically non-trivial.
"""

KEYWORDS: List[str] = list(KEYWORD_BIAS.keys())

# ─────────────────────────────────────────────────────────────────────────────
# 3. Asset Universe
# ─────────────────────────────────────────────────────────────────────────────

ASSET_CONFIGS: Dict[str, dict] = {
    "JKM": {
        "base_price": 15.0,        # USD/MMBtu — JKM spot (proxy)
        "volatility": 0.0006,      # per-bar return std dev
        "description": "Japan-Korea Marker — Asian LNG benchmark",
    },
    "TTF": {
        "base_price": 30.0,        # EUR/MWh
        "volatility": 0.0005,
        "description": "Title Transfer Facility — European gas hub",
    },
    "TEPCO": {
        "base_price": 650.0,       # JPY
        "volatility": 0.0004,
        "description": "Tokyo Electric Power — proxy for Japanese power demand",
    },
}
"""
Asset configs used by generators.py.  Each entry is passed as **kwargs to
SpotGenerator, so keys must match its constructor parameters.
"""

# Primary asset used for futures/options surface (single curve framework)
PRIMARY_ASSET: str = "JKM"

# ─────────────────────────────────────────────────────────────────────────────
# 4. Futures Curve
# ─────────────────────────────────────────────────────────────────────────────

FUTURES_TENORS: List[str] = ["1M", "3M", "6M", "12M"]
"""Futures contract tenors whose prices are synthetic-generated around events."""

TENOR_TO_YEARS: Dict[str, float] = {
    "1M":  1 / 12,
    "3M":  3 / 12,
    "6M":  6 / 12,
    "12M": 12 / 12,
}
"""Convert tenor label to time-to-expiry in years (used in F = S·exp(…·T))."""

# Risk-free rate (simplified, annualised)
RISK_FREE_RATE: float = 0.05

# Base cost-of-carry (annualised storage + financing, excluding convenience yield)
STORAGE_COST_BASE: float = 0.03      # 3% p.a. base rate

# Non-linear storage cost: kicks in above STORAGE_THRESHOLD utilisation
STORAGE_THRESHOLD: float = 0.90      # 90% tank utilisation
STORAGE_COST_SCALE: float = 5.0      # multiplier on incremental utilisation above threshold
"""
Non-linear storage cost parameters (stylised).
At utilisation u:
    c(u) = STORAGE_COST_BASE × (1 + STORAGE_COST_SCALE × max(0, u - STORAGE_THRESHOLD))

Docstring caveat: real LNG terminal storage costs include slot fees, boil-off,
and demurrage that are not constant. Calibration against GIIGNL annual reports
would be required for production use.
"""

# ─────────────────────────────────────────────────────────────────────────────
# 5. Options Surface
# ─────────────────────────────────────────────────────────────────────────────

# Strike grid: 85 % – 115 % of ATM price in 5-percentage-point steps
OPTION_STRIKE_MONEYNESS: List[float] = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]
"""
Relative strike levels (K / F_ATM).  Combined with per-tenor ATM prices,
these produce the absolute strike grid passed to Black-76 pricing.
"""

# Base ATM implied volatility (annualised)
BASE_ATM_VOL: float = 0.30           # 30% p.a.

# Smile curvature (quadratic coefficient on moneyness deviation)
SMILE_CURVATURE: float = 0.10
"""
Stylised vol smile: IV(k) = BASE_ATM_VOL + SMILE_CURVATURE × (k - 1)²
where k = K / F_ATM.
"""

# Risk-reversal convention: delta used for 25Δ risk reversal / butterfly
RR_DELTA: float = 0.25

# ─────────────────────────────────────────────────────────────────────────────
# 6. Market State Thresholds
# ─────────────────────────────────────────────────────────────────────────────

# Storage mean-reversion parameters (used in MarketStateGenerator)
STORAGE_MEAN: float = 0.70           # long-run mean utilisation
STORAGE_REVERSION_SPEED: float = 0.05  # per day mean-reversion coefficient

# Market tightness classification (based on z-score of the tightness composite)
TIGHTNESS_TIGHT_THRESHOLD: float   = +0.50   # z > +0.5  → "tight"
TIGHTNESS_LOOSE_THRESHOLD: float   = -0.50   # z < -0.5  → "loose"
# |z| ≤ 0.5 → "neutral"

# Spark spread: proxy for whether gas is the marginal fuel
# spark_spread = electricity_price − gas_price × HEAT_RATE
HEAT_RATE: float = 7.0               # MMBtu/MWh (CCGT efficiency)
ELECTRICITY_BASE_PRICE: float = 120.0 # JPY/kWh proxy (stylised)
# Raised from 90 → 120 so that spark_spread = elec − JKM×HEAT_RATE ≈ 120-105 = +15
# near the long-run mean, giving a balanced gas_is_marginal True/False split
# required for Prediction 3 test (Issue 7 fix).

# Electricity price volatility (for spark spread simulation)
ELECTRICITY_VOLATILITY: float = 0.0008  # per-bar

# ─────────────────────────────────────────────────────────────────────────────
# 7. Nuclear Restart — Structural Attributes
# ─────────────────────────────────────────────────────────────────────────────

NUCLEAR_CAPACITY_MIN_MW: float = 500.0
NUCLEAR_CAPACITY_MAX_MW: float = 1200.0
"""MW range for simulated reactor capacities (Prediction 2: capacity proportionality)."""

CREDIBILITY_TIERS: Dict[int, str] = {
    1: "NRA_formal_approval",    # highest credibility
    2: "company_announcement",   # medium
    3: "policy_signal",          # lowest
}
"""
Credibility tier for nuclear restart news (transmission channel friction).
Tier 1 events should have larger and faster price impact than Tier 3.
"""
