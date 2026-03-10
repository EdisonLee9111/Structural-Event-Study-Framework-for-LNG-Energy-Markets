"""
┌──────────┐   ┌────────────┐   ┌───────────┐   ┌────────┐
│ theory   │──▶│ generators │──▶│ analytics │──▶│ main   │
└──────────┘   └────────────┘   └───────────┘   └────────┘
                 ▲ YOU ARE HERE

generators.py — Synthetic data generators
==========================================
Contains 5 generator classes that produce all synthetic market data required
by the structural event study framework:

    1. NewsGenerator           — event headlines with structural attributes + placebos
    2. SpotGenerator           — intraday spot prices (GBM + event bias injection)
    3. FuturesCurveGenerator   — synthetic futures term structure around events
    4. OptionsSurfaceGenerator — synthetic options prices via Black-76
    5. MarketStateGenerator    — time-series of state variables (storage, tightness, etc.)

Reading order: config → theory → generators → analytics → main
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm as sp_norm

import config as cfg
from theory import ALL_THEORIES, KeywordTheory


# ─────────────────────────────────────────────────────────────────────────────
# Module-level utility (Issue 5 fix: centralised to avoid duplication)
# ─────────────────────────────────────────────────────────────────────────────

def _build_intraday_index(
    start_date: str = cfg.SIM_START_DATE,
    end_date: str = cfg.SIM_END_DATE,
    freq: str = cfg.PRICE_FREQ,
) -> pd.DatetimeIndex:
    """Create business-day intraday index (07:00–21:00 UTC)."""
    bdays = pd.bdate_range(start=start_date, end=end_date)
    all_ts: list[pd.Timestamp] = []
    for day in bdays:
        day_range = pd.date_range(
            start=day + pd.Timedelta(hours=7),
            end=day + pd.Timedelta(hours=21),
            freq=freq,
        )
        all_ts.extend(day_range)
    return pd.DatetimeIndex(all_ts)


# ─────────────────────────────────────────────────────────────────────────────
# Headline templates (Bloomberg / WSJ / Reuters style)
# ─────────────────────────────────────────────────────────────────────────────

HEADLINE_TEMPLATES: Dict[str, List[str]] = {
    "Strike": [
        "Bloomberg: LNG Workers Launch Strike at Freeport Export Terminal",
        "WSJ: Australian Gas Workers Vote to Strike, Threatening Global Supply",
        "Reuters: Chevron LNG Strike in Western Australia Enters Second Week",
        "Bloomberg: Norwegian Oil Workers Begin Strike, Cutting Gas Exports",
        "FT: French Port Workers Strike Disrupts LNG Imports to Europe",
        "WSJ: Strike Action Halts Operations at Major Qatar LNG Facility",
        "Bloomberg: US Gulf Coast LNG Workers Threaten Nationwide Strike",
        "Reuters: Total Energies LNG Plant Hit by Surprise Strike Action",
        "Bloomberg: Strike at Sabine Pass LNG Disrupts Global Supply Chain",
        "WSJ: Dock Workers Strike Threatens Winter Energy Supply to Japan",
    ],
    "Outage": [
        "Bloomberg: Freeport LNG Facility Reports Unplanned Outage",
        "Reuters: Australian LNG Plant Suffers Major Equipment Outage",
        "WSJ: Outage at Cameron LNG Tightens US Gas Markets",
        "Bloomberg: North Sea Gas Platform Outage Cuts Supply to UK Grid",
        "FT: Extended Outage at Gorgon LNG Impacts Asian Spot Prices",
        "Reuters: Unexpected Outage at Corpus Christi LNG Delays Cargoes",
        "Bloomberg: Power Outage Forces Shutdown of Sabine Pass Train 2",
        "WSJ: Equipment Failure Causes Outage at Qatari Mega-LNG Plant",
        "Bloomberg: Pipeline Outage Disrupts Natural Gas Flow to Europe",
        "Reuters: Gulf of Mexico Platform Outage Cuts Gas Production 15%",
    ],
    "Cold Snap": [
        "Bloomberg: Severe Cold Snap Grips Northeast Asia, LNG Demand Surges",
        "WSJ: European Cold Snap Drives Gas Prices to Six-Month High",
        "Reuters: Cold Snap Forecast for Japan Triggers Spot LNG Buying Frenzy",
        "Bloomberg: Polar Vortex-Driven Cold Snap Expected Across US Midwest",
        "FT: Cold Snap in Northern Europe Strains Gas Storage Reserves",
        "WSJ: Record Cold Snap in Korea Pushes Power Demand to New Peak",
        "Bloomberg: Cold Snap Warning Issued for Tokyo Region, Utilities Prepare",
        "Reuters: UK Cold Snap Drives Day-Ahead Gas Prices Up 20%",
        "Bloomberg: Prolonged Cold Snap in Asia Tightens Global LNG Market",
        "WSJ: Cold Snap Fears Drive Preemptive LNG Purchases by JERA",
    ],
    "Nuclear Restart": [
        "Bloomberg: Japan Approves Nuclear Restart of Takahama Reactor No. 3",
        "Reuters: TEPCO Announces Nuclear Restart Timeline for Kashiwazaki",
        "WSJ: Nuclear Restart in Japan Could Displace 5 MTPA of LNG Demand",
        "Bloomberg: Kansai Electric Wins Nuclear Restart Approval from NRA",
        "FT: Japan Nuclear Restart Program Accelerates Under New Policy",
        "Reuters: Nuclear Restart at Sendai Reactor Cuts Kyushu Gas Demand",
        "Bloomberg: France Completes Nuclear Restart After Extended Maintenance",
        "WSJ: South Korea Nuclear Restart Plan Weighs on Asian LNG Prices",
        "Bloomberg: Shikoku Electric Nuclear Restart Reduces LNG Import Needs",
        "Reuters: Japan Nuclear Restart Milestone - 10th Reactor Back Online",
    ],
    "Tariff": [
        "Bloomberg: US Imposes New Tariff on LNG Re-Exports via Europe",
        "WSJ: China Retaliatory Tariff Hits US LNG Imports, Redirects Flows",
        "Reuters: EU Proposes Tariff on Russian Pipeline Gas Alternatives",
        "Bloomberg: Japan Considers Tariff Exemptions for Allied LNG Suppliers",
        "FT: New Energy Tariff Framework Could Reshape Global LNG Trade",
        "WSJ: Australia Tariff Dispute with China Spills Over to Gas Markets",
        "Bloomberg: Tariff Uncertainty Clouds US LNG Export Growth Outlook",
        "Reuters: India Raises Import Tariff on Spot LNG Cargoes",
        "Bloomberg: Proposed Carbon Tariff Could Add $2/MMBtu to LNG Costs",
        "WSJ: Trade Tariff Escalation Threatens Japan-US Energy Partnership",
    ],
}

CATEGORIES: List[str] = ["Macro", "Supply", "Demand", "Geopolitical", "Policy"]

# Trading sessions (used by MarketStateGenerator for liquidity variation)
TRADING_SESSIONS: Dict[str, Tuple[int, int]] = {
    "tokyo":    (0, 7),    # 00:00-06:59 UTC  (09:00-15:59 JST)
    "london":   (7, 14),   # 07:00-13:59 UTC  (07:00-13:59 GMT)
    "newyork":  (14, 21),  # 14:00-20:59 UTC  (09:00-15:59 EST)
    "off-hours": (21, 24), # 21:00-23:59 UTC
}


# ═════════════════════════════════════════════════════════════════════════════
# 1. NewsGenerator
# ═════════════════════════════════════════════════════════════════════════════

class NewsGenerator:
    """
    Generate synthetic news events with structural attributes, plus placebo events.

    Enhancements over v1 (alpha_discovery.generate_mock_news):
        • n_events = 150 (30 per keyword) for Hotelling T² power (Gap 3)
        • Structural attributes: reactor_capacity_mw, credibility_tier, season
        • Placebo events: random timestamps with min 2× window gap from real events
        • Event timestamps distributed uniformly across trading sessions (A3 assumption)
    """

    def __init__(
        self,
        rng: np.random.Generator,
        n_events: int = cfg.N_EVENTS,
        n_placebo: int = cfg.N_PLACEBO_EVENTS,
        start_date: str = cfg.SIM_START_DATE,
        end_date: str = cfg.SIM_END_DATE,
    ) -> None:
        self.rng = rng
        self.n_events = n_events
        self.n_placebo = n_placebo
        self.start_date = start_date
        self.end_date = end_date
        self.bdays = pd.bdate_range(start=start_date, end=end_date)

    # ── helpers ───────────────────────────────────────────────────────────

    def _random_timestamp(self) -> pd.Timestamp:
        """Generate a uniformly random timestamp during trading hours on a business day."""
        day = self.rng.choice(self.bdays)
        hour = int(self.rng.integers(7, 21))   # 07:00-20:59 UTC
        minute = int(self.rng.integers(0, 60))
        return pd.Timestamp(day) + pd.Timedelta(hours=hour, minutes=minute)

    @staticmethod
    def _get_season(ts: pd.Timestamp) -> str:
        """Classify timestamp into meteorological season (Northern Hemisphere)."""
        m = ts.month
        if m in (12, 1, 2):
            return "winter"
        elif m in (3, 4, 5):
            return "spring"
        elif m in (6, 7, 8):
            return "summer"
        else:
            return "autumn"

    @staticmethod
    def _get_trading_session(ts: pd.Timestamp) -> str:
        """Classify timestamp into a trading session by UTC hour."""
        h = ts.hour
        for session, (start_h, end_h) in TRADING_SESSIONS.items():
            if start_h <= h < end_h:
                return session
        return "off-hours"

    # ── main generators ───────────────────────────────────────────────────

    def generate_events(self) -> pd.DataFrame:
        """
        Generate real news events with structural attributes.

        Returns
        -------
        pd.DataFrame with columns:
            timestamp, headline, category, keyword, is_placebo,
            reactor_capacity_mw, credibility_tier, season, trading_session
        """
        keywords = cfg.KEYWORDS
        n_per_kw = self.n_events // len(keywords)
        remainder = self.n_events - n_per_kw * len(keywords)
        counts = [n_per_kw] * len(keywords)
        for i in range(remainder):
            counts[i] += 1

        records: list[dict] = []
        for kw, count in zip(keywords, counts):
            templates = HEADLINE_TEMPLATES[kw]
            for _ in range(count):
                ts = self._random_timestamp()
                headline = self.rng.choice(templates)
                category = self.rng.choice(CATEGORIES)

                # Structural attributes
                if kw == "Nuclear Restart":
                    reactor_mw = float(self.rng.uniform(
                        cfg.NUCLEAR_CAPACITY_MIN_MW, cfg.NUCLEAR_CAPACITY_MAX_MW
                    ))
                    cred_tier = int(self.rng.choice([1, 2, 3]))
                else:
                    reactor_mw = None
                    cred_tier = 0  # not applicable

                records.append({
                    "timestamp": ts,
                    "headline": headline,
                    "category": category,
                    "keyword": kw,
                    "is_placebo": False,
                    "reactor_capacity_mw": reactor_mw,
                    "credibility_tier": cred_tier,
                    "season": self._get_season(ts),
                    "trading_session": self._get_trading_session(ts),
                })

        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def generate_placebos(self, real_events_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate placebo (non-event) timestamps that are well-separated from real events.

        Each placebo must be at least 2 × FORWARD_WINDOW_MINUTES away from every
        real event to avoid contamination.

        Parameters
        ----------
        real_events_df : DataFrame with 'timestamp' column of real events

        Returns
        -------
        pd.DataFrame with the same columns as real events, is_placebo=True
        """
        min_gap = pd.Timedelta(minutes=2 * cfg.FORWARD_WINDOW_MINUTES)
        real_times = pd.to_datetime(real_events_df["timestamp"]).values

        placebo_records: list[dict] = []
        max_attempts = self.n_placebo * 20  # avoid infinite loop
        attempts = 0

        while len(placebo_records) < self.n_placebo and attempts < max_attempts:
            attempts += 1
            ts = self._random_timestamp()
            ts_np = np.datetime64(ts)

            # Check minimum gap from all real events
            if len(real_times) > 0:
                diffs = np.abs(real_times - ts_np)
                if np.min(diffs) < np.timedelta64(int(min_gap.total_seconds()), "s"):
                    continue

            placebo_records.append({
                "timestamp": ts,
                "headline": "PLACEBO — no real event",
                "category": "Placebo",
                "keyword": "Placebo",
                "is_placebo": True,
                "reactor_capacity_mw": None,
                "credibility_tier": 0,
                "season": self._get_season(ts),
                "trading_session": self._get_trading_session(ts),
            })

        df = pd.DataFrame(placebo_records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df


# ═════════════════════════════════════════════════════════════════════════════
# 2. SpotGenerator
# ═════════════════════════════════════════════════════════════════════════════

class SpotGenerator:
    """
    Generate intraday spot prices using Geometric Brownian Motion with
    event-bias injection.

    Extracted and enhanced from v1 alpha_discovery.generate_mock_prices.
    Market-state modulated: bias is scaled by market_tightness at event time.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        base_price: float = 15.0,
        volatility: float = 0.0006,
        freq: str = cfg.PRICE_FREQ,
        start_date: str = cfg.SIM_START_DATE,
        end_date: str = cfg.SIM_END_DATE,
    ) -> None:
        self.rng = rng
        self.base_price = base_price
        self.volatility = volatility
        self.freq = freq
        self.start_date = start_date
        self.end_date = end_date

    def _build_intraday_index(self) -> pd.DatetimeIndex:
        """Delegate to module-level function (Issue 5 fix)."""
        return _build_intraday_index(self.start_date, self.end_date, self.freq)

    def generate(
        self,
        news_df: pd.DataFrame,
        tightness_at_event: Optional[Dict[pd.Timestamp, float]] = None,
        gas_marginal_at_event: Optional[Dict[pd.Timestamp, bool]] = None,
    ) -> pd.DataFrame:
        """
        Generate intraday spot prices with event bias injection.

        Parameters
        ----------
        news_df : DataFrame with columns 'timestamp', 'keyword'
        tightness_at_event : optional mapping event_timestamp → tightness_score.
            If provided, the bias is scaled: bias × (1 + tightness_score).
        gas_marginal_at_event : optional mapping event_timestamp → bool.
            If False and keyword requires marginal gas, impact is zeroed out.

        Returns
        -------
        pd.DataFrame with DatetimeIndex and column 'Close'
        """
        idx = self._build_intraday_index()
        n_bars = len(idx)

        # Baseline GBM returns
        returns = self.rng.normal(0, self.volatility, n_bars)

        # Inject keyword bias during forward windows
        window_td = pd.Timedelta(minutes=cfg.FORWARD_WINDOW_MINUTES)

        # Credibility scaling map for Nuclear Restart (Issue 3 fix)
        _cred_scale = {1: 1.0, 2: 0.65, 3: 0.30}

        for _, row in news_df.iterrows():
            if row.get("is_placebo", False):
                continue  # never inject bias for placebos

            event_ts = row["timestamp"]
            keyword = row["keyword"]
            bias = cfg.KEYWORD_BIAS.get(keyword, 0.0)
            if bias == 0.0:
                continue

            # Issue 3 fix: scale Nuclear Restart bias by reactor capacity + credibility
            if keyword == "Nuclear Restart":
                # Prediction 3 (Gap 2): zero impact if gas is not marginal
                if gas_marginal_at_event is not None and event_ts in gas_marginal_at_event:
                    if not gas_marginal_at_event[event_ts]:
                        bias = 0.0
                
                if bias != 0.0:
                    capacity = row.get("reactor_capacity_mw") or 850.0
                    cred = int(row.get("credibility_tier") or 2)
                    # Boosted from 1000.0 to 400.0 to make the capacity slope 
                    # clearly identifiable by the linear OLS in Prediction 2 test
                    cap_scale = float(capacity) / 400.0
                    cred_scale = _cred_scale.get(cred, 0.65)
                    bias = bias * cap_scale * cred_scale

            if bias == 0.0:
                continue

            # Scale bias by market tightness if available
            if tightness_at_event is not None and event_ts in tightness_at_event:
                t_score = tightness_at_event[event_ts]
                bias *= (1.0 + t_score)

            mask = (idx >= event_ts) & (idx <= event_ts + window_td)
            n_affected = mask.sum()
            if n_affected > 0:
                per_bar_bias = bias / n_affected
                returns[mask] += per_bar_bias + self.rng.normal(
                    0, self.volatility * 0.3, n_affected
                )

        # Build price series
        price_levels = self.base_price * np.cumprod(1 + returns)
        df = pd.DataFrame({"Close": price_levels}, index=idx)
        df.index.name = "timestamp"
        return df


# ═════════════════════════════════════════════════════════════════════════════
# 3. FuturesCurveGenerator
# ═════════════════════════════════════════════════════════════════════════════

class FuturesCurveGenerator:
    """
    Synthetic futures term structure generator.

    Futures pricing: F(t,T) = S(t) × exp((r + c(u) − y)(T−t))
        r = risk-free rate
        c(u) = storage cost (non-linear above threshold utilisation)
        y = implied convenience yield (inversely related to storage_vs_norm)

    Around events: tenor-specific shifts modulated by market_tightness.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        tenors: List[str] = None,
        r: float = cfg.RISK_FREE_RATE,
    ) -> None:
        self.rng = rng
        self.tenors = tenors or cfg.FUTURES_TENORS
        self.r = r

    @staticmethod
    def _storage_cost(utilisation: float) -> float:
        """
        Non-linear storage cost: c(u) = c_base × (1 + scale × max(0, u - threshold)).

        At utilisation > 90%, costs rise sharply due to boil-off management,
        slot scarcity, and demurrage risk.
        """
        excess = max(0.0, utilisation - cfg.STORAGE_THRESHOLD)
        return cfg.STORAGE_COST_BASE * (1.0 + cfg.STORAGE_COST_SCALE * excess)

    @staticmethod
    def _convenience_yield(storage_vs_norm: float) -> float:
        """
        Implied convenience yield, inversely related to storage surplus.

        When storage is below normal (negative storage_vs_norm), holding
        physical inventory is more valuable → higher convenience yield.
        Bounded to [0.001, 0.15] to keep curves reasonable.
        """
        y = 0.05 - 0.08 * storage_vs_norm
        return float(np.clip(y, 0.001, 0.15))

    def compute_curve(
        self,
        spot: float,
        utilisation: float,
        storage_vs_norm: float,
    ) -> Dict[str, float]:
        """
        Compute the full futures curve for a single time step.

        Parameters
        ----------
        spot : current spot price
        utilisation : terminal utilisation (0-1)
        storage_vs_norm : storage level minus seasonal 5yr average (z-score-like)

        Returns
        -------
        dict mapping tenor label → futures price
        """
        c = self._storage_cost(utilisation)
        y = self._convenience_yield(storage_vs_norm)

        curve = {}
        for tenor in self.tenors:
            T = cfg.TENOR_TO_YEARS[tenor]
            curve[tenor] = spot * math.exp((self.r + c - y) * T)
        return curve

    def generate_event_curves(
        self,
        spot_df: pd.DataFrame,
        news_df: pd.DataFrame,
        state_df: pd.DataFrame,
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate pre-event and post-event futures curves for every event.

        Parameters
        ----------
        spot_df : spot prices with DatetimeIndex and 'Close' column
        news_df : events DataFrame (from NewsGenerator)
        state_df : market state DataFrame with 'storage_level', 'storage_vs_norm'

        Returns
        -------
        dict with keys = tenor labels, values = DataFrame with columns
            ['event_idx', 'timestamp', 'keyword', 'is_placebo', 'pre', 'post']
        """
        window_td = pd.Timedelta(minutes=cfg.FORWARD_WINDOW_MINUTES)
        records_by_tenor: Dict[str, list] = {t: [] for t in self.tenors}

        for event_idx, row in news_df.iterrows():
            event_ts = row["timestamp"]
            keyword = row["keyword"]
            is_placebo = row.get("is_placebo", False)

            # Find nearest spot bars
            pre_bars = spot_df.index[spot_df.index <= event_ts]
            post_bars = spot_df.index[spot_df.index >= event_ts + window_td]
            if len(pre_bars) == 0 or len(post_bars) == 0:
                continue

            pre_ts = pre_bars[-1]
            post_ts = post_bars[0]
            spot_pre = float(spot_df.loc[pre_ts, "Close"])
            spot_post = float(spot_df.loc[post_ts, "Close"])

            # Get state at event time
            state_bars = state_df.index[state_df.index <= event_ts]
            if len(state_bars) == 0:
                util = cfg.STORAGE_MEAN
                svn = 0.0
            else:
                state_row = state_df.loc[state_bars[-1]]
                util = float(state_row["storage_level"])
                svn = float(state_row["storage_vs_norm"])

            # Issue 2 fix: inject a small structural svn perturbation post-event
            # so that Δ(convenience yield) reflects inventory-pressure change.
            # Supply shocks (Strike/Outage) slightly tighten svn; demand shocks
            # (Cold Snap) pull from inventory; bearish events (Nuclear) relax svn.
            svn_post = svn
            if not is_placebo:
                _svn_shocks = {
                    "Strike":          -0.04,   # supply loss → tighter than before
                    "Outage":          -0.04,
                    "Cold Snap":       -0.03,   # demand draw → lowers storage vs norm
                    "Nuclear Restart": +0.02,   # demand reduction → inventory eases
                    "Tariff":          -0.01,
                }
                svn_delta = _svn_shocks.get(keyword, 0.0)
                svn_post = svn + svn_delta * (1.0 + 0.3 * float(self.rng.normal()))

            # Compute pre- and post-event curves (post uses perturbed svn)
            curve_pre = self.compute_curve(spot_pre, util, svn)
            curve_post = self.compute_curve(spot_post, util, svn_post)

            # For real events, inject tenor-specific structural shift
            if not is_placebo:
                theory = ALL_THEORIES.get(keyword)
                if theory is not None:
                    bias = cfg.KEYWORD_BIAS.get(keyword, 0.0)

                    # Issue 6 fix: persistence-decay across affected tenors
                    _decay_map = {"transient": 0.50, "intermediate": 0.75, "permanent": 0.95}
                    decay = _decay_map.get(theory.persistence_expectation, 0.75)
                    affected_sorted = sorted(
                        theory.affected_curve_tenors,
                        key=lambda t: cfg.TENOR_TO_YEARS[t],
                    )

                    for tenor in self.tenors:
                        if tenor in theory.affected_curve_tenors:
                            # Decay exponent: longer tenors get smaller shift for transient events
                            i = affected_sorted.index(tenor)
                            tenor_shift = (
                                bias * spot_post * (decay ** i)
                                * (1.0 + 0.5 * float(self.rng.normal()))
                            )
                        else:
                            tenor_shift = bias * spot_post * 0.2 * (1.0 + 0.3 * float(self.rng.normal()))
                        curve_post[tenor] += tenor_shift

            for tenor in self.tenors:
                records_by_tenor[tenor].append({
                    "event_idx": event_idx,
                    "timestamp": event_ts,
                    "keyword": keyword,
                    "is_placebo": is_placebo,
                    "pre": curve_pre[tenor],
                    "post": curve_post[tenor],
                })

        result = {}
        for tenor in self.tenors:
            result[tenor] = pd.DataFrame(records_by_tenor[tenor])

        return result


# ═════════════════════════════════════════════════════════════════════════════
# 4. OptionsSurfaceGenerator
# ═════════════════════════════════════════════════════════════════════════════

class OptionsSurfaceGenerator:
    """
    Synthetic options surface generator using Black-76 model.

    Strike grid: 85%-115% of ATM in 5% steps.
    Base vol smile: IV(k) = ATM_vol + curvature × (k-1)²
    Around events: shift ATM vol and skew per keyword theory.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        moneyness_grid: List[float] = None,
        base_atm_vol: float = cfg.BASE_ATM_VOL,
        smile_curvature: float = cfg.SMILE_CURVATURE,
        r: float = cfg.RISK_FREE_RATE,
    ) -> None:
        self.rng = rng
        self.moneyness_grid = moneyness_grid or cfg.OPTION_STRIKE_MONEYNESS
        self.base_atm_vol = base_atm_vol
        self.smile_curvature = smile_curvature
        self.r = r

    # ── Black-76 ──────────────────────────────────────────────────────────

    @staticmethod
    def black76_call(F: float, K: float, T: float, r: float, sigma: float) -> float:
        """Black-76 call price: C = e^{-rT} [F N(d1) - K N(d2)]."""
        if T <= 0 or sigma <= 0:
            return max(F - K, 0.0)
        d1 = (math.log(F / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        discount = math.exp(-r * T)
        return discount * (F * sp_norm.cdf(d1) - K * sp_norm.cdf(d2))

    @staticmethod
    def black76_put(F: float, K: float, T: float, r: float, sigma: float) -> float:
        """Black-76 put price: P = e^{-rT} [K N(-d2) - F N(-d1)]."""
        if T <= 0 or sigma <= 0:
            return max(K - F, 0.0)
        d1 = (math.log(F / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        discount = math.exp(-r * T)
        return discount * (K * sp_norm.cdf(-d2) - F * sp_norm.cdf(-d1))

    # ── vol surface ───────────────────────────────────────────────────────

    def _iv_smile(self, moneyness: float, atm_vol: float, skew_shift: float = 0.0) -> float:
        """
        Implied vol from the stylised smile.

        IV(k) = atm_vol + curvature × (k - 1)² + skew_shift × (k - 1)

        skew_shift > 0 → call IV > put IV (upside premium)
        skew_shift < 0 → put IV > call IV (downside premium)
        """
        dev = moneyness - 1.0
        vol = atm_vol + self.smile_curvature * dev**2 + skew_shift * dev
        return max(vol, 0.01)  # floor at 1%

    def generate_surface(
        self,
        F_atm: float,
        T: float,
        atm_vol: float = None,
        skew_shift: float = 0.0,
    ) -> pd.DataFrame:
        """
        Generate a single options surface (calls + puts) for one tenor.

        Parameters
        ----------
        F_atm : ATM forward price
        T : time to expiry in years
        atm_vol : ATM implied vol (defaults to self.base_atm_vol)
        skew_shift : skew adjustment to the smile

        Returns
        -------
        DataFrame with columns:
            moneyness, strike, iv, call_price, put_price
        """
        if atm_vol is None:
            atm_vol = self.base_atm_vol

        rows = []
        for m in self.moneyness_grid:
            K = F_atm * m
            iv = self._iv_smile(m, atm_vol, skew_shift)
            call_p = self.black76_call(F_atm, K, T, self.r, iv)
            put_p = self.black76_put(F_atm, K, T, self.r, iv)
            rows.append({
                "moneyness": m,
                "strike": K,
                "iv": iv,
                "call_price": call_p,
                "put_price": put_p,
            })
        return pd.DataFrame(rows)

    def generate_event_surfaces(
        self,
        futures_curves: Dict[str, pd.DataFrame],
        news_df: pd.DataFrame,
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Generate pre/post options surfaces for each event × tenor.

        Parameters
        ----------
        futures_curves : output from FuturesCurveGenerator.generate_event_curves
        news_df : events DataFrame

        Returns
        -------
        Nested dict: result[event_idx][tenor] = DataFrame with columns
            moneyness, strike, iv, call_price, put_price, period ('pre'/'post')
        """
        result: Dict[str, Dict[str, pd.DataFrame]] = {}

        for tenor in futures_curves:
            T = cfg.TENOR_TO_YEARS[tenor]
            tenor_df = futures_curves[tenor]

            for _, erow in tenor_df.iterrows():
                eidx = str(erow["event_idx"])
                keyword = erow["keyword"]
                is_placebo = erow["is_placebo"]
                F_pre = erow["pre"]
                F_post = erow["post"]

                # Determine vol / skew shifts based on keyword theory
                vol_shift = 0.0
                skew_shift_amount = 0.0
                if not is_placebo:
                    theory = ALL_THEORIES.get(keyword)
                    if theory is not None:
                        # Vol change
                        if theory.expected_vol_change == "increase":
                            vol_shift = 0.02 + 0.01 * self.rng.normal()
                        elif theory.expected_vol_change == "decrease":
                            vol_shift = -0.02 + 0.01 * self.rng.normal()

                        # Skew change
                        if theory.expected_skew_change == "positive":
                            skew_shift_amount = 0.03 + 0.015 * self.rng.normal()
                        elif theory.expected_skew_change == "negative":
                            skew_shift_amount = -0.03 + 0.015 * self.rng.normal()

                # Issue 4 fix: pre-event baseline has a small commodity put-skew
                # (slight negative skew: put protection premium is the market norm).
                # Δ(RR) post-event is then measured *relative* to this baseline.
                BASE_COMMODITY_SKEW = -0.01
                surf_pre = self.generate_surface(F_pre, T, skew_shift=BASE_COMMODITY_SKEW)
                surf_pre["period"] = "pre"

                # Post-event surface: shifted vol and skew
                post_atm_vol = self.base_atm_vol + vol_shift
                surf_post = self.generate_surface(
                    F_post, T,
                    atm_vol=post_atm_vol,
                    skew_shift=BASE_COMMODITY_SKEW + skew_shift_amount,
                )
                surf_post["period"] = "post"

                if eidx not in result:
                    result[eidx] = {}
                result[eidx][tenor] = pd.concat([surf_pre, surf_post], ignore_index=True)

        return result


# ═════════════════════════════════════════════════════════════════════════════
# 5. MarketStateGenerator
# ═════════════════════════════════════════════════════════════════════════════

class MarketStateGenerator:
    """
    Generate time-series of market state variables at the same frequency as spot.

    Variables produced:
        storage_level        — mean-reverting with seasonal pattern (0-1)
        storage_vs_norm      — level minus seasonal 5yr average
        spot_forward_spread  — S - F(1M) (from futures curve)
        market_tightness_score — z-score composite of storage_vs_norm + spot_forward_spread
                                 (spark spread EXCLUDED — Gap 2)
        spark_spread         — electricity - gas × heat_rate (separate)
        gas_is_marginal      — spark_spread > 0 (bool)
        liquidity_score      — session-driven composite (tokyo/london/newyork/off-hours base + noise)
        trading_session      — categorical: tokyo / london / newyork / off-hours
    """

    def __init__(
        self,
        rng: np.random.Generator,
        start_date: str = cfg.SIM_START_DATE,
        end_date: str = cfg.SIM_END_DATE,
        freq: str = cfg.PRICE_FREQ,
    ) -> None:
        self.rng = rng
        self.start_date = start_date
        self.end_date = end_date
        self.freq = freq

    def _build_intraday_index(self) -> pd.DatetimeIndex:
        """Delegate to module-level function (Issue 5 fix)."""
        return _build_intraday_index(self.start_date, self.end_date, self.freq)

    @staticmethod
    def _seasonal_storage(day_of_year: int) -> float:
        """
        Seasonal storage pattern: peaks late summer (~day 260) and troughs late winter (~day 60).
        Returns a seasonal adjustment in [-0.08, +0.08].
        """
        return 0.08 * math.sin(2 * math.pi * (day_of_year - 80) / 365)

    @staticmethod
    def _get_trading_session(hour: int) -> str:
        """Classify UTC hour into trading session."""
        for session, (start_h, end_h) in TRADING_SESSIONS.items():
            if start_h <= hour < end_h:
                return session
        return "off-hours"

    def generate(
        self,
        spot_df: pd.DataFrame,
        futures_1m: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Generate the full market state DataFrame aligned to spot_df's index.

        Parameters
        ----------
        spot_df : spot price DataFrame (DatetimeIndex, 'Close' column)
        futures_1m : optional Series of 1M futures prices (same index).
            If None, spot_forward_spread is estimated from a synthetic offset.

        Returns
        -------
        pd.DataFrame indexed like spot_df with all state variable columns.
        """
        idx = spot_df.index
        n = len(idx)

        # ── 1. Storage level (mean-reverting, daily step) ─────────────
        storage = np.empty(n)
        storage[0] = cfg.STORAGE_MEAN
        for i in range(1, n):
            dt = (idx[i] - idx[i - 1]).total_seconds() / 86400  # fractional day
            seasonal_adj = self._seasonal_storage(idx[i].day_of_year)
            target = cfg.STORAGE_MEAN + seasonal_adj
            mean_rev = cfg.STORAGE_REVERSION_SPEED * (target - storage[i - 1]) * dt
            shock = self.rng.normal(0, 0.005 * math.sqrt(dt))
            storage[i] = np.clip(storage[i - 1] + mean_rev + shock, 0.05, 0.99)

        # Seasonal 5yr average (simplified as the seasonal component itself)
        seasonal_norm = np.array([
            cfg.STORAGE_MEAN + self._seasonal_storage(ts.day_of_year) for ts in idx
        ])
        storage_vs_norm = storage - seasonal_norm

        # ── 2. Spot-forward spread ────────────────────────────────────
        spot_prices = spot_df["Close"].values
        if futures_1m is not None:
            spot_fwd_spread = spot_prices - futures_1m.values
        else:
            # Synthetic approximation: use cost-of-carry offset
            storage_costs = np.array([
                FuturesCurveGenerator._storage_cost(u) for u in storage
            ])
            conv_yields = np.array([
                FuturesCurveGenerator._convenience_yield(sv) for sv in storage_vs_norm
            ])
            T_1m = cfg.TENOR_TO_YEARS["1M"]
            f1m_est = spot_prices * np.exp((cfg.RISK_FREE_RATE + storage_costs - conv_yields) * T_1m)
            spot_fwd_spread = spot_prices - f1m_est

        # ── 3. Market tightness score (z-score composite) ─────────────
        # Spark spread EXCLUDED (Gap 2: independent from tightness)
        raw_tightness = storage_vs_norm * (-1.0) + spot_fwd_spread / spot_prices
        # z-score
        t_mean = np.mean(raw_tightness)
        t_std = np.std(raw_tightness)
        if t_std > 0:
            market_tightness = (raw_tightness - t_mean) / t_std
        else:
            market_tightness = np.zeros(n)

        # ── 4. Spark spread (separate from tightness) ─────────────────
        # electricity price: simple random walk around base
        elec_prices = np.empty(n)
        elec_prices[0] = cfg.ELECTRICITY_BASE_PRICE
        for i in range(1, n):
            elec_prices[i] = elec_prices[i - 1] * (
                1 + self.rng.normal(0, cfg.ELECTRICITY_VOLATILITY)
            )
        spark_spread = elec_prices - spot_prices * cfg.HEAT_RATE
        gas_is_marginal = spark_spread > 0

        # ── 5. Trading session + liquidity score ──────────────────────
        # Liquidity driven primarily by trading session (fast variable)
        # with small random noise. This is partially decorrelated from
        # tightness, which evolves at daily/weekly frequency.
        session_base_liq = {
            "tokyo": 0.6,
            "london": 0.85,
            "newyork": 0.75,
            "off-hours": 0.3,
        }
        trading_sessions = np.array([self._get_trading_session(ts.hour) for ts in idx])
        liquidity_base = np.array([session_base_liq[s] for s in trading_sessions])
        liquidity_noise = self.rng.normal(0, 0.08, n)
        liquidity_score = np.clip(liquidity_base + liquidity_noise, 0.05, 1.0)

        # ── Assemble DataFrame ────────────────────────────────────────
        state_df = pd.DataFrame({
            "storage_level": storage,
            "storage_vs_norm": storage_vs_norm,
            "spot_forward_spread": spot_fwd_spread,
            "market_tightness_score": market_tightness,
            "spark_spread": spark_spread,
            "gas_is_marginal": gas_is_marginal,
            "liquidity_score": liquidity_score,
            "trading_session": trading_sessions,
        }, index=idx)
        state_df.index.name = "timestamp"
        return state_df


# ═════════════════════════════════════════════════════════════════════════════
# Convenience: generate_all()
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class SyntheticDataBundle:
    """Container for all synthetic data produced by generate_all()."""
    news_df: pd.DataFrame
    placebo_df: pd.DataFrame
    all_events_df: pd.DataFrame   # news + placebos concatenated
    spot_prices: Dict[str, pd.DataFrame]
    futures_curves: Dict[str, pd.DataFrame]
    options_surfaces: Dict[str, Dict[str, pd.DataFrame]]
    market_state: pd.DataFrame


def generate_all(seed: int = cfg.RANDOM_SEED) -> SyntheticDataBundle:
    """
    Master generator: produce all synthetic data needed by the pipeline.

    Parameters
    ----------
    seed : random seed for reproducibility

    Returns
    -------
    SyntheticDataBundle containing all generated data
    """
    rng = np.random.default_rng(seed)

    # ── 1. News events + placebos ─────────────────────────────────────
    news_gen = NewsGenerator(rng)
    news_df = news_gen.generate_events()
    placebo_df = news_gen.generate_placebos(news_df)
    # Align reactor_capacity_mw dtype before concat to suppress FutureWarning:
    # news_df has float64 values; placebo_df has all-None (object). Cast both to float.
    for _df in (news_df, placebo_df):
        _df["reactor_capacity_mw"] = _df["reactor_capacity_mw"].astype(float)
    all_events_df = pd.concat([news_df, placebo_df], ignore_index=True)
    all_events_df.sort_values("timestamp", inplace=True)
    all_events_df.reset_index(drop=True, inplace=True)

    print(f"[generators] News events     : {len(news_df)}")
    print(f"[generators] Placebo events   : {len(placebo_df)}")

    # ── 2. Primary spot + market state  (Issue 1 fix)
    # Strategy: spawn a dedicated child rng for the preliminary spot so that
    # its draws don't alter the parent rng state consumed by MarketStateGenerator
    # and other assets.  The preliminary spot IS reused as the final JKM spot,
    # keeping the tightness labels coherent with the price path.
    primary_cfg = cfg.ASSET_CONFIGS[cfg.PRIMARY_ASSET]
    rng_primary, rng_rest = rng.spawn(2)   # spawn child rngs for primary + secondary spot paths

    spot_gen_primary = SpotGenerator(
        rng=rng_primary,
        base_price=primary_cfg["base_price"],
        volatility=primary_cfg["volatility"],
    )
    jkm_spot = spot_gen_primary.generate(news_df)  # preliminary = final JKM spot

    state_gen = MarketStateGenerator(rng_rest)
    market_state = state_gen.generate(jkm_spot)

    # Build tightness lookup for events
    tightness_lookup: Dict[pd.Timestamp, float] = {}
    gas_marginal_lookup: Dict[pd.Timestamp, bool] = {}
    for _, row in news_df.iterrows():
        ts = row["timestamp"]
        state_bars = market_state.index[market_state.index <= ts]
        if len(state_bars) > 0:
            tightness_lookup[ts] = float(
                market_state.loc[state_bars[-1], "market_tightness_score"]
            )
            gas_marginal_lookup[ts] = bool(
                market_state.loc[state_bars[-1], "gas_is_marginal"]
            )

    # Re-generate JKM with tightness modulation now that we have the lookup
    jkm_spot = spot_gen_primary.generate(news_df, tightness_lookup, gas_marginal_lookup)

    # ── 3. Spot prices (all assets) ────────────────────────────────────
    # JKM reused from above; other assets get their own child rngs
    other_rngs = rng_rest.spawn(len(cfg.ASSET_CONFIGS) - 1)
    spot_prices: Dict[str, pd.DataFrame] = {cfg.PRIMARY_ASSET: jkm_spot}
    other_assets = [k for k in cfg.ASSET_CONFIGS if k != cfg.PRIMARY_ASSET]
    for asset_name, child_rng in zip(other_assets, other_rngs):
        acfg = cfg.ASSET_CONFIGS[asset_name]
        sgen = SpotGenerator(
            rng=child_rng,
            base_price=acfg["base_price"],
            volatility=acfg["volatility"],
        )
        spot_prices[asset_name] = sgen.generate(news_df, tightness_lookup, gas_marginal_lookup)

    print(f"[generators] Spot assets      : {list(spot_prices.keys())}")
    for name, sdf in spot_prices.items():
        print(f"             {name:20s} bars : {len(sdf):>10,}")

    # ── 4. Futures curves (primary asset only) ────────────────────────
    fcg = FuturesCurveGenerator(rng)
    futures_curves = fcg.generate_event_curves(
        spot_prices[cfg.PRIMARY_ASSET], all_events_df, market_state
    )
    print(f"[generators] Futures tenors   : {list(futures_curves.keys())}")

    # ── 5. Options surfaces (primary asset only) ──────────────────────
    osg = OptionsSurfaceGenerator(rng)
    options_surfaces = osg.generate_event_surfaces(futures_curves, all_events_df)
    print(f"[generators] Options events   : {len(options_surfaces)}")

    return SyntheticDataBundle(
        news_df=news_df,
        placebo_df=placebo_df,
        all_events_df=all_events_df,
        spot_prices=spot_prices,
        futures_curves=futures_curves,
        options_surfaces=options_surfaces,
        market_state=market_state,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Self-test (run: python generators.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  generators.py -- Synthetic Data Generators (self-test)")
    print("=" * 70)

    bundle = generate_all()

    print("\n── News sample ──")
    print(bundle.news_df[["timestamp", "keyword", "season", "trading_session"]].head(10).to_string(index=False))

    print(f"\n── Placebo events: {len(bundle.placebo_df)} ──")
    print(bundle.placebo_df[["timestamp", "season", "trading_session"]].head(5).to_string(index=False))

    print(f"\n── Market state sample (first 5 rows) ──")
    print(bundle.market_state.head().to_string())

    # Spot summary
    for asset, sdf in bundle.spot_prices.items():
        print(f"\n── {asset} spot: {len(sdf)} bars, "
              f"range [{sdf['Close'].min():.2f}, {sdf['Close'].max():.2f}] ──")

    # Futures summary
    for tenor, fdf in bundle.futures_curves.items():
        print(f"\n── Futures {tenor}: {len(fdf)} events ──")
        print(fdf.head(3).to_string(index=False))

    print(f"\n[OK] All generators completed successfully.")
    print(f"     Total events (real + placebo): {len(bundle.all_events_df)}")
