"""
================================================================================
  Project Alpha-Discovery
  Event Study Framework for Energy Markets & News-Driven Strategies
  ────────────────────────────────────────────────────────────────
  Author : [Your Name] - Graduate Quant Researcher Candidate
  Target : JERA Global Markets
  Python : 3.10+
  License: MIT
================================================================================

  This framework ingests historical news headlines and asset price data,
  identifies keyword-driven events, measures forward price impact via an
  Event Study methodology, and outputs professional-grade visualizations.

  Reference architecture inspired by the Bluesky custom-feed pattern
  (docs.bsky.app/docs/starter-templates/custom-feeds):
    • A "firehose" of news events is filtered by keyword rules
    • Each qualifying event triggers a forward-return calculation
    • Results are aggregated into a statistical summary and heatmap

  Modules
  -------
  1. Golden Dataset Generator  – synthetic news + price data
  2. Event Processor            – EventAnalyzer class
  3. Statistical Verification   – hypothesis testing & scoring
  4. Visualization              – Seaborn correlation heatmap
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

# ─── Global Config ────────────────────────────────────────────────────────────

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Event window: minutes after the news event to measure impact
FORWARD_WINDOW_MINUTES = 60

# Keywords and their expected directional bias for mock data generation
# positive bias → price tends to go UP after event
# negative bias → price tends to go DOWN after event
KEYWORD_BIAS: Dict[str, float] = {
    "Strike":          +0.012,   # supply disruption → bullish
    "Outage":          +0.008,   # unplanned outage  → bullish
    "Cold Snap":       +0.010,   # demand surge      → bullish
    "Nuclear Restart": -0.010,   # new baseload      → bearish for gas
    "Tariff":          +0.005,   # trade friction     → mild bullish
}

# Simulated asset tickers
ASSET_TICKERS = ["UNG", "TTF", "TEPCO (9501.T)"]


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1 ─ The Golden Dataset Generator
# ══════════════════════════════════════════════════════════════════════════════

# Bloomberg / WSJ / Reuters-style headline templates per keyword
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

CATEGORIES = ["Macro", "Supply", "Demand", "Geopolitical", "Policy"]


def generate_mock_news(
    n_events: int = 60,
    start_date: str = "2025-02-11",
    end_date: str = "2026-02-11",
) -> pd.DataFrame:
    """
    Generate a realistic mock news DataFrame with Bloomberg/WSJ-style headlines.

    Parameters
    ----------
    n_events   : total number of news events to generate
    start_date : start of the date range (inclusive)
    end_date   : end of the date range (inclusive)

    Returns
    -------
    pd.DataFrame with columns [timestamp, headline, category, keyword]
    """
    keywords = list(HEADLINE_TEMPLATES.keys())
    records: list[dict] = []

    # Distribute events roughly evenly across keywords, with some randomness
    base_per_kw = n_events // len(keywords)
    remainder = n_events - base_per_kw * len(keywords)
    counts = [base_per_kw] * len(keywords)
    for i in range(remainder):
        counts[i] += 1

    bdays = pd.bdate_range(start=start_date, end=end_date)

    for kw, count in zip(keywords, counts):
        templates = HEADLINE_TEMPLATES[kw]
        for _ in range(count):
            day = np.random.choice(bdays)
            # Random hour during London/NY/Tokyo trading hours (07-21 UTC)
            hour = np.random.randint(7, 21)
            minute = np.random.randint(0, 60)
            ts = pd.Timestamp(day) + pd.Timedelta(hours=hour, minutes=minute)
            headline = np.random.choice(templates)
            category = np.random.choice(CATEGORIES)
            records.append(
                {"timestamp": ts, "headline": headline, "category": category, "keyword": kw}
            )

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def generate_mock_prices(
    news_df: pd.DataFrame,
    asset_name: str = "UNG",
    base_price: float = 10.0,
    volatility: float = 0.0005,
    freq: str = "5min",
    start_date: str = "2025-02-11",
    end_date: str = "2026-02-11",
) -> pd.DataFrame:
    """
    Generate mock intraday price data with intentional bias around news events.

    The price follows a Geometric Brownian Motion (GBM) baseline, with an
    injected drift near keyword events to ensure the Event Study detects
    statistically significant patterns.

    Parameters
    ----------
    news_df    : news DataFrame (must contain 'timestamp' and 'keyword')
    asset_name : ticker label (used for logging)
    base_price : starting price level
    volatility : per-bar return standard deviation
    freq       : bar frequency string (e.g., "5min", "1h")
    start_date : price series start
    end_date   : price series end

    Returns
    -------
    pd.DataFrame with DatetimeIndex and column 'Close'
    """
    # Create intraday index: business days, trading hours 07:00-21:00 UTC
    bdays = pd.bdate_range(start=start_date, end=end_date)
    all_timestamps: list[pd.Timestamp] = []
    for day in bdays:
        day_range = pd.date_range(
            start=day + pd.Timedelta(hours=7),
            end=day + pd.Timedelta(hours=21),
            freq=freq,
        )
        all_timestamps.extend(day_range)

    idx = pd.DatetimeIndex(all_timestamps)
    n_bars = len(idx)

    # Baseline returns: GBM
    returns = np.random.normal(0, volatility, n_bars)

    # Inject keyword bias near event timestamps
    for _, row in news_df.iterrows():
        event_ts = row["timestamp"]
        keyword = row["keyword"]
        bias = KEYWORD_BIAS.get(keyword, 0.0)

        if bias == 0.0:
            continue

        # Apply bias to the FORWARD_WINDOW_MINUTES after the event
        mask = (idx >= event_ts) & (idx <= event_ts + pd.Timedelta(minutes=FORWARD_WINDOW_MINUTES))
        n_affected = mask.sum()
        if n_affected > 0:
            # Spread the total bias across the window bars with some noise
            per_bar_bias = bias / n_affected
            returns[mask] += per_bar_bias + np.random.normal(0, volatility * 0.3, n_affected)

    # Build price series from returns
    price_levels = base_price * np.cumprod(1 + returns)

    df = pd.DataFrame({"Close": price_levels}, index=idx)
    df.index.name = "timestamp"
    return df


def generate_mock_data(
    asset_configs: Optional[Dict[str, dict]] = None,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Master generator: creates news + multi-asset price DataFrames.

    Parameters
    ----------
    asset_configs : dict mapping asset_name -> kwargs for generate_mock_prices
                    If None, defaults to UNG / TTF / TEPCO configs.

    Returns
    -------
    (news_df, price_dict)
        news_df    : pd.DataFrame of news events
        price_dict : dict[asset_name] -> pd.DataFrame of prices
    """
    if asset_configs is None:
        asset_configs = {
            "UNG":            {"base_price": 10.0, "volatility": 0.0006},
            "TTF":            {"base_price": 30.0, "volatility": 0.0005},
            "TEPCO (9501.T)": {"base_price": 650.0, "volatility": 0.0004},
        }

    news_df = generate_mock_news(n_events=60)

    price_dict: Dict[str, pd.DataFrame] = {}
    for asset_name, kwargs in asset_configs.items():
        price_dict[asset_name] = generate_mock_prices(
            news_df=news_df, asset_name=asset_name, **kwargs
        )

    return news_df, price_dict


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2 ─ The Event Processor
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EventResult:
    """Container for a single event's forward-return measurement."""
    event_time: pd.Timestamp
    headline: str
    keyword: str
    price_at_t: float
    price_at_t_plus: float
    forward_return: float
    window_minutes: int


@dataclass
class KeywordStats:
    """Aggregated statistics for one keyword on one asset."""
    keyword: str
    asset: str
    n_events: int
    mean_return: float
    median_return: float
    std_return: float
    win_rate: float                  # directional win rate (auto-detected)
    directional_consistency: float   # % of events moving in the dominant direction
    signal_strength: float           # |Mean Return| x Consistency (signed)
    sharpe_ratio: float              # Mean Return / Std Dev  (risk-adjusted signal)
    direction: str                   # "LONG" or "SHORT"
    t_stat: float
    p_value: float


class EventAnalyzer:
    """
    Core engine for Event Study analysis.

    Workflow
    --------
    1. Filter news by keyword
    2. For each event at time T, find Close(T) and Close(T + window)
    3. Compute forward return  =  (Close_T+w - Close_T) / Close_T
    4. Aggregate into KeywordStats with statistical tests
    """

    def __init__(
        self,
        news_df: pd.DataFrame,
        price_df: pd.DataFrame,
        asset_name: str = "UNG",
        forward_window: int = FORWARD_WINDOW_MINUTES,
    ) -> None:
        self.news_df = news_df.copy()
        self.price_df = price_df.copy()
        self.asset_name = asset_name
        self.forward_window = forward_window
        self._results: Dict[str, List[EventResult]] = {}

    # ── Core Methods ──────────────────────────────────────────────────────

    def _snap_to_nearest_bar(self, ts: pd.Timestamp) -> Optional[pd.Timestamp]:
        """
        Snap a news timestamp to the nearest available price bar.

        If the event occurs outside trading hours, return the next available bar.
        Returns None if no valid bar is found within 24 hours.
        """
        idx = self.price_df.index
        # Forward-fill: find first bar >= event time
        future_bars = idx[idx >= ts]
        if len(future_bars) == 0:
            return None
        nearest = future_bars[0]
        # Reject if the nearest bar is more than 24h away
        if (nearest - ts) > pd.Timedelta(hours=24):
            return None
        return nearest

    def analyze_keyword(self, keyword: str) -> List[EventResult]:
        """
        Run the event study for a single keyword.

        Parameters
        ----------
        keyword : target keyword to filter headlines

        Returns
        -------
        List of EventResult objects
        """
        # Vectorized keyword filter (case-insensitive)
        mask = self.news_df["headline"].str.contains(keyword, case=False, na=False)
        events = self.news_df.loc[mask]

        results: List[EventResult] = []

        for _, row in events.iterrows():
            event_ts = row["timestamp"]

            # Snap to nearest price bar
            bar_ts = self._snap_to_nearest_bar(event_ts)
            if bar_ts is None:
                continue

            # Target exit time
            exit_ts = bar_ts + pd.Timedelta(minutes=self.forward_window)

            # Find the closest bar to the exit time
            exit_bars = self.price_df.index[self.price_df.index >= exit_ts]
            if len(exit_bars) == 0:
                continue
            exit_bar = exit_bars[0]

            # Reject if exit bar is unreasonably far (next day)
            if (exit_bar - bar_ts) > pd.Timedelta(hours=24):
                continue

            price_t = self.price_df.loc[bar_ts, "Close"]
            price_t_plus = self.price_df.loc[exit_bar, "Close"]

            fwd_return = (price_t_plus - price_t) / price_t

            results.append(
                EventResult(
                    event_time=event_ts,
                    headline=row["headline"],
                    keyword=keyword,
                    price_at_t=price_t,
                    price_at_t_plus=price_t_plus,
                    forward_return=fwd_return,
                    window_minutes=self.forward_window,
                )
            )

        self._results[keyword] = results
        return results

    def analyze_keywords(self, keywords: List[str]) -> Dict[str, List[EventResult]]:
        """Run event study for multiple keywords."""
        for kw in keywords:
            self.analyze_keyword(kw)
        return self._results

    # ══════════════════════════════════════════════════════════════════════
    # MODULE 3 ─ Statistical Verification
    # ══════════════════════════════════════════════════════════════════════

    def calculate_stats(self, keyword: str) -> Optional[KeywordStats]:
        """
        Compute aggregate statistics for a keyword's event study results.

        Metrics
        -------
        - Mean Return               : average forward return across all events
        - Median Return             : median forward return (robust to outliers)
        - Std Return                : standard deviation of forward returns
        - Direction                 : auto-detected signal direction (LONG/SHORT)
        - Win Rate                  : % of events moving in the *detected* direction
        - Directional Consistency   : same as win rate (explicit naming for clarity)
        - Signal Strength           : sign(mean) * |Mean Return| * Consistency
        - t-statistic               : one-sample t-test (H0: mean return = 0)
        - p-value                   : two-tailed p-value from t-test

        Key Fix (v2)
        ------------
        The original formula Signal_Strength = Mean_Return * Win_Rate collapsed
        to 0 for bearish keywords (e.g. Nuclear Restart) because:
          Win_Rate = count(return > 0) / N  -->  ~0 for consistently negative returns
          Signal   = (-0.9%) * 0            -->  0  (incorrect!)

        Corrected formula:
          direction = LONG if mean > 0 else SHORT
          consistency = count(return in expected direction) / N
          Signal_Strength = sign(mean) * |Mean Return| * Consistency

        This correctly scores both bullish AND bearish signals.

        Returns
        -------
        KeywordStats or None if no events found
        """
        if keyword not in self._results or len(self._results[keyword]) == 0:
            return None

        returns = np.array([r.forward_return for r in self._results[keyword]])
        n = len(returns)

        mean_ret = float(np.mean(returns))
        median_ret = float(np.median(returns))
        std_ret = float(np.std(returns, ddof=1)) if n > 1 else 0.0

        # ── Auto-detect signal direction from the data ──────────────────
        direction = "LONG" if mean_ret >= 0 else "SHORT"

        # ── Directional Win Rate ────────────────────────────────────────
        # LONG  signal: "win" = return > 0  (price went up as expected)
        # SHORT signal: "win" = return < 0  (price went down as expected)
        if direction == "LONG":
            win_rate = float(np.mean(returns > 0))
        else:
            win_rate = float(np.mean(returns < 0))

        # Directional consistency is the same value, named explicitly
        directional_consistency = win_rate

        # ── Signal Strength (v2) ────────────────────────────────────────
        # Formula: sign(mean) * |Mean Return| * Directional Consistency
        # - Preserves the sign (positive for LONG, negative for SHORT)
        # - Magnitude reflects both the average impact AND its reliability
        sign = 1.0 if direction == "LONG" else -1.0
        signal_strength = sign * abs(mean_ret) * directional_consistency

        # ── Sharpe Ratio (risk-adjusted signal) ──────────────────────
        # Formula: Mean Return / Std Dev
        # 赚 1% 但波动很小（稳赚）>> 赚 5% 但时常亏 10%（赌博）
        # 值越大越好，带符号：正=做多信号，负=做空信号
        if n > 1 and std_ret > 0:
            sharpe_ratio = mean_ret / std_ret
        else:
            sharpe_ratio = 0.0

        # ── One-sample t-test: is the mean return != 0? ─────────────────
        if n > 1 and std_ret > 0:
            t_stat, p_val = stats.ttest_1samp(returns, 0)
            t_stat = float(t_stat)
            p_val = float(p_val)
        else:
            t_stat, p_val = 0.0, 1.0

        return KeywordStats(
            keyword=keyword,
            asset=self.asset_name,
            n_events=n,
            mean_return=mean_ret,
            median_return=median_ret,
            std_return=std_ret,
            win_rate=win_rate,
            directional_consistency=directional_consistency,
            signal_strength=signal_strength,
            sharpe_ratio=sharpe_ratio,
            direction=direction,
            t_stat=t_stat,
            p_value=p_val,
        )

    def get_all_stats(self, keywords: List[str]) -> List[KeywordStats]:
        """Calculate stats for all analyzed keywords."""
        all_stats: List[KeywordStats] = []
        for kw in keywords:
            s = self.calculate_stats(kw)
            if s is not None:
                all_stats.append(s)
        return all_stats


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 4 ─ Visualization
# ══════════════════════════════════════════════════════════════════════════════

def _build_heatmap_matrix(
    all_stats: List[KeywordStats],
    metric: str = "mean_return",
) -> pd.DataFrame:
    """
    Pivot a list of KeywordStats into a matrix suitable for Seaborn heatmap.

    Rows = Assets, Columns = Keywords, Values = chosen metric.
    """
    records = [
        {"Asset": s.asset, "Keyword": s.keyword, metric: getattr(s, metric)}
        for s in all_stats
    ]
    df = pd.DataFrame(records)
    matrix = df.pivot(index="Asset", columns="Keyword", values=metric)
    return matrix


def plot_correlation_heatmap(
    all_stats: List[KeywordStats],
    save_path: Optional[str] = None,
) -> None:
    """
    Plot a professional-grade heatmap of keyword-asset correlations.

    Parameters
    ----------
    all_stats : list of KeywordStats across all assets and keywords
    save_path : if provided, save the figure to this path
    """
    # ── Build matrices for the three panels ─────────────────────────────
    mean_return_matrix = _build_heatmap_matrix(all_stats, "mean_return") * 100  # → %
    signal_matrix = _build_heatmap_matrix(all_stats, "signal_strength") * 100
    sharpe_matrix = _build_heatmap_matrix(all_stats, "sharpe_ratio")

    fig, axes = plt.subplots(1, 3, figsize=(24, 5.5), gridspec_kw={"wspace": 0.35})

    # ── Common style ────────────────────────────────────────────────────
    cmap_diverging = sns.diverging_palette(10, 150, s=90, l=50, as_cmap=True)

    # ── Panel 1: Mean Forward Return (%) ────────────────────────────────
    ax1 = axes[0]
    vmax = max(abs(mean_return_matrix.values.min()), abs(mean_return_matrix.values.max()))
    sns.heatmap(
        mean_return_matrix,
        annot=True,
        fmt=".3f",
        cmap=cmap_diverging,
        center=0,
        vmin=-vmax,
        vmax=vmax,
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": "Mean Return (%)", "shrink": 0.8},
        ax=ax1,
    )
    ax1.set_title("Mean Forward Return by Keyword × Asset (%)", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("News Keyword", fontsize=11)
    ax1.set_ylabel("Asset", fontsize=11)
    ax1.tick_params(axis="x", rotation=30)
    ax1.tick_params(axis="y", rotation=0)

    # ── Panel 2: Signal Strength (v2: signed, supports SHORT signals) ──
    ax2 = axes[1]
    vmax_sig = max(abs(signal_matrix.values.min()), abs(signal_matrix.values.max()))
    sns.heatmap(
        signal_matrix,
        annot=True,
        fmt=".4f",
        cmap=cmap_diverging,
        center=0,
        vmin=-vmax_sig,
        vmax=vmax_sig,
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": "Signal Strength (%)", "shrink": 0.8},
        ax=ax2,
    )
    ax2.set_title("Signal Strength (|Mean Ret| x Consistency, signed)", fontsize=12, fontweight="bold", pad=12)
    ax2.set_xlabel("News Keyword", fontsize=11)
    ax2.set_ylabel("Asset", fontsize=11)
    ax2.tick_params(axis="x", rotation=30)
    ax2.tick_params(axis="y", rotation=0)

    # ── Panel 3: Sharpe Ratio (Mean / Std — risk-adjusted signal) ────
    ax3 = axes[2]
    vmax_sr = max(abs(sharpe_matrix.values.min()), abs(sharpe_matrix.values.max()))
    if vmax_sr == 0:
        vmax_sr = 1.0  # avoid degenerate color range
    sns.heatmap(
        sharpe_matrix,
        annot=True,
        fmt=".3f",
        cmap=cmap_diverging,
        center=0,
        vmin=-vmax_sr,
        vmax=vmax_sr,
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": "Sharpe Ratio", "shrink": 0.8},
        ax=ax3,
    )
    ax3.set_title("Sharpe Ratio (Mean Ret / Std Dev)", fontsize=12, fontweight="bold", pad=12)
    ax3.set_xlabel("News Keyword", fontsize=11)
    ax3.set_ylabel("Asset", fontsize=11)
    ax3.tick_params(axis="x", rotation=30)
    ax3.tick_params(axis="y", rotation=0)

    # ── Suptitle ────────────────────────────────────────────────────────
    fig.suptitle(
        "Project Alpha-Discovery | News-Keyword Event Study Heatmap",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"\n[OK] Heatmap saved -> {save_path}")

    plt.show()


def plot_event_returns_distribution(
    all_stats: List[KeywordStats],
    results_dict: Dict[str, Dict[str, List[EventResult]]],
    keywords: List[str],
    save_path: Optional[str] = None,
) -> None:
    """
    Plot the distribution of forward returns for each keyword (box + strip plot).
    """
    records: list[dict] = []
    for asset_name, kw_results in results_dict.items():
        for kw in keywords:
            if kw in kw_results:
                for er in kw_results[kw]:
                    records.append(
                        {
                            "Asset": asset_name,
                            "Keyword": kw,
                            "Forward Return (%)": er.forward_return * 100,
                        }
                    )

    if not records:
        print("[!] No event results to plot distribution.")
        return

    df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(
        data=df,
        x="Keyword",
        y="Forward Return (%)",
        hue="Asset",
        palette="Set2",
        linewidth=1.2,
        fliersize=3,
        ax=ax,
    )
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_title(
        "Forward Return Distribution by Keyword × Asset",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(title="Asset", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"[OK] Distribution plot saved -> {save_path}")

    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION BLOCK
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    End-to-end pipeline:
      1. Generate mock data
      2. Run event study across multiple assets & keywords
      3. Print statistical summary
      4. Plot heatmap and distribution chart
    """
    print("=" * 72)
    print("  Project Alpha-Discovery | Event Study Framework")
    print("  Targeting: JERA Global Markets - Graduate Quant Role")
    print("=" * 72)

    # ── Step 1: Generate Mock Data ──────────────────────────────────────
    print("\n[1/4] Generating mock news & price data ...")
    news_df, price_dict = generate_mock_data()

    print(f"      News events generated : {len(news_df)}")
    for asset, pdf in price_dict.items():
        print(f"      {asset:20s} bars : {len(pdf):>10,}")

    print("\n-- Sample Headlines --")
    print(news_df[["timestamp", "headline", "keyword"]].head(8).to_string(index=False))

    # ── Step 2: Run Event Study ─────────────────────────────────────────
    target_keywords = ["Strike", "Cold Snap", "Nuclear Restart"]
    print(f"\n[2/4] Running event study for keywords: {target_keywords}")
    print(f"      Forward window: {FORWARD_WINDOW_MINUTES} minutes")

    all_stats: List[KeywordStats] = []
    all_results: Dict[str, Dict[str, List[EventResult]]] = {}

    for asset_name, price_df in price_dict.items():
        print(f"\n      Analyzing: {asset_name} ...")
        analyzer = EventAnalyzer(
            news_df=news_df,
            price_df=price_df,
            asset_name=asset_name,
            forward_window=FORWARD_WINDOW_MINUTES,
        )
        analyzer.analyze_keywords(target_keywords)
        stats_list = analyzer.get_all_stats(target_keywords)
        all_stats.extend(stats_list)
        all_results[asset_name] = analyzer._results

        for s in stats_list:
            print(f"        {s.keyword:20s}  [{s.direction:5s}]  n={s.n_events:3d}  "
                  f"mean={s.mean_return*100:+.4f}%  "
                  f"consist={s.directional_consistency:.1%}  "
                  f"signal={s.signal_strength*100:+.4f}%  "
                  f"sharpe={s.sharpe_ratio:+.3f}  "
                  f"p={s.p_value:.4f}")

    # ── Step 3: Print Summary Table ─────────────────────────────────────
    print("\n[3/4] Statistical Summary Table\n")

    summary_records = [
        {
            "Asset": s.asset,
            "Keyword": s.keyword,
            "Direction": s.direction,
            "N": s.n_events,
            "Mean Ret(%)": round(s.mean_return * 100, 4),
            "Median Ret(%)": round(s.median_return * 100, 4),
            "Std(%)": round(s.std_return * 100, 4),
            "Win Rate": f"{s.win_rate:.1%}",
            "Consistency": f"{s.directional_consistency:.1%}",
            "Signal Str(%)": round(s.signal_strength * 100, 4),
            "Sharpe Ratio": round(s.sharpe_ratio, 4),
            "t-stat": round(s.t_stat, 3),
            "p-value": round(s.p_value, 4),
            "Sig": "***" if s.p_value < 0.01 else ("**" if s.p_value < 0.05 else ("*" if s.p_value < 0.1 else "")),
        }
        for s in all_stats
    ]
    summary_df = pd.DataFrame(summary_records)
    print(summary_df.to_string(index=False))

    # ── Step 4: Visualize ───────────────────────────────────────────────
    print("\n[4/4] Generating visualizations ...")
    plot_correlation_heatmap(all_stats, save_path="heatmap_alpha_discovery.png")
    plot_event_returns_distribution(
        all_stats, all_results, target_keywords,
        save_path="distribution_alpha_discovery.png",
    )

    print("\n" + "=" * 72)
    print("  Pipeline complete. Ready for interview presentation.  ")
    print("=" * 72)


if __name__ == "__main__":
    main()
