"""
┌──────────┐   ┌────────────┐   ┌───────────┐   ┌────────┐
│ theory   │──▶│ generators │──▶│ analytics │──▶│ main   │
└──────────┘   └────────────┘   └───────────┘   └────────┘
  ▲ YOU ARE HERE

theory.py — Structural model definitions
=========================================
Defines the causal / transmission-channel theories that connect LNG market
news events to observable price outcomes.

Key design principle
--------------------
Each keyword maps to a *KeywordTheory*, which is a fully explicit causal
hypothesis: it names the intermediate steps (TransmissionChannels), identifies
which parts of the term structure should move (affected_curve_tenors), and
predicts the direction of changes in implied volatility and skew.

Reading order: config → theory → generators → analytics → main
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Core Structural Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TransmissionChannel:
    """
    One step in the causal chain from news to price.

    Attributes
    ----------
    name : str
        Human-readable identifier (e.g. "reactor_timeline_update").
    input_signal : str
        What triggers / enters this channel step.
    output_signal : str
        What this step produces as output for the next channel.
    friction_variables : list[str]
        Variables that can block or attenuate the signal passing through
        this channel. If a friction variable is in an adverse state, the
        channel may not transmit (conditional on market state).

    Example
    -------
    TransmissionChannel(
        name="nuclear_to_gas_dispatch",
        input_signal="expected_nuclear_mw_online",
        output_signal="gas_fired_dispatch_reduction_mw",
        friction_variables=["gas_is_marginal"],   # gas must be on merit order
    )
    """
    name: str
    input_signal: str
    output_signal: str
    friction_variables: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class KeywordTheory:
    """
    Full structural theory for one keyword type.

    Attributes
    ----------
    keyword : str
        Must match a key in config.KEYWORD_BIAS.
    channels : list[TransmissionChannel]
        Ordered list of causal steps; the output of step n is the input
        of step n+1.
    affected_curve_tenors : list[str]
        Which futures tenors should exhibit significant shifts after
        events of this keyword type (e.g. ["1M", "3M"] = front-biased).
    curve_shift_direction : str
        Expected direction of level shift: "up" | "down" | "neutral".
    expected_vol_change : str
        Direction of implied ATM vol change: "increase" | "decrease" | "neutral".
    expected_skew_change : str
        Direction of risk-reversal change: "positive" | "negative" | "neutral".
        Positive RR = call IV > put IV (upside tail premium).
    marginal_fuel_required : bool
        If True, the full signal requires gas to be the marginal fuel
        (spark_spread > 0) — see Prediction 3.
    persistence_expectation : str
        Expected persistence_ratio (12M shift / 1M shift):
        "permanent" (≈1), "transient" (≈0), "intermediate".
    notes : str
        Free-text economic rationale.
    """
    keyword: str
    channels: List[TransmissionChannel]
    affected_curve_tenors: List[str]
    curve_shift_direction: str           # "up" / "down" / "neutral"
    expected_vol_change: str             # "increase" / "decrease" / "neutral"
    expected_skew_change: str            # "positive" / "negative" / "neutral"
    marginal_fuel_required: bool = False
    persistence_expectation: str = "intermediate"
    notes: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Keyword Theory Definitions
# ─────────────────────────────────────────────────────────────────────────────

# ── 1. Strike ─────────────────────────────────────────────────────────────
THEORY_STRIKE = KeywordTheory(
    keyword="Strike",
    channels=[
        TransmissionChannel(
            name="strike_announcement",
            input_signal="news_of_industrial_action",
            output_signal="expected_export_volume_reduction",
            friction_variables=["contract_labour_negotiation_stage"],
        ),
        TransmissionChannel(
            name="volume_to_global_supply",
            input_signal="expected_export_volume_reduction",
            output_signal="global_spot_supply_tightening",
            friction_variables=["alternative_cargo_availability", "storage_buffer"],
        ),
        TransmissionChannel(
            name="supply_tightening_to_price",
            input_signal="global_spot_supply_tightening",
            output_signal="jkm_spot_price_increase",
            friction_variables=["storage_vs_norm"],  # high storage absorbs shock
        ),
    ],
    affected_curve_tenors=["1M", "3M"],   # front-end: strike is usually short-lived
    curve_shift_direction="up",
    expected_vol_change="increase",        # uncertainty about duration → vol up
    expected_skew_change="positive",       # upside tail risk dominates
    marginal_fuel_required=False,
    persistence_expectation="transient",   # supply resumes after resolution
    notes=(
        "A labour strike at an LNG export terminal primarily affects near-term "
        "supply. Longer tenors are less affected unless the strike is prolonged "
        "or signals structural labour disputes. Vol rises on outcome uncertainty. "
        "Upside call skew premium reflects risk of extended disruption."
    ),
)

# ── 2. Outage ─────────────────────────────────────────────────────────────
THEORY_OUTAGE = KeywordTheory(
    keyword="Outage",
    channels=[
        TransmissionChannel(
            name="technical_failure",
            input_signal="equipment_failure_report",
            output_signal="production_capacity_offline_mw",
            friction_variables=["redundancy_systems", "maintenance_schedule"],
        ),
        TransmissionChannel(
            name="capacity_offline_to_supply_gap",
            input_signal="production_capacity_offline_mw",
            output_signal="spot_supply_shortfall",
            friction_variables=["pipeline_interconnect_availability", "storage_buffer"],
        ),
        TransmissionChannel(
            name="supply_gap_to_spot_price",
            input_signal="spot_supply_shortfall",
            output_signal="jkm_spot_price_increase",
            friction_variables=["storage_vs_norm"],
        ),
    ],
    affected_curve_tenors=["1M", "3M"],
    curve_shift_direction="up",
    expected_vol_change="increase",
    expected_skew_change="positive",
    marginal_fuel_required=False,
    persistence_expectation="transient",
    notes=(
        "Unplanned outages are typically short-lived (days to weeks) but create "
        "immediate spot scarcity. Impact is attenuated by high storage levels. "
        "3M tenor moves if repair timeline is unclear; 6M+ tenors generally "
        "unaffected unless the outage signals systemic infrastructure issues."
    ),
)

# ── 3. Cold Snap ──────────────────────────────────────────────────────────
THEORY_COLD_SNAP = KeywordTheory(
    keyword="Cold Snap",
    channels=[
        TransmissionChannel(
            name="temperature_signal",
            input_signal="meteorological_forecast_below_normal",
            output_signal="residential_and_power_heating_demand_increase",
            friction_variables=["forecast_accuracy", "hedging_positions"],
        ),
        TransmissionChannel(
            name="heating_demand_to_gas_burn",
            input_signal="residential_and_power_heating_demand_increase",
            output_signal="gas_consumption_increase",
            friction_variables=["renewable_generation_level", "demand_response"],
        ),
        TransmissionChannel(
            name="gas_burn_to_inventory_draw",
            input_signal="gas_consumption_increase",
            output_signal="storage_drawdown",
            friction_variables=["storage_vs_norm"],  # starting level matters
        ),
        TransmissionChannel(
            name="inventory_draw_to_jkm",
            input_signal="storage_drawdown",
            output_signal="jkm_spot_price_increase",
            friction_variables=["lng_spot_cargo_availability"],
        ),
    ],
    affected_curve_tenors=["1M", "3M"],    # weather effect is seasonal
    curve_shift_direction="up",
    expected_vol_change="increase",
    expected_skew_change="positive",
    marginal_fuel_required=False,
    persistence_expectation="transient",
    notes=(
        "Cold snaps are a canonical LNG demand shock. Impact is asymmetrically "
        "large when storage levels are already below seasonal norms. The "
        "market_tightness_score (storage + backwardation) directly conditions "
        "impact magnitude — Prediction 1. Tenors beyond 3M rarely move "
        "unless the cold snap extends into a multi-week event."
    ),
)

# ── 4. Nuclear Restart ────────────────────────────────────────────────────
#
# This is the structurally richest theory and is used for Predictions 2 & 3.
#
# 5-step transmission chain (matches the implementation_plan table):
#
# Step | Channel                          | Key Friction
# -----|----------------------------------|-------------------------------------------
#  1   | News → timeline expectation      | credibility_tier (NRA formal >> intent)
#  2   | Timeline → nuclear MW online     | reactor_capacity_mw
#  3   | Nuclear ↑ → gas dispatch ↓       | gas_is_marginal (spark_spread > 0)
#  4   | Gas dispatch ↓ → LNG demand ↓   | long_term_contract_rigidity
#  5   | LNG demand ↓ → JKM adjusts       | storage_vs_norm (high storage absorbs)
#
THEORY_NUCLEAR_RESTART = KeywordTheory(
    keyword="Nuclear Restart",
    channels=[
        TransmissionChannel(
            name="news_to_timeline_expectation",
            input_signal="nuclear_restart_news_headline",
            output_signal="expected_timeline_revision",
            friction_variables=["credibility_tier"],
            # Tier 1 (NRA formal) = low friction; Tier 3 (policy signal) = high friction
        ),
        TransmissionChannel(
            name="timeline_to_nuclear_mw",
            input_signal="expected_timeline_revision",
            output_signal="expected_nuclear_capacity_mw_online",
            friction_variables=["reactor_capacity_mw"],
            # Larger reactor → larger output signal
        ),
        TransmissionChannel(
            name="nuclear_to_gas_dispatch",
            input_signal="expected_nuclear_capacity_mw_online",
            output_signal="expected_gas_fired_dispatch_reduction_gw",
            friction_variables=["gas_is_marginal"],
            # KEY FRICTION: if spark_spread < 0, gas is NOT on merit order
            # → nuclear replacing gas has ~0 LNG demand impact
            # → this is Prediction 3 (marginal fuel condition)
        ),
        TransmissionChannel(
            name="gas_dispatch_to_lng_demand",
            input_signal="expected_gas_fired_dispatch_reduction_gw",
            output_signal="expected_lng_import_volume_reduction",
            friction_variables=["long_term_contract_rigidity"],
        ),
        TransmissionChannel(
            name="lng_demand_to_jkm",
            input_signal="expected_lng_import_volume_reduction",
            output_signal="jkm_spot_price_decrease",
            friction_variables=["storage_vs_norm"],
        ),
    ],
    affected_curve_tenors=["3M", "6M", "12M"],   # restart has lasting impact
    curve_shift_direction="down",
    expected_vol_change="decrease",     # baseload certainty → vol down
    expected_skew_change="negative",    # downside risk for gas longs
    marginal_fuel_required=True,        # Prediction 3: gas must be marginal
    persistence_expectation="permanent",
    notes=(
        "Nuclear Restart is the most structurally important keyword because it "
        "links to a verifiable physical mechanism (MW displacement). Full "
        "price impact is conditional on two separate gate conditions: "
        "(1) gas must be the marginal fuel (spark_spread > 0), and "
        "(2) news credibility must be high (Tier 1 or 2). "
        "Prediction 2 tests proportionality to reactor_capacity_mw. "
        "Prediction 3 tests the marginal-fuel gate (gas_is_marginal). "
        "Long curve tenors (6M, 12M) should show persistent downward shift "
        "because nuclear restart changes the structural supply-demand balance."
    ),
)

# ── 5. Tariff ─────────────────────────────────────────────────────────────
THEORY_TARIFF = KeywordTheory(
    keyword="Tariff",
    channels=[
        TransmissionChannel(
            name="tariff_announcement",
            input_signal="trade_policy_news",
            output_signal="expected_trade_flow_redirection",
            friction_variables=["policy_implementation_lag", "contract_flexibility"],
        ),
        TransmissionChannel(
            name="flow_redirection_to_netback",
            input_signal="expected_trade_flow_redirection",
            output_signal="basis_differential_change",
            friction_variables=["shipping_optionality"],
        ),
        TransmissionChannel(
            name="basis_change_to_price",
            input_signal="basis_differential_change",
            output_signal="jkm_spot_price_mild_increase",
            friction_variables=["alternative_supply_routes"],
        ),
    ],
    affected_curve_tenors=["1M", "3M", "6M"],
    curve_shift_direction="up",
    expected_vol_change="increase",      # policy uncertainty → vol up
    expected_skew_change="positive",
    marginal_fuel_required=False,
    persistence_expectation="intermediate",
    notes=(
        "Tariffs create trade flow distortions but the net LNG price effect "
        "depends heavily on whether flows can be rerouted. High shipping "
        "optionality attenuates the impact. Mid-curve (3M-6M) moves on "
        "uncertainty about implementation; longer tenors less responsive."
    ),
)

# ─────────────────────────────────────────────────────────────────────────────
# Registry: all theories indexed by keyword
# ─────────────────────────────────────────────────────────────────────────────

ALL_THEORIES: dict[str, KeywordTheory] = {
    t.keyword: t
    for t in [
        THEORY_STRIKE,
        THEORY_OUTAGE,
        THEORY_COLD_SNAP,
        THEORY_NUCLEAR_RESTART,
        THEORY_TARIFF,
    ]
}
"""
Registry mapping keyword → KeywordTheory.
Usage:
    from theory import ALL_THEORIES
    nuclear_theory = ALL_THEORIES["Nuclear Restart"]
    channel_frictions = nuclear_theory.channels[2].friction_variables
"""


def get_theory(keyword: str) -> KeywordTheory:
    """Return the KeywordTheory for a given keyword, raising KeyError if unknown."""
    if keyword not in ALL_THEORIES:
        raise KeyError(
            f"No theory defined for keyword '{keyword}'. "
            f"Available: {list(ALL_THEORIES.keys())}"
        )
    return ALL_THEORIES[keyword]


def describe_theory(keyword: str) -> str:
    """
    Return a human-readable summary of a keyword's transmission chain.

    Useful for logging / README generation.
    """
    theory = get_theory(keyword)
    lines = [
        f"Theory: {theory.keyword}",
        f"  Curve direction   : {theory.curve_shift_direction}",
        f"  Affected tenors   : {', '.join(theory.affected_curve_tenors)}",
        f"  Vol change        : {theory.expected_vol_change}",
        f"  Skew change       : {theory.expected_skew_change}",
        f"  Marginal fuel req : {theory.marginal_fuel_required}",
        f"  Persistence       : {theory.persistence_expectation}",
        f"  Channels:",
    ]
    for i, ch in enumerate(theory.channels, 1):
        frictions = ", ".join(ch.friction_variables) if ch.friction_variables else "none"
        lines.append(f"    [{i}] {ch.name}")
        lines.append(f"         in  : {ch.input_signal}")
        lines.append(f"         out : {ch.output_signal}")
        lines.append(f"         friction: {frictions}")
    if theory.notes:
        lines.append(f"  Notes: {theory.notes}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Self-test (run: python theory.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  theory.py — Structural Theory Definitions")
    print("=" * 70)
    for kw in ALL_THEORIES:
        print()
        print(describe_theory(kw))
        print("-" * 70)
    print(f"\n[OK] {len(ALL_THEORIES)} keyword theories defined.")
