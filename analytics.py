"""
┌──────────┐   ┌────────────┐   ┌───────────┐   ┌────────┐
│ theory   │──▶│ generators │──▶│ analytics │──▶│ main   │
└──────────┘   └────────────┘   └───────────┘   └────────┘
                                   ▲ YOU ARE HERE

analytics.py — Analytics Engine
=================================
Contains 4 analysis classes:
  - CurveDecomposer
  - ImpliedDistribution
  - ConvenienceYieldCalculator
  - StructuralEventAnalyzer
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.stats import norm as sp_norm
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

import config as cfg


# ═════════════════════════════════════════════════════════════════════════════
# 1. CurveDecomposer
# ═════════════════════════════════════════════════════════════════════════════
class CurveDecomposer:
    """Decomposes futures curve shifts into level, slope, and curvature."""
    
    @staticmethod
    def compute_curve_shift(pre_curve: Dict[str, float], post_curve: Dict[str, float]) -> Dict[str, float]:
        """Compute the absolute shift per tenor."""
        return {tenor: post_curve[tenor] - pre_curve.get(tenor, np.nan) for tenor in post_curve}

    @staticmethod
    def decompose_level_slope_curvature(shift: Dict[str, float]) -> Tuple[float, float, float]:
        """
        Level = mean shift across tenors
        Slope = shift(12M) - shift(1M)
        Curvature = shift(1M) + shift(12M) - 2 * shift(6M)
        """
        level = np.nanmean(list(shift.values()))
        slope = shift.get("12M", 0.0) - shift.get("1M", 0.0)
        curvature = shift.get("1M", 0.0) + shift.get("12M", 0.0) - 2 * shift.get("6M", 0.0)
        return level, slope, curvature

    @staticmethod
    def persistence_ratio(shift: Dict[str, float]) -> float:
        """ratio = shift(12M) / shift(1M). ≈1 = permanent, ≈0 = transient."""
        s1 = shift.get("1M", 0.0)
        s12 = shift.get("12M", 0.0)
        if abs(s1) < 1e-8:
            return 0.0
        return s12 / s1


# ═════════════════════════════════════════════════════════════════════════════
# 2. ImpliedDistribution
# ═════════════════════════════════════════════════════════════════════════════
class ImpliedDistribution:
    """Extracts risk-neutral density and implied moments from options surface."""
    
    @staticmethod
    def breeden_litzenberger(strikes: np.ndarray, call_prices: np.ndarray, r: float, T: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        d²C/dK² -> risk-neutral PDF.
        Uses second-order central finite differences.
        """
        if len(strikes) < 3:
            return strikes, np.zeros_like(strikes)
        
        dK = np.diff(strikes)
        if not np.allclose(dK, dK[0], rtol=1e-4):
            # If not uniformly spaced, we use midpoints for stable derivative
            pass
        
        # 1st deriv at midpoints:
        dC = np.diff(call_prices) / dK
        K_mid = strikes[:-1] + dK / 2.0
        
        # 2nd deriv:
        dK_mid = np.diff(K_mid)
        ddC = np.diff(dC) / dK_mid
        
        K_inner = strikes[1:-1]
        pdf = np.exp(r * T) * ddC
        pdf = np.maximum(pdf, 0) # PDF must be non-negative
        
        # Normalize
        norm = np.trapezoid(pdf, K_inner)
        if norm > 0:
            pdf /= norm
            
        return K_inner, pdf

    @staticmethod
    def extract_moments(pdf: np.ndarray, strikes: np.ndarray) -> Tuple[float, float, float, float]:
        """mean, variance, skewness, kurtosis"""
        if np.sum(pdf) == 0:
            return 0.0, 0.0, 0.0, 0.0
            
        dx = np.gradient(strikes) if len(strikes) > 1 else 1.0
        norm = np.sum(pdf * dx)
        if norm == 0: norm = 1.0
            
        mean = np.sum(strikes * pdf * dx) / norm
        var = np.sum((strikes - mean)**2 * pdf * dx) / norm
        std = np.sqrt(var) if var > 0 else 1.0
        
        skew = np.sum(((strikes - mean)/std)**3 * pdf * dx) / norm
        kurt = np.sum(((strikes - mean)/std)**4 * pdf * dx) / norm
        
        return mean, var, skew, kurt

    @staticmethod
    def compare_distributions(pdf_pre: np.ndarray, strikes_pre: np.ndarray, pdf_post: np.ndarray, strikes_post: np.ndarray) -> Tuple[float, float, float]:
        m1, v1, s1, _ = ImpliedDistribution.extract_moments(pdf_pre, strikes_pre)
        m2, v2, s2, _ = ImpliedDistribution.extract_moments(pdf_post, strikes_post)
        return m2 - m1, v2 - v1, s2 - s1
        
    @staticmethod
    def _black76_delta_call(F: float, K: float, T: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0:
            return 1.0 if F > K else 0.0
        d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
        return sp_norm.cdf(d1)
        
    @staticmethod
    def _black76_delta_put(F: float, K: float, T: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0:
            return -1.0 if F < K else 0.0
        d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
        return sp_norm.cdf(d1) - 1.0

    @staticmethod
    def compute_risk_reversal(iv_surface: pd.DataFrame, F_atm: float, T: float, delta: float = 0.25) -> float:
        """IV(25Δ call) - IV(25Δ put), model-free testing."""
        calls_delta = [ImpliedDistribution._black76_delta_call(F_atm, r['strike'], T, r['iv']) for _, r in iv_surface.iterrows()]
        puts_delta = [ImpliedDistribution._black76_delta_put(F_atm, r['strike'], T, r['iv']) for _, r in iv_surface.iterrows()]
        
        calls_iv = iv_surface['iv'].values
        
        try:
            # Note: Call delta decreases with strike [1 to 0]. Interp requires strictly increasing x-coords.
            call_25_iv = np.interp(delta, calls_delta[::-1], calls_iv[::-1])
            # Put delta goes from 0 to -1 as strike increases, so it's also decreasing. Need to reverse.
            put_25_iv = np.interp(-delta, puts_delta[::-1], calls_iv[::-1])
            return call_25_iv - put_25_iv
        except:
            return 0.0

    @staticmethod
    def compute_butterfly(iv_surface: pd.DataFrame, F_atm: float, T: float, delta: float = 0.25) -> float:
        """0.5 * (IV_25call + IV_25put) - IV_ATM"""
        calls_delta = [ImpliedDistribution._black76_delta_call(F_atm, r['strike'], T, r['iv']) for _, r in iv_surface.iterrows()]
        puts_delta = [ImpliedDistribution._black76_delta_put(F_atm, r['strike'], T, r['iv']) for _, r in iv_surface.iterrows()]
        
        calls_iv = iv_surface['iv'].values
        
        try:
            call_25_iv = np.interp(delta, calls_delta[::-1], calls_iv[::-1])
            put_25_iv = np.interp(-delta, puts_delta[::-1], calls_iv[::-1])
            idx_atm = (iv_surface['moneyness'] - 1.0).abs().idxmin()
            atm_iv = iv_surface.loc[idx_atm, 'iv']
            return 0.5 * (call_25_iv + put_25_iv) - atm_iv
        except:
            return 0.0


# ═════════════════════════════════════════════════════════════════════════════
# 3. ConvenienceYieldCalculator
# ═════════════════════════════════════════════════════════════════════════════
class ConvenienceYieldCalculator:
    """
    y = r + c - ln(F/S)/T
    
    Terminology: always "implied convenience yield", acknowledging it's a composite residual.
    Contains at least three unseparated components:
    1. True Kaldor convenience yield (value of holding physical inventory)
    2. Shipping optionality (cargo diversion right, JKM <-> TTF)
    3. Geographic basis risk (JKM DES Japan/Korea Asia-Pacific premium)
    
    State-dependent bias: all three contaminating components amplify in tight markets, 
    so the residual overstates the textbook concept in tight regimes and understates 
    it in loose regimes.
    
    Usage boundary: suitable as a directional tightness classifier (tight / neutral / loose) 
    only; coefficient magnitudes on delta_implied_convenience_yield in the interaction 
    regression must not be interpreted as cross-state structural elasticities.
    
    Note: Calibration against real terminal utilization data (e.g., GIIGNL annual reports) 
    would be required for production use.
    """

    @staticmethod
    def compute_implied_yield(spot: float, futures: float, T: float, r: float, c: float) -> float:
        if spot <= 0 or futures <= 0 or T <= 0:
            return 0.0
        return r + c - math.log(futures / spot) / T


# ═════════════════════════════════════════════════════════════════════════════
# 4. StructuralEventAnalyzer
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class StructuralEventResult:
    # Identity
    event_idx: int
    event_time: pd.Timestamp
    headline: str
    keyword: str
    is_placebo: bool                        # Gap 1: placebo flag
    # Market state (tightness — for Prediction 1)
    market_tightness: float                 # storage + backwardation only
    storage_vs_norm: float
    # Transmission condition (separate — for Prediction 3, Gap 2)
    gas_is_marginal: bool
    spark_spread: float
    season: str
    # Spot response
    spot_forward_return: float
    # Curve responses
    curve_level_shift: float
    curve_slope_change: float
    curve_curvature_change: float
    persistence_ratio: float
    # Options responses (Gap 4: RR and B-L separated)
    delta_implied_vol: float
    delta_risk_reversal: float              # model-free, for testing
    delta_bl_skewness: float                # B-L third moment, for visualization
    # Convenience yield
    delta_implied_convenience_yield: float  # Gap 5: "implied" prefix
    # Structural attributes
    reactor_capacity_mw: Optional[float]
    credibility_tier: int
    liquidity_score: float

class StructuralEventAnalyzer:

    @staticmethod
    def analyze_event(
        event_idx: int,
        event_row: pd.Series,
        spot_df: pd.DataFrame,
        futures_curves: Dict[str, pd.DataFrame],
        options_surfaces: Dict[str, Dict[str, pd.DataFrame]],
        state_df: pd.DataFrame,
    ) -> Optional[StructuralEventResult]:
        """Runs full analysis for a single event."""
        
        event_time = event_row["timestamp"]
        keyword = event_row["keyword"]
        is_placebo = event_row.get("is_placebo", False)
        
        # Determine Pre and Post indices
        window_td = pd.Timedelta(minutes=cfg.FORWARD_WINDOW_MINUTES)
        pre_bars = spot_df.index[spot_df.index <= event_time]
        post_bars = spot_df.index[spot_df.index >= event_time + window_td]
        
        if len(pre_bars) == 0 or len(post_bars) == 0:
            return None
            
        pre_ts = pre_bars[-1]
        post_ts = post_bars[0]
        
        # Spot Response
        spot_pre = spot_df.loc[pre_ts, "Close"]
        spot_post = spot_df.loc[post_ts, "Close"]
        spot_fwd_ret = (spot_post / spot_pre) - 1.0 if spot_pre > 0 else 0.0

        # Market State
        state_bars = state_df.index[state_df.index <= event_time]
        if len(state_bars) == 0:
            return None
        state_pre = state_df.loc[state_bars[-1]]
        
        # Curve Responses
        pre_curve = {}
        post_curve = {}
        eidx_str = str(event_idx)
        
        for tenor, fdf in futures_curves.items():
            row = fdf[fdf['event_idx'] == event_idx]
            if not row.empty:
                pre_curve[tenor] = float(row.iloc[0]['pre'])
                post_curve[tenor] = float(row.iloc[0]['post'])
                
        if len(pre_curve) == 0: return None
            
        shifts = CurveDecomposer.compute_curve_shift(pre_curve, post_curve)
        lvl, slp, curv = CurveDecomposer.decompose_level_slope_curvature(shifts)
        pers_ratio = CurveDecomposer.persistence_ratio(shifts)
        
        # Implied Convenience Yield (1M Tenor as proxy)
        T_1m = cfg.TENOR_TO_YEARS["1M"]
        c = 0.03 # approximated fixed baseline cost for yield calc
        if "storage_level" in state_df.columns:
            util = state_pre["storage_level"]
            excess = max(0.0, util - cfg.STORAGE_THRESHOLD)
            c = cfg.STORAGE_COST_BASE * (1.0 + cfg.STORAGE_COST_SCALE * excess)

        iy_pre = ConvenienceYieldCalculator.compute_implied_yield(spot_pre, pre_curve.get("1M", spot_pre), T_1m, cfg.RISK_FREE_RATE, c)
        iy_post = ConvenienceYieldCalculator.compute_implied_yield(spot_post, post_curve.get("1M", spot_post), T_1m, cfg.RISK_FREE_RATE, c)
        delta_icy = iy_post - iy_pre

        # Options Responses (we use 1M tenor, if available)
        delta_iv = 0.0
        delta_rr = 0.0
        delta_bl_skew = 0.0
        
        if eidx_str in options_surfaces and "1M" in options_surfaces[eidx_str]:
            surf_df = options_surfaces[eidx_str]["1M"]
            surf_pre = surf_df[surf_df["period"] == "pre"].copy().reset_index(drop=True)
            surf_post = surf_df[surf_df["period"] == "post"].copy().reset_index(drop=True)
            
            if not surf_pre.empty and not surf_post.empty:
                idx_atm_pre = (surf_pre['moneyness'] - 1.0).abs().idxmin()
                idx_atm_post = (surf_post['moneyness'] - 1.0).abs().idxmin()
                
                atm_iv_pre = surf_pre.loc[idx_atm_pre, 'iv']
                atm_iv_post = surf_post.loc[idx_atm_post, 'iv']
                delta_iv = atm_iv_post - atm_iv_pre
                
                rr_pre = ImpliedDistribution.compute_risk_reversal(surf_pre, pre_curve["1M"], T_1m)
                rr_post = ImpliedDistribution.compute_risk_reversal(surf_post, post_curve["1M"], T_1m)
                delta_rr = rr_post - rr_pre
                
                K_pre, pdf_pre = ImpliedDistribution.breeden_litzenberger(surf_pre['strike'].values, surf_pre['call_price'].values, cfg.RISK_FREE_RATE, T_1m)
                K_post, pdf_post = ImpliedDistribution.breeden_litzenberger(surf_post['strike'].values, surf_post['call_price'].values, cfg.RISK_FREE_RATE, T_1m)
                
                _, _, skew_delta = ImpliedDistribution.compare_distributions(pdf_pre, K_pre, pdf_post, K_post)
                delta_bl_skew = skew_delta
        
        # Attributes
        capacity = event_row.get("reactor_capacity_mw")
        capacity = float(capacity) if pd.notna(capacity) and capacity is not None else None
        
        cred_tier = event_row.get("credibility_tier")
        cred_tier = int(cred_tier) if pd.notna(cred_tier) and cred_tier is not None else 0

        return StructuralEventResult(
            event_idx=event_idx,
            event_time=event_time,
            headline=event_row["headline"],
            keyword=keyword,
            is_placebo=is_placebo,
            market_tightness=float(state_pre.get("market_tightness_score", 0.0)),
            storage_vs_norm=float(state_pre.get("storage_vs_norm", 0.0)),
            gas_is_marginal=bool(state_pre.get("gas_is_marginal", False)),
            spark_spread=float(state_pre.get("spark_spread", 0.0)),
            season=event_row.get("season", "unknown"),
            spot_forward_return=spot_fwd_ret,
            curve_level_shift=lvl,
            curve_slope_change=slp,
            curve_curvature_change=curv,
            persistence_ratio=pers_ratio,
            delta_implied_vol=delta_iv,
            delta_risk_reversal=delta_rr,
            delta_bl_skewness=delta_bl_skew,
            delta_implied_convenience_yield=delta_icy,
            reactor_capacity_mw=capacity,
            credibility_tier=cred_tier,
            liquidity_score=float(state_pre.get("liquidity_score", 1.0))
        )

    @staticmethod
    def run_interaction_regression(results: List[StructuralEventResult]) -> sm.regression.linear_model.RegressionResultsWrapper:
        """
        R = α + β₁·KeywordDummy + β₂·Tightness + β₃·(KeywordDummy × Tightness) 
            + β₄·GasMarginal + β₅·(NuclearRestartDummy × GasMarginal)
            + β₆·LiquidityScore + ε
        """
        df = pd.DataFrame([vars(r) for r in results if not r.is_placebo])
        if df.empty:
            return None
            
        keywords = df['keyword'].unique()
        if len(keywords) > 1:
            dummy_k = pd.get_dummies(df['keyword'], drop_first=True, dtype=float)
            df = pd.concat([df, dummy_k], axis=1)
            kw_cols = list(dummy_k.columns)
        else:
            df['KwDummy'] = 1.0
            kw_cols = ['KwDummy']

        df['NuclearRestartDummy'] = (df['keyword'] == 'Nuclear Restart').astype(float)
        
        for kw in kw_cols:
            df[f'Int_{kw}_Tightness'] = df[kw] * df['market_tightness']
        
        df['Int_Nuclear_GasMarginal'] = df['NuclearRestartDummy'] * df['gas_is_marginal'].astype(float)
        
        exog_cols = kw_cols + ['market_tightness'] + [f'Int_{kw}_Tightness' for kw in kw_cols] + \
                    ['gas_is_marginal', 'Int_Nuclear_GasMarginal', 'liquidity_score']
        
        exog_cols = [c for c in exog_cols if c in df.columns]
        
        X = df[exog_cols].astype(float)
        X = sm.add_constant(X)
        y = df['spot_forward_return'].astype(float)
        
        model = sm.OLS(y, X).fit()
        return model

    @staticmethod
    def check_identification_quality(results: List[StructuralEventResult]) -> Dict[str, any]:
        """Pre-regression diagnostics."""
        df = pd.DataFrame([vars(r) for r in results])
        if df.empty:
            return {}
            
        X = df[['market_tightness', 'liquidity_score']].astype(float)
        X = sm.add_constant(X)
        vif_tight = variance_inflation_factor(X.values, 1) if X.shape[1] > 1 else 1.0
        vif_liq = variance_inflation_factor(X.values, 2) if X.shape[1] > 2 else 1.0

        corr = df['market_tightness'].corr(df['liquidity_score'])
        
        responses = df[['spot_forward_return', 'curve_level_shift', 'curve_slope_change', 
                        'delta_implied_vol', 'delta_risk_reversal', 'delta_bl_skewness', 
                        'delta_implied_convenience_yield']].astype(float)
        
        cond_num = np.linalg.cond(responses.cov().values) if len(responses) > 10 else np.nan
        low_liq_count = (df['liquidity_score'] < 0.2).sum()
        
        return {
            "vif_tightness": vif_tight,
            "vif_liquidity": vif_liq,
            "correlation_tight_liq": corr,
            "vif_warning": vif_tight > 5 or vif_liq > 5,
            "corr_warning": abs(corr) > 0.6,
            "condition_number": cond_num,
            "low_liquidity_n": int(low_liq_count)
        }

    @staticmethod
    def compute_conditional_stats(results: List[StructuralEventResult], tightness_bins: List[str] = ["tight", "neutral", "loose"]) -> pd.DataFrame:
        df = pd.DataFrame([vars(r) for r in results])
        if df.empty:
            return pd.DataFrame()
            
        df['tightness_state'] = 'neutral'
        df.loc[df['market_tightness'] > cfg.TIGHTNESS_TIGHT_THRESHOLD, 'tightness_state'] = 'tight'
        df.loc[df['market_tightness'] < cfg.TIGHTNESS_LOOSE_THRESHOLD, 'tightness_state'] = 'loose'
        
        grouped = df.groupby(['keyword', 'tightness_state'])['spot_forward_return'].agg(['mean', 'std', 'count']).reset_index()
        return grouped


# ═════════════════════════════════════════════════════════════════════════════
# 5. StatisticalTester
# ═════════════════════════════════════════════════════════════════════════════
class StatisticalTester:
    """
    Implements the Statistical Testing Framework (Gap 3).
    Layer 1: Hotelling's T^2 joint test on the 7-dimensional response vector.
    Layer 2: Benjamini-Hochberg FDR control on individual dimensions.
    """

    RESPONSE_DIMS = [
        'spot_forward_return',
        'curve_level_shift',
        'curve_slope_change',
        'delta_implied_vol',
        'delta_risk_reversal',
        'delta_bl_skewness',
        'delta_implied_convenience_yield'
    ]

    @staticmethod
    def hotellings_t2(data: pd.DataFrame, columns: List[str] = None) -> Tuple[float, float]:
        """
        Hotelling's T-squared test for one-sample mean: H0: mu = 0.
        Returns (T2_stat, p_value).
        """
        # We need statsmodels or spicy/numpy to compute F-test from T2
        from scipy.stats import f as sp_f
        
        cols = columns or StatisticalTester.RESPONSE_DIMS
        X = data[cols].dropna().values
        n, p = X.shape
        if n <= p:
            return np.nan, np.nan
            
        x_bar = np.mean(X, axis=0)
        cov_matrix = np.cov(X, rowvar=False)
        try:
            inv_cov = np.linalg.pinv(cov_matrix)
        except np.linalg.LinAlgError:
            return np.nan, np.nan
            
        t2_stat = n * x_bar.T @ inv_cov @ x_bar
        
        rank = np.linalg.matrix_rank(cov_matrix)
        if rank == 0:
            return 0.0, 1.0
        
        # Convert to F-statistic using the actual rank
        f_stat = t2_stat * (n - rank) / (rank * (n - 1))
        df1 = rank
        df2 = n - rank
        p_val = sp_f.sf(f_stat, df1, df2)
        
        return float(t2_stat), float(p_val)

    @staticmethod
    def ttest_1samp_dims(data: pd.DataFrame, columns: List[str] = None) -> Dict[str, float]:
        """
        Runs one-sample t-test (H0: mu=0) for each dimension.
        Returns dict of dimension -> raw_p_value.
        """
        from scipy.stats import ttest_1samp
        
        cols = columns or StatisticalTester.RESPONSE_DIMS
        p_vals = {}
        for col in cols:
            series = data[col].dropna()
            if len(series) < 2:
                p_vals[col] = np.nan
            else:
                _, p = ttest_1samp(series, 0.0)
                p_vals[col] = float(p)
        return p_vals

    @staticmethod
    def apply_fdr_bh(p_values: Dict[str, float], alpha: float = 0.05) -> pd.DataFrame:
        """
        Applies Benjamini-Hochberg FDR control.
        """
        dims = list(p_values.keys())
        raw_p = [p_values[d] for d in dims]
        
        valid_idx = [i for i, p in enumerate(raw_p) if not np.isnan(p)]
        valid_p = [raw_p[i] for i in valid_idx]
        
        if not valid_p:
            df = pd.DataFrame({"Dimension": dims, "Raw_p": raw_p, "BH_adj_p": np.nan, "Significant": False})
            return df
            
        reject, pvals_corrected, _, _ = multipletests(valid_p, alpha=alpha, method='fdr_bh')
        
        adj_p = np.full(len(raw_p), np.nan)
        sig = np.zeros(len(raw_p), dtype=bool)
        
        for i, v_idx in enumerate(valid_idx):
            adj_p[v_idx] = pvals_corrected[i]
            sig[v_idx] = reject[i]
            
        return pd.DataFrame({
            "Dimension": dims,
            "Raw_p": raw_p,
            "BH_adj_p": adj_p,
            "Significant": sig
        })

