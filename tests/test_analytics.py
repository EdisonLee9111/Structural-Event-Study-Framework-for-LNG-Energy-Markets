import pytest
import numpy as np
import pandas as pd
import math

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analytics import CurveDecomposer, ImpliedDistribution, ConvenienceYieldCalculator
import config as cfg

def test_curve_decomposition_sums_correctly():
    """
    Test 5: Curve decomposition sums correctly.
    Verifies that compute_curve_shift, decompose_level_slope_curvature, 
    and persistence_ratio work appropriately.
    """
    pre_curve = {"1M": 10.0, "3M": 11.0, "6M": 12.0, "12M": 14.0}
    post_curve = {"1M": 12.0, "3M": 13.0, "6M": 15.0, "12M": 18.0}
    
    shifts = CurveDecomposer.compute_curve_shift(pre_curve, post_curve)
    assert shifts["1M"] == 2.0
    assert shifts["3M"] == 2.0
    assert shifts["6M"] == 3.0
    assert shifts["12M"] == 4.0
    
    lvl, slp, curv = CurveDecomposer.decompose_level_slope_curvature(shifts)
    
    # Level = mean shift = (2+2+3+4)/4 = 11/4 = 2.75
    assert np.isclose(lvl, 2.75)
    
    # Slope = shift(12M) - shift(1M) = 4 - 2 = 2.0
    assert np.isclose(slp, 2.0)
    
    # Curvature = shift(1M) + shift(12M) - 2*shift(6M) = 2 + 4 - 2*3 = 0.0
    assert np.isclose(curv, 0.0)
    
    # Persistence ratio = shift(12M) / shift(1M) = 4 / 2 = 2.0
    pers_ratio = CurveDecomposer.persistence_ratio(shifts)
    assert np.isclose(pers_ratio, 2.0)

def test_futures_no_arbitrage():
    """
    Test 1: Futures no-arbitrage.
    F = S * exp((r + c - y) * T) -> ln(F/S)/T = r + c - y  =>  y = r + c - ln(F/S)/T
    We test the relationship holds.
    """
    S = 10.0
    T = 0.25  # 3M
    r = 0.05
    c = 0.02
    y = 0.03
    
    # Calculate No-arbitrage Future Price
    F = S * math.exp((r + c - y) * T)
    
    # Recalculate y using ConvenienceYieldCalculator
    implied_y = ConvenienceYieldCalculator.compute_implied_yield(S, F, T, r, c)
    assert np.isclose(y, implied_y), f"Expected yield {y}, got {implied_y}"

def test_put_call_parity():
    """
    Test 2: Put-call parity for Black-76.
    C - P = (F - K) * e^{-rT}
    This is essentially verifying our synthetic prices if we generated them, 
    but we will mock the synthetic formula here and verify.
    """
    from scipy.stats import norm

    def black76_price(F, K, T, r, sigma, option_type="call"):
        d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == "call":
            return np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))
        else:
            return np.exp(-r * T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

    F = 10.0
    K = 10.5
    T = 0.5
    r = 0.05
    sigma = 0.3
    
    call_price = black76_price(F, K, T, r, sigma, "call")
    put_price = black76_price(F, K, T, r, sigma, "put")
    
    left_side = call_price - put_price
    right_side = (F - K) * np.exp(-r * T)
    
    assert np.isclose(left_side, right_side), f"Parity failed: C-P={left_side}, RHS={right_side}"

def test_breeden_litzenberger_recovers_known_distribution():
    """
    Test 3: B-L recovers known distribution from Black-76.
    """
    from scipy.stats import norm
    
    F = 10.0
    r = 0.0
    T = 1.0
    sigma = 0.2
    
    strikes = np.linspace(1.0, 30.0, 501)
    
    def black76_call(F, K, T, r, sigma):
        if T <= 0: return max(F - K, 0)
        d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))
        
    call_prices = np.array([black76_call(F, k, T, r, sigma) for k in strikes])
    
    K_inner, pdf = ImpliedDistribution.breeden_litzenberger(strikes, call_prices, r, T)
    
    # B-76 underlying asset X_T at maturity is Lognormal, but B-76 is usually on Futures.
    # Risk-neutral PDF of F_T given F_0 is Lognormal.
    expected_mean = F
    # The expected variance of a lognormal variable F_T = F_0 * exp(-sigma^2 T / 2 + sigma W)
    expected_var = (F**2) * (math.exp(sigma**2 * T) - 1)
    
    m, v, s, k = ImpliedDistribution.extract_moments(pdf, K_inner)
    
    # Checking Mean
    assert np.isclose(m, expected_mean, rtol=1e-2)
    # Checking Variance
    assert np.isclose(v, expected_var, rtol=1e-2)

def test_convenience_yield_sensitivity():
    """
    Test 6: Convenience yield sensitivity
    Verify qualitatively different behavior at utilization 80% vs 95%.
    Valides non-linear storage cost.
    Cost logic: base * (1 + scale * max(0, util - threshold))
    threshold = 0.90, base = cfg.STORAGE_COST_BASE
    """
    S = 10.0
    F = 10.5
    T = 1/12
    r = 0.05
    
    base_cost = cfg.STORAGE_COST_BASE
    scale = cfg.STORAGE_COST_SCALE
    threshold = cfg.STORAGE_THRESHOLD
    
    util_80 = 0.80
    util_95 = 0.95
    
    c_80 = base_cost * (1.0 + scale * max(0.0, util_80 - threshold))
    c_95 = base_cost * (1.0 + scale * max(0.0, util_95 - threshold))
    
    y_80 = ConvenienceYieldCalculator.compute_implied_yield(S, F, T, r, c_80)
    y_95 = ConvenienceYieldCalculator.compute_implied_yield(S, F, T, r, c_95)
    
    # 80% is below threshold of 0.90, so cost should just be base_cost.
    assert np.isclose(c_80, base_cost)
    
    # 95% is above threshold, so cost c_95 > c_80
    assert c_95 > c_80
    
    # Since cost increases, the implied convenience yield y = r + c - spread also increases
    # in tight markets to justify the same contango F/S.
    assert y_95 > y_80
