import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

import config as cfg
from generators import generate_all
from analytics import StructuralEventAnalyzer, StatisticalTester, ImpliedDistribution
from tests.test_falsification import (
    test_prediction_0_placebo,
    test_prediction_1_tightness_interaction,
    test_prediction_2_capacity_proportionality,
    test_prediction_3_marginal_fuel_condition
)

# Set styling
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

def run_falsification_tests(event_results):
    print("\n--- Running Falsification Tests (Predictions 0-3) ---")
    tests = [
        ("Prediction 0 (Placebo Non-Significance)", test_prediction_0_placebo),
        ("Prediction 1 (Tightness Interaction)", test_prediction_1_tightness_interaction),
        ("Prediction 2 (Capacity Proportionality)", test_prediction_2_capacity_proportionality),
        ("Prediction 3 (Marginal Fuel Condition)", test_prediction_3_marginal_fuel_condition),
    ]
    
    for name, func in tests:
        try:
            func(event_results)
            print(f"[PASS] {name}")
        except Exception as e:
            print(f"[FAIL] {name}: {str(e)}")

def plot_conditional_signal_surface(results, output_dir):
    df = pd.DataFrame([vars(r) for r in results if not r.is_placebo])
    if df.empty: return
    
    df['tightness_state'] = 'neutral'
    df.loc[df['market_tightness'] > cfg.TIGHTNESS_TIGHT_THRESHOLD, 'tightness_state'] = 'tight'
    df.loc[df['market_tightness'] < cfg.TIGHTNESS_LOOSE_THRESHOLD, 'tightness_state'] = 'loose'
    
    metrics = [
        'spot_forward_return', 
        'curve_level_shift', 
        'delta_implied_vol', 
        'delta_implied_convenience_yield'
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    for i, state in enumerate(['tight', 'loose']):
        subset = df[df['tightness_state'] == state]
        if subset.empty: continue
        
        agg = subset.groupby('keyword')[metrics].mean()
        
        sns.heatmap(agg, annot=True, fmt=".4f", cmap="coolwarm", center=0, ax=axes[i], cbar=(i==1))
        axes[i].set_title(f"State: {state.capitalize()} Market")
        axes[i].set_ylabel("")
        
    plt.suptitle("Conditional Signal Surface: Metric Impact by Keyword and Market State")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "conditional_signal_surface.png"))
    plt.close()

def plot_futures_curve_shift(bundle, results, output_dir):
    real_results = [r for r in results if not r.is_placebo]
    df = pd.DataFrame([vars(r) for r in real_results])
    
    tenors = cfg.FUTURES_TENORS
    numeric_tenors = [cfg.TENOR_TO_YEARS[t] for t in tenors]
    
    keywords = df['keyword'].unique()
    fig, axes = plt.subplots(1, len(keywords), figsize=(5 * len(keywords), 5), sharey=True)
    if len(keywords) == 1: axes = [axes]
    
    for i, kw in enumerate(keywords):
        kw_events = df[df['keyword'] == kw]
        if kw_events.empty: continue
        med_idx = kw_events['curve_level_shift'].abs().sort_values().index[len(kw_events)//2]
        event_idx = kw_events.loc[med_idx, 'event_idx']
        
        pre_curve = []
        post_curve = []
        for t in tenors:
            row = bundle.futures_curves[t][bundle.futures_curves[t]['event_idx'] == event_idx].iloc[0]
            pre_curve.append(row['pre'])
            post_curve.append(row['post'])
            
        axes[i].plot(numeric_tenors, pre_curve, marker='o', label='Pre-Event', linestyle='--')
        axes[i].plot(numeric_tenors, post_curve, marker='s', label='Post-Event', linestyle='-')
        axes[i].set_title(f"{kw} (Representative Event)")
        axes[i].set_xlabel("Years to Maturity")
        if i == 0: axes[i].set_ylabel("Price")
        axes[i].legend()
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "futures_curve_shifts.png"))
    plt.close()

def plot_implied_density_comparison(bundle, results, output_dir):
    real_results = [r for r in results if not r.is_placebo]
    df = pd.DataFrame([vars(r) for r in real_results])
    
    keywords = df['keyword'].unique()
    fig, axes = plt.subplots(1, len(keywords), figsize=(5 * len(keywords), 5), sharey=True)
    if len(keywords) == 1: axes = [axes]
    
    for i, kw in enumerate(keywords):
        kw_events = df[df['keyword'] == kw]
        if kw_events.empty: continue
        
        max_idx = kw_events['delta_bl_skewness'].abs().idxmax()
        event_idx = kw_events.loc[max_idx, 'event_idx']
        
        try:
            surf_df = bundle.options_surfaces[str(event_idx)]["1M"]
            surf_pre = surf_df[surf_df["period"] == "pre"].copy().reset_index(drop=True)
            surf_post = surf_df[surf_df["period"] == "post"].copy().reset_index(drop=True)
            
            T_1m = cfg.TENOR_TO_YEARS["1M"]
            K_pre, pdf_pre = ImpliedDistribution.breeden_litzenberger(surf_pre['strike'].values, surf_pre['call_price'].values, cfg.RISK_FREE_RATE, T_1m)
            K_post, pdf_post = ImpliedDistribution.breeden_litzenberger(surf_post['strike'].values, surf_post['call_price'].values, cfg.RISK_FREE_RATE, T_1m)
            
            axes[i].plot(K_pre, pdf_pre, label='Pre-Event', color='blue')
            axes[i].plot(K_post, pdf_post, label='Post-Event', color='red', linestyle='--')
            axes[i].set_title(f"{kw} (Max Skew Event)")
            axes[i].set_xlabel("Strike")
            if i == 0: axes[i].set_ylabel("Probability Density")
            axes[i].legend()
        except KeyError:
            axes[i].set_title(f"{kw} (No Options Data)")
            
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "implied_density_comparison.png"))
    plt.close()

def plot_level_slope_curvature(results, output_dir):
    df = pd.DataFrame([vars(r) for r in results if not r.is_placebo])
    if df.empty: return
    
    if 'curve_curvature_change' in df.columns:
        agg = df.groupby('keyword')[['curve_level_shift', 'curve_slope_change', 'curve_curvature_change']].mean()
    else:
        agg = df.groupby('keyword')[['curve_level_shift', 'curve_slope_change']].mean()
        
    agg.plot(kind='bar', figsize=(10, 6))
    plt.title("Level / Slope / Curvature Decomposition by Keyword")
    plt.ylabel("Average Shift")
    plt.axhline(0, color='black', linewidth=1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "level_slope_curvature.png"))
    plt.close()

def plot_unconditional_vs_conditional(results, output_dir):
    df = pd.DataFrame([vars(r) for r in results if not r.is_placebo])
    if df.empty: return
    
    df['tightness_state'] = 'neutral'
    df.loc[df['market_tightness'] > cfg.TIGHTNESS_TIGHT_THRESHOLD, 'tightness_state'] = 'tight'
    df.loc[df['market_tightness'] < cfg.TIGHTNESS_LOOSE_THRESHOLD, 'tightness_state'] = 'loose'
    
    uc_agg = df.groupby('keyword')['spot_forward_return'].mean().rename('Unconditional')
    c_agg_tight = df[df['tightness_state'] == 'tight'].groupby('keyword')['spot_forward_return'].mean().rename('Conditional (Tight)')
    c_agg_loose = df[df['tightness_state'] == 'loose'].groupby('keyword')['spot_forward_return'].mean().rename('Conditional (Loose)')
    
    comb = pd.concat([uc_agg, c_agg_tight, c_agg_loose], axis=1)
    
    comb.plot(kind='bar', figsize=(10, 6))
    plt.title("Unconditional vs Conditional Spot Return by Keyword")
    plt.ylabel("Mean Forward Return")
    plt.axhline(0, color='black', linewidth=1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "unconditional_vs_conditional.png"))
    plt.close()

def print_structural_summary_table(results):
    real_results = [r for r in results if not r.is_placebo]
    df = pd.DataFrame([vars(r) for r in real_results])
    
    print("\n--- Structural Event Study Results (Layer 1 & 2 Testing) ---")
    
    for kw in df['keyword'].unique():
        kw_df = df[df['keyword'] == kw]
        t2_stat, p_val_t2 = StatisticalTester.hotellings_t2(kw_df)
        
        print(f"\nKeyword: {kw}")
        print(f"Hotelling's T^2 p-value: {p_val_t2:.4f} " + ("(Significant)" if p_val_t2 < 0.05 else "(Not Significant)"))
        
        p_vals_dict = StatisticalTester.ttest_1samp_dims(kw_df)
        bh_df = StatisticalTester.apply_fdr_bh(p_vals_dict)
        
        means = kw_df[StatisticalTester.RESPONSE_DIMS].mean().to_dict()
        bh_df['Mean Impact'] = bh_df['Dimension'].map(means)
        
        display_df = bh_df[['Dimension', 'Mean Impact', 'Raw_p', 'BH_adj_p', 'Significant']].copy()
        print(tabulate(display_df, headers='keys', tablefmt='psql', showindex=False, floatfmt=".4f"))

def main():
    print("""
Pipeline: theory -> generators -> analytics -> main
YOU ARE HERE
""")
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    print("1. Generating synthetic dataset...")
    bundle = generate_all(seed=cfg.RANDOM_SEED)
    
    print("2. Running structural event study...")
    results = []
    for idx, row in bundle.all_events_df.iterrows():
        res = StructuralEventAnalyzer.analyze_event(
            event_idx=idx,
            event_row=row,
            spot_df=bundle.spot_prices[cfg.PRIMARY_ASSET],
            futures_curves=bundle.futures_curves,
            options_surfaces=bundle.options_surfaces,
            state_df=bundle.market_state
        )
        if res is not None:
            results.append(res)
            
    print(f"   Processed {len(results)} events (including placebos).")
    
    print("\n3. Evaluating Statistical Framework...")
    print_structural_summary_table(results)
    
    print("\n4. Running Falsification Predictions...")
    run_falsification_tests(results)
    
    print("\n5. Generating Visualizations...")
    plot_conditional_signal_surface(results, output_dir)
    plot_futures_curve_shift(bundle, results, output_dir)
    plot_implied_density_comparison(bundle, results, output_dir)
    plot_level_slope_curvature(results, output_dir)
    plot_unconditional_vs_conditional(results, output_dir)
    
    print(f"\nPipeline complete! All visualizations saved to '{output_dir}/'")

if __name__ == "__main__":
    main()
