"""
Exemple Complet d'Utilisation - Composite Score QAAF v2.0

Ce script démontre l'utilisation complète du composite_score
dans un workflow réaliste de stratégie de trading.

Author: QAAF Team
Date: 2025-10-24
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime

# Add paths (adjust to your project structure)
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'tools'))

# Import composite score
from composite_score import (
    CompositeScore,
    quick_composite,
    composite_signal,
    validate_metrics_alignment,
    get_info
)


# ============================================================
# Scenario 1: Basic Usage with Synthetic Data
# ============================================================

def scenario_1_basic():
    """Scénario basique avec données synthétiques."""
    print("=" * 60)
    print("Scénario 1: Basic Usage")
    print("=" * 60)
    
    # Generate synthetic metrics
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=252, freq='D')  # 1 year
    
    vol_ratio = pd.Series(
        np.random.uniform(0.8, 1.5, 252),
        index=dates,
        name='vol_ratio'
    )
    
    bound_coherence = pd.Series(
        np.random.uniform(0.3, 0.7, 252),
        index=dates,
        name='bound_coherence'
    )
    
    alpha_stability = pd.Series(
        np.random.uniform(0.4, 0.8, 252),
        index=dates,
        name='alpha_stability'
    )
    
    # Create composite with default weights
    print("\n1. Creating composite with default weights...")
    composite = CompositeScore()
    print(f"   Weights: {composite.weights}")
    
    # Compute score
    print("\n2. Computing composite score...")
    score = composite.compute(vol_ratio, bound_coherence, alpha_stability)
    print(f"   Score range: [{score.min():.3f}, {score.max():.3f}]")
    print(f"   Score mean: {score.mean():.3f}")
    print(f"   Score std: {score.std():.3f}")
    
    # Generate signals
    print("\n3. Generating trading signals...")
    signals = composite_signal(score, buy_threshold=0.6, sell_threshold=0.4)
    print(f"   Buy signals: {(signals == 1).sum()} ({(signals == 1).sum()/len(signals):.1%})")
    print(f"   Hold signals: {(signals == 0).sum()} ({(signals == 0).sum()/len(signals):.1%})")
    print(f"   Sell signals: {(signals == -1).sum()} ({(signals == -1).sum()/len(signals):.1%})")
    
    print("\n✅ Scenario 1 complete\n")
    
    return score, signals


# ============================================================
# Scenario 2: Custom Weights & Metadata
# ============================================================

def scenario_2_custom_weights():
    """Scénario avec poids personnalisés et métadonnées."""
    print("=" * 60)
    print("Scénario 2: Custom Weights & Metadata")
    print("=" * 60)
    
    # Generate synthetic metrics
    np.random.seed(123)
    dates = pd.date_range('2020-01-01', periods=252, freq='D')
    
    vol_ratio = pd.Series(np.random.uniform(0.8, 1.5, 252), index=dates)
    bound_coherence = pd.Series(np.random.uniform(0.3, 0.7, 252), index=dates)
    alpha_stability = pd.Series(np.random.uniform(0.4, 0.8, 252), index=dates)
    
    # Try 3 different weight strategies
    strategies = {
        'Equal': {'vol_ratio': 1/3, 'bound_coherence': 1/3, 'alpha_stability': 1/3},
        'Vol-Heavy': {'vol_ratio': 0.5, 'bound_coherence': 0.3, 'alpha_stability': 0.2},
        'Alpha-Heavy': {'vol_ratio': 0.2, 'bound_coherence': 0.3, 'alpha_stability': 0.5}
    }
    
    results = []
    
    for strategy_name, weights in strategies.items():
        print(f"\n{strategy_name} Strategy:")
        print(f"  Weights: {weights}")
        
        composite = CompositeScore(weights=weights)
        score, meta = composite.compute_with_metadata(
            vol_ratio, bound_coherence, alpha_stability
        )
        
        print(f"  Score mean: {meta['mean']:.3f}")
        print(f"  Score std: {meta['std']:.3f}")
        print(f"  Correlations:")
        for metric, corr in meta['correlation_with_inputs'].items():
            print(f"    {metric}: {corr:.3f}")
        
        results.append({
            'strategy': strategy_name,
            'mean': meta['mean'],
            'std': meta['std'],
            'score': score
        })
    
    print("\n✅ Scenario 2 complete\n")
    
    return results


# ============================================================
# Scenario 3: Real-World Simulation (BTC/PAXG)
# ============================================================

def scenario_3_realistic_backtest():
    """Simulation réaliste avec métriques stylisées."""
    print("=" * 60)
    print("Scénario 3: Realistic Backtest Simulation")
    print("=" * 60)
    
    # Simulate realistic metrics (based on BTC/PAXG characteristics)
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=504, freq='D')  # 2 years
    
    # Vol_ratio: Trending with regime shifts
    vol_ratio_base = np.ones(504)
    vol_ratio_base[:168] *= np.random.uniform(0.8, 1.2, 168)  # Regime 1
    vol_ratio_base[168:336] *= np.random.uniform(1.3, 1.8, 168)  # Regime 2 (high vol)
    vol_ratio_base[336:] *= np.random.uniform(0.9, 1.3, 168)  # Regime 3
    vol_ratio = pd.Series(vol_ratio_base, index=dates)
    
    # Bound_coherence: Mean-reverting around 0.5
    bc_noise = np.random.randn(504) * 0.1
    bound_coherence = pd.Series(
        0.5 + bc_noise,
        index=dates
    ).clip(0, 1)
    
    # Alpha_stability: Slightly autocorrelated
    alpha_base = np.random.uniform(0.4, 0.8, 504)
    alpha_smooth = pd.Series(alpha_base, index=dates).rolling(10).mean().fillna(0.6)
    alpha_stability = alpha_smooth.clip(0, 1)
    
    print("\n1. Metric statistics:")
    print(f"   vol_ratio: mean={vol_ratio.mean():.3f}, std={vol_ratio.std():.3f}")
    print(f"   bound_coherence: mean={bound_coherence.mean():.3f}, std={bound_coherence.std():.3f}")
    print(f"   alpha_stability: mean={alpha_stability.mean():.3f}, std={alpha_stability.std():.3f}")
    
    # Create composite with vol-heavy weights (typical for BTC/PAXG)
    print("\n2. Creating composite (vol-heavy strategy)...")
    composite = CompositeScore(weights={
        'vol_ratio': 0.5,
        'bound_coherence': 0.3,
        'alpha_stability': 0.2
    })
    
    score, meta = composite.compute_with_metadata(
        vol_ratio, boun