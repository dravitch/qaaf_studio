"""
metis_q2_permutation.py — MÉTIS Q2
Layer 3 QAAF Studio 3.0

Test de permutation : significativité statistique vs B_5050.
10 000 itérations, permutation des signaux d'allocation OOS.
Métrique : CNSR-USD (pas Sharpe paire brute).

Critère : p-value < 0.05.
Justification : le bull run 2023-2024 gonfle tous les Sharpe.
La permutation isole ce qui vient de la règle, pas du régime de marché.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from layer1_engine.metrics_engine import compute_cnsr

N_PERMUTATIONS = 10_000
PVALUE_MAX     = 0.05


@dataclass
class PermutationResult:
    passed:    bool
    p_value:   float
    cnsr_obs:  float
    cnsr_bench: float
    perm_mean: float
    perm_std:  float
    notes:     str


def run_q2(
    strategy_fn:  Callable[[pd.Series, dict], pd.Series],
    params:       dict,
    r_pair_oos:   pd.Series,
    r_base_oos:   pd.Series,
    rf_annual:    float = 0.04,
    n_perm:       int   = N_PERMUTATIONS,
) -> PermutationResult:
    """
    Exécute le test de permutation sur la période OOS.

    Paramètres
    ----------
    n_perm : nombre de permutations (défaut 10 000, réduire à 500 pour tests)
    """
    print(f"\n📊 MÉTIS Q2 — Permutation ({n_perm:,} itérations, "
          f"p-value < {PVALUE_MAX}) ...")

    # Stratégie observée
    try:
        alloc_oos = strategy_fn(r_pair_oos, params)
        common    = r_pair_oos.index.intersection(alloc_oos.index)
        r_port    = alloc_oos.reindex(common).ffill() * r_pair_oos.loc[common]
        cnsr_obs  = compute_cnsr(r_port, r_base_oos.reindex(common),
                                 rf_annual)["cnsr_usd_fed"]
    except Exception as e:
        notes = f"Exception stratégie : {e}"
        return PermutationResult(False, np.nan, np.nan, np.nan, np.nan, np.nan, notes)

    # Benchmark B_5050
    r_bench    = 0.5 * r_pair_oos.loc[common]
    cnsr_bench = compute_cnsr(r_bench, r_base_oos.reindex(common),
                               rf_annual)["cnsr_usd_fed"]

    print(f"    CNSR observé = {cnsr_obs:.4f} | CNSR B_5050 = {cnsr_bench:.4f}")
    print(f"    Permutation de {n_perm:,} signaux OOS ...")

    # Distribution nulle — permuter les allocations
    rng          = np.random.default_rng(42)
    alloc_arr    = alloc_oos.reindex(common).ffill().values
    r_pair_arr   = r_pair_oos.loc[common].values
    r_base_arr   = r_base_oos.reindex(common).values
    perm_cnsrs   = []

    for _ in range(n_perm):
        perm_a = rng.permutation(alloc_arr)
        r_p    = pd.Series(perm_a * r_pair_arr)
        r_b    = pd.Series(r_base_arr)
        try:
            c = compute_cnsr(r_p, r_b, rf_annual)["cnsr_usd_fed"]
            if np.isfinite(c):
                perm_cnsrs.append(c)
        except Exception:
            pass

    perm_arr  = np.array(perm_cnsrs)
    p_value   = float(np.mean(perm_arr >= cnsr_obs)) if len(perm_arr) > 0 else np.nan
    passed    = np.isfinite(p_value) and p_value < PVALUE_MAX

    notes = (f"p-value={p_value:.4f} | CNSR_obs={cnsr_obs:.3f} | "
             f"bench={cnsr_bench:.3f} | perm_mean={perm_arr.mean():.3f}")

    emoji = "✅" if passed else "🔴"
    print(f"  {emoji} Q2 : {notes}")

    return PermutationResult(
        passed    = passed,
        p_value   = p_value,
        cnsr_obs  = float(cnsr_obs),
        cnsr_bench= float(cnsr_bench),
        perm_mean = float(perm_arr.mean()),
        perm_std  = float(perm_arr.std()),
        notes     = notes,
    )
