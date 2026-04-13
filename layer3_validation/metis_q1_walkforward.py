"""
metis_q1_walkforward.py — MÉTIS Q1
Layer 3 QAAF Studio 3.0

Walk-forward : robustesse temporelle sur 5 fenêtres glissantes.
Métrique : CNSR-USD (pas Sharpe paire brute).

Critère : CNSR-USD OOS > 0.5 sur au moins 4/5 fenêtres.
Justification : une fenêtre robuste est conjoncturelle,
quatre sur cinq suggère une propriété structurelle.

Règle : Layer 3 utilise uniquement les données réelles IS+OOS.
Ni les données synthétiques (Layer 2) ni des splits redéfinis.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List

import numpy as np
import pandas as pd

from layer1_engine.metrics_engine import compute_cnsr

MIN_CNSR_PER_WINDOW = 0.50
MIN_WINDOWS_PASS    = 4
N_WINDOWS           = 5


@dataclass
class WalkForwardResult:
    passed:          bool
    n_windows_pass:  int
    window_cnsrs:    List[float]
    median_cnsr:     float
    notes:           str


def run_q1(
    strategy_fn:   Callable[[pd.Series, dict], pd.Series],
    params:        dict,
    r_pair_full:   pd.Series,   # IS + OOS concaténés
    r_base_full:   pd.Series,
    rf_annual:     float = 0.04,
) -> WalkForwardResult:
    """
    Exécute le walk-forward 5 fenêtres sur r_pair_full (IS + OOS).
    Chaque fenêtre utilise un split 70/30 interne.
    """
    print(f"\n📊 MÉTIS Q1 — Walk-forward ({N_WINDOWS} fenêtres, "
          f"critère ≥ {MIN_WINDOWS_PASS}/{N_WINDOWS} avec CNSR > {MIN_CNSR_PER_WINDOW}) ...")

    T      = len(r_pair_full)
    w_size = T // N_WINDOWS
    cnsrs  = []

    for i in range(N_WINDOWS):
        start     = i * (T // (N_WINDOWS + 1))
        end       = min(start + w_size, T)
        split_70  = start + int((end - start) * 0.70)

        if end - split_70 < 30:
            cnsrs.append(np.nan)
            continue

        r_pair_is  = r_pair_full.iloc[start:split_70]
        r_pair_oos = r_pair_full.iloc[split_70:end]
        r_base_oos = r_base_full.iloc[split_70:end]

        try:
            alloc_oos = strategy_fn(r_pair_oos, params)
            common    = r_pair_oos.index.intersection(alloc_oos.index)
            r_port    = alloc_oos.reindex(common).ffill() * r_pair_oos.loc[common]
            cnsr_val  = compute_cnsr(r_port, r_base_oos.reindex(common),
                                     rf_annual)["cnsr_usd_fed"]
        except Exception as e:
            cnsr_val = np.nan
            print(f"    ⚠️  Fenêtre {i+1} exception : {e}")

        is_pass = np.isfinite(cnsr_val) and cnsr_val >= MIN_CNSR_PER_WINDOW
        emoji   = "✅" if is_pass else "🔴"
        d0      = r_pair_full.index[start]
        d1      = r_pair_full.index[end - 1]
        print(f"    {emoji} Fenêtre {i+1} ({d0} → {d1}) : CNSR={cnsr_val:.3f}")
        cnsrs.append(cnsr_val)

    n_pass     = sum(1 for c in cnsrs if np.isfinite(c) and c >= MIN_CNSR_PER_WINDOW)
    passed     = n_pass >= MIN_WINDOWS_PASS
    median_c   = float(np.nanmedian(cnsrs))
    notes      = (f"{n_pass}/{N_WINDOWS} fenêtres ≥ {MIN_CNSR_PER_WINDOW} | "
                  f"médiane CNSR={median_c:.3f}")

    emoji = "✅" if passed else "🔴"
    print(f"  {emoji} Q1 : {notes}")

    return WalkForwardResult(passed, n_pass, cnsrs, median_c, notes)
