"""
metis_q4_dsr.py — MÉTIS Q4
Layer 3 QAAF Studio 3.0

Deflated Sharpe Ratio — protection contre le multiple testing.
N_trials alimenté automatiquement par SplitManager.

Critère : DSR ≥ 0.95.
Si DSR < 0.95 : stratégie marquée SUSPECT_DSR — non archivée,
mais non déployable sans investigation supplémentaire.

Note : un DSR < 0.95 n'est pas un échec définitif. Il peut être réévalué
si N_trials diminue ou si de nouvelles données OOS deviennent disponibles.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from layer1_engine.metrics_engine import deflated_sharpe_ratio, compute_cnsr

DSR_THRESHOLD = 0.95


@dataclass
class DSRResult:
    passed:    bool
    dsr:       float
    n_trials:  int
    cnsr_oos:  float
    status:    str    # "PASS" | "SUSPECT_DSR"
    notes:     str


def run_q4(
    strategy_fn:  Callable[[pd.Series, dict], pd.Series],
    params:       dict,
    r_pair_oos:   pd.Series,
    r_base_oos:   pd.Series,
    n_trials:     int,
    rf_annual:    float = 0.04,
) -> DSRResult:
    """
    Calcule le DSR sur la période OOS avec N_trials fourni par SplitManager.

    Paramètres
    ----------
    n_trials : compteur cumulatif de la famille (ex. 101 pour EMA_span_variants)
    """
    print(f"\n📊 MÉTIS Q4 — DSR (N_trials={n_trials}, seuil ≥ {DSR_THRESHOLD}) ...")

    try:
        alloc_oos = strategy_fn(r_pair_oos, params)
        common    = r_pair_oos.index.intersection(alloc_oos.index)
        r_port    = alloc_oos.reindex(common).ffill() * r_pair_oos.loc[common]
        r_base_c  = r_base_oos.reindex(common)

        cnsr_oos = compute_cnsr(r_port, r_base_c, rf_annual)["cnsr_usd_fed"]
        r_usd    = r_port + r_base_c
        dsr      = deflated_sharpe_ratio(r_usd.dropna(), n_trials, rf_annual)
    except Exception as e:
        notes = f"Exception : {e}"
        return DSRResult(False, np.nan, n_trials, np.nan, "ERROR", notes)

    passed = np.isfinite(dsr) and dsr >= DSR_THRESHOLD
    status = "PASS" if passed else "SUSPECT_DSR"
    notes  = (f"DSR={dsr:.4f} | N_trials={n_trials} | "
              f"CNSR_OOS={cnsr_oos:.3f} | seuil={DSR_THRESHOLD}")

    emoji = "✅" if passed else "⚠️"
    print(f"  {emoji} Q4 : {notes}")

    if not passed:
        print(f"    ⚠️  Stratégie marquée {status}.")
        print(f"    Non déployable en l'état.")
        print(f"    Réévaluer si N_trials ↓ ou si T (données OOS) ↑.")

    return DSRResult(
        passed=passed,
        dsr=float(dsr) if np.isfinite(dsr) else np.nan,
        n_trials=n_trials,
        cnsr_oos=float(cnsr_oos) if np.isfinite(cnsr_oos) else np.nan,
        status=status,
        notes=notes,
    )
