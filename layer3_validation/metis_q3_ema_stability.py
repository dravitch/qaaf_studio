"""
metis_q3_ema_stability.py — MÉTIS Q3
Layer 3 QAAF Studio 3.0

Stabilité du span EMA : grille 20j→120j sur IS uniquement.
Optimisation sur IS seulement — OOS vu une seule fois en Q final.

Critère : performance monotone ou plateau identifiable autour de 60j,
pas un spike isolé.
Justification : un optimum ponctuel sur IS n'est pas défendable en déploiement.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict

import numpy as np
import pandas as pd

from layer1_engine.metrics_engine import compute_cnsr

EMA_MIN  = 20
EMA_MAX  = 120
EMA_STEP = 5     # réduit à 10 pour les tests rapides


@dataclass
class EMAStabilityResult:
    passed:              bool
    target_span:         int
    target_cnsr:         float
    neighbourhood_median: float
    is_spike:            bool
    span_cnsrs:          Dict[int, float] = field(default_factory=dict)
    notes:               str = ""


def run_q3(
    strategy_fn: Callable[[pd.Series, dict], pd.Series],
    params:      dict,
    r_pair_is:   pd.Series,
    r_base_is:   pd.Series,
    rf_annual:   float = 0.04,
    ema_step:    int   = EMA_STEP,
) -> EMAStabilityResult:
    """
    Exécute la grille EMA sur IS et détecte les spikes.

    Paramètres
    ----------
    ema_step : pas de grille (défaut 5, utiliser 10 pour tests rapides)
    """
    target_span = params.get("ema_span", 60)
    spans       = list(range(EMA_MIN, EMA_MAX + 1, ema_step))

    print(f"\n📊 MÉTIS Q3 — Stabilité EMA (grille {EMA_MIN}→{EMA_MAX}j "
          f"pas={ema_step}, IS uniquement) ...")

    span_cnsrs: Dict[int, float] = {}

    for span in spans:
        p_s = {**params, "ema_span": span}
        try:
            alloc    = strategy_fn(r_pair_is, p_s)
            common   = r_pair_is.index.intersection(alloc.index)
            r_port   = alloc.reindex(common).ffill() * r_pair_is.loc[common]
            cnsr_val = compute_cnsr(r_port, r_base_is.reindex(common),
                                    rf_annual)["cnsr_usd_fed"]
            span_cnsrs[span] = float(cnsr_val) if np.isfinite(cnsr_val) else np.nan
        except Exception:
            span_cnsrs[span] = np.nan

    # Spike detection : CNSR(target) > médiane voisinage ±20j + 0.30
    neighbourhood = [span_cnsrs[s] for s in span_cnsrs
                     if abs(s - target_span) <= 20 and np.isfinite(span_cnsrs.get(s, np.nan))]
    nbh_median    = float(np.nanmedian(neighbourhood)) if neighbourhood else np.nan
    target_cnsr   = span_cnsrs.get(target_span, np.nan)

    is_spike = (np.isfinite(target_cnsr) and np.isfinite(nbh_median)
                and target_cnsr > nbh_median + 0.30)
    passed   = not is_spike and np.isfinite(target_cnsr)

    # Visualisation ASCII
    _ascii_curve(span_cnsrs, target_span)

    notes = (f"CNSR(span={target_span})={target_cnsr:.3f} | "
             f"médiane_voisinage={nbh_median:.3f} | "
             f"spike={'OUI 🚨' if is_spike else 'NON'}")

    emoji = "✅" if passed else "🔴"
    print(f"  {emoji} Q3 : {notes}")

    return EMAStabilityResult(
        passed=passed,
        target_span=target_span,
        target_cnsr=float(target_cnsr) if np.isfinite(target_cnsr) else np.nan,
        neighbourhood_median=nbh_median,
        is_spike=is_spike,
        span_cnsrs=span_cnsrs,
        notes=notes,
    )


def _ascii_curve(span_cnsrs: Dict[int, float], target_span: int,
                 height: int = 6, width: int = 40) -> None:
    """Affichage ASCII minimaliste de la courbe CNSR IS vs span."""
    spans  = sorted(span_cnsrs.keys())
    cnsrs  = [span_cnsrs[s] for s in spans]
    valid  = [(s, c) for s, c in zip(spans, cnsrs) if np.isfinite(c)]
    if not valid:
        return

    s_arr, c_arr = zip(*valid)
    mn, mx = min(c_arr), max(c_arr)
    rng    = mx - mn if mx != mn else 1.0

    print(f"\n    Courbe CNSR IS — EMA {min(s_arr)}→{max(s_arr)}j "
          f"(★ = span cible {target_span}j)")
    for row in range(height, -1, -1):
        threshold = mn + (row / height) * rng
        line = ""
        for s, c in zip(s_arr, c_arr):
            if abs(c - threshold) < rng / height:
                line += "★" if s == target_span else "·"
            else:
                line += " "
        val_str = f"{threshold:+.2f}" if row % 2 == 0 else "      "
        print(f"    {val_str} │{line}")
    print(f"           └{'─'*len(s_arr)}▶ span")
