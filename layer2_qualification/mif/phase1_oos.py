"""
phase1_oos.py — MIF Phase 1
Layer 2 QAAF Studio 3.0

Généralisation OOS (G1-G5) sur régimes variés — données synthétiques.
Détecte l'overfitting sur le régime d'entraînement.

Règle d'arrêt : FAIL Phase 1 → archiver la limitation, reformuler possible.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List

import numpy as np
import pandas as pd

from layer1_engine.metrics_engine    import compute_cnsr
from layer2_qualification.mif.synthetic_data import generate_synthetic_paxgbtc


@dataclass
class GeneralizationResult:
    label:        str
    regime:       str
    passed:       bool
    cnsr_strat:   float | None
    cnsr_bench:   float | None
    delta:        float | None
    notes:        str


def run_phase1(
    strategy_fn: Callable[[pd.Series, dict], pd.Series],
    params: dict,
    rf_annual: float = 0.04,
) -> List[GeneralizationResult]:
    """
    Exécute G1-G5 sur 5 régimes synthétiques variés.
    Critère : CNSR > -1.0 et algorithme stable sur chaque régime.
    """
    print("\n⚙️  MIF Phase 1 — Généralisation OOS (G1-G5) ...")
    results = []

    tests = [
        ("G1", "Bear market",     "bear",     101),
        ("G2", "Marché latéral",  "lateral",  202),
        ("G3", "Crash",           "crash",    303),
        ("G4", "Standard",        "standard", 404),
        ("G5", "Standard long",   "standard", 505),
    ]
    Ts = [400, 400, 400, 400, 800]

    for (label, name, regime, seed), T in zip(tests, Ts):
        r_pair, r_base = generate_synthetic_paxgbtc(T=T, seed=seed, regime=regime)
        try:
            alloc        = strategy_fn(r_pair, params)
            common       = r_pair.index.intersection(alloc.index)
            r_port       = alloc.reindex(common).ffill() * r_pair.loc[common]
            r_base_c     = r_base.reindex(common)
            cnsr_s       = compute_cnsr(r_port, r_base_c, rf_annual)["cnsr_usd_fed"]
            cnsr_b       = compute_cnsr(0.5 * r_pair.loc[common], r_base_c,
                                        rf_annual)["cnsr_usd_fed"]
            delta        = (cnsr_s - cnsr_b) if np.isfinite(cnsr_s) and np.isfinite(cnsr_b) else None
            passed       = np.isfinite(cnsr_s) and cnsr_s > -1.0
            notes        = f"bench={cnsr_b:.3f} | delta={delta:+.3f}" if delta is not None else "—"
        except Exception as e:
            cnsr_s, cnsr_b, delta, passed, notes = None, None, None, False, f"Exception: {e}"

        res   = GeneralizationResult(f"{label} {name}", regime, passed,
                                     cnsr_s, cnsr_b, delta, notes)
        emoji = "✅" if passed else "🔴"
        c_str = f"{cnsr_s:.3f}" if cnsr_s is not None else "—"
        print(f"  {emoji} {label} {name:25s} CNSR={c_str}  {notes}")
        results.append(res)

    return results
