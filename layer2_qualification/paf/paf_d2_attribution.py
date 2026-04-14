"""
PAF Direction 2 — Attribution de performance.

Question : quelle composante du signal explique la performance observée en D1 ?
Protocole : signal_complet vs signal_sans_composante_X, à allocation comparable.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict

from layer1_engine.backtester     import Backtester
from layer1_engine.metrics_engine import compute_cnsr


@dataclass
class D2Result:
    verdict: str              # COMPOSANTE_ACTIVE | NEUTRE | DEGRADANTE
    composante: str
    cnsr_avec:  float         # signal complet
    cnsr_sans:  float         # signal sans la composante
    delta:      float         # cnsr_avec - cnsr_sans
    std_alloc_avec: float
    std_alloc_sans: float
    details: dict


def run_d2(
    prices_oos: pd.DataFrame,
    r_btc_oos: pd.Series,
    signal_complet_fn: Callable,
    signal_sans_fn: Callable,
    composante_name: str,
    backtester: Backtester,
    seuil_actif: float = 0.05,
) -> D2Result:
    """
    Exécute PAF Direction 2 pour une composante donnée.

    Répéter pour chaque composante à tester (régimes, PhaseCoherence, etc.)
    """
    def _run(alloc_fn):
        result = backtester.run(alloc_fn, prices_oos, r_btc_oos)
        zero = pd.Series(0.0, index=result["r_portfolio_usd"].index)
        metrics = compute_cnsr(result["r_portfolio_usd"], zero)
        return metrics["cnsr_usd_fed"], result["std_alloc"]

    cnsr_avec,  std_avec = _run(signal_complet_fn)
    cnsr_sans, std_sans  = _run(signal_sans_fn)
    delta = cnsr_avec - cnsr_sans

    if delta > seuil_actif:
        verdict = "COMPOSANTE_ACTIVE"
    elif delta < -seuil_actif:
        verdict = "DEGRADANTE"
    else:
        verdict = "NEUTRE"

    return D2Result(
        verdict=verdict,
        composante=composante_name,
        cnsr_avec=round(cnsr_avec, 4),
        cnsr_sans=round(cnsr_sans, 4),
        delta=round(delta, 4),
        std_alloc_avec=round(std_avec, 4),
        std_alloc_sans=round(std_sans, 4),
        details={
            "cnsr_complet": cnsr_avec,
            "cnsr_sans":    cnsr_sans,
            "delta":        delta,
            "verdict":      verdict,
        }
    )
