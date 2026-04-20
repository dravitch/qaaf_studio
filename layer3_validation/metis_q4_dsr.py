"""
MÉTIS Q4 — DSR (Deflated Sharpe Ratio).

Protocole : calculer le DSR avec N_trials = nombre de variantes
            testées dans la même famille de stratégies.
Critère   : DSR ≥ 0.95 pour PASS, 0.80–0.95 pour SUSPECT_DSR.
Justification : sans correction pour le nombre d'essais, un CNSR
                élevé peut simplement refléter du cherry-picking.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass

from layer1_engine.metrics_engine import deflated_sharpe_ratio


@dataclass
class Q4Result:
    verdict:     str    # PASS | SUSPECT_DSR | FAIL
    dsr:         float
    n_trials:    int
    cnsr_oos:    float
    threshold:   float  # seuil PASS (0.95)
    threshold_suspect: float  # seuil SUSPECT (0.80)


def run_q4(
    r_portfolio_usd: pd.Series,
    cnsr_oos: float,
    n_trials: int,
    rf_annual: float = 0.04,
    threshold: float = 0.95,
    threshold_suspect: float = 0.80,
) -> Q4Result:
    """
    Calcule le DSR et rend un verdict.

    n_trials : nombre total de variantes testées dans la famille.
               Pour la famille EMA_span_variants (grille 20-120j) : 101.
               Pour une hypothèse isolée : 1.
    """
    dsr = deflated_sharpe_ratio(r_portfolio_usd, n_trials, rf_annual)

    if np.isnan(dsr):
        verdict = "FAIL"
    elif dsr >= threshold:
        verdict = "PASS"
    elif dsr >= threshold_suspect:
        verdict = "SUSPECT_DSR"
    else:
        verdict = "FAIL"

    return Q4Result(
        verdict           = verdict,
        dsr               = round(float(dsr), 4) if not np.isnan(dsr) else None,
        n_trials          = n_trials,
        cnsr_oos          = round(float(cnsr_oos), 4),
        threshold         = threshold,
        threshold_suspect = threshold_suspect,
    )
