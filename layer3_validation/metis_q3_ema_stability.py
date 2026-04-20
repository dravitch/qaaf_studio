"""
MÉTIS Q3 — Stabilité du span EMA (sur-ajustement des paramètres).

Protocole : grille EMA 20j→120j sur IS uniquement.
Critère   : pas de spike isolé (le span cible ne doit pas être
            un optimum ponctuel par rapport à ses voisins).
Justification : si 60j est un optimum isolé, le paramètre est
                sur-ajusté et ne généralisera pas.

Note : ce test tourne sur IS uniquement. L'OOS n'est jamais vu.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Callable, List

from layer1_engine.backtester     import Backtester
from layer1_engine.metrics_engine import compute_cnsr


@dataclass
class Q3Result:
    verdict:          str     # PASS | FAIL
    target_span:      int     # span cible (60j par défaut)
    cnsr_target:      float   # CNSR IS pour le span cible
    median_neighbors: float   # médiane des spans voisins ±2
    is_spike:         bool    # True si spike détecté
    cnsr_by_span:     dict    # {span: cnsr} pour toute la grille
    ema_step:         int


def _h9_ema_signal(prices: pd.DataFrame, span: int, window: int = 60) -> pd.Series:
    """Signal H9+EMA pour un span donné."""
    log_ratio = np.log(prices["paxg"] / prices["btc"])
    q25 = log_ratio.rolling(window, min_periods=window // 2).quantile(0.25)
    q75 = log_ratio.rolling(window, min_periods=window // 2).quantile(0.75)
    iqr = (q75 - q25).replace(0, np.nan)
    h9  = (1.0 - ((log_ratio - q25) / iqr).clip(0, 1)).clip(0, 1).fillna(0.5)
    return h9.ewm(span=span, adjust=False).mean().clip(0, 1)


def run_q3(
    prices_is: pd.DataFrame,
    r_btc_is: pd.Series,
    target_span: int = 60,
    backtester: Backtester = None,
    span_min: int = 20,
    span_max: int = 120,
    ema_step: int = 10,
    spike_ratio: float = 1.5,
) -> Q3Result:
    """
    Exécute la grille EMA sur IS uniquement.

    spike_ratio : si CNSR(target) > spike_ratio × médiane_voisins,
                  c'est un spike (sur-ajustement).
    """
    spans        = list(range(span_min, span_max + 1, ema_step))
    cnsr_by_span = {}

    for span in spans:
        alloc_fn = lambda df, s=span: _h9_ema_signal(df, s)
        try:
            result = backtester.run(alloc_fn, prices_is, r_btc_is)
            zero   = pd.Series(0.0, index=result["r_portfolio_usd"].index)
            m      = compute_cnsr(result["r_portfolio_usd"], zero)
            cnsr_by_span[span] = round(float(m["cnsr_usd_fed"]), 4)
        except Exception:
            cnsr_by_span[span] = None

    # Span cible
    cnsr_target = cnsr_by_span.get(target_span, float("nan"))
    if cnsr_target is None:
        cnsr_target = float("nan")

    # Voisins : spans ±2*step autour de target
    neighbor_spans = [
        s for s in spans
        if s != target_span and abs(s - target_span) <= 2 * ema_step
    ]
    neighbor_vals = [cnsr_by_span[s] for s in neighbor_spans if cnsr_by_span.get(s) is not None]
    median_neighbors = float(np.median(neighbor_vals)) if neighbor_vals else float("nan")

    # Détection de spike
    is_spike = False
    if not np.isnan(cnsr_target) and not np.isnan(median_neighbors) and median_neighbors > 0:
        is_spike = cnsr_target > spike_ratio * median_neighbors

    return Q3Result(
        verdict          = "FAIL" if is_spike else "PASS",
        target_span      = target_span,
        cnsr_target      = round(float(cnsr_target), 4) if not np.isnan(cnsr_target) else None,
        median_neighbors = round(median_neighbors, 4) if not np.isnan(median_neighbors) else None,
        is_spike         = is_spike,
        cnsr_by_span     = cnsr_by_span,
        ema_step         = ema_step,
    )
