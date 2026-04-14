"""
PAF Direction 3 — Test de la source minimale.

Question : la performance vient-elle d'un signal d'information,
           ou d'un artefact de lissage/réduction de friction ?
Protocole : reproduire la variance d'allocation par un EMA trivial,
            comparer à iso-variance.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Callable

from layer1_engine.backtester     import Backtester
from layer1_engine.metrics_engine import compute_cnsr


@dataclass
class D3Result:
    verdict: str           # SIGNAL_INFORMATIF | ARTEFACT_LISSAGE
    cnsr_signal:  float    # signal à tester
    cnsr_trivial: float    # EMA triviale sur H9 (même variance approx)
    delta:        float    # cnsr_signal - cnsr_trivial
    std_alloc_signal:  float
    std_alloc_trivial: float
    ema_span_used: int
    details: dict


def _find_ema_span_matching_variance(
    prices: pd.DataFrame,
    base_alloc_fn: Callable,
    target_std: float,
    spans: list = None,
) -> tuple:
    """
    Trouve le span EMA qui produit une std_alloc proche de target_std.
    Retourne (span_optimal, alloc_fn, std_obtenu).
    """
    if spans is None:
        spans = list(range(10, 130, 10))

    log_ratio = np.log(prices["paxg"] / prices["btc"])
    q25 = log_ratio.rolling(60, min_periods=30).quantile(0.25)
    q75 = log_ratio.rolling(60, min_periods=30).quantile(0.75)
    iqr = (q75 - q25).replace(0, np.nan)
    h9_raw = ((log_ratio - q25) / iqr).clip(0, 1)
    h9_signal = 1.0 - h9_raw

    best_span, best_std, best_fn = spans[0], float("inf"), None
    for span in spans:
        alloc = h9_signal.ewm(span=span, adjust=False).mean().clip(0, 1).fillna(0.5)
        std = float(alloc.std())
        if abs(std - target_std) < abs(best_std - target_std):
            best_span = span
            best_std = std
            def make_fn(sp):
                a = h9_signal.ewm(span=sp, adjust=False).mean().clip(0, 1).fillna(0.5)
                def fn(df): return a.reindex(df.index).fillna(0.5)
                return fn
            best_fn = make_fn(span)

    return best_span, best_fn, best_std


def run_d3(
    prices_oos: pd.DataFrame,
    r_btc_oos: pd.Series,
    signal_fn: Callable,
    backtester: Backtester,
) -> D3Result:
    """
    Exécute PAF Direction 3.

    Algorithme :
    1. Calculer la std_alloc du signal testé.
    2. Trouver le span EMA de H9 qui produit une std_alloc similaire.
    3. Comparer les CNSR à iso-variance.
    """
    # std_alloc du signal
    result_signal = backtester.run(signal_fn, prices_oos, r_btc_oos)
    target_std = result_signal["std_alloc"]
    zero = pd.Series(0.0, index=result_signal["r_portfolio_usd"].index)
    cnsr_signal = compute_cnsr(result_signal["r_portfolio_usd"], zero)["cnsr_usd_fed"]

    # Trouver EMA triviale à iso-variance
    span, trivial_fn, actual_std = _find_ema_span_matching_variance(
        prices_oos, signal_fn, target_std
    )

    result_trivial = backtester.run(trivial_fn, prices_oos, r_btc_oos)
    zero_t = pd.Series(0.0, index=result_trivial["r_portfolio_usd"].index)
    cnsr_trivial = compute_cnsr(result_trivial["r_portfolio_usd"], zero_t)["cnsr_usd_fed"]

    delta = cnsr_signal - cnsr_trivial
    verdict = "SIGNAL_INFORMATIF" if delta > 0.05 else "ARTEFACT_LISSAGE"

    return D3Result(
        verdict=verdict,
        cnsr_signal=round(cnsr_signal, 4),
        cnsr_trivial=round(cnsr_trivial, 4),
        delta=round(delta, 4),
        std_alloc_signal=round(target_std, 4),
        std_alloc_trivial=round(actual_std, 4),
        ema_span_used=span,
        details={
            "target_std": target_std,
            "actual_std": actual_std,
            "span":       span,
            "cnsr_signal": cnsr_signal,
            "cnsr_trivial": cnsr_trivial,
        }
    )
