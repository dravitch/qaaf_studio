"""
studio/oracle.py
Signal oracle asymétrique pour la certification du pipeline QAAF Studio 3.1.

Ce n'est pas un signal de production — c'est un étalon conçu pour être
certifiable par un pipeline bien calibré (MIF G1/G3, MÉTIS Q2/Q4, DSR).

Invariants :
  - Aucun lookahead : la moyenne 120j est calculée avec .shift(1).
  - alloc_btc ∈ [alloc_low, alloc_high] à tout instant.
  - Signal stationnaire sur la durée complète des données.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


ORACLE_PARAMS: dict = {
    "trend_window":  120,   # Fenêtre de tendance (jours)
    "vol_window":     30,   # Fenêtre de volatilité réalisée (jours)
    "vol_threshold":  2.0,  # Multiplicateur médiane vol → neutre
    "alloc_high":    0.75,  # Allocation BTC max (position longue BTC)
    "alloc_low":     0.25,  # Allocation BTC min (position longue PAXG)
}


def compute_oracle_signal(
    prices_pair: pd.Series,
    params: dict | None = None,
) -> pd.Series:
    """
    Calcule l'allocation BTC du signal oracle.

    Logique asymétrique :
      1. Long BTC  (alloc_high) quand log(ratio) ≥ MA_{trend_window}  d'hier.
      2. Long PAXG (alloc_low)  quand log(ratio) <  MA_{trend_window}  d'hier.
      3. Neutre    (0.5)        quand vol_réalisée > vol_threshold × médiane.

    Args:
        prices_pair : prix de la paire PAXG/BTC (ou tout ratio base/quote).
        params      : surcharge de ORACLE_PARAMS (facultatif).

    Returns:
        pd.Series alloc_btc ∈ [alloc_low, alloc_high], même index que prices_pair.
    """
    p = {**ORACLE_PARAMS, **(params or {})}

    log_ratio  = np.log(prices_pair)

    # Tendance — décalée d'1 jour pour éviter le lookahead
    trend = log_ratio.rolling(p["trend_window"]).mean().shift(1)

    # Volatilité réalisée — décalée d'1 jour
    vol        = log_ratio.diff().rolling(p["vol_window"]).std().shift(1)
    vol_median = vol.expanding().median()
    high_vol   = vol > p["vol_threshold"] * vol_median

    # Allocation de base : position PAXG par défaut
    alloc = pd.Series(p["alloc_low"], index=prices_pair.index, dtype=float)

    # Warmup (trend non disponible) → neutre
    alloc = alloc.where(trend.notna(), other=0.5)

    # Long BTC quand ratio sous la tendance (BTC surperforme)
    long_btc = trend.notna() & (log_ratio < trend)
    alloc = alloc.where(~long_btc, other=p["alloc_high"])

    # Neutre en régime haute volatilité — priorité sur la tendance
    alloc = alloc.where(~high_vol, other=0.5)

    return alloc.clip(p["alloc_low"], p["alloc_high"]).rename("alloc_oracle")
