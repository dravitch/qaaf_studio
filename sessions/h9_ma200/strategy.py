"""Signal H9+EMA60j+MA200 — logique de stratégie isolée."""
import numpy as np
import pandas as pd

from sessions.h9_ema60j.strategy import h9_ema_strategy

MA200_PERIOD = 200


def apply_ma200_filter(
    alloc_h9: pd.Series,
    btc_prices: pd.Series,
    mode: str,
) -> pd.Series:
    """Applies MA200 BTC filter to H9 allocation."""
    ma200 = btc_prices.rolling(MA200_PERIOD, min_periods=MA200_PERIOD // 2).mean()
    bear  = btc_prices < ma200
    if mode == "hard":
        return alloc_h9.where(~bear, 0.0)
    if mode == "soft":
        return alloc_h9.where(~bear, alloc_h9 * 0.5)
    raise ValueError(f"mode inconnu : {mode}")


def h9_ma200_strategy(r_pair: pd.Series, params: dict) -> pd.Series:
    """
    H9+EMA60j + filtre MA200 optionnel — wrapper MIF-compatible.
    Si params['btc_prices'] est absent (données synthétiques MIF),
    retourne l'allocation H9 de base sans filtre.
    """
    alloc_base = h9_ema_strategy(r_pair, params)
    btc_prices = params.get("btc_prices")
    if btc_prices is not None:
        mode = params.get("ma200_mode", "hard")
        btc_aligned = btc_prices.reindex(r_pair.index).ffill().bfill()
        return apply_ma200_filter(alloc_base, btc_aligned, mode)
    return alloc_base
