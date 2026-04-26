"""Signal H9+EMA60j — logique de stratégie isolée."""
import numpy as np
import pandas as pd

EMA_SPAN = 60


def h9_ema_strategy(r_pair: pd.Series, params: dict) -> pd.Series:
    """
    H9 : allocation PAXG proportionnelle à la déviation du ratio de sa MA.
    Lissage EMA span.

    Paramètres
    ----------
    r_pair : log-rendements PAXG/BTC
    params : {"ema_span": int, "h9_lookback": int}
    """
    span     = params.get("ema_span",    EMA_SPAN)
    lookback = params.get("h9_lookback", 20)

    ratio = r_pair.cumsum().apply(np.exp)
    ma    = ratio.rolling(lookback, min_periods=max(1, lookback // 2)).mean()
    std   = ratio.rolling(lookback, min_periods=max(1, lookback // 2)).std()
    std   = std.replace(0, np.nan).ffill().fillna(1e-4)
    z     = (ratio - ma) / std

    raw      = 0.5 - 0.25 * np.tanh(z * 0.8)
    smoothed = raw.ewm(span=span, min_periods=span // 2).mean()
    return smoothed.clip(0.1, 0.9).rename("alloc_h9_ema")
