"""Signal H9_brut — H9 sans lissage EMA (signal brut)."""
import numpy as np
import pandas as pd


def h9_brut_strategy(r_pair: pd.Series, params: dict) -> pd.Series:
    lookback = params.get("h9_lookback", 20)
    ratio = r_pair.cumsum().apply(np.exp)
    ma  = ratio.rolling(lookback, min_periods=max(1, lookback // 2)).mean()
    std = ratio.rolling(lookback, min_periods=max(1, lookback // 2)).std()
    std = std.replace(0, np.nan).ffill().fillna(1e-4)
    z   = (ratio - ma) / std
    alloc = (0.5 - 0.25 * np.tanh(z * 0.8)).clip(0.1, 0.9)
    return alloc.rename("alloc_h9_brut")
