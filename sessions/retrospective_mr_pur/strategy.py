"""Signal MR_pur — mean-reversion simple sur ratio PAXG/BTC."""
import numpy as np
import pandas as pd


def mr_pur_strategy(r_pair: pd.Series, params: dict) -> pd.Series:
    lookback = params.get("lookback", 20)
    log_p = r_pair.cumsum()
    ma  = log_p.rolling(lookback, min_periods=max(1, lookback // 2)).mean()
    std = log_p.rolling(lookback, min_periods=max(1, lookback // 2)).std()
    std = std.replace(0, np.nan).ffill().fillna(1e-4)
    z   = (log_p - ma) / std
    alloc = (0.5 - 0.25 * np.tanh(z * 0.8)).clip(0.1, 0.9)
    return alloc.rename("alloc_mr_pur")
