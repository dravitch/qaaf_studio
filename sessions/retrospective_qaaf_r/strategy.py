"""Signal QAAF-R — proxy phase-cohérence géométrique (rétrospective Studio)."""
import numpy as np
import pandas as pd


def qaaf_r_strategy(r_pair: pd.Series, params: dict) -> pd.Series:
    T = params.get("T", 30)
    log_p = r_pair.cumsum()
    ma  = log_p.rolling(T, min_periods=max(1, T // 2)).mean()
    std = log_p.rolling(T, min_periods=max(1, T // 2)).std()
    std = std.replace(0, np.nan).ffill().fillna(1e-4)
    z   = (log_p - ma) / std

    r_base = params.get("r_base")
    if r_base is not None:
        r_base_aligned = r_base.reindex(r_pair.index).ffill().bfill()
        log_btc = r_base_aligned.cumsum()
        ma_btc  = log_btc.rolling(T, min_periods=max(1, T // 2)).mean()
        std_btc = log_btc.rolling(T, min_periods=max(1, T // 2)).std()
        std_btc = std_btc.replace(0, np.nan).ffill().fillna(1e-4)
        z_btc   = (log_btc - ma_btc) / std_btc
        # Phase difference proxy — artefact de lissage identifié PAF D3
        coherence = np.cos(np.pi * (z - z_btc).clip(-2, 2) / 4)
        alloc = (0.5 + 0.25 * coherence).clip(0.1, 0.9)
    else:
        # MIF fallback (données synthétiques sans BTC)
        alloc = (0.5 - 0.25 * np.tanh(z * 0.8)).clip(0.1, 0.9)

    return alloc.rename("alloc_qaaf_r")
