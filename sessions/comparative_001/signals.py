"""
Définitions des signaux pour la session comparative_001.

Chaque signal est une fonction :
    fn(prices_df: pd.DataFrame, config: dict) -> pd.Series
    où la Series retournée est l'allocation PAXG [0, 1] par jour.

Convention : 1.0 = 100% PAXG, 0.0 = 100% BTC.
"""

import pandas as pd
import numpy as np


def signal_passive(prices_df: pd.DataFrame, config: dict) -> pd.Series:
    alloc = config.get("alloc", 0.5)
    return pd.Series(float(alloc), index=prices_df.index)


def signal_h9_ema(prices_df: pd.DataFrame, config: dict) -> pd.Series:
    span     = config.get("span", 60)
    window   = config.get("window", 60)
    log_ratio = np.log(prices_df["paxg"] / prices_df["btc"])
    q25 = log_ratio.rolling(window, min_periods=window // 2).quantile(0.25)
    q75 = log_ratio.rolling(window, min_periods=window // 2).quantile(0.75)
    iqr = (q75 - q25).replace(0, np.nan)
    h9_raw = ((log_ratio - q25) / iqr).clip(0, 1)
    h9_signal = 1.0 - h9_raw
    alloc = h9_signal.ewm(span=span, adjust=False).mean()
    return alloc.clip(0, 1).fillna(0.5)


def signal_h9_ma200_filter(prices_df: pd.DataFrame, config: dict) -> pd.Series:
    window   = config.get("window", 60)
    bear_cap = config.get("bear_cap", 0.20)
    log_ratio = np.log(prices_df["paxg"] / prices_df["btc"])
    q25 = log_ratio.rolling(window, min_periods=window // 2).quantile(0.25)
    q75 = log_ratio.rolling(window, min_periods=window // 2).quantile(0.75)
    iqr = (q75 - q25).replace(0, np.nan)
    h9_raw    = ((log_ratio - q25) / iqr).clip(0, 1)
    h9_signal = (1.0 - h9_raw).clip(0, 1)
    ma200   = prices_df["btc"].rolling(200, min_periods=100).mean()
    is_bull = prices_df["btc"] > ma200
    alloc   = h9_signal.where(is_bull, h9_signal.clip(0, bear_cap))
    return alloc.fillna(0.5)


def signal_h9_ema_ma200(prices_df: pd.DataFrame, config: dict) -> pd.Series:
    span     = config.get("span", 60)
    window   = config.get("window", 60)
    bear_cap = config.get("bear_cap", 0.20)
    alloc_ema = signal_h9_ema(prices_df, {"span": span, "window": window})
    ma200     = prices_df["btc"].rolling(200, min_periods=100).mean()
    is_bull   = prices_df["btc"] > ma200
    alloc     = alloc_ema.where(is_bull, alloc_ema.clip(0, bear_cap))
    return alloc.clip(0, 1)


SIGNAL_REGISTRY = {
    "passive":         signal_passive,
    "h9_ema":          signal_h9_ema,
    "h9_ma200_filter": signal_h9_ma200_filter,
    "h9_ema_ma200":    signal_h9_ema_ma200,
}