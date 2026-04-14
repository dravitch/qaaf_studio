import pytest
import pandas as pd
import numpy as np
from sessions.comparative_001.signals import (
    signal_passive, signal_h9_ema,
    signal_h9_ma200_filter, signal_h9_ema_ma200
)

@pytest.fixture
def prices_df():
    np.random.seed(42)
    n   = 300
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    btc  = pd.Series(1000 * np.exp(np.cumsum(np.random.randn(n) * 0.03)), index=idx)
    paxg = pd.Series(1800 * np.exp(np.cumsum(np.random.randn(n) * 0.01)), index=idx)
    return pd.DataFrame({"paxg": paxg, "btc": btc})

def test_passive_alloc_is_constant(prices_df):
    alloc = signal_passive(prices_df, {"alloc": 0.5})
    assert (alloc == 0.5).all()

def test_h9_ema_alloc_in_range(prices_df):
    alloc = signal_h9_ema(prices_df, {"span": 60})
    assert alloc.dropna().between(0, 1).all()

def test_ma200_filter_caps_bear_allocation(prices_df):
    """En régime bear artificiel, allocation plafonnée à bear_cap."""
    btc_bear = pd.Series(
        1000 * np.exp(np.cumsum(np.full(300, -0.01))),
        index=prices_df.index
    )
    df_bear = pd.DataFrame({"paxg": prices_df["paxg"], "btc": btc_bear})
    alloc   = signal_h9_ma200_filter(df_bear, {"bear_cap": 0.20})
    assert (alloc.iloc[250:] <= 0.21).all(), f"Bear cap exceeded: max={alloc.iloc[250:].max():.3f}"