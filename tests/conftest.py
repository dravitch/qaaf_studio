import pandas as pd
import numpy as np

def make_synthetic_prices():
    np.random.seed(42)
    n   = 200
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    r_btc  = np.random.randn(n) * 0.03 + 0.001
    r_paxg = np.random.randn(n) * 0.01 + 0.0005
    btc_prices  = pd.Series(1000 * np.exp(np.cumsum(r_btc)),  index=idx, name="btc")
    paxg_prices = pd.Series(1800 * np.exp(np.cumsum(r_paxg)), index=idx, name="paxg")
    return paxg_prices, btc_prices

def make_synthetic_returns():
    paxg, btc = make_synthetic_prices()
    r_paxg = np.log(paxg / paxg.shift(1)).dropna()
    r_btc  = np.log(btc  / btc.shift(1)).dropna()
    common = r_paxg.index.intersection(r_btc.index)
    return r_paxg.loc[common], r_btc.loc[common]

try:
    import pytest
    @pytest.fixture
    def synthetic_prices():
        return make_synthetic_prices()
    @pytest.fixture
    def synthetic_returns(synthetic_prices):
        paxg, btc = synthetic_prices
        r_paxg = np.log(paxg / paxg.shift(1)).dropna()
        r_btc  = np.log(btc  / btc.shift(1)).dropna()
        common = r_paxg.index.intersection(r_btc.index)
        return r_paxg.loc[common], r_btc.loc[common]
except ImportError:
    pass
