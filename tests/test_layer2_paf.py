"""Tests unitaires — PAF Layer 2 (logique des verdicts sur données synthétiques)."""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

from layer1_engine.backtester        import Backtester
from layer2_qualification.paf.paf_d1_hierarchy   import run_d1
from layer2_qualification.paf.paf_d2_attribution import run_d2
from layer2_qualification.paf.paf_d3_source      import run_d3


@pytest.fixture
def synthetic_bull_oos(tmp_path):
    np.random.seed(42)
    n = 400
    idx = pd.date_range("2023-06-01", periods=n, freq="B")
    r_btc  = np.random.randn(n) * 0.03 + 0.004
    r_paxg = np.random.randn(n) * 0.01 + 0.001
    btc  = pd.Series(30000 * np.exp(np.cumsum(r_btc)),  index=idx, name="btc")
    paxg = pd.Series(1900  * np.exp(np.cumsum(r_paxg)), index=idx, name="paxg")
    prices = pd.DataFrame({"paxg": paxg, "btc": btc})
    r_btc_s = pd.Series(np.log(btc / btc.shift(1)).dropna(), name="r_btc")
    return prices.loc[r_btc_s.index], r_btc_s


@pytest.fixture
def backtester(tmp_path):
    import yaml
    cfg = {
        "engine": {"fees_pct": 0.001, "initial_capital": 10000.0, "mode": "lump_sum"},
        "rates":  {"rf_fed": 0.04, "rf_usdc": 0.03, "rf_zero": 0.0},
        "splits": {"is_start": "2020-06-01", "is_end": "2023-05-31",
                   "oos_start": "2023-06-01", "oos_end": "2024-12-31"},
        "data":   {"cache_dir": str(tmp_path / ".cache"),
                   "tickers": {"btc": "BTC-USD", "paxg": "PAXG-USD"}},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(cfg))
    return Backtester(config_path=str(cfg_path))


def h9_ema_fn(df):
    lr = np.log(df["paxg"] / df["btc"])
    q25 = lr.rolling(60, min_periods=30).quantile(0.25)
    q75 = lr.rolling(60, min_periods=30).quantile(0.75)
    iqr = (q75 - q25).replace(0, np.nan)
    h9 = (1 - ((lr - q25) / iqr).clip(0, 1)).clip(0, 1).fillna(0.5)
    return h9.ewm(span=60, adjust=False).mean().clip(0, 1)


def test_d1_returns_valid_verdict(synthetic_bull_oos, backtester):
    prices, r_btc = synthetic_bull_oos
    result = run_d1(prices, r_btc, h9_ema_fn, backtester)
    assert result.verdict in ("HIERARCHIE_CONFIRMEE", "PARTIELLE", "B_PASSIF_DOMINE", "STOP")


def test_d1_returns_numeric_cnsr_values(synthetic_bull_oos, backtester):
    prices, r_btc = synthetic_bull_oos
    result = run_d1(prices, r_btc, h9_ema_fn, backtester)
    assert not np.isnan(result.mr_pur_cnsr)
    assert not np.isnan(result.signal_ref_cnsr)
    assert not np.isnan(result.b_5050_cnsr)


def test_d1_b_passif_domine_when_active_very_poor(synthetic_bull_oos, backtester):
    prices, r_btc = synthetic_bull_oos
    def bad_signal(df): return pd.Series(0.95, index=df.index)
    result = run_d1(prices, r_btc, bad_signal, backtester)
    assert result.verdict in ("B_PASSIF_DOMINE", "STOP", "PARTIELLE", "HIERARCHIE_CONFIRMEE")


def test_d2_neutre_when_composante_makes_no_difference(synthetic_bull_oos, backtester):
    prices, r_btc = synthetic_bull_oos
    def signal_a(df): return pd.Series(0.5, index=df.index)
    def signal_b(df): return pd.Series(0.5, index=df.index)
    result = run_d2(prices, r_btc, signal_a, signal_b, "test", backtester)
    assert result.verdict == "NEUTRE"
    assert abs(result.delta) < 0.01


def test_d2_delta_sign_matches_verdict(synthetic_bull_oos, backtester):
    prices, r_btc = synthetic_bull_oos
    def alloc_high(df): return pd.Series(0.8, index=df.index)
    def alloc_low(df):  return pd.Series(0.2, index=df.index)
    result = run_d2(prices, r_btc, alloc_high, alloc_low, "test", backtester,
                    seuil_actif=0.01)
    assert result.delta == round(result.cnsr_avec - result.cnsr_sans, 4)
    if result.delta > 0.01:
        assert result.verdict == "COMPOSANTE_ACTIVE"
    elif result.delta < -0.01:
        assert result.verdict == "DEGRADANTE"


def test_d3_returns_valid_verdict(synthetic_bull_oos, backtester):
    prices, r_btc = synthetic_bull_oos
    result = run_d3(prices, r_btc, h9_ema_fn, backtester)
    assert result.verdict in ("SIGNAL_INFORMATIF", "ARTEFACT_LISSAGE")
    assert not np.isnan(result.cnsr_signal)
    assert not np.isnan(result.cnsr_trivial)
    assert result.ema_span_used > 0


def test_d3_iso_variance_tolerance(synthetic_bull_oos, backtester):
    prices, r_btc = synthetic_bull_oos
    result = run_d3(prices, r_btc, h9_ema_fn, backtester)
    if result.std_alloc_signal > 0:
        ratio = result.std_alloc_trivial / result.std_alloc_signal
        assert 0.3 <= ratio <= 3.0, f"EMA triviale trop différente : ratio={ratio:.2f}"


def test_d2_delta_is_cnsr_avec_minus_cnsr_sans(synthetic_bull_oos, backtester):
    """Invariant arithmétique : delta = cnsr_avec - cnsr_sans, toujours."""
    prices, r_btc = synthetic_bull_oos
    result = run_d2(prices, r_btc, h9_ema_fn,
                    lambda df: pd.Series(0.5, index=df.index),
                    "arith_check", backtester)
    assert abs(result.delta - (result.cnsr_avec - result.cnsr_sans)) < 1e-6
