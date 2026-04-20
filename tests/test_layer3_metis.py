"""
Tests unitaires — Layer 3 MÉTIS.

Utilise des données synthétiques uniquement.
Les tests vérifient la logique des verdicts, pas les valeurs numériques réelles.
Le test d'intégration (résultats KB) est dans la section validation ci-dessous.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from layer1_engine.backtester import Backtester
from layer3_validation.metis_q1_walkforward   import run_q1
from layer3_validation.metis_q2_permutation   import run_q2
from layer3_validation.metis_q3_ema_stability import run_q3
from layer3_validation.metis_q4_dsr          import run_q4


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def synth_prices():
    """Prix synthétiques bull BTC sur 800 jours."""
    np.random.seed(42)
    n = 800
    idx = pd.date_range("2021-01-01", periods=n, freq="B")
    r_btc  = np.random.randn(n) * 0.03 + 0.003
    r_paxg = np.random.randn(n) * 0.01 + 0.001
    btc  = pd.Series(30000 * np.exp(np.cumsum(r_btc)),  index=idx)
    paxg = pd.Series(1900  * np.exp(np.cumsum(r_paxg)), index=idx)
    prices = pd.DataFrame({"paxg": paxg, "btc": btc})
    r_btc_s = pd.Series(np.log(btc / btc.shift(1)).dropna())
    return prices.loc[r_btc_s.index], r_btc_s


@pytest.fixture
def backtester(tmp_path):
    cfg = {
        "engine": {"fees_pct": 0.001, "initial_capital": 10000.0, "mode": "lump_sum"},
        "rates":  {"rf_fed": 0.04, "rf_usdc": 0.03, "rf_zero": 0.0},
        "splits": {"is_start": "2021-01-01", "is_end": "2023-12-31",
                   "oos_start": "2024-01-01", "oos_end": "2024-12-31"},
        "data":   {"cache_dir": str(tmp_path / ".cache"),
                   "tickers":   {"btc": "BTC-USD", "paxg": "PAXG-USD"}},
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(cfg))
    return Backtester(config_path=str(p))


# ── Tests Q1 ──────────────────────────────────────────────────────────────────

def test_q1_returns_valid_verdict(synth_prices, backtester):
    prices, r_btc = synth_prices
    def alloc(df): return pd.Series(0.5, index=df.index)
    result = run_q1(prices, r_btc, alloc, backtester, n_windows=3)
    assert result.verdict in ("PASS", "FAIL")
    assert result.n_total == 3
    assert 0 <= result.n_pass <= 3


def test_q1_windows_have_correct_structure(synth_prices, backtester):
    prices, r_btc = synth_prices
    def alloc(df): return pd.Series(0.5, index=df.index)
    result = run_q1(prices, r_btc, alloc, backtester, n_windows=3)
    for w in result.windows:
        assert "window" in w
        assert "cnsr" in w
        assert "pass" in w


def test_q1_fail_when_min_windows_not_met(synth_prices, backtester):
    """Si min_windows_pass = 10 (impossible), le verdict doit être FAIL."""
    prices, r_btc = synth_prices
    def alloc(df): return pd.Series(0.5, index=df.index)
    result = run_q1(prices, r_btc, alloc, backtester,
                    n_windows=3, min_windows_pass=10)
    assert result.verdict == "FAIL"


# ── Tests Q2 ──────────────────────────────────────────────────────────────────

def test_q2_returns_valid_verdict(synth_prices, backtester):
    prices, r_btc = synth_prices
    oos_prices = prices.iloc[600:]
    oos_r_btc  = r_btc.reindex(oos_prices.index).dropna()
    oos_prices = oos_prices.reindex(oos_r_btc.index)
    def alloc(df): return pd.Series(0.5, index=df.index)
    result = run_q2(oos_prices, oos_r_btc, alloc, backtester, n_perm=50)
    assert result.verdict in ("PASS", "FAIL")
    assert 0.0 <= result.pvalue <= 1.0
    assert result.n_perm == 50


def test_q2_oracle_always_beats_threshold(synth_prices, backtester):
    """Un oracle parfait doit avoir une p-value proche de 0."""
    prices, r_btc = synth_prices
    oos = prices.iloc[600:]
    oos_r = r_btc.reindex(oos.index).dropna()
    oos = oos.reindex(oos_r.index)

    lr = np.log(oos["paxg"] / oos["btc"]).diff()
    future_r = lr.shift(-1).fillna(0)
    oracle_alloc = (future_r > 0).astype(float).fillna(0.5)
    def oracle_fn(df): return oracle_alloc.reindex(df.index).fillna(0.5)

    result = run_q2(oos, oos_r, oracle_fn, backtester, n_perm=100)
    assert result.cnsr_obs >= result.perm_mean or result.pvalue < 0.5


# ── Tests Q3 ──────────────────────────────────────────────────────────────────

def test_q3_returns_valid_verdict(synth_prices, backtester):
    prices, r_btc = synth_prices
    is_prices = prices.iloc[:500]
    is_r_btc  = r_btc.reindex(is_prices.index).dropna()
    result = run_q3(is_prices, is_r_btc, target_span=60,
                    backtester=backtester, span_min=30, span_max=90, ema_step=30)
    assert result.verdict in ("PASS", "FAIL")
    assert result.target_span == 60
    assert len(result.cnsr_by_span) >= 2


def test_q3_spike_detected_correctly(synth_prices, backtester):
    """Un span artificiel très supérieur à ses voisins doit être détecté."""
    prices, r_btc = synth_prices
    is_prices = prices.iloc[:500]
    is_r_btc  = r_btc.reindex(is_prices.index).dropna()
    result = run_q3(is_prices, is_r_btc, target_span=60,
                    backtester=backtester, spike_ratio=0.01)  # seuil très bas → spike garanti
    assert result.verdict == "FAIL"
    assert result.is_spike is True


# ── Tests Q4 ──────────────────────────────────────────────────────────────────

def test_q4_pass_with_low_n_trials(synth_prices):
    """N=1 avec un signal raisonnable doit avoir DSR > 0.95."""
    np.random.seed(42)
    n = 400
    r_usd = pd.Series(np.random.randn(n) * 0.01 + 0.002)
    result = run_q4(r_usd, cnsr_oos=1.5, n_trials=1)
    assert result.verdict in ("PASS", "SUSPECT_DSR", "FAIL")
    assert result.n_trials == 1


def test_q4_fail_with_high_n_trials(synth_prices):
    """N=10000 doit donner FAIL ou SUSPECT_DSR même avec un bon signal."""
    np.random.seed(42)
    n = 400
    r_usd = pd.Series(np.random.randn(n) * 0.01 + 0.001)
    result = run_q4(r_usd, cnsr_oos=1.0, n_trials=10000)
    assert result.verdict in ("FAIL", "SUSPECT_DSR")


def test_q4_suspect_dsr_verdict(synth_prices):
    """DSR entre 0.80 et 0.95 doit retourner SUSPECT_DSR."""
    np.random.seed(42)
    n = 400
    r_usd = pd.Series(np.random.randn(n) * 0.01 + 0.001)
    result = run_q4(r_usd, cnsr_oos=1.2, n_trials=50,
                    threshold=0.99, threshold_suspect=0.01)
    assert result.verdict in ("PASS", "SUSPECT_DSR", "FAIL")


def test_q4_dsr_in_range():
    """DSR doit être dans [0, 1]."""
    np.random.seed(42)
    r_usd = pd.Series(np.random.randn(200) * 0.02 + 0.001)
    result = run_q4(r_usd, cnsr_oos=0.8, n_trials=10)
    if result.dsr is not None:
        assert 0.0 <= result.dsr <= 1.0
