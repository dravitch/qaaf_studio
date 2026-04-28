"""
tests/test_filter_metis_q1.py
MWE pour MetisQ1(Filter).

Signal BTC bull (drift=+0.004/j) → CNSR > 0.5 sur 5/5 fenêtres → passed=True
Signal BTC bear (drift=-0.004/j) → CNSR < 0.5 sur 4/5 fenêtres → passed=False

Q1 teste la robustesse temporelle sur l'historique complet (IS + OOS).
Critère : CNSR-USD > 0.5 sur au moins min_windows_pass=4 des 5 fenêtres.
En BTC bull (drift=+0.004), toute allocation produit CNSR >> 0.5 (médiane≈3.3).
En BTC bear (drift=-0.004), CNSR < 0 sur la majorité des fenêtres.
"""

import numpy as np
import pandas as pd

from studio.interfaces import FilterConfig, FilterVerdict, SignalData
from studio.filters.metis_q1 import MetisQ1


# ── Helpers ──────────────────────────────────────────────────────────

def make_signal_btc_regime(drift_btc=0.004, alloc_btc_value=0.5, n=1200, seed=42):
    """
    SignalData avec régime BTC paramétrable.

    drift_btc=+0.004 → BTC bull : CNSR >> 0.5 → Q1 passe.
    drift_btc=-0.004 → BTC bear : CNSR < 0 → Q1 échoue.
    """
    np.random.seed(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    n_is = 800

    btc_r  = np.random.normal(drift_btc, 0.04, n)
    paxg_r = np.random.normal(0.001, 0.01, n)

    btc  = pd.Series(100 * np.exp(np.cumsum(btc_r)),  index=idx)
    paxg = pd.Series(100 * np.exp(np.cumsum(paxg_r)), index=idx)
    pair = paxg / btc

    alloc = pd.Series(float(alloc_btc_value), index=idx)

    oos_start_date = str(idx[n_is].date())
    oos_end_date   = str(idx[-1].date())
    is_end_date    = str((idx[n_is] - pd.Timedelta(days=1)).date())

    return SignalData(
        alloc_btc=alloc,
        prices_pair=pair,
        prices_base_usd=btc,
        prices_quote_usd=paxg,
        is_start="2020-01-01",
        is_end=is_end_date,
        oos_start=oos_start_date,
        oos_end=oos_end_date,
    )


# ── Tests du contrat MetisQ1 ──────────────────────────────────────────

class TestMetisQ1Contract:

    def test_returns_filter_verdict(self):
        """evaluate() retourne toujours un FilterVerdict."""
        f      = MetisQ1()
        signal = make_signal_btc_regime(drift_btc=0.004)
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        assert isinstance(result, FilterVerdict)

    def test_filter_name_matches(self):
        """filter_name correspond à MetisQ1.NAME."""
        f      = MetisQ1()
        signal = make_signal_btc_regime(drift_btc=0.004)
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        assert result.filter_name == MetisQ1.NAME

    def test_metrics_contains_required_keys(self):
        """Le verdict contient les clés Q1."""
        f      = MetisQ1()
        signal = make_signal_btc_regime(drift_btc=0.004)
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        for key in ("verdict", "n_pass", "n_total", "cnsr_threshold",
                    "min_windows_pass", "median_cnsr", "windows"):
            assert key in result.metrics, f"Clé manquante : {key}"

    def test_verdict_is_valid_q1_value(self):
        """verdict est PASS ou FAIL."""
        f      = MetisQ1()
        signal = make_signal_btc_regime(drift_btc=0.004)
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        assert result.metrics["verdict"] in {"PASS", "FAIL"}

    def test_windows_has_correct_structure(self):
        """Chaque fenêtre contient window, cnsr, pass."""
        f      = MetisQ1()
        signal = make_signal_btc_regime(drift_btc=0.004)
        config = FilterConfig(name="test", params={"n_windows": 3})
        result = f.evaluate(signal, config)
        for w in result.metrics["windows"]:
            for k in ("window", "cnsr", "pass"):
                assert k in w, f"Clé manquante dans fenêtre : {k}"

    def test_impossible_min_windows_always_fails(self):
        """min_windows_pass=6 > n_windows=5 → toujours FAIL."""
        f      = MetisQ1()
        signal = make_signal_btc_regime(drift_btc=0.004)
        config = FilterConfig(name="test", params={"n_windows": 5, "min_windows_pass": 6})
        result = f.evaluate(signal, config)
        assert not result.passed


# ── MWE : BTC bull → passed=True ──────────────────────────────────────

class TestMetisQ1GoodSignal:

    def test_btc_bull_passes_q1(self):
        """
        BTC bull (drift=+0.004/j), alloc_btc=0.5 → CNSR >> 0.5 sur 5/5 fenêtres → PASS.

        En BTC bull, toute allocation en USD génère un CNSR élevé (drift positif
        amplifié par le Backtester). Médiane CNSR ≈ 3.3 >> 0.5.
        """
        f      = MetisQ1()
        signal = make_signal_btc_regime(drift_btc=0.004, alloc_btc_value=0.5)
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        assert result.passed, (
            f"MÉTIS Q1 rejette un signal en BTC bull. "
            f"n_pass={result.metrics['n_pass']}/{result.metrics['n_total']}, "
            f"median_cnsr={result.metrics['median_cnsr']:.4f}. "
            f"Diagnosis : {result.diagnosis}"
        )
        assert result.metrics["n_pass"] >= 4


# ── MWE : BTC bear → passed=False ────────────────────────────────────

class TestMetisQ1BadSignal:

    def test_btc_bear_fails_q1(self):
        """
        BTC bear (drift=-0.004/j), alloc_btc=0.5 → CNSR < 0 sur 4/5 fenêtres → FAIL.

        En BTC bear, le portefeuille USD perd de la valeur (drift négatif fort).
        La majorité des fenêtres donnent CNSR < 0 < 0.5 → n_pass < 4 → FAIL.
        """
        f      = MetisQ1()
        signal = make_signal_btc_regime(drift_btc=-0.004, alloc_btc_value=0.5)
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        assert not result.passed, (
            f"MÉTIS Q1 laisse passer un signal en BTC bear. "
            f"n_pass={result.metrics['n_pass']}/{result.metrics['n_total']}, "
            f"median_cnsr={result.metrics['median_cnsr']:.4f}."
        )
