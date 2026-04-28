"""
tests/test_filter_gate0.py
MWE pour Gate0DataQuality(Filter).

Signal propre  → passed=True
Signal avec NaN > 5% dans les prix OOS → passed=False
"""

import numpy as np
import pandas as pd
import pytest

from studio.interfaces import FilterConfig, FilterVerdict, SignalData
from studio.filters.gate0_data_quality import Gate0DataQuality


# ── Helpers ──────────────────────────────────────────────────────────

def make_clean_signal(n=1200, seed=42):
    """SignalData avec données propres (aucun NaN)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    btc  = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.001, 0.03, n))),  index=idx)
    paxg = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.002, 0.02, n))),  index=idx)
    pair = paxg / btc
    alloc = pd.Series(0.5, index=idx)
    return SignalData(
        alloc_btc=alloc,
        prices_pair=pair,
        prices_base_usd=btc,
        prices_quote_usd=paxg,
        is_start="2020-01-01",
        is_end="2022-12-31",
        oos_start="2023-01-01",
        oos_end="2023-04-10",
    )


def make_nan_signal(nan_ratio=0.06, seed=42):
    """SignalData avec NaN > 5% dans les prix (ratio sur la série complète)."""
    rng = np.random.default_rng(seed)
    n = 1200
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    btc  = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.001, 0.03, n))),  index=idx)
    paxg = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.002, 0.02, n))),  index=idx)

    # Injecter des NaN dans les derniers jours : nan_ratio * n ≥ 5% de la série
    btc_nan  = btc.copy()
    paxg_nan = paxg.copy()
    n_nan    = max(int(n * nan_ratio), int(n * 0.06))  # au moins 6% de n
    btc_nan.iloc[-n_nan:]  = np.nan
    paxg_nan.iloc[-n_nan:] = np.nan

    pair  = paxg_nan / btc_nan
    alloc = pd.Series(0.5, index=idx)
    return SignalData(
        alloc_btc=alloc,
        prices_pair=pair,
        prices_base_usd=btc_nan,
        prices_quote_usd=paxg_nan,
        is_start="2020-01-01",
        is_end="2022-12-31",
        oos_start="2023-01-01",
        oos_end="2023-04-10",
    )


# ── Tests du contrat Gate0 ────────────────────────────────────────────

class TestGate0Contract:

    def test_returns_filter_verdict(self):
        """evaluate() retourne toujours un FilterVerdict."""
        f      = Gate0DataQuality()
        signal = make_clean_signal()
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        assert isinstance(result, FilterVerdict)

    def test_filter_name_matches(self):
        """filter_name dans le verdict correspond à Gate0DataQuality.NAME."""
        f      = Gate0DataQuality()
        signal = make_clean_signal()
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        assert result.filter_name == Gate0DataQuality.NAME

    def test_metrics_contains_required_keys(self):
        """Le verdict contient n_issues, issues, nan_threshold, min_points."""
        f      = Gate0DataQuality()
        signal = make_clean_signal()
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        for key in ("n_issues", "issues", "nan_threshold", "min_points"):
            assert key in result.metrics, f"Clé manquante : {key}"

    def test_issues_is_list(self):
        """metrics['issues'] est toujours une liste."""
        f      = Gate0DataQuality()
        signal = make_clean_signal()
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        assert isinstance(result.metrics["issues"], list)

    def test_n_issues_matches_issues_length(self):
        """n_issues == len(issues)."""
        f      = Gate0DataQuality()
        signal = make_nan_signal()
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        assert result.metrics["n_issues"] == len(result.metrics["issues"])

    def test_date_incoherence_detected(self):
        """IS/OOS chevauchement → issue détecté et passed=False."""
        rng = np.random.default_rng(42)
        n = 1000
        idx = pd.date_range("2020-01-01", periods=n, freq="D")
        btc  = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.001, 0.03, n))), index=idx)
        paxg = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.001, 0.02, n))), index=idx)
        signal = SignalData(
            alloc_btc=pd.Series(0.5, index=idx),
            prices_pair=paxg / btc,
            prices_base_usd=btc,
            prices_quote_usd=paxg,
            is_start="2020-01-01",
            is_end="2023-06-01",   # is_end > oos_start
            oos_start="2022-01-01",
            oos_end="2023-12-31",
        )
        f      = Gate0DataQuality()
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        assert not result.passed, "Chevauchement IS/OOS doit être détecté"


# ── MWE : données propres → passed=True ──────────────────────────────

class TestGate0CleanData:

    def test_clean_data_passes_gate0(self):
        """Données sans NaN, index monotone → Gate 0 passe."""
        f      = Gate0DataQuality()
        signal = make_clean_signal()
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        assert result.passed, (
            f"Gate 0 rejette des données propres. "
            f"Issues : {result.metrics['issues']}"
        )
        assert result.metrics["n_issues"] == 0


# ── MWE : données avec NaN → passed=False ────────────────────────────

class TestGate0NaNData:

    def test_nan_data_fails_gate0(self):
        """NaN > 5% dans les prix → Gate 0 échoue."""
        f      = Gate0DataQuality()
        signal = make_nan_signal(nan_ratio=0.10)  # 10% NaN
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        assert not result.passed, (
            f"Gate 0 laisse passer des données avec >5% NaN. "
            f"n_issues={result.metrics['n_issues']}"
        )
        assert result.metrics["n_issues"] > 0

    def test_diagnosis_actionable_on_fail(self):
        """Diagnosis contient un mot d'action quand Gate 0 échoue."""
        f      = Gate0DataQuality()
        signal = make_nan_signal(nan_ratio=0.10)
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        assert not result.passed
        assert any(
            w in result.diagnosis.lower()
            for w in ["corriger", "vérifier", "données", "pipeline"]
        ), f"Diagnosis non actionnable : {result.diagnosis}"
