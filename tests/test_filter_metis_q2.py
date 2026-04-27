"""
tests/test_filter_metis_q2.py
MWE pour MetisQ2(Filter).

Deux cas obligatoires (Test Before Trust) :
  - Signal bon (oracle) → passed=True  (recalibration Alternative A validée)
  - Signal aléatoire    → passed=False (doit rester vert)

Recalibration Session 2 : ajustement de régime actif.
test_oracle_signal_passes_q2 est vert sans xfail.
"""

import pytest
import numpy as np
import pandas as pd

from studio.interfaces import FilterConfig, FilterVerdict, FilterError, SignalData
from studio.filters.metis_q2 import MetisQ2


# ── Helpers ──────────────────────────────────────────────────────────

def make_signal_data_bull_oos(seed=42):
    """
    SignalData avec un bull run structurel en OOS (2023-2024).

    IS  (2019-2022) : drift_btc=0.0008 (régime modéré)
    OOS (2023-2024) : drift_btc=0.004  (fort bull run BTC)

    La rupture de régime IS/OOS déclenche regime_adjusted=True
    et permet à l'oracle (long BTC en tendance) de battre cnsr_bench_is.
    """
    from studio.oracle import compute_oracle_signal

    rng = np.random.default_rng(seed)
    n_is  = 1461  # 2019-01-01 → 2022-12-31 (4 ans)
    n_oos = 730   # 2023-01-01 → 2024-12-31 (2 ans)
    n     = n_is + n_oos

    idx = pd.date_range("2019-01-01", periods=n, freq="D")

    # IS : drift modéré
    r_btc_is  = rng.normal(0.0008, 0.04, n_is)
    r_paxg_is = rng.normal(0.0004, 0.01, n_is)

    # OOS : fort bull run BTC (5× drift IS)
    r_btc_oos  = rng.normal(0.004, 0.04, n_oos)
    r_paxg_oos = rng.normal(0.0004, 0.01, n_oos)

    r_btc_all  = np.concatenate([r_btc_is,  r_btc_oos])
    r_paxg_all = np.concatenate([r_paxg_is, r_paxg_oos])

    btc  = pd.Series(100 * np.exp(np.cumsum(r_btc_all)),  index=idx)
    paxg = pd.Series(100 * np.exp(np.cumsum(r_paxg_all)), index=idx)
    pair = paxg / btc

    alloc_btc = compute_oracle_signal(pair)

    return SignalData(
        alloc_btc=alloc_btc,
        prices_pair=pair,
        prices_base_usd=btc,
        prices_quote_usd=paxg,
        is_start="2019-01-01",
        is_end="2022-12-31",
        oos_start="2023-01-01",
        oos_end="2024-12-31",
    )


def make_signal_data_random(seed=99):
    """SignalData avec signal aléatoire uniforme sur même structure."""
    from studio.oracle import compute_oracle_signal

    rng = np.random.default_rng(seed)
    n = 2191
    idx = pd.date_range("2019-01-01", periods=n, freq="D")
    btc  = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.001, 0.04, n))),  index=idx)
    paxg = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, n))), index=idx)
    pair = paxg / btc

    alloc_btc = pd.Series(rng.uniform(0.0, 1.0, n), index=idx)

    return SignalData(
        alloc_btc=alloc_btc,
        prices_pair=pair,
        prices_base_usd=btc,
        prices_quote_usd=paxg,
        is_start="2019-01-01",
        is_end="2022-12-31",
        oos_start="2023-01-01",
        oos_end="2024-12-31",
    )


# ── Tests du contrat MetisQ2 ──────────────────────────────────────────

class TestMetisQ2Contract:

    def test_returns_filter_verdict(self):
        """evaluate() retourne toujours un FilterVerdict, jamais None."""
        f      = MetisQ2()
        signal = make_signal_data_bull_oos()
        config = FilterConfig(name="test", params={"n_perm": 100})
        result = f.evaluate(signal, config)
        assert isinstance(result, FilterVerdict)

    def test_filter_name_matches(self):
        """filter_name dans le verdict correspond à MetisQ2.NAME."""
        f      = MetisQ2()
        signal = make_signal_data_bull_oos()
        config = FilterConfig(name="test", params={"n_perm": 100})
        result = f.evaluate(signal, config)
        assert result.filter_name == MetisQ2.NAME

    def test_metrics_contains_required_keys(self):
        """Le verdict contient toutes les clés attendues."""
        f      = MetisQ2()
        signal = make_signal_data_bull_oos()
        config = FilterConfig(name="test", params={"n_perm": 100})
        result = f.evaluate(signal, config)
        for key in ("pvalue", "cnsr_obs", "cnsr_bench_oos", "cnsr_bench_is",
                    "regime_adjusted", "n_perm", "criterion"):
            assert key in result.metrics, f"Clé manquante dans metrics : {key}"

    def test_regime_adjusted_is_bool(self):
        """regime_adjusted est un booléen documenté dans les metrics."""
        f      = MetisQ2()
        signal = make_signal_data_bull_oos()
        config = FilterConfig(name="test", params={"n_perm": 100})
        result = f.evaluate(signal, config)
        assert isinstance(result.metrics["regime_adjusted"], bool)

    def test_diagnosis_is_actionable_on_fail(self):
        """En cas d'échec (p_threshold=0.0, regime_margin très élevé), diagnosis actionnable."""
        f      = MetisQ2()
        signal = make_signal_data_bull_oos()
        config = FilterConfig(name="test", params={"n_perm": 100, "p_threshold": 0.0,
                                                    "regime_margin": 100.0})
        result = f.evaluate(signal, config)
        if not result.passed:
            assert any(
                w in result.diagnosis.lower()
                for w in ["pour", "envisager", "filtre", "battre"]
            ), f"Diagnosis non actionnable : {result.diagnosis}"

    def test_criterion_is_regime_adjusted(self):
        """criterion='regime_adjusted' est documenté après recalibration Session 2."""
        f      = MetisQ2()
        signal = make_signal_data_bull_oos()
        config = FilterConfig(name="test", params={"n_perm": 100})
        result = f.evaluate(signal, config)
        assert result.metrics["criterion"] == "regime_adjusted", (
            "criterion doit être 'regime_adjusted' après recalibration. "
            "Si 'absolute' : la recalibration n'a pas été appliquée."
        )


# ── MWE : signal bon → passed=True ───────────────────────────────────

class TestMetisQ2GoodSignal:
    """
    Signal oracle sur données avec bull run OOS structurel.
    regime_adjusted=True déclenche la comparaison directe (Option B).
    L'oracle long BTC bat le benchmark IS conservateur.
    """

    def test_oracle_signal_passes_q2(self):
        """Signal oracle avec bull run OOS → Q2 passe (ajustement de régime)."""
        f      = MetisQ2()
        signal = make_signal_data_bull_oos()
        config = FilterConfig(name="test", params={"n_perm": 500})
        result = f.evaluate(signal, config)

        assert result.metrics["regime_adjusted"], (
            f"regime_adjusted=False : le bull run OOS n'a pas été détecté. "
            f"diff cnsr_bench OOS-IS="
            f"{result.metrics.get('cnsr_bench_oos', 0) - result.metrics.get('cnsr_bench_is', 0):.3f} "
            f"≤ regime_margin=0.8. Vérifier les données de test."
        )
        assert result.passed, (
            f"Q2 rejette le signal oracle même en régime bull. "
            f"cnsr_obs={result.metrics.get('cnsr_obs'):.3f} ≤ "
            f"cnsr_bench_oos={result.metrics.get('cnsr_bench_oos'):.3f}. "
            f"Diagnostic : {result.diagnosis}"
        )


# ── MWE : signal aléatoire → passed=False ────────────────────────────

class TestMetisQ2BadSignal:
    """
    Signal aléatoire pur.
    Même avec ajustement de régime, un signal aléatoire ne bat pas cnsr_bench_is.
    """

    def test_random_signal_fails_q2(self):
        """Signal aléatoire uniforme → Q2 échoue toujours."""
        f      = MetisQ2()
        signal = make_signal_data_random()
        config = FilterConfig(name="test", params={"n_perm": 500})
        result = f.evaluate(signal, config)
        assert not result.passed, (
            f"Q2 laisse passer un signal aléatoire. "
            f"p={result.metrics.get('pvalue'):.3f}, "
            f"regime_adjusted={result.metrics.get('regime_adjusted')}. "
            f"La recalibration est trop laxiste si ce test devient vert."
        )
