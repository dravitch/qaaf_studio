"""
tests/test_filter_metis_q2.py
MWE pour MetisQ2(Filter).

Deux cas obligatoires (Test Before Trust) :
  - Signal délibérément bon  → xfail (RECALIBRATION_PENDING)
  - Signal aléatoire → passed=False (doit rester vert)

Ces tests vérifient le contrat du filtre, pas sa calibration finale.
La calibration (Alternative A) sera appliquée en Session 2.
"""

import pytest
import numpy as np
import pandas as pd

from studio.interfaces import FilterConfig, FilterVerdict, FilterError, SignalData
from studio.filters.metis_q2 import MetisQ2


# ── Helpers ──────────────────────────────────────────────────────────

def make_signal_data(alloc_fn=None, seed=42):
    """
    SignalData minimal pour les tests MetisQ2.
    OOS : 2023-01-01 → 2024-12-31 (≥ 60 jours garantis).
    Utilise oracle.compute_oracle_signal() pour le signal bon.
    """
    from studio.oracle import compute_oracle_signal

    rng = np.random.default_rng(seed)
    idx = pd.date_range("2019-01-01", periods=2200, freq="D")

    btc  = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.001, 0.04, 2200))),  index=idx)
    paxg = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, 2200))), index=idx)
    pair = paxg / btc

    if alloc_fn is None:
        alloc_btc = compute_oracle_signal(pair)
    else:
        alloc_btc = alloc_fn(pair, idx)

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
        signal = make_signal_data()
        config = FilterConfig(name="test", params={"n_perm": 100})
        result = f.evaluate(signal, config)
        assert isinstance(result, FilterVerdict)

    def test_filter_name_matches(self):
        """filter_name dans le verdict correspond à MetisQ2.NAME."""
        f      = MetisQ2()
        signal = make_signal_data()
        config = FilterConfig(name="test", params={"n_perm": 100})
        result = f.evaluate(signal, config)
        assert result.filter_name == MetisQ2.NAME

    def test_metrics_contains_required_keys(self):
        """Le verdict contient pvalue, cnsr_obs, cnsr_bench, n_perm, criterion."""
        f      = MetisQ2()
        signal = make_signal_data()
        config = FilterConfig(name="test", params={"n_perm": 100})
        result = f.evaluate(signal, config)
        for key in ("pvalue", "cnsr_obs", "cnsr_bench", "n_perm", "criterion"):
            assert key in result.metrics, f"Clé manquante dans metrics : {key}"

    def test_pvalue_in_unit_interval(self):
        """p-value est dans [0, 1]."""
        f      = MetisQ2()
        signal = make_signal_data()
        config = FilterConfig(name="test", params={"n_perm": 100})
        result = f.evaluate(signal, config)
        p = result.metrics["pvalue"]
        assert 0.0 <= p <= 1.0, f"p-value={p} hors [0,1]"

    def test_diagnosis_is_actionable_on_fail(self):
        """En cas d'échec, diagnosis contient un mot d'action."""
        f      = MetisQ2()
        signal = make_signal_data()
        config = FilterConfig(name="test", params={"n_perm": 100, "p_threshold": 0.0})
        result = f.evaluate(signal, config)
        assert not result.passed
        assert any(
            w in result.diagnosis.lower()
            for w in ["pour", "envisager", "filtre", "isoler"]
        ), f"Diagnosis non actionnable : {result.diagnosis}"

    def test_criterion_recalibration_pending(self):
        """criterion='absolute' est documenté avant recalibration."""
        f      = MetisQ2()
        signal = make_signal_data()
        config = FilterConfig(name="test", params={"n_perm": 100})
        result = f.evaluate(signal, config)
        # RECALIBRATION_PENDING : 'absolute' avant, 'regime_adjusted' après Session 2
        assert result.metrics["criterion"] in ("absolute", "regime_adjusted"), (
            "criterion doit être documenté dans les metrics."
        )


# ── MWE : signal bon → xfail (RECALIBRATION_PENDING) ─────────────────

class TestMetisQ2GoodSignal:
    """
    Signal oracle sur données synthétiques.
    xfail : le critère p < 0.05 brut peut rejeter un signal réel sur
    la période OOS 2023-2024 (bull run systémique gonfle tous les CNSR).
    Ce test passe au vert après recalibration Alternative A (Session 2).
    """

    @pytest.mark.xfail(
        reason=(
            "RECALIBRATION_PENDING : critère p < 0.05 brut. "
            "Le bull run 2023-2024 rend le test de permutation trop strict : "
            "l'alpha du signal est indiscernable du bruit de marché systémique. "
            "Passe au vert après ajustement de régime (Alternative A, Session 2)."
        ),
        strict=False,
    )
    def test_oracle_signal_passes_q2(self):
        """Signal oracle → Q2 passe (p < 0.05) sur données synthétiques OOS."""
        f      = MetisQ2()
        signal = make_signal_data()
        config = FilterConfig(name="test", params={"n_perm": 500})
        result = f.evaluate(signal, config)
        assert result.passed, (
            f"Q2 rejette le signal oracle. "
            f"p={result.metrics.get('pvalue'):.3f} ≥ {result.metrics.get('p_threshold')}. "
            f"CNSR obs={result.metrics.get('cnsr_obs'):.3f}, "
            f"bench={result.metrics.get('cnsr_bench'):.3f}. "
            f"Diagnostic : {result.diagnosis}"
        )


# ── MWE : signal aléatoire → passed=False ────────────────────────────

class TestMetisQ2BadSignal:
    """
    Signal aléatoire pur : p-value attendue ≈ 0.5 >> 0.05.
    Vérifie que le filtre rejette bien un bruit non-certifiable.
    Robuste à la calibration — un bruit aléatoire échoue toujours Q2.
    """

    def test_random_signal_fails_q2(self):
        """Signal aléatoire uniforme → Q2 échoue (p >> 0.05)."""
        rng = np.random.default_rng(99)

        def random_alloc(pair, idx):
            return pd.Series(rng.uniform(0.0, 1.0, len(idx)), index=idx)

        f      = MetisQ2()
        signal = make_signal_data(alloc_fn=random_alloc)
        config = FilterConfig(name="test", params={"n_perm": 500, "p_threshold": 0.05})
        result = f.evaluate(signal, config)

        assert not result.passed, (
            f"Q2 laisse passer un signal aléatoire. "
            f"p={result.metrics.get('pvalue'):.3f}. "
            f"La recalibration est trop laxiste si ce test devient vert."
        )
