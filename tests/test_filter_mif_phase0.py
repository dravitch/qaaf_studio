"""
tests/test_filter_mif_phase0.py
MWE pour MIFPhase0(Filter).

Signal oracle → 6/6 tests T1-T6 → passed=True
Seuil impossible (min_pass=7 > 6 tests) → passed=False

Les 6 tests T1-T6 vérifient la stabilité algorithmique :
- T1-T4, T6 : _stable(alloc) + CNSR fini
- T3 : std_alloc > 0.01
- T5 : asymétrie directionnelle (half_delta > 0.001)
L'algorithme oracle (trend-following asymétrique) passe tous les tests.
"""

import numpy as np
import pandas as pd

from studio.interfaces import FilterConfig, FilterVerdict, SignalData
from studio.filters.mif_phase0 import MIFPhase0


# ── Helpers ──────────────────────────────────────────────────────────

def make_signal_data(n=1200, seed=42):
    """
    SignalData minimal pour MIF Phase 0.

    Les allocations ne sont pas utilisées par le filtre (il re-calcule
    l'algorithme oracle sur données synthétiques). Les prix servent
    uniquement à satisfaire le contrat SignalData.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    btc  = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.001, 0.04, n))), index=idx)
    paxg = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.001, 0.01, n))), index=idx)
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


# ── Tests du contrat MIFPhase0 ────────────────────────────────────────

class TestMIFPhase0Contract:

    def test_returns_filter_verdict(self):
        """evaluate() retourne toujours un FilterVerdict."""
        f      = MIFPhase0()
        signal = make_signal_data()
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        assert isinstance(result, FilterVerdict)

    def test_filter_name_matches(self):
        """filter_name correspond à MIFPhase0.NAME."""
        f      = MIFPhase0()
        signal = make_signal_data()
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        assert result.filter_name == MIFPhase0.NAME

    def test_metrics_contains_required_keys(self):
        """Le verdict contient n_pass, n_total, min_pass, failed, tests."""
        f      = MIFPhase0()
        signal = make_signal_data()
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        for key in ("n_pass", "n_total", "min_pass", "failed", "tests"):
            assert key in result.metrics, f"Clé manquante : {key}"

    def test_n_total_is_six(self):
        """n_total == 6 tests T1-T6."""
        f      = MIFPhase0()
        signal = make_signal_data()
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        assert result.metrics["n_total"] == 6

    def test_failed_is_list(self):
        """failed est toujours une liste."""
        f      = MIFPhase0()
        signal = make_signal_data()
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        assert isinstance(result.metrics["failed"], list)

    def test_n_pass_coherent_with_failed(self):
        """n_pass + len(failed) == n_total."""
        f      = MIFPhase0()
        signal = make_signal_data()
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        m = result.metrics
        assert m["n_pass"] + len(m["failed"]) == m["n_total"]


# ── MWE : algorithme oracle → passed=True ────────────────────────────

class TestMIFPhase0GoodSignal:

    def test_oracle_algorithm_passes_all_isolation_tests(self):
        """
        Algorithme oracle (trend-following asymétrique) → T1-T6 : 6/6 → passed=True.

        L'oracle varie son allocation (std_alloc > 0.01 ✓)
        et réagit différemment dans la 1ère et 2ème moitié (half_delta > 0.001 ✓).
        """
        f      = MIFPhase0()
        signal = make_signal_data()
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        assert result.passed, (
            f"MIF Phase 0 rejette l'algorithme oracle. "
            f"n_pass={result.metrics['n_pass']}/{result.metrics['n_total']}. "
            f"Tests échoués : {result.metrics['failed']}. "
            f"Diagnosis : {result.diagnosis}"
        )
        assert result.metrics["n_pass"] == 6


# ── MWE : seuil impossible → passed=False ────────────────────────────

class TestMIFPhase0BadSignal:

    def test_impossible_threshold_fails_phase0(self):
        """
        min_pass=7 pour 6 tests → impossible → passed=False.

        Test structurel : vérifie que le filtre rejette quand le seuil
        ne peut pas être atteint, indépendamment du signal.
        """
        f      = MIFPhase0()
        signal = make_signal_data()
        config = FilterConfig(name="test", params={"min_pass": 7})
        result = f.evaluate(signal, config)
        assert not result.passed, (
            f"MIF Phase 0 a passé avec min_pass=7 (impossible sur 6 tests). "
            f"n_pass={result.metrics['n_pass']}, n_total={result.metrics['n_total']}."
        )

    def test_impossible_threshold_has_actionable_diagnosis(self):
        """Diagnosis après échec est actionnable (≥ 20 caractères)."""
        f      = MIFPhase0()
        signal = make_signal_data()
        config = FilterConfig(name="test", params={"min_pass": 7})
        result = f.evaluate(signal, config)
        assert not result.passed
        assert len(result.diagnosis) >= 20
        assert any(
            w in result.diagnosis.lower()
            for w in ["corriger", "minimum", "tests", "échoués", "revoir"]
        ), f"Diagnosis non actionnable : {result.diagnosis}"
