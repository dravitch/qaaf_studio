"""
tests/test_filter_mif_phase2.py
MWE pour MIFPhase2(Filter).

Signal oracle → 4/4 paires M1-M4 → passed=True (gate_ratio=0.75 par défaut)
Seuil impossible (gate_ratio=1.01 > 4/4=1.0) → passed=False

Gate 75% : 3/4 paires minimum doivent passer.
L'algorithme oracle (trend-following asymétrique) produit des allocations
finies sur toutes les paires → 4/4 passent → gate 100% ≥ 75%.
"""

import numpy as np
import pandas as pd

from studio.interfaces import FilterConfig, FilterVerdict, SignalData
from studio.filters.mif_phase2 import MIFPhase2


# ── Helpers ──────────────────────────────────────────────────────────

def make_signal_data(n=1200, seed=42):
    """
    SignalData minimal pour MIF Phase 2.

    Le filtre re-calcule l'algorithme oracle sur données synthétiques.
    Les prix servent uniquement à satisfaire le contrat SignalData.
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


# ── Tests du contrat MIFPhase2 ────────────────────────────────────────

class TestMIFPhase2Contract:

    def test_returns_filter_verdict(self):
        """evaluate() retourne toujours un FilterVerdict."""
        f      = MIFPhase2()
        signal = make_signal_data()
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        assert isinstance(result, FilterVerdict)

    def test_filter_name_matches(self):
        """filter_name correspond à MIFPhase2.NAME."""
        f      = MIFPhase2()
        signal = make_signal_data()
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        assert result.filter_name == MIFPhase2.NAME

    def test_metrics_contains_required_keys(self):
        """Le verdict contient n_pass, n_total, gate_ratio, ratio, failed, pairs."""
        f      = MIFPhase2()
        signal = make_signal_data()
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        for key in ("n_pass", "n_total", "gate_ratio", "ratio", "failed", "pairs"):
            assert key in result.metrics, f"Clé manquante : {key}"

    def test_n_total_is_four(self):
        """n_total == 4 paires M1-M4."""
        f      = MIFPhase2()
        signal = make_signal_data()
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        assert result.metrics["n_total"] == 4

    def test_ratio_coherent_with_n_pass(self):
        """ratio == n_pass / n_total."""
        f      = MIFPhase2()
        signal = make_signal_data()
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        m = result.metrics
        expected = round(m["n_pass"] / m["n_total"], 4)
        assert abs(m["ratio"] - expected) < 1e-6

    def test_failed_is_list(self):
        """failed est toujours une liste."""
        f      = MIFPhase2()
        signal = make_signal_data()
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        assert isinstance(result.metrics["failed"], list)


# ── MWE : algorithme oracle → passed=True ────────────────────────────

class TestMIFPhase2GoodSignal:

    def test_oracle_algorithm_passes_four_pairs(self):
        """
        Algorithme oracle → 4/4 paires avec CNSR fini → gate 100% ≥ 75% → passed=True.

        L'oracle produit des allocations valides sur M1 (PAXG/BTC replica),
        M2 (ETH/BTC), M3 (SOL/BTC), M4 (BNB/BTC).
        """
        f      = MIFPhase2()
        signal = make_signal_data()
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        assert result.passed, (
            f"MIF Phase 2 rejette l'algorithme oracle. "
            f"n_pass={result.metrics['n_pass']}/{result.metrics['n_total']}. "
            f"Paires échouées : {result.metrics['failed']}. "
            f"Diagnosis : {result.diagnosis}"
        )
        assert result.metrics["n_pass"] == 4


# ── MWE : seuil impossible → passed=False ────────────────────────────

class TestMIFPhase2BadSignal:

    def test_impossible_gate_ratio_fails_phase2(self):
        """
        gate_ratio=1.01 → 4/4=1.0 < 1.01 → impossible → passed=False.

        Test structurel : vérifie que le filtre rejette quand le gate
        ne peut pas être atteint, indépendamment du signal.
        """
        f      = MIFPhase2()
        signal = make_signal_data()
        config = FilterConfig(name="test", params={"gate_ratio": 1.01})
        result = f.evaluate(signal, config)
        assert not result.passed, (
            f"MIF Phase 2 a passé avec gate_ratio=1.01 (impossible sur 4/4=100%). "
            f"ratio={result.metrics['ratio']}, gate_ratio={result.metrics['gate_ratio']}."
        )

    def test_impossible_gate_has_actionable_diagnosis(self):
        """Diagnosis après échec est actionnable (≥ 20 caractères)."""
        f      = MIFPhase2()
        signal = make_signal_data()
        config = FilterConfig(name="test", params={"gate_ratio": 1.01})
        result = f.evaluate(signal, config)
        assert not result.passed
        assert len(result.diagnosis) >= 20
        assert any(
            w in result.diagnosis.lower()
            for w in ["vérifier", "paires", "échouées", "gate_ratio", "finies"]
        ), f"Diagnosis non actionnable : {result.diagnosis}"
