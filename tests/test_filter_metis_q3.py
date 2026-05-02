"""
tests/test_filter_metis_q3.py
MWE pour MetisQ3(Filter).

Signal standard (régime neutre IS) + spike_ratio=1.5 → pas de spike → passed=True
Même signal + spike_ratio=0.5 → spike détecté (cnsr_target > 0.5×médiane) → passed=False

Q3 teste la stabilité du span EMA sur grille [20, 30, ..., 120] appliquée à H9.
Sur un régime neutre/PAXG-favorable, tous les spans donnent CNSR ≈ 0.7-0.8.
spike_ratio=1.5 : 0.76 > 1.5 × 0.76 = 1.14 → False → PASS.
spike_ratio=0.5 : 0.76 > 0.5 × 0.76 = 0.38 → True  → FAIL.
"""

import numpy as np
import pandas as pd

from studio.interfaces import FilterConfig, FilterVerdict, SignalData
from studio.filters.metis_q3 import MetisQ3


# ── Helpers ──────────────────────────────────────────────────────────

def make_signal_is_data(n=1200, n_is=800, seed=42):
    """
    SignalData avec IS neutre (drift≈0) et OOS quelconque.

    Q3 utilise uniquement la période IS. drift≈0 assure que tous les
    spans H9+EMA donnent des CNSR similaires (pas de spike à span=60).
    """
    np.random.seed(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")

    btc_r  = np.random.normal(0.001, 0.04, n)
    paxg_r = np.random.normal(0.0015, 0.01, n)

    btc  = pd.Series(100 * np.exp(np.cumsum(btc_r)),  index=idx)
    paxg = pd.Series(100 * np.exp(np.cumsum(paxg_r)), index=idx)
    pair = paxg / btc
    alloc = pd.Series(0.5, index=idx)

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


# ── Tests du contrat MetisQ3 ──────────────────────────────────────────

class TestMetisQ3Contract:

    def test_returns_filter_verdict(self):
        """evaluate() retourne toujours un FilterVerdict."""
        f      = MetisQ3()
        signal = make_signal_is_data()
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        assert isinstance(result, FilterVerdict)

    def test_filter_name_matches(self):
        """filter_name correspond à MetisQ3.NAME."""
        f      = MetisQ3()
        signal = make_signal_is_data()
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        assert result.filter_name == MetisQ3.NAME

    def test_metrics_contains_required_keys(self):
        """Le verdict contient les clés Q3."""
        f      = MetisQ3()
        signal = make_signal_is_data()
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        for key in ("verdict", "target_span", "cnsr_target", "median_neighbors",
                    "is_spike", "spike_ratio", "cnsr_by_span"):
            assert key in result.metrics, f"Clé manquante : {key}"

    def test_verdict_is_valid_q3_value(self):
        """verdict est PASS ou FAIL."""
        f      = MetisQ3()
        signal = make_signal_is_data()
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        assert result.metrics["verdict"] in {"PASS", "FAIL"}

    def test_cnsr_by_span_contains_multiple_spans(self):
        """cnsr_by_span contient plusieurs spans (au moins 5)."""
        f      = MetisQ3()
        signal = make_signal_is_data()
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        assert len(result.metrics["cnsr_by_span"]) >= 5

    def test_is_spike_consistent_with_verdict(self):
        """is_spike=True ↔ verdict='FAIL'."""
        f      = MetisQ3()
        signal = make_signal_is_data()
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        m = result.metrics
        if m["is_spike"]:
            assert m["verdict"] == "FAIL"
        else:
            assert m["verdict"] == "PASS"


# ── MWE : spike_ratio=1.5 → PASS ──────────────────────────────────────

class TestMetisQ3GoodSignal:

    def test_smooth_profile_passes_q3(self):
        """
        Régime neutre IS, spike_ratio=1.5 → tous les spans CNSR ≈ 0.76.
        cnsr_target(span=60) ≈ 0.76 ≤ 1.5 × médiane_voisins(0.76) = 1.14 → PASS.
        """
        f      = MetisQ3()
        signal = make_signal_is_data()
        config = FilterConfig(name="test", params={"spike_ratio": 1.5})
        result = f.evaluate(signal, config)
        assert result.passed, (
            f"MÉTIS Q3 détecte un spike inexistant. "
            f"cnsr_target={result.metrics['cnsr_target']}, "
            f"median_neighbors={result.metrics['median_neighbors']}, "
            f"is_spike={result.metrics['is_spike']}. "
            f"Diagnosis : {result.diagnosis}"
        )
        assert not result.metrics["is_spike"]


# ── MWE : spike_ratio=0.5 → FAIL ────────────────────────────────────

class TestMetisQ3BadSignal:

    def test_low_spike_ratio_fails_q3(self):
        """
        Même régime, spike_ratio=0.5 → cnsr_target(0.76) > 0.5 × médiane(0.76) → spike → FAIL.

        Test structurel : spike_ratio=0.5 détecte tout span positif
        comme sur-ajusté dès que ses voisins sont aussi positifs.
        """
        f      = MetisQ3()
        signal = make_signal_is_data()
        config = FilterConfig(name="test", params={"spike_ratio": 0.5})
        result = f.evaluate(signal, config)
        assert not result.passed, (
            f"MÉTIS Q3 ne détecte pas le spike avec spike_ratio=0.5. "
            f"cnsr_target={result.metrics['cnsr_target']}, "
            f"median_neighbors={result.metrics['median_neighbors']}, "
            f"is_spike={result.metrics['is_spike']}."
        )
        assert result.metrics["is_spike"]
