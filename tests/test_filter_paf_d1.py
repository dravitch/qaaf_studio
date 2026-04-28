"""
tests/test_filter_paf_d1.py
MWE pour PAFD1Hierarchy(Filter).

Signal 75% PAXG sur régime BTC bear → HIERARCHIE_CONFIRMEE → passed=True
Signal 25% PAXG (lourd BTC) sur même régime → B_PASSIF_DOMINE → passed=False

Régime MWE OOS : fort bear BTC (drift=-0.004/j), PAXG stable (+0.001/j).
MR_pur achète BTC sur chaque rebond → se retrouve lourd BTC → détruit.
Signal 75% PAXG évite BTC → bat MR_pur et B_5050 → HIERARCHIE_CONFIRMEE.
Signal 25% PAXG (75% BTC) est dominé par B_5050 → B_PASSIF_DOMINE.
"""

import numpy as np
import pandas as pd

from studio.interfaces import FilterConfig, FilterVerdict, SignalData
from studio.filters.paf_d1_hierarchy import PAFD1Hierarchy


# ── Helpers ──────────────────────────────────────────────────────────

def make_signal_btc_bear(alloc_btc_value=0.25, n=1200, seed=42):
    """
    SignalData avec BTC bear en OOS (drift_btc=-0.004/j, drift_paxg=+0.001/j).

    alloc_btc=0.25 → signal_ref = 75% PAXG (évite BTC bear) → passe D1.
    alloc_btc=0.75 → signal_ref = 25% PAXG (lourd BTC bear)  → échoue D1.

    IS neutre (drift≈0), OOS bear : les 400 derniers jours.
    """
    np.random.seed(seed)
    idx  = pd.date_range("2020-01-01", periods=n, freq="D")
    n_is = 800

    btc_r  = np.concatenate([
        np.random.normal(0.0, 0.02, n_is),
        np.random.normal(-0.004, 0.02, n - n_is),
    ])
    paxg_r = np.concatenate([
        np.random.normal(0.0, 0.01, n_is),
        np.random.normal(0.001, 0.01, n - n_is),
    ])

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


# ── Tests du contrat PAFD1 ────────────────────────────────────────────

class TestPAFD1Contract:

    def test_returns_filter_verdict(self):
        """evaluate() retourne toujours un FilterVerdict."""
        f      = PAFD1Hierarchy()
        signal = make_signal_btc_bear()
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        assert isinstance(result, FilterVerdict)

    def test_filter_name_matches(self):
        """filter_name correspond à PAFD1Hierarchy.NAME."""
        f      = PAFD1Hierarchy()
        signal = make_signal_btc_bear()
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        assert result.filter_name == PAFD1Hierarchy.NAME

    def test_metrics_contains_required_keys(self):
        """Le verdict contient les clés de D1Result."""
        f      = PAFD1Hierarchy()
        signal = make_signal_btc_bear()
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        for key in ("verdict", "mr_pur_cnsr", "signal_ref_cnsr",
                    "b_5050_cnsr", "delta_ref_vs_mr"):
            assert key in result.metrics, f"Clé manquante : {key}"

    def test_verdict_is_valid_d1_value(self):
        """verdict est une des 4 valeurs valides de D1."""
        f      = PAFD1Hierarchy()
        signal = make_signal_btc_bear()
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        assert result.metrics["verdict"] in {
            "HIERARCHIE_CONFIRMEE", "PARTIELLE",
            "B_PASSIF_DOMINE", "STOP"
        }

    def test_cnsr_values_are_numeric(self):
        """Tous les CNSR dans metrics sont des float non-NaN."""
        f      = PAFD1Hierarchy()
        signal = make_signal_btc_bear()
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        import math
        for key in ("mr_pur_cnsr", "signal_ref_cnsr", "b_5050_cnsr"):
            val = result.metrics[key]
            assert isinstance(val, float) and not math.isnan(val), f"{key} invalide"


# ── MWE : oracle sur PAXG bull → passed=True ────────────────────────

class TestPAFD1GoodSignal:

    def test_paxg_heavy_btc_bear_passes_d1(self):
        """
        Signal 75% PAXG (alloc_btc=0.25) sur régime BTC bear → HIERARCHIE_CONFIRMEE → passed=True.

        En BTC bear, MR_pur rachète BTC à chaque rebond → se retrouve lourd BTC → détruit.
        75% PAXG évite BTC → bat MR_pur et B_5050 (50% BTC) → HIERARCHIE_CONFIRMEE.
        """
        f      = PAFD1Hierarchy()
        signal = make_signal_btc_bear(alloc_btc_value=0.25)
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        assert result.passed, (
            f"PAF D1 rejette 75% PAXG sur BTC bear. "
            f"verdict={result.metrics['verdict']}, "
            f"signal_ref={result.metrics['signal_ref_cnsr']:.3f}, "
            f"MR_pur={result.metrics['mr_pur_cnsr']:.3f}, "
            f"B_5050={result.metrics['b_5050_cnsr']:.3f}. "
            f"Diagnosis : {result.diagnosis}"
        )


# ── MWE : signal lourd BTC en bear → passed=False ────────────────────

class TestPAFD1BadSignal:

    def test_btc_heavy_btc_bear_fails_d1(self):
        """
        Signal 25% PAXG (alloc_btc=0.75, lourd BTC) sur régime BTC bear → passed=False.

        En BTC bear, tenir 75% BTC est pire que B_5050 (50% BTC).
        B_5050 domine → B_PASSIF_DOMINE ou STOP.
        """
        f      = PAFD1Hierarchy()
        signal = make_signal_btc_bear(alloc_btc_value=0.75)
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        assert not result.passed, (
            f"PAF D1 laisse passer un signal 25% PAXG en BTC bear. "
            f"verdict={result.metrics['verdict']}, "
            f"signal_ref={result.metrics['signal_ref_cnsr']:.3f}, "
            f"B_5050={result.metrics['b_5050_cnsr']:.3f}."
        )
