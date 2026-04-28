"""
tests/test_filter_paf_d2.py
MWE pour PAFD2Attribution(Filter).

Signal 75% PAXG vs 50/50 sur régime BTC bear → COMPOSANTE_ACTIVE → passed=True
Signal constant 50/50 vs 50/50 → NEUTRE → passed=False

Régime MWE OOS : BTC bear (drift=-0.004/j), PAXG stable.
Signal 75% PAXG (alloc_btc=0.25) >> 50/50 → delta CNSR > seuil_actif=0.05.
Signal 50/50 (alloc_btc=0.5) vs 50/50 → delta=0 → NEUTRE.
"""

import numpy as np
import pandas as pd

from studio.interfaces import FilterConfig, FilterVerdict, SignalData
from studio.filters.paf_d2_attribution import PAFD2Attribution


# ── Helpers ──────────────────────────────────────────────────────────

def make_signal_btc_bear(alloc_btc_value=0.25, n=1200, seed=42):
    """
    SignalData avec BTC bear en OOS (drift_btc=-0.004/j, drift_paxg=+0.001/j).

    alloc_btc=0.25 → signal_complet = 75% PAXG (ajoute de la valeur en BTC bear).
    alloc_btc=0.50 → signal_complet = 50/50 = signal_sans (delta=0 → NEUTRE).
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


# ── Tests du contrat PAFD2 ────────────────────────────────────────────

class TestPAFD2Contract:

    def test_returns_filter_verdict(self):
        """evaluate() retourne toujours un FilterVerdict."""
        f      = PAFD2Attribution()
        signal = make_signal_btc_bear()
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        assert isinstance(result, FilterVerdict)

    def test_filter_name_matches(self):
        """filter_name correspond à PAFD2Attribution.NAME."""
        f      = PAFD2Attribution()
        signal = make_signal_btc_bear()
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        assert result.filter_name == PAFD2Attribution.NAME

    def test_metrics_contains_required_keys(self):
        """Le verdict contient verdict, cnsr_avec, cnsr_sans, delta."""
        f      = PAFD2Attribution()
        signal = make_signal_btc_bear()
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        for key in ("verdict", "cnsr_avec", "cnsr_sans", "delta", "seuil_actif"):
            assert key in result.metrics, f"Clé manquante : {key}"

    def test_verdict_is_valid_d2_value(self):
        """verdict est une des 3 valeurs valides de D2."""
        f      = PAFD2Attribution()
        signal = make_signal_btc_bear()
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        assert result.metrics["verdict"] in {"COMPOSANTE_ACTIVE", "NEUTRE", "DEGRADANTE"}

    def test_delta_equals_cnsr_avec_minus_sans(self):
        """delta == cnsr_avec - cnsr_sans (invariant D2Result)."""
        f      = PAFD2Attribution()
        signal = make_signal_btc_bear()
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        computed = round(result.metrics["cnsr_avec"] - result.metrics["cnsr_sans"], 4)
        assert abs(computed - result.metrics["delta"]) < 1e-6, (
            f"Invariant delta brisé : {computed} ≠ {result.metrics['delta']}"
        )

    def test_neutral_signal_is_neutre(self):
        """Allocation 50/50 vs signal_sans (50/50) → NEUTRE (delta=0)."""
        f      = PAFD2Attribution()
        signal = make_signal_btc_bear(alloc_btc_value=0.5)
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        assert result.metrics["verdict"] == "NEUTRE"
        assert result.metrics["delta"] == 0.0


# ── MWE : 75% PAXG vs 50/50 en BTC bear → passed=True ───────────────

class TestPAFD2GoodSignal:

    def test_paxg_heavy_btc_bear_passes_d2(self):
        """
        75% PAXG (alloc_btc=0.25) vs 50/50 statique sur BTC bear → COMPOSANTE_ACTIVE → passed=True.

        La règle d'allocation (éviter BTC bear) ajoute de la valeur vs neutre 50/50.
        delta CNSR > seuil_actif=0.05.
        """
        f      = PAFD2Attribution()
        signal = make_signal_btc_bear(alloc_btc_value=0.25)
        config = FilterConfig(name="test", params={"seuil_actif": 0.05})
        result = f.evaluate(signal, config)
        assert result.passed, (
            f"PAF D2 rejette 75% PAXG vs 50/50 sur BTC bear. "
            f"verdict={result.metrics['verdict']}, "
            f"delta={result.metrics['delta']:+.4f} ≤ {result.metrics['seuil_actif']}. "
            f"Diagnosis : {result.diagnosis}"
        )


# ── MWE : signal 50/50 constant → passed=False ───────────────────────

class TestPAFD2BadSignal:

    def test_neutral_signal_fails_d2(self):
        """
        Signal 50/50 (alloc_btc=0.5) → signal_complet = signal_sans → delta=0 → NEUTRE → passed=False.

        La règle d'allocation est inexistante — identique à la référence passive.
        """
        f      = PAFD2Attribution()
        signal = make_signal_btc_bear(alloc_btc_value=0.5)
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        assert not result.passed, (
            f"PAF D2 laisse passer un signal constant (50/50). "
            f"verdict={result.metrics['verdict']}, delta={result.metrics['delta']:+.4f}."
        )
