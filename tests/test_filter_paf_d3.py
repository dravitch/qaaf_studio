"""
tests/test_filter_paf_d3.py
MWE pour PAFD3SignalInformatif(Filter).

Signal oracle 75% PAXG sur BTC bear → SIGNAL_INFORMATIF → passed=True
Signal H9 EMA triviale → même variance → ARTEFACT_LISSAGE → passed=False

Régime MWE OOS : BTC bear (drift=-0.004/j), PAXG stable (+0.001/j).
Signal oracle (alloc_btc=0.25) évite BTC → bat EMA triviale H9 → SIGNAL_INFORMATIF.
Signal H9 EMA span=30 == EMA triviale à iso-variance → delta=0 → ARTEFACT_LISSAGE.
"""

import numpy as np
import pandas as pd

from studio.interfaces import FilterConfig, FilterVerdict, SignalData
from studio.filters.paf_d3_signal_informatif import PAFD3SignalInformatif


# ── Helpers ──────────────────────────────────────────────────────────

def make_signal_btc_bear(alloc_btc_value=0.25, n=1200, seed=42):
    """
    SignalData avec BTC bear en OOS (drift_btc=-0.004/j, drift_paxg=+0.001/j).

    alloc_btc=0.25 → signal = 75% PAXG (actif, évite BTC bear) → SIGNAL_INFORMATIF.
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


def make_signal_h9_artefact(n=1200, seed=42, span=30):
    """
    SignalData dont alloc_paxg = H9 EMA span=30 sur les prix OOS.

    run_d3 triviale = H9 EMA du même span → delta=0 → ARTEFACT_LISSAGE garanti.

    Construction :
      - Générer les mêmes prix que le régime BTC bear
      - Calculer H9 EMA span=30 sur la tranche OOS
      - Stocker comme alloc_btc = 1 - alloc_paxg_h9
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

    oos_start = idx[n_is]
    oos_end   = idx[-1]

    # Calcul du signal H9 EMA span=<span> sur les prix OOS
    # (même formule que _find_ema_span_matching_variance dans run_d3)
    btc_oos  = btc.loc[oos_start:oos_end]
    paxg_oos = paxg.loc[oos_start:oos_end]
    log_ratio = np.log(paxg_oos / btc_oos)
    q25 = log_ratio.rolling(60, min_periods=30).quantile(0.25)
    q75 = log_ratio.rolling(60, min_periods=30).quantile(0.75)
    iqr = (q75 - q25).replace(0, np.nan)
    h9  = 1.0 - ((log_ratio - q25) / iqr).clip(0, 1)
    alloc_paxg_oos = h9.ewm(span=span, adjust=False).mean().clip(0, 1).fillna(0.5)

    # alloc_btc pour toute la série (IS + OOS) ; OOS porte le signal H9
    alloc_btc_full = pd.Series(0.5, index=idx)
    alloc_btc_full.loc[oos_start:oos_end] = (1.0 - alloc_paxg_oos).clip(0.0, 1.0)

    oos_start_date = str(oos_start.date())
    oos_end_date   = str(oos_end.date())
    is_end_date    = str((oos_start - pd.Timedelta(days=1)).date())

    return SignalData(
        alloc_btc=alloc_btc_full,
        prices_pair=pair,
        prices_base_usd=btc,
        prices_quote_usd=paxg,
        is_start="2020-01-01",
        is_end=is_end_date,
        oos_start=oos_start_date,
        oos_end=oos_end_date,
    )


# ── Tests du contrat PAFD3 ────────────────────────────────────────────

class TestPAFD3Contract:

    def test_returns_filter_verdict(self):
        """evaluate() retourne toujours un FilterVerdict."""
        f      = PAFD3SignalInformatif()
        signal = make_signal_btc_bear()
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        assert isinstance(result, FilterVerdict)

    def test_filter_name_matches(self):
        """filter_name correspond à PAFD3SignalInformatif.NAME."""
        f      = PAFD3SignalInformatif()
        signal = make_signal_btc_bear()
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        assert result.filter_name == PAFD3SignalInformatif.NAME

    def test_metrics_contains_required_keys(self):
        """Le verdict contient les clés de D3Result."""
        f      = PAFD3SignalInformatif()
        signal = make_signal_btc_bear()
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        for key in ("verdict", "cnsr_signal", "cnsr_trivial", "delta",
                    "ema_span_used", "std_alloc_signal", "std_alloc_trivial"):
            assert key in result.metrics, f"Clé manquante : {key}"

    def test_verdict_is_valid_d3_value(self):
        """verdict est une des 2 valeurs valides de D3."""
        f      = PAFD3SignalInformatif()
        signal = make_signal_btc_bear()
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        assert result.metrics["verdict"] in {"SIGNAL_INFORMATIF", "ARTEFACT_LISSAGE"}

    def test_delta_equals_cnsr_signal_minus_trivial(self):
        """delta == cnsr_signal - cnsr_trivial (invariant D3Result)."""
        f      = PAFD3SignalInformatif()
        signal = make_signal_btc_bear()
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        computed = round(result.metrics["cnsr_signal"] - result.metrics["cnsr_trivial"], 4)
        # Tolérance 2e-4 : delta stocké avant arrondi individuel de cnsr_signal/cnsr_trivial
        assert abs(computed - result.metrics["delta"]) < 2e-4, (
            f"Invariant delta brisé : {computed} ≠ {result.metrics['delta']}"
        )

    def test_ema_span_is_positive_integer(self):
        """ema_span_used est un entier > 0."""
        f      = PAFD3SignalInformatif()
        signal = make_signal_btc_bear()
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        span = result.metrics["ema_span_used"]
        assert isinstance(span, int) and span > 0, f"ema_span_used invalide : {span}"


# ── MWE : oracle 75% PAXG → SIGNAL_INFORMATIF → passed=True ─────────

class TestPAFD3GoodSignal:

    def test_oracle_btc_bear_passes_d3(self):
        """
        Signal oracle 75% PAXG (alloc_btc=0.25) sur BTC bear → SIGNAL_INFORMATIF → passed=True.

        En BTC bear, l'oracle évite BTC → surperforme l'EMA H9 triviale
        (qui fait de la mean-reversion et rachète BTC sur chaque rebond).
        delta CNSR >> 0.05.
        """
        f      = PAFD3SignalInformatif()
        signal = make_signal_btc_bear(alloc_btc_value=0.25)
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        assert result.passed, (
            f"PAF D3 rejette l'oracle 75% PAXG sur BTC bear. "
            f"verdict={result.metrics['verdict']}, "
            f"cnsr_signal={result.metrics['cnsr_signal']:.4f}, "
            f"cnsr_trivial={result.metrics['cnsr_trivial']:.4f}, "
            f"delta={result.metrics['delta']:+.4f}. "
            f"Diagnosis : {result.diagnosis}"
        )


# ── MWE : signal H9 EMA triviale → ARTEFACT_LISSAGE → passed=False ──

class TestPAFD3BadSignal:

    def test_h9_ema_artefact_fails_d3(self):
        """
        Signal = H9 EMA span=30 sur prix OOS → run_d3 triviale = même H9 span=30 → delta=0.

        Le signal est exactement reproductible par l'EMA triviale interne à run_d3.
        delta=0 ≤ 0.05 → ARTEFACT_LISSAGE → passed=False.
        """
        f      = PAFD3SignalInformatif()
        signal = make_signal_h9_artefact(span=30)
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        assert not result.passed, (
            f"PAF D3 laisse passer un signal H9 EMA trivial. "
            f"verdict={result.metrics['verdict']}, "
            f"delta={result.metrics['delta']:+.4f}."
        )
        assert abs(result.metrics["delta"]) < 1e-3, (
            f"delta={result.metrics['delta']:+.4f} inattendu pour signal H9 EMA trivial. "
            f"Attendu : delta ≈ 0."
        )
