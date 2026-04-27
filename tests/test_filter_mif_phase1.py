"""
tests/test_filter_mif_phase1.py
MWE pour MIFPhase1(Filter).

Deux cas obligatoires (Test Before Trust) :
  - Signal délibérément bon  → passed=True  (xfail : peut échouer avant recalibration)
  - Signal délibérément mauvais → passed=False

Ces tests vérifient le contrat du filtre, pas sa calibration.
La calibration sera validée dans test_signal_oracle_certified.py
après la recalibration (Session 2).
"""

import pytest
import numpy as np
import pandas as pd

from studio.interfaces import FilterConfig, FilterVerdict, FilterError, SignalData
from studio.filters.mif_phase1 import MIFPhase1


# ── Helpers ──────────────────────────────────────────────────────────

def make_signal_data(alloc_value=None, n=1200, seed=42):
    """
    Construit un SignalData minimal pour les tests MIF Phase 1.

    Les prix servent à calculer alloc_btc. run_phase1() génère ses propres
    données synthétiques — le filtre réindexe alloc_btc sur ces dates.
    n=1200 couvre 2019-09-27 → 2023-01, chevauchant les dates synthétiques
    (2020-01-01 → 2022-03-12 pour G5 le plus long).
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2019-09-27", periods=n, freq="D")

    btc  = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.001, 0.04, n))),   index=idx)
    paxg = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, n))),  index=idx)
    pair = paxg / btc

    if alloc_value is None:
        # Oracle convention (alloc_btc) : trend-following asymétrique
        log_ratio = np.log(pair)
        trend     = log_ratio.rolling(120).mean().shift(1)
        vol       = log_ratio.diff().rolling(30).std()
        vol_med   = vol.expanding().median()
        high_vol  = vol > 2.0 * vol_med

        alloc = pd.Series(0.5, index=idx)
        # where(cond, other) : garde où cond=True, remplace où cond=False
        alloc = alloc.where(log_ratio >= trend, other=0.75)  # BTC outperforms → alloc_btc=0.75
        alloc = alloc.where(log_ratio < trend,  other=0.25)  # PAXG outperforms → alloc_btc=0.25
        alloc = alloc.where(~high_vol, other=0.5)
        alloc = alloc.fillna(0.5).clip(0.25, 0.75)
    else:
        alloc = pd.Series(float(alloc_value), index=idx)

    return SignalData(
        alloc_btc=alloc,
        prices_pair=pair,
        prices_base_usd=btc,
        prices_quote_usd=paxg,
        is_start="2020-06-01",
        is_end="2023-05-31",
        oos_start="2023-06-01",
        oos_end="2024-12-30",
    )


# ── Tests du contrat MIFPhase1 ────────────────────────────────────────

class TestMIFPhase1Contract:

    def test_returns_filter_verdict(self):
        """evaluate() retourne toujours un FilterVerdict, jamais None."""
        f      = MIFPhase1()
        signal = make_signal_data(alloc_value=0.5)
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        assert isinstance(result, FilterVerdict)

    def test_filter_name_matches(self):
        """filter_name dans le verdict correspond à MIFPhase1.NAME."""
        f      = MIFPhase1()
        signal = make_signal_data(alloc_value=0.5)
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        assert result.filter_name == MIFPhase1.NAME

    def test_metrics_contains_required_keys(self):
        """Le verdict contient n_pass, n_total, failed, delta_mode."""
        f      = MIFPhase1()
        signal = make_signal_data(alloc_value=0.5)
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        for key in ("n_pass", "n_total", "failed", "delta_mode"):
            assert key in result.metrics, f"Clé manquante dans metrics : {key}"

    def test_regime_detail_has_five_regimes(self):
        """regime_detail contient exactement 5 régimes (G1-G5)."""
        f      = MIFPhase1()
        signal = make_signal_data(alloc_value=0.5)
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        assert result.metrics["n_total"] == 5

    def test_diagnosis_is_actionable_on_fail(self):
        """En cas d'échec, diagnosis contient un mot d'action."""
        f      = MIFPhase1()
        signal = make_signal_data(alloc_value=0.5)
        config = FilterConfig(name="test", params={"min_pass": 6})  # impossible à satisfaire
        result = f.evaluate(signal, config)
        if not result.passed:
            assert any(
                w in result.diagnosis.lower()
                for w in ["pour", "envisager", "réduire", "filtre"]
            ), f"Diagnosis non actionnable : {result.diagnosis}"

    def test_delta_mode_is_documented(self):
        """delta_mode est documenté dans les metrics (trace l'état avant recalibration)."""
        f      = MIFPhase1()
        signal = make_signal_data(alloc_value=0.5)
        config = FilterConfig(name="test", params={})
        result = f.evaluate(signal, config)
        # RECALIBRATION_PENDING : 'absolute' avant recalibration, 'relative' après
        assert result.metrics["delta_mode"] in ("absolute", "relative"), (
            "delta_mode doit être 'absolute' (avant) ou 'relative' (après recalibration)."
        )


# ── MWE : signal bon → passed=True ───────────────────────────────────

class TestMIFPhase1GoodSignal:
    """
    Signal délibérément bon : suiveur de tendance avec filtre de volatilité.
    Construit selon la même logique que le signal oracle.

    xfail souple : peut échouer si MIF Phase 1 est trop strict (critère absolu).
    Rouge = démonstration empirique que la recalibration Session 2 est nécessaire.
    Vert = MIF Phase 1 était déjà bien calibré pour ce signal → recalibration simplifiée.
    """

    @pytest.mark.xfail(
        reason=(
            "MIF Phase 1 avec critère cnsr_strat > -1.0 (absolu) peut rejeter un signal "
            "trend-following raisonnable sur données synthétiques crypto (kurtosis élevé, F2). "
            "Ce test passe au vert après recalibration vers critère relatif (delta). "
            "Voir # RECALIBRATION_PENDING dans mif_phase1.py."
        ),
        strict=False,
    )
    def test_good_signal_passes_phase1(self):
        """Signal oracle (trend-following + filtre vol) → passe Phase 1."""
        f      = MIFPhase1()
        signal = make_signal_data()  # alloc_btc oracle convention
        config = FilterConfig(name="test", params={"min_pass": 3})
        result = f.evaluate(signal, config)
        assert result.passed, (
            f"MIF Phase 1 rejette un signal trend-following raisonnable. "
            f"{result.metrics.get('n_pass')}/{result.metrics.get('n_total')} régimes passent. "
            f"Échecs : {result.metrics.get('failed')}. "
            f"Diagnostic : {result.diagnosis}"
        )


# ── MWE : signal mauvais → passed=False ──────────────────────────────

class TestMIFPhase1BadSignal:
    """
    Signal délibérément mauvais : critère impossible (min_pass=6 pour 5 régimes).
    Vérifie que le filtre rejette bien un signal non-certifiable.
    Indépendant de la calibration — purement structurel.
    """

    def test_impossible_threshold_fails(self):
        """min_pass=6 avec 5 régimes → toujours passed=False."""
        f      = MIFPhase1()
        signal = make_signal_data(alloc_value=0.5)
        config = FilterConfig(name="test", params={"min_pass": 6})
        result = f.evaluate(signal, config)
        assert not result.passed, (
            f"Phase 1 a passé avec min_pass=6 (impossible sur 5 régimes). "
            f"n_pass={result.metrics.get('n_pass')}, n_total={result.metrics.get('n_total')}. "
            f"Bug dans la logique de comptage."
        )

    def test_impossible_threshold_has_actionable_diagnosis(self):
        """Diagnosis après échec min_pass=6 est actionnable."""
        f      = MIFPhase1()
        signal = make_signal_data(alloc_value=0.5)
        config = FilterConfig(name="test", params={"min_pass": 6})
        result = f.evaluate(signal, config)
        assert not result.passed
        assert len(result.diagnosis) >= 20
        assert any(
            w in result.diagnosis.lower()
            for w in ["pour", "envisager", "réduire", "filtre"]
        )
