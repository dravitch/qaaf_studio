"""
tests/test_signal_oracle_certified.py
Certification du signal oracle QAAF Studio 3.1.

Structure :
  TestOracleSignalInvariants   — 5 tests invariants (toujours VERTS)
  TestOracleCertifiedByPipeline — 3 tests pipeline (SKIPPED avant recalibration)
  test_pipeline_certification_marker_exists — 1 test marqueur (toujours VERT)

Critère de succès Phase 2 :
  - AVANT recalibration : invariants VERTS, pipeline SKIPPED.
  - APRÈS recalibration (Livrable 4) : tous VERTS.
"""

import pytest
import numpy as np
import pandas as pd

from studio.oracle import compute_oracle_signal, ORACLE_PARAMS


# ─── Tentative d'import du pipeline ──────────────────────────────────────────
# Les noms ci-dessous sont volontairement ceux de l'API cible post-recalibration.
# Ils ne correspondent pas encore à l'API actuelle → PIPELINE_AVAILABLE = False
# → les tests de certification sont SKIPPED jusqu'au Livrable 4.

try:
    from layer2_qualification.mif import MetisQ1WalkForward          # noqa: F401
    from layer1_engine.data_loader import make_synthetic_paxg_btc    # noqa: F401
    from layer3_validation.metis import MetisQ2Permutation           # noqa: F401
    from layer3_validation.dsr import DSRFilter                      # noqa: F401
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False

_SKIP_PIPELINE = pytest.mark.skipif(
    not PIPELINE_AVAILABLE,
    reason=(
        "Pipeline non recalibré (MIF G1/G3, MÉTIS Q2, DSR N_effectif). "
        "Ces tests passent au vert après le Livrable 4."
    ),
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

def _make_prices(n: int = 800, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2019-01-01", periods=n, freq="D")
    return pd.Series(
        rng.lognormal(0, 0.02, n).cumprod() * 2000.0,
        index=idx,
        name="paxg_btc",
    )


# ─── TestOracleSignalInvariants ───────────────────────────────────────────────

class TestOracleSignalInvariants:
    """Invariants du signal oracle — indépendants du pipeline."""

    def test_alloc_within_bounds(self):
        """alloc_btc est toujours dans [alloc_low, alloc_high]."""
        alloc = compute_oracle_signal(_make_prices())
        lo, hi = ORACLE_PARAMS["alloc_low"], ORACLE_PARAMS["alloc_high"]
        assert alloc.between(lo, hi).all(), (
            f"Allocations hors [{lo}, {hi}] : {alloc[~alloc.between(lo, hi)]}"
        )

    def test_no_nan_after_warmup(self):
        """Aucun NaN après la période de chauffe (trend_window + vol_window)."""
        prices = _make_prices(n=800)
        warmup = ORACLE_PARAMS["trend_window"] + ORACLE_PARAMS["vol_window"]
        alloc  = compute_oracle_signal(prices)
        assert alloc.iloc[warmup:].isna().sum() == 0, (
            "NaN présents après la période de chauffe."
        )

    def test_signal_is_variable(self):
        """Le signal produit au moins 2 valeurs distinctes (pas constant)."""
        alloc  = compute_oracle_signal(_make_prices())
        warmup = ORACLE_PARAMS["trend_window"] + ORACLE_PARAMS["vol_window"]
        assert alloc.iloc[warmup:].nunique() >= 2, (
            "Le signal est constant — il ne module pas l'allocation."
        )

    def test_no_lookahead(self):
        """Ajouter un point futur ne modifie pas les allocations passées."""
        prices_base  = _make_prices(n=500)
        prices_extra = pd.concat([
            prices_base,
            pd.Series([prices_base.iloc[-1] * 10.0],
                      index=[prices_base.index[-1] + pd.Timedelta(days=1)]),
        ])
        alloc_base  = compute_oracle_signal(prices_base)
        alloc_extra = compute_oracle_signal(prices_extra)
        shared      = alloc_base.index
        pd.testing.assert_series_equal(
            alloc_base.loc[shared],
            alloc_extra.loc[shared],
            check_names=False,
            check_freq=False,
        )

    def test_positive_cnsr_on_synthetic(self):
        """Le signal oracle produit un CNSR > 0 sur des données synthétiques tendancielles."""
        rng = np.random.default_rng(42)
        n   = 1500
        idx = pd.date_range("2019-01-01", periods=n, freq="D")
        # Série avec tendance baissière marquée (BTC surperforme → long BTC profitable)
        drift  = -0.0015
        prices = pd.Series(
            np.exp(np.cumsum(rng.normal(drift, 0.02, n))) * 2000.0,
            index=idx,
        )
        alloc    = compute_oracle_signal(prices)
        log_ret  = np.log(prices / prices.shift(1)).dropna()
        oos_start = pd.Timestamp("2022-01-01")
        alloc_oos = alloc.loc[oos_start:].shift(1).dropna()
        ret_oos   = log_ret.loc[alloc_oos.index]
        # Excès vs neutre 50/50 : (alloc_btc - 0.5) * (-r_pair)
        # Positif quand l'oracle penche du bon côté de la tendance.
        excess = (alloc_oos - 0.5) * (-ret_oos)
        cnsr   = excess.mean() / (excess.std() + 1e-12) * np.sqrt(252)
        assert cnsr > 0, (
            f"CNSR OOS = {cnsr:.3f} ≤ 0 sur données synthétiques tendancielles. "
            f"Vérifier la logique de compute_oracle_signal."
        )


# ─── TestOracleCertifiedByPipeline ───────────────────────────────────────────

@pytest.mark.pipeline_certification
class TestOracleCertifiedByPipeline:
    """
    Tests de certification complète — SKIPPED avant recalibration du pipeline.

    Ces tests passent au vert quand MIF G1/G3, MÉTIS Q2 et DSR N_effectif
    sont recalibrés (Livrable 4).
    """

    @_SKIP_PIPELINE
    def test_mif_phase1_generalization(self):
        """MIF Phase 1 : le signal oracle généralise (G1 trend + G3 vol filter)."""
        prices = _make_prices(n=1800)
        alloc  = compute_oracle_signal(prices)
        filter_mif = MetisQ1WalkForward()  # type: ignore[name-defined]
        signal = make_synthetic_paxg_btc(alloc)  # type: ignore[name-defined]
        verdict = filter_mif.evaluate(signal, config=None)
        assert verdict.passed, f"MIF Phase 1 FAIL : {verdict.diagnosis}"

    @_SKIP_PIPELINE
    def test_metis_q2_permutation(self):
        """MÉTIS Q2 : le signal oracle résiste au test de permutation."""
        prices = _make_prices(n=1800)
        alloc  = compute_oracle_signal(prices)
        q2     = MetisQ2Permutation()  # type: ignore[name-defined]
        signal = make_synthetic_paxg_btc(alloc)  # type: ignore[name-defined]
        verdict = q2.evaluate(signal, config=None)
        assert verdict.passed, f"MÉTIS Q2 FAIL : {verdict.diagnosis}"

    @_SKIP_PIPELINE
    def test_metis_q4_dsr(self):
        """MÉTIS Q4 : le DSR de l'oracle dépasse 0.95 avec N_effectif correct."""
        prices = _make_prices(n=1800)
        alloc  = compute_oracle_signal(prices)
        dsr_f  = DSRFilter()  # type: ignore[name-defined]
        signal = make_synthetic_paxg_btc(alloc)  # type: ignore[name-defined]
        verdict = dsr_f.evaluate(signal, config=None)
        assert verdict.passed, f"MÉTIS Q4 DSR FAIL : {verdict.diagnosis}"


# ─── Marqueur ────────────────────────────────────────────────────────────────

def test_pipeline_certification_marker_exists():
    """Le marqueur pytest pipeline_certification est enregistré dans pyproject.toml."""
    import subprocess
    result = subprocess.run(
        ["python", "-m", "pytest", "--markers"],
        capture_output=True, text=True,
    )
    assert "pipeline_certification" in result.stdout, (
        "Marqueur 'pipeline_certification' absent. "
        "Ajouter markers = [...] dans [tool.pytest.ini_options] de pyproject.toml."
    )
