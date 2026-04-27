"""
tests/test_filter_metis_q4.py
MWE pour MetisQ4(Filter).

Signal oracle (N_trials=1) → DSR élevé → passed=True
Signal aléatoire (N_trials=101, non corrigé) → DSR bas → passed=False
"""

import numpy as np
import pandas as pd

from studio.interfaces import FilterConfig, FilterVerdict, SignalData
from studio.filters.metis_q4 import MetisQ4, compute_n_effectif


# ── Fixture partagée ─────────────────────────────────────────────────

def make_signal_data(alloc_fn=None, n=1922, seed=42):
    """Données synthétiques avec bull run structurel en OOS."""
    np.random.seed(seed)
    idx = pd.date_range("2019-09-27", periods=n, freq="D")

    n_is  = int(n * 0.65)
    n_oos = n - n_is

    btc_r = np.concatenate([
        np.random.normal(0.001, 0.04, n_is),
        np.random.normal(0.004, 0.04, n_oos),
    ])
    paxg_r = np.concatenate([
        np.random.normal(0.0005, 0.01, n_is),
        np.random.normal(0.001,  0.01, n_oos),
    ])

    btc  = pd.Series(100 * np.exp(np.cumsum(btc_r)),  index=idx)
    paxg = pd.Series(100 * np.exp(np.cumsum(paxg_r)), index=idx)
    pair = paxg / btc

    if alloc_fn is None:
        log_r  = np.log(pair)
        trend  = log_r.rolling(120).mean().shift(1)
        vol    = log_r.diff().rolling(30).std()
        vmed   = vol.expanding().median()
        high_v = vol > 2.0 * vmed
        alloc  = pd.Series(0.5, index=idx)
        alloc  = alloc.where(log_r >= trend, other=0.75)
        alloc  = alloc.where(log_r <  trend, other=0.25)
        alloc  = alloc.where(~high_v, other=0.5)
        alloc  = alloc.fillna(0.5).clip(0.25, 0.75)
    else:
        alloc = alloc_fn(pair, btc, paxg, idx)

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


# ── Tests compute_n_effectif ─────────────────────────────────────────

class TestComputeNEffectif:

    def test_single_trial_unchanged(self):
        """N=1 → N_effectif=1 quelle que soit la corrélation."""
        assert compute_n_effectif(1, avg_correlation=0.97) == 1

    def test_no_correction_when_no_correlation(self):
        """Sans corrélation fournie → N_effectif = N_trials."""
        assert compute_n_effectif(101, avg_correlation=None) == 101

    def test_high_correlation_reduces_n(self):
        """Corrélation 0.97 sur N=101 → N_effectif ≈ 3."""
        n_eff = compute_n_effectif(101, avg_correlation=0.97)
        assert 1 <= n_eff <= 10, (
            f"N_effectif={n_eff} hors de la plage attendue [1, 10] "
            f"pour N=101 et corrélation=0.97."
        )

    def test_zero_correlation_unchanged(self):
        """Corrélation 0 → variantes indépendantes → N_effectif = N_trials."""
        assert compute_n_effectif(101, avg_correlation=0.0) == 101

    def test_result_always_at_least_one(self):
        """N_effectif est toujours ≥ 1."""
        for avg_corr in [0.99, 0.999, 1.0]:
            assert compute_n_effectif(101, avg_correlation=avg_corr) >= 1


# ── MWE : signal oracle (N=1) → passed=True ─────────────────────────

class TestMetisQ4OracleSignal:

    def test_oracle_n1_passes_q4(self):
        """
        Signal oracle avec N_trials=1 → DSR élevé → passed=True.

        N=1 signifie : signal unique, pas d'optimisation de paramètre.
        Le DSR avec N=1 est proche de la probabilité que le CNSR
        soit positif — élevé si le signal est raisonnable.
        """
        f      = MetisQ4()
        signal = make_signal_data()
        config = FilterConfig(
            name="test",
            params={"n_trials": 1, "dsr_threshold": 0.95},
        )
        result = f.evaluate(signal, config)

        assert result.passed, (
            f"Signal oracle échoue Q4 avec N=1. "
            f"DSR={result.metrics['dsr']:.4f} < {result.metrics['dsr_threshold']}. "
            f"Le signal oracle doit être statistiquement défendable "
            f"pour un signal unique non optimisé. "
            f"Diagnosis : {result.diagnosis}"
        )

    def test_n_effectif_is_one_for_oracle(self):
        """Avec N_trials=1, N_effectif=1 (pas de correction)."""
        f      = MetisQ4()
        signal = make_signal_data()
        config = FilterConfig(name="test", params={"n_trials": 1})
        result = f.evaluate(signal, config)
        assert result.metrics["n_effectif"] == 1

    def test_n_effectif_corrected_for_ema_family(self):
        """
        Avec N_trials=101 et corrélation=0.97 (famille EMA),
        N_effectif ≤ 10 — correction Bailey-LdP active.
        """
        f      = MetisQ4()
        signal = make_signal_data()
        config = FilterConfig(
            name="test",
            params={
                "n_trials":        101,
                "avg_correlation": 0.97,
                "dsr_threshold":   0.95,
            },
        )
        result = f.evaluate(signal, config)
        assert result.metrics["n_effectif"] <= 10, (
            f"N_effectif={result.metrics['n_effectif']} — "
            f"la correction Bailey-LdP n'est pas appliquée correctement."
        )


# ── MWE : signal aléatoire → passed=False ───────────────────────────

class TestMetisQ4BadSignal:

    def test_random_signal_fails_q4(self):
        """
        Signal aléatoire avec N_trials=101 (non corrigé) → DSR bas → passed=False.

        Ce test vérifie que Q4 rejette un signal optimisé sur 101 variantes.
        Il doit rester vert même après recalibration (N_effectif ne
        sauve pas un signal vraiment aléatoire).
        """
        np.random.seed(77)

        def random_alloc_fn(pair, btc, paxg, idx):
            return pd.Series(
                np.random.uniform(0.25, 0.75, len(idx)),
                index=idx,
            )

        signal = make_signal_data(alloc_fn=random_alloc_fn)
        f      = MetisQ4()
        config = FilterConfig(
            name="test",
            params={
                "n_trials":        101,
                "avg_correlation": None,
                "dsr_threshold":   0.95,
            },
        )
        result = f.evaluate(signal, config)

        assert not result.passed, (
            f"Q4 laisse passer un signal aléatoire avec N=101. "
            f"DSR={result.metrics['dsr']:.4f}. "
            f"Si ce test est vert, Q4 est trop laxiste."
        )
