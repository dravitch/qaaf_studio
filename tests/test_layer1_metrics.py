"""
Tests Layer 1 — MetricsEngine (CNSR, DSR)

Fonctionne avec pytest ET unittest :
    python3 -m unittest tests/test_layer1_metrics.py -v
    pytest tests/test_layer1_metrics.py -v          # si pytest installé
"""

import sys
import math
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from tests.conftest import make_synthetic_returns
from layer1_engine.metrics_engine import compute_cnsr, deflated_sharpe_ratio


class TestComputeCNSR(unittest.TestCase):

    def setUp(self):
        self.r_paxg, self.r_btc = make_synthetic_returns()
        self.r_pair = self.r_paxg - self.r_btc

    def test_identity_log_returns(self):
        """r_USD = r_pair + r_base est une identité exacte (< 1e-10 d'erreur)."""
        r_usd_direct   = self.r_paxg
        r_usd_computed = self.r_pair + self.r_btc
        diff = (r_usd_direct - r_usd_computed).abs().max()
        self.assertLess(diff, 1e-10,
                        f"CNSR identity violated: max diff = {diff}")

    def test_returns_expected_keys(self):
        result   = compute_cnsr(self.r_pair, self.r_btc)
        required = {"cnsr_usd_fed", "cnsr_usd_usdc", "cnsr_usd_0",
                    "sortino", "calmar", "omega", "max_dd_pct", "n_obs"}
        self.assertTrue(required.issubset(set(result.keys())))

    def test_higher_rf_lowers_sharpe(self):
        """Rf=0 doit donner un Sharpe ≥ Rf=4%."""
        m0 = compute_cnsr(self.r_pair, self.r_btc, rf_annual=0.0)
        m4 = compute_cnsr(self.r_pair, self.r_btc, rf_annual=0.04)
        self.assertGreaterEqual(
            m0["cnsr_usd_0"], m4["cnsr_usd_fed"] - 1e-6,
            "Rf=0 should give higher or equal Sharpe than Rf=4%"
        )

    def test_max_dd_non_negative(self):
        result = compute_cnsr(self.r_pair, self.r_btc)
        self.assertGreaterEqual(result["max_dd_pct"], 0.0)

    def test_std_zero_returns_nan(self):
        r_zero = pd.Series(np.zeros(100))
        result = compute_cnsr(r_zero, r_zero)
        self.assertTrue(math.isnan(result["cnsr_usd_fed"]))

    def test_misaligned_indexes_handled(self):
        """Index partiellement différents — doit s'aligner sans erreur."""
        idx1   = pd.date_range("2022-01-01", periods=200, freq="B")
        idx2   = pd.date_range("2022-02-01", periods=200, freq="B")
        r_pair = pd.Series(np.random.normal(0, 0.02, 200), index=idx1)
        r_base = pd.Series(np.random.normal(0, 0.03, 200), index=idx2)
        result = compute_cnsr(r_pair, r_base)
        # n_obs doit être l'intersection, pas 200
        self.assertLess(result["n_obs"], 200)


class TestDSR(unittest.TestCase):

    def setUp(self):
        self.r_paxg, _ = make_synthetic_returns()

    def test_range_0_to_1(self):
        dsr = deflated_sharpe_ratio(self.r_paxg, n_trials=10)
        self.assertGreaterEqual(dsr, 0.0)
        self.assertLessEqual(dsr,   1.0)

    def test_decreases_with_more_trials(self):
        """Plus N est grand, plus le DSR diminue pour le même signal."""
        dsr_1   = deflated_sharpe_ratio(self.r_paxg, n_trials=1)
        dsr_10  = deflated_sharpe_ratio(self.r_paxg, n_trials=10)
        dsr_100 = deflated_sharpe_ratio(self.r_paxg, n_trials=100)
        self.assertGreaterEqual(dsr_1,  dsr_10  - 1e-6,
                                f"dsr_1={dsr_1:.3f} dsr_10={dsr_10:.3f}")
        self.assertGreaterEqual(dsr_10, dsr_100 - 1e-6,
                                f"dsr_10={dsr_10:.3f} dsr_100={dsr_100:.3f}")

    def test_short_series_returns_nan(self):
        """T < 30 → nan."""
        r_short = pd.Series(np.random.normal(0, 0.02, 5))
        dsr     = deflated_sharpe_ratio(r_short, n_trials=1)
        self.assertTrue(math.isnan(dsr))

    def test_n101_with_realistic_signal(self):
        """N=101 (grille EMA) avec signal réaliste → DSR fini."""
        rng   = np.random.default_rng(42)
        r_usd = pd.Series(rng.normal(0.0005, 0.018, 380))
        dsr   = deflated_sharpe_ratio(r_usd, n_trials=101)
        self.assertFalse(math.isnan(dsr))
        self.assertGreaterEqual(dsr, 0.0)
        self.assertLessEqual(dsr,   1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
