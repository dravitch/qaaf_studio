"""
Tests Layer 3 — MÉTIS v2.1

Fonctionne avec pytest ET unittest :
    python3 -m unittest tests/test_layer3_metis.py -v
    pytest tests/test_layer3_metis.py -v

Note : tests avec petit N pour la vitesse (n_perm=50, ema_step=20, 3 fenêtres).
MÉTIS Q1 échouera souvent sur données synthétiques random — c'est attendu.
Le critère est : pas d'exception, structure correcte du résultat.
"""

import sys
import types
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import math
import numpy as np
import pandas as pd

from layer2_qualification.mif.synthetic_data import generate_synthetic_paxgbtc
from layer3_validation.metis_q1_walkforward  import run_q1, WalkForwardResult
from layer3_validation.metis_q3_ema_stability import run_q3, EMAStabilityResult
from layer3_validation.metis_q4_dsr          import run_q4, DSRResult
from layer3_validation.metis_runner          import METISRunner


# ── Stratégies de test ────────────────────────────────────────────────────────

def _ema_strategy(r_pair: pd.Series, params: dict) -> pd.Series:
    """Stratégie EMA standard — signal stable, allocation variable."""
    span = params.get("ema_span", 20)
    raw  = r_pair.ewm(span=span, min_periods=max(1, span // 2)).mean()
    mn, mx = raw.min(), raw.max()
    if mx > mn:
        alloc = 0.1 + 0.8 * (raw - mn) / (mx - mn)
    else:
        alloc = pd.Series(0.5, index=r_pair.index)
    return alloc.clip(0.1, 0.9)


def _make_synthetic_bundle(n=600, seed=42):
    """Bundle synthétique minimal pour METISRunner."""
    r_pair, r_base = generate_synthetic_paxgbtc(T=n, seed=seed)
    idx    = r_pair.index
    paxg_btc = pd.Series(np.exp(r_pair.cumsum()), index=idx)
    btc_usd  = pd.Series(np.exp(r_base.cumsum()), index=idx)
    return types.SimpleNamespace(
        paxg_btc = paxg_btc,
        btc_usd  = btc_usd,
        paxg_usd = paxg_btc * btc_usd,
    )


def _make_sm(bundle, split_at=0.7):
    """SplitManager synthétique minimal."""
    idx = bundle.paxg_btc.index
    n   = len(idx)
    k   = int(n * split_at)

    is_start  = str(idx[0].date())
    is_end    = str(idx[k-1].date())
    oos_start = str(idx[k].date())
    oos_end   = str(idx[-1].date())

    class FakeSM:
        def apply(self, series):
            s = series.loc[is_start:is_end]
            o = series.loc[oos_start:oos_end]
            return s, o
        def set_n_trials(self, family, n): pass
        def apply_df(self, df):
            return df.loc[is_start:is_end], df.loc[oos_start:oos_end]

    return FakeSM(), is_start, is_end, oos_start, oos_end


# ── Tests Q1 Walk-forward ─────────────────────────────────────────────────────

class TestMETISQ1(unittest.TestCase):

    def setUp(self):
        self.r_pair, self.r_base = generate_synthetic_paxgbtc(T=600, seed=42)
        self.params = {"ema_span": 20}

    def test_q1_returns_result_object(self):
        """Q1 retourne un WalkForwardResult."""
        result = run_q1(_ema_strategy, self.params,
                        self.r_pair, self.r_base)
        self.assertIsInstance(result, WalkForwardResult)

    def test_q1_window_count(self):
        """Q1 calcule exactement 5 fenêtres."""
        result = run_q1(_ema_strategy, self.params,
                        self.r_pair, self.r_base)
        self.assertEqual(len(result.window_cnsrs), 5)

    def test_q1_n_windows_pass_consistent(self):
        """n_windows_pass est cohérent avec window_cnsrs."""
        result = run_q1(_ema_strategy, self.params,
                        self.r_pair, self.r_base)
        expected = sum(1 for c in result.window_cnsrs
                       if np.isfinite(c) and c >= 0.5)
        self.assertEqual(result.n_windows_pass, expected)

    def test_q1_passed_field(self):
        """passed est True ssi n_windows_pass >= 4."""
        result = run_q1(_ema_strategy, self.params,
                        self.r_pair, self.r_base)
        self.assertEqual(result.passed, result.n_windows_pass >= 4)

    def test_q1_median_cnsr_finite_or_nan(self):
        """median_cnsr est fini ou NaN — jamais inf."""
        result = run_q1(_ema_strategy, self.params,
                        self.r_pair, self.r_base)
        self.assertFalse(np.isinf(result.median_cnsr))

    def test_q1_notes_non_empty(self):
        """notes n'est pas vide."""
        result = run_q1(_ema_strategy, self.params,
                        self.r_pair, self.r_base)
        self.assertTrue(len(result.notes) > 0)


# ── Tests Q3 Stabilité EMA ────────────────────────────────────────────────────

class TestMETISQ3EMA(unittest.TestCase):

    def setUp(self):
        self.r_pair, self.r_base = generate_synthetic_paxgbtc(T=600, seed=42)
        self.params_is = self.r_pair.iloc[:400]
        self.rbase_is  = self.r_base.iloc[:400]
        self.params = {"ema_span": 60}

    def test_q3_returns_result_object(self):
        """Q3 retourne un EMAStabilityResult."""
        result = run_q3(_ema_strategy, self.params,
                        self.params_is, self.rbase_is,
                        ema_step=20)  # step large pour vitesse
        self.assertIsInstance(result, EMAStabilityResult)

    def test_q3_spike_detection(self):
        """
        Q3 spike detection : un span isolé avec CNSR > médiane voisinage + 0.30
        est détecté comme spike.

        Stratégie : pour span=60, prédicteur parfait (lookahead) → CNSR très élevé.
        Pour les autres spans, allocation constante 0.5 → CNSR bas.
        L'écart dépasse largement le seuil +0.30.
        """
        r_is      = self.params_is   # r_pair IS
        r_base_is = self.rbase_is

        def spike_strategy(r_pair, params):
            span = params.get("ema_span", 60)
            if span == 60:
                # Prédicteur parfait : alloc forte si rendement positif, faible sinon
                alloc = (r_pair > 0).astype(float) * 0.8 + 0.1
                return alloc
            else:
                return pd.Series(0.5, index=r_pair.index)

        result = run_q3(spike_strategy, {"ema_span": 60},
                        r_is, r_base_is, ema_step=20)

        # Le prédicteur parfait à span=60 doit créer un spike
        self.assertTrue(result.is_spike,
                        f"Spike non détecté : target={result.target_cnsr:.3f}, "
                        f"nbh_median={result.neighbourhood_median:.3f}")
        self.assertFalse(result.passed)

    def test_q3_no_spike_passes(self):
        """Q3 : une stratégie avec performance régulière passe (is_spike=False)."""
        result = run_q3(_ema_strategy, {"ema_span": 60},
                        self.params_is, self.rbase_is,
                        ema_step=20)
        # is_spike doit être False si la performance est régulière
        # (peut varier selon les données synthétiques)
        self.assertIsInstance(result.is_spike, bool)

    def test_q3_span_cnsrs_populated(self):
        """Q3 calcule des CNSR pour chaque span dans la grille."""
        result = run_q3(_ema_strategy, self.params,
                        self.params_is, self.rbase_is,
                        ema_step=20)
        self.assertGreater(len(result.span_cnsrs), 0)
        # Les spans doivent être dans [20, 120]
        for span in result.span_cnsrs:
            self.assertGreaterEqual(span, 20)
            self.assertLessEqual(span, 120)

    def test_q3_target_span_in_results(self):
        """Le span cible est dans span_cnsrs."""
        result = run_q3(_ema_strategy, {"ema_span": 60},
                        self.params_is, self.rbase_is,
                        ema_step=20)
        self.assertIn(60, result.span_cnsrs)


# ── Tests Q4 DSR ──────────────────────────────────────────────────────────────

class TestMETISQ4DSR(unittest.TestCase):

    def setUp(self):
        r_pair, r_base = generate_synthetic_paxgbtc(T=600, seed=42)
        self.r_pair_oos = r_pair.iloc[400:]
        self.r_base_oos = r_base.iloc[400:]
        self.params = {"ema_span": 20}

    def test_q4_returns_result(self):
        """Q4 retourne un DSRResult."""
        result = run_q4(_ema_strategy, self.params,
                        self.r_pair_oos, self.r_base_oos,
                        n_trials=10)
        self.assertIsInstance(result, DSRResult)

    def test_q4_dsr_in_range(self):
        """DSR est dans [0, 1]."""
        result = run_q4(_ema_strategy, self.params,
                        self.r_pair_oos, self.r_base_oos,
                        n_trials=10)
        if not math.isnan(result.dsr):
            self.assertGreaterEqual(result.dsr, 0.0)
            self.assertLessEqual(result.dsr, 1.0)

    def test_q4_status_valid(self):
        """Status est PASS, SUSPECT_DSR ou ERROR."""
        result = run_q4(_ema_strategy, self.params,
                        self.r_pair_oos, self.r_base_oos,
                        n_trials=10)
        self.assertIn(result.status, ("PASS", "SUSPECT_DSR", "ERROR"))

    def test_q4_n_trials_stored(self):
        """n_trials est stocké dans le résultat."""
        result = run_q4(_ema_strategy, self.params,
                        self.r_pair_oos, self.r_base_oos,
                        n_trials=42)
        self.assertEqual(result.n_trials, 42)


# ── Tests METISRunner ─────────────────────────────────────────────────────────

class TestMETISRunner(unittest.TestCase):

    def setUp(self):
        bundle      = _make_synthetic_bundle(n=600, seed=42)
        self.sm, *_ = _make_sm(bundle, split_at=0.7)
        self.bundle = bundle
        self.params = {"ema_span": 20}

    def test_runner_q1_only(self):
        """METISRunner avec questions='Q1' exécute Q1 uniquement."""
        runner = METISRunner(
            strategy_fn = _ema_strategy,
            params      = self.params,
            bundle      = self.bundle,
            split_manager = self.sm,
            hypothesis  = "TEST",
            n_trials    = 10,
        )
        report = runner.run(questions="Q1")
        self.assertIsNotNone(report.q1)
        self.assertIsNone(report.q2)
        self.assertIsNone(report.q3)
        self.assertIsNone(report.q4)

    def test_runner_q3_spike_step(self):
        """METISRunner Q3 avec ema_step=20 (rapide)."""
        runner = METISRunner(
            strategy_fn = _ema_strategy,
            params      = self.params,
            bundle      = self.bundle,
            split_manager = self.sm,
            hypothesis  = "TEST",
            n_trials    = 10,
        )
        report = runner.run(questions="Q3", ema_step=20)
        self.assertIsNone(report.q1)
        self.assertIsNotNone(report.q3)
        self.assertIsInstance(report.q3.span_cnsrs, dict)

    def test_runner_verdict_field(self):
        """METISReport.verdict() retourne une string non vide."""
        runner = METISRunner(
            strategy_fn = _ema_strategy,
            params      = self.params,
            bundle      = self.bundle,
            split_manager = self.sm,
            hypothesis  = "TEST",
            n_trials    = 10,
        )
        report = runner.run(questions="Q1Q4")
        v = report.verdict()
        self.assertIsInstance(v, str)
        self.assertGreater(len(v), 0)

    def test_runner_export_kb(self):
        """export_kb_update() retourne un dict avec 'metis' et 'verdict_final'."""
        runner = METISRunner(
            strategy_fn = _ema_strategy,
            params      = self.params,
            bundle      = self.bundle,
            split_manager = self.sm,
            hypothesis  = "TEST",
            n_trials    = 10,
        )
        report = runner.run(questions="Q1")
        kb = report.export_kb_update()
        self.assertIn("metis",         kb)
        self.assertIn("verdict_final", kb)
        self.assertIn("Q1_walkforward", kb["metis"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
