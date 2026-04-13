"""
Tests Layer 4 — D-SIG v0.5

Fonctionne avec pytest ET unittest :
    python3 -m unittest tests/test_layer4_dsig.py -v
    pytest tests/test_layer4_dsig.py -v

Critères :
- Triple-Réduction : score ∈ [0,100], label ∈ {EXCELLENT,GOOD,DEGRADED,CRITICAL}
- Fail-fast DSR cap : dsr=0.79 → score ≤ 59
- Fail-fast PAF STOP : score ≤ 20
- Cohérence score/label (85+ = EXCELLENT, 60-84 = GOOD, 35-59 = DEGRADED, 0-34 = CRITICAL)
- Aggregateur : min score drive le résultat global
"""

import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from layer4_decision.dsig.mapper    import strategy_to_dsig, score_to_label_color, DSIGSignal
from layer4_decision.dsig.aggregator import aggregate


# ── Métriques de test ─────────────────────────────────────────────────────────

GOOD_METRICS = {
    "cnsr_usd_fed":     1.5,
    "sortino":          2.0,
    "calmar":           1.0,
    "max_dd_pct":       15.0,
    "walk_forward_score": 0.8,
    "dsr":              0.97,
}

POOR_METRICS = {
    "cnsr_usd_fed":    -0.8,
    "sortino":         -1.0,
    "calmar":           0.0,
    "max_dd_pct":      55.0,
    "walk_forward_score": 0.0,
    "dsr":              0.50,
}

AVERAGE_METRICS = {
    "cnsr_usd_fed":     0.6,
    "sortino":          0.8,
    "calmar":           0.5,
    "max_dd_pct":       25.0,
    "walk_forward_score": 0.5,
    "dsr":              0.85,
}


# ── Tests score_to_label_color ─────────────────────────────────────────────────

class TestScoreToLabelColor(unittest.TestCase):

    def test_85_plus_is_excellent(self):
        label, color = score_to_label_color(85)
        self.assertEqual(label, "EXCELLENT")
        self.assertEqual(color, "GREEN")

    def test_100_is_excellent(self):
        label, color = score_to_label_color(100)
        self.assertEqual(label, "EXCELLENT")

    def test_60_to_84_is_good(self):
        for score in (60, 70, 84):
            label, color = score_to_label_color(score)
            self.assertEqual(label, "GOOD",   f"score={score}: expected GOOD")
            self.assertEqual(color, "YELLOW", f"score={score}: expected YELLOW")

    def test_35_to_59_is_degraded(self):
        for score in (35, 45, 59):
            label, color = score_to_label_color(score)
            self.assertEqual(label, "DEGRADED", f"score={score}: expected DEGRADED")
            self.assertEqual(color, "ORANGE",   f"score={score}: expected ORANGE")

    def test_0_to_34_is_critical(self):
        for score in (0, 10, 34):
            label, color = score_to_label_color(score)
            self.assertEqual(label, "CRITICAL", f"score={score}: expected CRITICAL")
            self.assertEqual(color, "RED",       f"score={score}: expected RED")


# ── Tests strategy_to_dsig ────────────────────────────────────────────────────

class TestStrategyToDSIG(unittest.TestCase):

    def test_score_in_range(self):
        """Triple-Réduction : score ∈ [0, 100]."""
        sig = strategy_to_dsig(GOOD_METRICS, "HIERARCHIE_CONFIRMEE", 10)
        self.assertGreaterEqual(sig.score, 0)
        self.assertLessEqual(sig.score, 100)

    def test_label_valid(self):
        """Label ∈ {EXCELLENT, GOOD, DEGRADED, CRITICAL}."""
        for metrics in (GOOD_METRICS, POOR_METRICS, AVERAGE_METRICS):
            sig = strategy_to_dsig(metrics, "HIERARCHIE_CONFIRMEE", 10)
            self.assertIn(sig.label, ("EXCELLENT", "GOOD", "DEGRADED", "CRITICAL"))

    def test_color_valid(self):
        """Color ∈ {GREEN, YELLOW, ORANGE, RED}."""
        for metrics in (GOOD_METRICS, POOR_METRICS, AVERAGE_METRICS):
            sig = strategy_to_dsig(metrics, "HIERARCHIE_CONFIRMEE", 10)
            self.assertIn(sig.color, ("GREEN", "YELLOW", "ORANGE", "RED"))

    def test_score_label_coherence_excellent(self):
        """Score 85+ → label EXCELLENT."""
        sig = strategy_to_dsig(GOOD_METRICS, "HIERARCHIE_CONFIRMEE", 10)
        if sig.score >= 85:
            self.assertEqual(sig.label, "EXCELLENT")

    def test_score_label_coherence_critical(self):
        """Score ≤ 34 → label CRITICAL."""
        sig = strategy_to_dsig(POOR_METRICS, "HIERARCHIE_CONFIRMEE", 10)
        if sig.score <= 34:
            self.assertEqual(sig.label, "CRITICAL")

    def test_dsr_cap_at_59(self):
        """Fail-fast DSR cap : dsr=0.79 → score ≤ 59 (DEGRADED max — SUSPECT_DSR)."""
        metrics_dsr_low = {**GOOD_METRICS, "dsr": 0.79}
        sig = strategy_to_dsig(metrics_dsr_low, "HIERARCHIE_CONFIRMEE", 10)
        self.assertLessEqual(sig.score, 59,
                             f"DSR cap échoué : score={sig.score} > 59 malgré dsr=0.79")

    def test_paf_stop_caps_score_at_20(self):
        """Fail-fast PAF STOP : verdict STOP → score ≤ 20."""
        sig = strategy_to_dsig(GOOD_METRICS, "STOP_PASSIF_DOMINE", 10)
        self.assertLessEqual(sig.score, 20,
                             f"PAF STOP cap échoué : score={sig.score} > 20")

    def test_cnsr_negative_caps_at_20(self):
        """Fail-fast CNSR < -0.5 → score ≤ 20."""
        metrics_bad_cnsr = {**GOOD_METRICS, "cnsr_usd_fed": -0.6}
        sig = strategy_to_dsig(metrics_bad_cnsr, "HIERARCHIE_CONFIRMEE", 10)
        self.assertLessEqual(sig.score, 20,
                             f"CNSR<-0.5 cap échoué : score={sig.score} > 20")

    def test_max_dd_high_caps_at_20(self):
        """Fail-fast max_dd > 40% → score ≤ 20."""
        metrics_high_dd = {**GOOD_METRICS, "max_dd_pct": 45.0}
        sig = strategy_to_dsig(metrics_high_dd, "HIERARCHIE_CONFIRMEE", 10)
        self.assertLessEqual(sig.score, 20,
                             f"MaxDD cap échoué : score={sig.score} > 20")

    def test_good_signal_gives_high_score(self):
        """Un signal avec bonnes métriques donne un score > 60."""
        sig = strategy_to_dsig(GOOD_METRICS, "HIERARCHIE_CONFIRMEE", 10)
        self.assertGreater(sig.score, 60,
                           f"Bon signal sous-estimé : score={sig.score}")

    def test_trend_stable_without_prev(self):
        """Sans prev_score → trend = STABLE."""
        sig = strategy_to_dsig(GOOD_METRICS, "HIERARCHIE_CONFIRMEE", 10)
        self.assertEqual(sig.trend, "STABLE")

    def test_trend_improving(self):
        """Score +10 vs précédent → IMPROVING."""
        sig = strategy_to_dsig(GOOD_METRICS, "HIERARCHIE_CONFIRMEE", 10,
                               prev_score=50)
        if sig.score >= 50 + 10:
            self.assertEqual(sig.trend, "IMPROVING")

    def test_trend_critical_fall(self):
        """Score -20 vs précédent → CRITICAL_FALL."""
        sig = strategy_to_dsig(POOR_METRICS, "HIERARCHIE_CONFIRMEE", 10,
                               prev_score=80)
        if sig.score <= 80 - 20:
            self.assertEqual(sig.trend, "CRITICAL_FALL")

    def test_dimensions_all_present(self):
        """Les 5 dimensions sont présentes dans le signal."""
        sig = strategy_to_dsig(GOOD_METRICS, "HIERARCHIE_CONFIRMEE", 10)
        for dim in ("cnsr", "sortino", "calmar", "drawdown", "stability"):
            self.assertIn(dim, sig.dimensions)

    def test_source_id_stored(self):
        """source_id est bien enregistré."""
        sig = strategy_to_dsig(GOOD_METRICS, "HIERARCHIE_CONFIRMEE", 10,
                               source_id="test-source")
        self.assertEqual(sig.source_id, "test-source")

    def test_nan_metrics_handled(self):
        """Métriques NaN → signal calculé sans exception."""
        metrics_nan = {
            "cnsr_usd_fed": float("nan"),
            "sortino":      float("nan"),
            "calmar":       float("nan"),
            "max_dd_pct":   float("nan"),
            "walk_forward_score": 0.0,
            "dsr":          float("nan"),
        }
        sig = strategy_to_dsig(metrics_nan, "HIERARCHIE_CONFIRMEE", 10)
        self.assertGreaterEqual(sig.score, 0)
        self.assertLessEqual(sig.score, 100)


# ── Tests Aggregateur ─────────────────────────────────────────────────────────

class TestAggregator(unittest.TestCase):

    def _make_signal(self, score, source="test"):
        label, color = score_to_label_color(score)
        return DSIGSignal(
            score      = score,
            label      = label,
            color      = color,
            trend      = "STABLE",
            source_id  = source,
            dimensions = {"cnsr": {"score": score}},
            raw        = {},
        )

    def test_aggregate_single_returns_same(self):
        """Un seul signal → retourné tel quel."""
        sig = self._make_signal(75)
        result = aggregate([sig])
        self.assertEqual(result.score, 75)

    def test_aggregate_min_score_drives(self):
        """Le signal au score minimum drive le résultat global."""
        sigs = [
            self._make_signal(80, "mif"),
            self._make_signal(40, "metis"),
            self._make_signal(70, "paf"),
        ]
        result = aggregate(sigs)
        self.assertEqual(result.score, 40,
                         f"Min score not used: got {result.score}, expected 40")

    def test_aggregate_label_coherent(self):
        """Le label du résultat est cohérent avec le score min."""
        sigs   = [self._make_signal(90), self._make_signal(30)]
        result = aggregate(sigs)
        label, _ = score_to_label_color(result.score)
        self.assertEqual(result.label, label)

    def test_aggregate_source_id(self):
        """source_id du résultat est 'aggregated'."""
        sigs   = [self._make_signal(70), self._make_signal(50)]
        result = aggregate(sigs)
        self.assertEqual(result.source_id, "aggregated")

    def test_aggregate_empty_raises(self):
        """Agréger une liste vide lève ValueError."""
        with self.assertRaises(ValueError):
            aggregate([])

    def test_aggregate_dimensions_merged(self):
        """Les dimensions de tous les signaux sont fusionnées."""
        sig1 = DSIGSignal(score=80, label="GOOD", color="YELLOW",
                          trend="STABLE", source_id="s1",
                          dimensions={"cnsr": {"score": 80}}, raw={})
        sig2 = DSIGSignal(score=50, label="DEGRADED", color="ORANGE",
                          trend="STABLE", source_id="s2",
                          dimensions={"stability": {"score": 50}}, raw={})
        result = aggregate([sig1, sig2])
        self.assertIn("cnsr",      result.dimensions)
        self.assertIn("stability", result.dimensions)

    def test_aggregate_raw_has_scores(self):
        """raw du résultat agrégé contient les scores par source."""
        sigs   = [self._make_signal(70, "src1"), self._make_signal(40, "src2")]
        result = aggregate(sigs)
        self.assertIn("scores", result.raw)
        self.assertIn("src1", result.raw["scores"])
        self.assertIn("src2", result.raw["scores"])

    def test_aggregate_all_same_score(self):
        """Tous les signaux avec le même score → résultat avec ce score."""
        sigs   = [self._make_signal(65) for _ in range(3)]
        result = aggregate(sigs)
        self.assertEqual(result.score, 65)


if __name__ == "__main__":
    unittest.main(verbosity=2)
