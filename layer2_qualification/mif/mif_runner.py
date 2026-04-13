"""
mif_runner.py — MIF v5.0 Layer 2 QAAF Studio 3.0

Orchestre Phase 0 → Phase 1 → Phase 2.
Règle d'arrêt à chaque phase : FAIL → stopper immédiat.

Note nomenclature : les phases MIF (0, 1, 2) sont internes à Layer 2.
Elles ne correspondent PAS aux phases du plan de mise en œuvre (1-4).
Voir README.md pour la correspondance exacte.

Note Phase 3 (I1-I5 intégration) : hors périmètre v1.0 (post-déploiement).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List

from layer2_qualification.mif.phase0_isolation  import run_phase0
from layer2_qualification.mif.phase1_oos        import run_phase1
from layer2_qualification.mif.phase2_multiasset import run_phase2, GATE_RATIO

PHASE2_GATE = GATE_RATIO


@dataclass
class MIFSummary:
    hypothesis:  str
    phase_stop:  int          # dernière phase exécutée (0/1/2), 3 = complet
    verdict:     str          # "PASS" | "FAIL_PHASE_0" | "FAIL_PHASE_1" | "FAIL_PHASE_2"
    results_p0:  list = field(default_factory=list)
    results_p1:  list = field(default_factory=list)
    results_p2:  list = field(default_factory=list)

    def print_summary(self) -> None:
        print(f"\n{'='*55}")
        print(f"MIF v5.0 SUMMARY — {self.hypothesis}")
        print(f"{'='*55}")
        for r in self.results_p0 + self.results_p1 + self.results_p2:
            emoji = "✅" if r.passed else "🔴"
            c_str = f"CNSR={r.cnsr:.3f}" if getattr(r, "cnsr", None) is not None else "—"
            print(f"  {emoji} {r.label:40s} {c_str}")
        print(f"\nVERDICT : {self.verdict}")
        if self.phase_stop < 2:
            print(f"  Arrêt Phase {self.phase_stop} — corriger avant de continuer.")


class MIFRunner:
    """
    Pipeline MIF v5.0 — exécute Phase 0 → 1 → 2 avec règle d'arrêt.

    Usage
    -----
    runner = MIFRunner(strategy_fn=my_fn, params={"ema_span": 60},
                       hypothesis="H9+EMA60j")
    summary = runner.run(max_phase=2)
    summary.print_summary()
    """

    def __init__(
        self,
        strategy_fn: Callable,
        params:      dict,
        hypothesis:  str  = "unnamed",
        rf_annual:   float = 0.04,
    ):
        self._fn  = strategy_fn
        self._p   = params
        self._hyp = hypothesis
        self._rf  = rf_annual

    def run(self, max_phase: int = 2) -> MIFSummary:
        print(f"\n{'='*55}")
        print(f"MIF v5.0 — {self._hyp}")
        print(f"{'='*55}")

        summary = MIFSummary(hypothesis=self._hyp, phase_stop=0, verdict="")

        # ── Phase 0 ───────────────────────────────────────────────────
        r0 = run_phase0(self._fn, self._p, self._rf)
        summary.results_p0 = r0
        summary.phase_stop = 0

        if not all(r.passed for r in r0):
            summary.verdict = "FAIL_PHASE_0"
            return summary

        if max_phase < 1:
            summary.verdict = "PARTIAL"
            return summary

        # ── Phase 1 ───────────────────────────────────────────────────
        r1 = run_phase1(self._fn, self._p, self._rf)
        summary.results_p1 = r1
        summary.phase_stop = 1

        if not all(r.passed for r in r1):
            summary.verdict = "FAIL_PHASE_1"
            return summary

        if max_phase < 2:
            summary.verdict = "PARTIAL"
            return summary

        # ── Phase 2 ───────────────────────────────────────────────────
        r2 = run_phase2(self._fn, self._p, self._rf)
        summary.results_p2 = r2
        summary.phase_stop = 2

        n_pass = sum(r.passed for r in r2)
        if n_pass / len(r2) < PHASE2_GATE:
            summary.verdict = "FAIL_PHASE_2"
        else:
            summary.verdict = "PASS"

        return summary
