"""
metis_runner.py — MÉTIS v2.1 Layer 3 QAAF Studio 3.0

Orchestre Q1 → Q2 → Q3 → Q4.
Métrique de référence : CNSR-USD sur toutes les questions.

Ce que MÉTIS ne fait PAS : générer de nouvelles hypothèses.
Si toutes les questions échouent, on remonte en Layer 2 avec une hypothèse différente.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

import numpy as np
import pandas as pd

from layer3_validation.metis_q1_walkforward import run_q1, WalkForwardResult
from layer3_validation.metis_q2_permutation  import run_q2, PermutationResult
from layer3_validation.metis_q3_ema_stability import run_q3, EMAStabilityResult
from layer3_validation.metis_q4_dsr          import run_q4, DSRResult


@dataclass
class METISReport:
    hypothesis: str
    q1: Optional[WalkForwardResult]  = None
    q2: Optional[PermutationResult]  = None
    q3: Optional[EMAStabilityResult] = None
    q4: Optional[DSRResult]          = None

    def verdict(self) -> str:
        qs = [q for q in [self.q1, self.q2, self.q3, self.q4] if q is not None]
        if not qs:
            return "EN_COURS"
        if all(q.passed for q in qs):
            return "CERTIFIE"
        # DSR seul échoue après Q1/Q2/Q3 → SUSPECT_DSR
        q123 = [q for q in [self.q1, self.q2, self.q3] if q is not None]
        if q123 and all(q.passed for q in q123) and self.q4 and not self.q4.passed:
            return "SUSPECT_DSR"
        failed = []
        if self.q1 and not self.q1.passed: failed.append("Q1")
        if self.q2 and not self.q2.passed: failed.append("Q2")
        if self.q3 and not self.q3.passed: failed.append("Q3")
        if self.q4 and not self.q4.passed: failed.append("Q4")
        return f"ARCHIVE_FAIL_{'_'.join(failed)}"

    def print_summary(self) -> None:
        print(f"\n{'='*60}")
        print(f"MÉTIS v2.1 REPORT — {self.hypothesis}")
        print(f"{'='*60}")
        q_map = [
            ("Q1 Walk-forward",  self.q1, lambda q: f"{q.median_cnsr:.3f}"),
            ("Q2 Permutation",   self.q2, lambda q: f"p={q.p_value:.4f}"),
            ("Q3 Stabilité EMA", self.q3, lambda q: f"{q.target_cnsr:.3f}"),
            ("Q4 DSR",           self.q4, lambda q: f"DSR={q.dsr:.4f}"),
        ]
        for name, q, fmt in q_map:
            if q is None:
                print(f"  ⬜ {name:30s} — non exécuté")
                continue
            emoji  = "✅" if q.passed else ("⚠️" if "DSR" in name else "🔴")
            metric = fmt(q) if q.passed or not q.passed else "—"
            try:
                metric = fmt(q)
            except Exception:
                metric = "—"
            print(f"  {emoji} {name:30s} → {metric}")
            print(f"       {q.notes}")
        v = self.verdict()
        print(f"\n{'='*60}")
        print(f"  VERDICT : {v}")
        print(f"{'='*60}")

    def export_kb_update(self) -> dict:
        """Dict prêt pour mise à jour du nœud 'metis' dans le YAML KB."""
        def _q(q, metric_fn):
            if q is None:
                return {"statut": "en_cours", "resultat": None, "notes": ""}
            try:
                m = metric_fn(q)
            except Exception:
                m = None
            return {
                "statut":   "pass" if q.passed else "fail",
                "resultat": m,
                "notes":    q.notes,
                "date":     datetime.now().strftime("%Y-%m-%d"),
            }
        return {
            "metis": {
                "Q1_walkforward":   _q(self.q1, lambda q: q.median_cnsr),
                "Q2_permutation":   _q(self.q2, lambda q: q.p_value),
                "Q3_stabilite_ema": _q(self.q3, lambda q: q.target_cnsr),
                "Q4_dsr":           _q(self.q4, lambda q: q.dsr),
            },
            "verdict_final": self.verdict(),
            "date_metis":    datetime.now().strftime("%Y-%m-%d"),
        }


class METISRunner:
    """
    Orchestre MÉTIS v2.1 — Q1 → Q2 → Q3 → Q4.

    Usage
    -----
    runner = METISRunner(
        strategy_fn=my_fn, params={"ema_span": 60},
        bundle=bundle, split_manager=sm,
        hypothesis="H9+EMA60j", n_trials=101,
    )
    report = runner.run(questions="Q1Q2Q3Q4")
    report.print_summary()
    """

    def __init__(
        self,
        strategy_fn:   Callable,
        params:        dict,
        bundle,
        split_manager,
        hypothesis:    str   = "unnamed",
        n_trials:      int   = 1,
        rf_annual:     float = 0.04,
    ):
        self._fn  = strategy_fn
        self._p   = params
        self._b   = bundle
        self._sm  = split_manager
        self._hyp = hypothesis
        self._N   = n_trials
        self._rf  = rf_annual

        # Log-rendements pré-calculés — Layer 3 utilise uniquement les données réelles
        import numpy as _np
        r_pair_full = _np.log(self._b.paxg_btc / self._b.paxg_btc.shift(1)).dropna()
        r_base_full = _np.log(self._b.btc_usd  / self._b.btc_usd.shift(1)).dropna()
        self._r_pair_is,  self._r_pair_oos  = split_manager.apply(r_pair_full)
        self._r_base_is,  self._r_base_oos  = split_manager.apply(r_base_full)
        self._r_pair_full = r_pair_full
        self._r_base_full = r_base_full

    def run(
        self,
        questions: str = "Q1Q2Q3Q4",
        n_perm:    int = 10_000,
        ema_step:  int = 5,
    ) -> METISReport:
        """
        Paramètres
        ----------
        questions : sous-ensemble à exécuter ex. "Q1Q3Q4"
        n_perm    : itérations permutation (réduire à 500 pour tests)
        ema_step  : pas grille EMA (réduire à 10 pour tests rapides)
        """
        print(f"\n{'='*60}")
        print(f"MÉTIS v2.1 — {self._hyp}")
        print(f"IS  : {self._r_pair_is.index[0]} → {self._r_pair_is.index[-1]}")
        print(f"OOS : {self._r_pair_oos.index[0]} → {self._r_pair_oos.index[-1]}")
        print(f"N_trials : {self._N} | Questions : {questions}")
        print(f"{'='*60}")

        report = METISReport(hypothesis=self._hyp)

        if "Q1" in questions:
            report.q1 = run_q1(self._fn, self._p,
                               self._r_pair_full, self._r_base_full, self._rf)

        if "Q2" in questions:
            report.q2 = run_q2(self._fn, self._p,
                               self._r_pair_oos, self._r_base_oos,
                               self._rf, n_perm=n_perm)

        if "Q3" in questions:
            report.q3 = run_q3(self._fn, self._p,
                               self._r_pair_is, self._r_base_is,
                               self._rf, ema_step=ema_step)

        if "Q4" in questions:
            report.q4 = run_q4(self._fn, self._p,
                               self._r_pair_oos, self._r_base_oos,
                               self._N, self._rf)

        return report
