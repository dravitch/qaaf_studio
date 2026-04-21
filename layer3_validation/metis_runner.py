"""
METISRunner — Orchestre Q1→Q2→Q3→Q4 avec règles d'arrêt.

Deux modes d'utilisation :

Mode config (données chargées automatiquement) :
    runner = METISRunner(config_path="config.yaml")
    report = runner.run(allocation_fn, n_perm=500)

Mode bundle (données pré-chargées) :
    runner = METISRunner(
        strategy_fn=h9_fn, params={"ema_span": 60},
        bundle=bundle, split_manager=sm,
        hypothesis="H9+EMA60j", n_trials=101,
    )
    report = runner.run(questions="Q1Q2Q3Q4", n_perm=500)
"""

import numpy as np
import pandas as pd
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from layer1_engine.data_loader    import DataLoader
from layer1_engine.backtester     import Backtester
from layer1_engine.split_manager  import SplitManager
from layer1_engine.benchmark_factory import BenchmarkFactory
from layer1_engine.metrics_engine import compute_cnsr

from layer3_validation.metis_q1_walkforward   import run_q1, Q1Result
from layer3_validation.metis_q2_permutation   import run_q2, Q2Result
from layer3_validation.metis_q3_ema_stability import run_q3, Q3Result
from layer3_validation.metis_q4_dsr           import run_q4, Q4Result


@dataclass
class METISReport:
    verdict_global: str
    q1: Optional[Q1Result] = None
    q2: Optional[Q2Result] = None
    q3: Optional[Q3Result] = None
    q4: Optional[Q4Result] = None
    cnsr_oos: Optional[float] = None
    notes: list = field(default_factory=list)

    def verdict(self) -> str:
        return self.verdict_global

    def print_summary(self) -> None:
        print(f"\n{'─'*55}")
        print(f"MÉTIS — {self.verdict_global}")
        if self.q1:
            icon = "✅" if self.q1.verdict == "PASS" else "🔴"
            print(f"  {icon} Q1 Walk-forward : {self.q1.n_pass}/{self.q1.n_total} "
                  f"— CNSR médiane={self.q1.median_cnsr}")
        if self.q2:
            icon = "✅" if self.q2.verdict == "PASS" else "🔴"
            print(f"  {icon} Q2 Permutation  : p-value={self.q2.pvalue} "
                  f"| CNSR_obs={self.q2.cnsr_obs}")
        if self.q3:
            icon = "✅" if self.q3.verdict == "PASS" else "🔴"
            print(f"  {icon} Q3 Stabilité EMA: spike={'OUI' if self.q3.is_spike else 'NON'} "
                  f"| CNSR(target)={self.q3.cnsr_target}")
        if self.q4:
            icon = "✅" if self.q4.verdict == "PASS" else (
                "⚠️" if self.q4.verdict == "SUSPECT_DSR" else "🔴"
            )
            print(f"  {icon} Q4 DSR          : {self.q4.dsr} | N={self.q4.n_trials}")
        print(f"{'─'*55}")

    def export_kb_update(self) -> dict:
        return {"metis": {
            "verdict": self.verdict_global,
            "cnsr_oos": self.cnsr_oos,
            "q1": {"verdict": self.q1.verdict, "n_pass": self.q1.n_pass,
                   "median_cnsr": self.q1.median_cnsr} if self.q1 else None,
            "q2": {"verdict": self.q2.verdict, "pvalue": self.q2.pvalue,
                   "cnsr_obs": self.q2.cnsr_obs} if self.q2 else None,
            "q3": {"verdict": self.q3.verdict, "is_spike": self.q3.is_spike,
                   "cnsr_target": self.q3.cnsr_target} if self.q3 else None,
            "q4": {"verdict": self.q4.verdict, "dsr": self.q4.dsr,
                   "n_trials": self.q4.n_trials} if self.q4 else None,
        }}

    def to_dict(self) -> dict:
        return {
            "verdict_global": self.verdict_global,
            "cnsr_oos":       self.cnsr_oos,
            "q1": {
                "verdict":      self.q1.verdict if self.q1 else None,
                "n_pass":       self.q1.n_pass if self.q1 else None,
                "median_cnsr":  self.q1.median_cnsr if self.q1 else None,
            } if self.q1 else None,
            "q2": {
                "verdict":      self.q2.verdict if self.q2 else None,
                "pvalue":       self.q2.pvalue if self.q2 else None,
                "cnsr_obs":     self.q2.cnsr_obs if self.q2 else None,
            } if self.q2 else None,
            "q3": {
                "verdict":      self.q3.verdict if self.q3 else None,
                "is_spike":     self.q3.is_spike if self.q3 else None,
                "cnsr_target":  self.q3.cnsr_target if self.q3 else None,
            } if self.q3 else None,
            "q4": {
                "verdict":      self.q4.verdict if self.q4 else None,
                "dsr":          self.q4.dsr if self.q4 else None,
                "n_trials":     self.q4.n_trials if self.q4 else None,
            } if self.q4 else None,
            "notes":          self.notes,
        }


class METISRunner:
    def __init__(
        self,
        config_path: str = "config.yaml",
        strategy_fn: Callable = None,
        params: dict = None,
        bundle=None,
        split_manager=None,
        hypothesis: str = "",
        n_trials: int = 1,
    ):
        self.config_path = config_path
        self._strategy_fn = strategy_fn
        self._params = params or {}
        self._bundle = bundle
        self._split_manager_ext = split_manager
        self._hypothesis = hypothesis
        self._n_trials = n_trials

        self.backtester = Backtester(config_path)
        self.sm = split_manager or SplitManager(config_path)

        self._prices_full = None
        self._r_btc_full  = None
        self._prices_oos  = None
        self._r_btc_oos   = None
        self._prices_is   = None
        self._r_btc_is    = None

    def _load_data(self):
        if self._prices_full is not None:
            return
        loader = DataLoader(self.config_path)
        paxg, btc, r_paxg, r_btc = loader.load_prices()
        prices_full = pd.DataFrame({"paxg": paxg, "btc": btc})
        self._prices_full = prices_full
        self._r_btc_full  = r_btc
        self._prices_is,  self._prices_oos  = self.sm.apply_df(prices_full)
        self._r_btc_is,   self._r_btc_oos   = self.sm.apply(r_btc)

    def _setup_from_bundle(self):
        if self._prices_full is not None:
            return
        bundle = self._bundle
        sm = self._split_manager_ext or self.sm
        prices_df = pd.DataFrame({"paxg": bundle.paxg_usd, "btc": bundle.btc_usd})
        r_btc_full = np.log(bundle.btc_usd / bundle.btc_usd.shift(1)).dropna()
        self._prices_full = prices_df
        self._r_btc_full  = r_btc_full
        self._prices_is,  self._prices_oos  = sm.apply_df(prices_df)
        self._r_btc_is,   self._r_btc_oos   = sm.apply(r_btc_full)

    def _make_alloc_fn(self) -> Callable:
        fn = self._strategy_fn
        p = self._params

        def alloc_fn(df):
            r_pair = np.log(df["paxg"] / df["paxg"].shift(1)).dropna()
            return fn(r_pair, p)

        return alloc_fn

    def run(
        self,
        allocation_fn: Callable = None,
        n_perm: int = 10000,
        n_trials: int = None,
        target_span: int = None,
        ema_step: int = 10,
        checkpoint_dir: Optional[Path] = None,
        questions: str = "Q1Q2Q3Q4",
        verbose: bool = True,
    ) -> METISReport:
        # Resolve allocation_fn
        if allocation_fn is None and self._strategy_fn is not None:
            allocation_fn = self._make_alloc_fn()

        # Resolve n_trials and target_span from stored params
        if n_trials is None:
            n_trials = self._n_trials
        if target_span is None:
            target_span = self._params.get("ema_span", 60)

        # Load data
        if self._bundle is not None:
            self._setup_from_bundle()
        else:
            self._load_data()

        report = METISReport(verdict_global="EN_COURS")

        # ── CNSR OOS du signal ────────────────────────────────────────────────
        result_oos = self.backtester.run(
            allocation_fn, self._prices_oos, self._r_btc_oos
        )
        zero = pd.Series(0.0, index=result_oos["r_portfolio_usd"].index)
        cnsr_oos = compute_cnsr(result_oos["r_portfolio_usd"], zero)["cnsr_usd_fed"]
        report.cnsr_oos = round(float(cnsr_oos), 4)

        # ── Q1 Walk-forward ───────────────────────────────────────────────────
        if "Q1" in questions:
            if verbose:
                print("\n📊 MÉTIS Q1 — Walk-forward "
                      "(5 fenêtres, critère ≥ 4/5 avec CNSR > 0.5) ...")
            q1 = run_q1(
                self._prices_full, self._r_btc_full,
                allocation_fn, self.backtester,
            )
            report.q1 = q1
            for w in q1.windows:
                icon = "✅" if w["pass"] else "🔴"
                cnsr_s = f"CNSR={w['cnsr']:.3f}" if w["cnsr"] is not None else "CNSR=N/A"
                if verbose:
                    print(f"    {icon} Fenêtre {w['window']} "
                          f"({w['start']} → {w['end']}) : {cnsr_s}")
            verdict_q1 = "✅" if q1.verdict == "PASS" else "🔴"
            if verbose:
                print(f"  {verdict_q1} Q1 : {q1.n_pass}/{q1.n_total} "
                      f"fenêtres ≥ 0.5 | médiane CNSR={q1.median_cnsr}")

        # ── Q2 Permutation ────────────────────────────────────────────────────
        if "Q2" in questions:
            cp_path = None
            if checkpoint_dir:
                checkpoint_dir = Path(checkpoint_dir)
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                cp_path = checkpoint_dir / "q2_checkpoint.yaml"

            if verbose:
                print(f"\n📊 MÉTIS Q2 — Permutation "
                      f"({n_perm} itérations, p-value < 0.05) ...")
            q2 = run_q2(
                self._prices_oos, self._r_btc_oos,
                allocation_fn, self.backtester,
                n_perm=n_perm, checkpoint_path=cp_path,
            )
            report.q2 = q2
            verdict_q2 = "✅" if q2.verdict == "PASS" else "🔴"
            if verbose:
                print(f"  {verdict_q2} Q2 : p-value={q2.pvalue} "
                      f"| CNSR_obs={q2.cnsr_obs} | bench={q2.cnsr_bench}")

        # ── Q3 Stabilité EMA ──────────────────────────────────────────────────
        if "Q3" in questions:
            if verbose:
                print(f"\n📊 MÉTIS Q3 — Stabilité EMA "
                      f"(grille 20→120j pas={ema_step}, IS uniquement) ...")
            q3 = run_q3(
                self._prices_is, self._r_btc_is,
                target_span=target_span,
                backtester=self.backtester,
                ema_step=ema_step,
            )
            report.q3 = q3
            verdict_q3 = "✅" if q3.verdict == "PASS" else "🔴"
            if verbose:
                print(f"  {verdict_q3} Q3 : CNSR(span={target_span})={q3.cnsr_target} | "
                      f"médiane_voisinage={q3.median_neighbors} | "
                      f"spike={'OUI' if q3.is_spike else 'NON'}")

        # ── Q4 DSR ────────────────────────────────────────────────────────────
        if "Q4" in questions:
            if verbose:
                print(f"\n📊 MÉTIS Q4 — DSR (N_trials={n_trials}, seuil ≥ 0.95) ...")
            q4 = run_q4(
                result_oos["r_portfolio_usd"],
                cnsr_oos=cnsr_oos,
                n_trials=n_trials,
            )
            report.q4 = q4
            verdict_q4 = "✅" if q4.verdict == "PASS" else (
                "⚠️" if q4.verdict == "SUSPECT_DSR" else "🔴"
            )
            if verbose:
                print(f"  {verdict_q4} Q4 : DSR={q4.dsr} "
                      f"| N_trials={n_trials} | seuil=0.95")

        # ── Verdict global ────────────────────────────────────────────────────
        fails = []
        if report.q1 and report.q1.verdict == "FAIL":    fails.append("Q1")
        if report.q2 and report.q2.verdict == "FAIL":    fails.append("Q2")
        if report.q3 and report.q3.verdict == "FAIL":    fails.append("Q3")
        if report.q4 and report.q4.verdict == "FAIL":    fails.append("Q4")

        suspect_dsr = (report.q4 and report.q4.verdict == "SUSPECT_DSR")

        if not fails and not suspect_dsr:
            report.verdict_global = "CERTIFIE"
        elif not fails and suspect_dsr:
            report.verdict_global = "SUSPECT_DSR"
        else:
            label = "_".join(fails)
            report.verdict_global = f"ARCHIVE_FAIL_{label}"
            if suspect_dsr:
                report.verdict_global += "_Q4"

        return report
