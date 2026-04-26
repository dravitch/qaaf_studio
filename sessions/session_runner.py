"""
session_runner.py — Orchestrateur générique de certification QAAF Studio 3.0

Usage
-----
    from sessions.session_runner import SessionRunner

    runner = SessionRunner(
        hypothesis="H9+EMA60j",
        family="EMA_span_variants",
        kb_path=Path("sessions/h9_ema60j/kb.yaml"),
        signal_fn=h9_ema_strategy,
        params={"ema_span": 60, "h9_lookback": 20},
    )
    result = runner.run(fast=False, force_metis=False)
"""

from __future__ import annotations

import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding and sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import types
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from layer1_engine.data_loader        import DataLoader
from layer1_engine.split_manager      import SplitManager
from layer1_engine.benchmark_factory  import BenchmarkFactory
from layer1_engine.backtester         import Backtester
from layer1_engine.metrics_engine     import compute_cnsr, deflated_sharpe_ratio
from layer2_qualification.mif.mif_runner import MIFRunner
from layer2_qualification.mif.synthetic_data import generate_synthetic_paxgbtc
from layer3_validation.metis_runner   import METISRunner
from layer4_decision.kb_manager       import KBManager
from layer4_decision.n_trials_tracker import NTrialsTracker
from layer4_decision.dsig.mapper      import strategy_to_dsig

_CONFIG_PATH = str(_ROOT / "config.yaml")
_KB_INV      = str(_ROOT / "layer4_decision" / "lentilles_inventory.yaml")


class SessionRunner:
    """
    Protocole complet de certification QAAF Studio 3.0 :
    KB check → données → benchmarks → PAF (KB) → métriques OOS →
    MIF → MÉTIS → D-SIG → verdict → KB update.
    """

    def __init__(
        self,
        hypothesis: str,
        family: str,
        kb_path: Path,
        signal_fn: Callable,
        params: dict,
        n_trials: int | None = None,
        params_hook: Callable | None = None,
    ):
        self.hypothesis  = hypothesis
        self.family      = family
        self.kb_path     = Path(kb_path)
        self.signal_fn   = signal_fn
        self.params      = params
        self._n_trials   = n_trials   # None = read from KB
        self._params_hook = params_hook  # (bundle) -> dict, augments params after load

    # ── Public ───────────────────────────────────────────────────────────────

    def run(
        self,
        fast: bool = False,
        force_metis: bool = False,
        skip_q2: bool = False,
        update_kb: bool = True,
    ) -> dict:
        n_perm   = 500 if fast else 10_000
        ema_step = 10  if fast else 5
        metis_q  = "Q1Q3Q4" if skip_q2 else "Q1Q2Q3Q4"

        kb      = KBManager(str(self.kb_path), _KB_INV)
        tracker = NTrialsTracker(str(self.kb_path))
        n_trials = self._n_trials or tracker.get_family_n_trials(self.family)

        print(f"\n{'='*62}")
        print(f"  QAAF STUDIO 3.0 — SESSION DE CERTIFICATION")
        print(f"  Hypothèse : {self.hypothesis}")
        print(f"  Famille   : {self.family} | N_trials = {n_trials}")
        print(f"  MÉTIS     : {metis_q} | n_perm={n_perm} | ema_step={ema_step}")
        print(f"  Mode      : {'FAST' if fast else 'COMPLET'}")
        print(f"{'='*62}")

        # ── 1. KB check ──────────────────────────────────────────────────────
        check = kb.pre_session_check(self.hypothesis, self.family)
        if check["recommendation"] == "SKIP_DUPLICATE":
            print("\n⏹️  Hypothèse déjà certifiée dans KB.")
            return check

        # ── 2. Données ───────────────────────────────────────────────────────
        print("\n📡 Chargement données ...")
        bundle, r_paxg_usd, r_btc_usd = self._load_data()
        params = {**self.params, **(self._params_hook(bundle) if self._params_hook else {})}

        sm = SplitManager(config_path=_CONFIG_PATH)
        sm.set_n_trials(self.family, n_trials)

        # ── 3. Benchmarks OOS ────────────────────────────────────────────────
        print("\n📊 Benchmarks passifs (CNSR-USD OOS) ...")
        prices_df = pd.DataFrame({"paxg": bundle.paxg_usd, "btc": bundle.btc_usd})
        _, prices_oos = sm.apply_df(prices_df)
        _, r_btc_oos  = sm.apply(r_btc_usd)

        backtester = Backtester(config_path=_CONFIG_PATH)
        bf         = BenchmarkFactory(backtester)
        b5050      = bf.b_5050(prices_oos, r_btc_oos)
        b_btc      = bf.b_btc( prices_oos, r_btc_oos)
        b_paxg     = bf.b_paxg(prices_oos, r_btc_oos)
        print(f"  B_5050 CNSR = {b5050['cnsr_usd_fed']:.3f}")
        print(f"  B_BTC  CNSR = {b_btc['cnsr_usd_fed']:.3f}")
        print(f"  B_PAXG CNSR = {b_paxg['cnsr_usd_fed']:.3f}")

        # ── 4. PAF — chargé depuis KB ────────────────────────────────────────
        paf_verdict = self._load_paf_from_kb(kb)

        # ── 5. Métriques OOS brutes ──────────────────────────────────────────
        print(f"\n{'─'*62}")
        r_pair_full = np.log(bundle.paxg_btc / bundle.paxg_btc.shift(1)).dropna()
        r_base_full = np.log(bundle.btc_usd  / bundle.btc_usd.shift(1)).dropna()
        _, r_pair_oos = sm.apply(r_pair_full)
        _, r_base_oos = sm.apply(r_base_full)

        alloc_oos  = self.signal_fn(r_pair_oos, params)
        common_oos = r_pair_oos.index.intersection(alloc_oos.index)
        r_port_oos = alloc_oos.reindex(common_oos).ffill() * r_pair_oos.loc[common_oos]
        r_usd_oos  = r_port_oos + r_base_oos.reindex(common_oos)

        metrics_oos = compute_cnsr(r_port_oos, r_base_oos.reindex(common_oos))
        dsr_oos     = deflated_sharpe_ratio(r_usd_oos.dropna(), n_trials)

        print(f"Métriques OOS {self.hypothesis} :")
        print(f"  CNSR-USD-Fed : {metrics_oos['cnsr_usd_fed']:.3f}")
        print(f"  Sortino      : {metrics_oos['sortino']:.3f}")
        print(f"  Max DD %     : {metrics_oos['max_dd_pct']:.1f}%")
        print(f"  DSR(N={n_trials}) : {dsr_oos:.4f}  "
              f"{'✅' if dsr_oos >= 0.95 else '⚠️'}")

        # ── 6. MIF ───────────────────────────────────────────────────────────
        print(f"\n{'─'*62}")
        mif_runner = MIFRunner(
            strategy_fn=self.signal_fn,
            params=params,
            hypothesis=self.hypothesis,
        )
        mif_summary = mif_runner.run(max_phase=2)
        mif_summary.print_summary()

        if "FAIL" in mif_summary.verdict:
            if not fast and not force_metis:
                kb.record_verdict(self.hypothesis, "suspendu",
                                  notes=f"MIF {mif_summary.verdict}")
                return {"verdict": mif_summary.verdict, "stopped": "MIF"}
            reason = "mode fast" if fast else "--force-metis"
            print(f"\n⚠️  MIF {mif_summary.verdict} — {reason}, pipeline continue...")

        # ── 7. MÉTIS ─────────────────────────────────────────────────────────
        print(f"\n{'─'*62}")
        metis_runner = METISRunner(
            strategy_fn=self.signal_fn,
            params=params,
            bundle=bundle,
            split_manager=sm,
            hypothesis=self.hypothesis,
            n_trials=n_trials,
        )
        metis_report = metis_runner.run(
            questions=metis_q,
            n_perm=n_perm,
            ema_step=ema_step,
        )
        metis_report.print_summary()

        # ── 8. D-SIG ─────────────────────────────────────────────────────────
        wf_score = (metis_report.q1.n_windows_pass / 5) if metis_report.q1 else 0.0
        signal   = strategy_to_dsig(
            metrics     = {**metrics_oos, "dsr": dsr_oos,
                           "walk_forward_score": wf_score},
            paf_verdict = paf_verdict,
            n_trials    = n_trials,
            source_id   = f"qaaf-studio::{self.hypothesis}",
        )
        print(f"\n{'─'*62}")
        print(f"D-SIG v0.5 : score={signal.score}/100 | {signal.label} | "
              f"{signal.color} | {signal.trend}")
        for dim, v in signal.dimensions.items():
            print(f"  {dim:12s} : {v['score']:3d}/100")

        # ── 9. Verdict final + KB update ─────────────────────────────────────
        verdict = metis_report.verdict()
        if update_kb:
            kb.update_metis(metis_report.export_kb_update().get("metis", {}))
            kb.record_verdict(
                hypothesis=self.hypothesis,
                verdict=verdict.lower().split("_fail_")[0],
                metrics={
                    "cnsr_usd_fed": round(float(metrics_oos["cnsr_usd_fed"]), 3),
                    "sortino":      round(float(metrics_oos["sortino"]),       3),
                    "max_dd_pct":   round(float(metrics_oos["max_dd_pct"]),    2),
                    "dsr":          round(float(dsr_oos),                      4),
                    "dsig_score":   signal.score,
                },
                notes=f"MÉTIS {metis_q} | N={n_trials} | MIF {mif_summary.verdict}",
            )
            lentille_statut = "active" if "CERTIFIE" in verdict else "archivee"
            kb.update_lentille(
                nom           = self.hypothesis,
                statut        = lentille_statut,
                cnsr_oos      = round(float(metrics_oos["cnsr_usd_fed"]), 3),
                paf_verdict   = paf_verdict,
                metis_verdict = verdict,
                dsig_score    = signal.score,
            )

        emoji = {"CERTIFIE": "🏆", "SUSPECT_DSR": "⚠️"}.get(verdict, "📁")
        print(f"\n{'='*62}")
        print(f"  {emoji}  VERDICT FINAL : {verdict}")
        print(f"  D-SIG : {signal.score}/100 — {signal.label} ({signal.color})")
        print(f"{'='*62}")

        return {
            "hypothesis": self.hypothesis,
            "verdict":    verdict,
            "dsig":       {"score": signal.score, "label": signal.label},
            "cnsr_oos":   round(float(metrics_oos["cnsr_usd_fed"]), 3),
            "dsr":        round(float(dsr_oos), 4),
            "mif":        mif_summary.verdict,
            "metis_q":    metis_q,
            "metis_q1":   metis_report.q1.verdict if metis_report.q1 else "N/A",
            "metis_q2":   metis_report.q2.verdict if metis_report.q2 else "N/A",
        }

    # ── Private ───────────────────────────────────────────────────────────────

    def _load_data(self):
        try:
            loader = DataLoader(config_path=_CONFIG_PATH)
            paxg_usd, btc_usd, r_paxg_usd, r_btc_usd = loader.load_prices(
                start="2019-01-01", end="2024-12-31"
            )
            bundle = types.SimpleNamespace(
                paxg_usd=paxg_usd,
                btc_usd=btc_usd,
                paxg_btc=paxg_usd / btc_usd,
            )
            print(f"   {len(r_paxg_usd)} jours chargés "
                  f"({r_paxg_usd.index[0]} → {r_paxg_usd.index[-1]})")
        except Exception as e:
            print(f"   ⚠️  Données réelles indisponibles ({e.__class__.__name__}: {e})")
            print("   → Utilisation de données synthétiques (résultats non certifiants)")
            r_pair_s, r_base_s = generate_synthetic_paxgbtc(T=1500, seed=42)
            idx      = r_pair_s.index
            btc_usd  = pd.Series(1000 * np.exp(r_base_s.cumsum()),               index=idx, name="btc")
            paxg_usd = pd.Series(1800 * np.exp((r_pair_s + r_base_s).cumsum()),  index=idx, name="paxg")
            r_paxg_usd = np.log(paxg_usd / paxg_usd.shift(1)).dropna()
            r_btc_usd  = np.log(btc_usd  / btc_usd.shift(1)).dropna()
            common     = r_paxg_usd.index.intersection(r_btc_usd.index)
            r_paxg_usd, r_btc_usd = r_paxg_usd.loc[common], r_btc_usd.loc[common]
            paxg_usd,   btc_usd   = paxg_usd.loc[common],   btc_usd.loc[common]
            bundle = types.SimpleNamespace(
                paxg_usd=paxg_usd,
                btc_usd=btc_usd,
                paxg_btc=paxg_usd / btc_usd,
            )
            print(f"   {len(r_paxg_usd)} jours synthétiques "
                  f"({r_paxg_usd.index[0].date()} → {r_paxg_usd.index[-1].date()})")
        return bundle, r_paxg_usd, r_btc_usd

    def _load_paf_from_kb(self, kb: KBManager) -> str:
        """Reads PAF verdict from KB (hypothese.paf.D1/D2/D3) and prints summary."""
        hyp_data = kb._load_hyp()
        hypothese = hyp_data.get("hypothese", {})
        paf = hypothese.get("paf", {})
        d1 = paf.get("D1", "N_A")
        d2 = paf.get("D2", "N_A")
        d3 = paf.get("D3", "N_A")
        date_paf = paf.get("date_paf", "")
        print(f"\n{'─'*62}")
        label = f"certifié {date_paf}" if date_paf else "chargé depuis KB"
        print(f"PAF : résultats {label}")
        print(f"  D1 : {d1}")
        print(f"  D2 : {d2}")
        print(f"  D3 : {d3}")
        return d3 if d3 != "N_A" else d1
