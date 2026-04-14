"""
PAF Runner — Orchestre D1→D2→D3 avec règles d'arrêt.

Précondition : toutes les données doivent avoir passé DQF (Layer 0)
avant d'entrer dans PAF. Le runner vérifie que le DataLoader
a produit des données avec statut DQF != FAIL.
"""

import pandas as pd
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Optional, Dict

from layer1_engine.data_loader     import DataLoader
from layer1_engine.backtester      import Backtester
from layer1_engine.split_manager   import SplitManager
from layer2_qualification.paf.paf_d1_hierarchy   import run_d1, D1Result
from layer2_qualification.paf.paf_d2_attribution import run_d2, D2Result
from layer2_qualification.paf.paf_d3_source      import run_d3, D3Result


@dataclass
class PAFBundle:
    """Bundle de données validées prêtes pour PAF."""
    prices_oos:  pd.DataFrame
    r_btc_oos:   pd.Series
    prices_is:   pd.DataFrame
    r_btc_is:    pd.Series
    dqf_status:  str   # PASS | WARNING — jamais FAIL ici


@dataclass
class PAFReport:
    verdict_global: str
    d1: Optional[D1Result] = None
    d2: Optional[D2Result] = None
    d3: Optional[D3Result] = None
    stopped_at: Optional[str] = None
    notes: list = field(default_factory=list)


def load_paf_bundle(config_path: str = "config.yaml") -> PAFBundle:
    loader = DataLoader(config_path)
    paxg_usd, btc_usd, r_paxg, r_btc = loader.load_prices()

    for ticker, report in loader.dqf_reports.items():
        if report.get("status") == "FAIL":
            raise ValueError(
                f"DQF FAIL pour {ticker} — données invalides. "
                f"Issues : {report.get('issues', [])}"
            )

    prices_full = pd.DataFrame({"paxg": paxg_usd, "btc": btc_usd})
    sm = SplitManager(config_path)
    prices_is,  prices_oos  = sm.apply_df(prices_full)
    r_btc_is,   r_btc_oos   = sm.apply(r_btc)

    statuses = [r.get("status", "PASS") for r in loader.dqf_reports.values()]
    dqf_global = "WARNING" if "WARNING" in statuses else "PASS"

    return PAFBundle(
        prices_oos=prices_oos, r_btc_oos=r_btc_oos,
        prices_is=prices_is,   r_btc_is=r_btc_is,
        dqf_status=dqf_global,
    )


def run_paf(
    bundle: PAFBundle,
    signal_ref_fn: Callable,
    backtester: Backtester,
    signal_candidat_fn: Optional[Callable] = None,
    composantes_d2: Optional[Dict[str, tuple]] = None,
) -> PAFReport:
    report = PAFReport(verdict_global="EN_COURS")

    print("\n── PAF D1 : Hiérarchie de signal ──")
    d1 = run_d1(
        prices_oos=bundle.prices_oos, r_btc_oos=bundle.r_btc_oos,
        signal_ref_fn=signal_ref_fn, backtester=backtester,
        signal_candidat_fn=signal_candidat_fn,
    )
    report.d1 = d1
    print(f"  MR_pur : {d1.mr_pur_cnsr:.4f} | Signal_ref : {d1.signal_ref_cnsr:.4f} | "
          f"B_5050 : {d1.b_5050_cnsr:.4f}")
    print(f"  → D1 verdict : {d1.verdict}")

    if d1.verdict == "B_PASSIF_DOMINE":
        report.verdict_global = "REQUALIFIER_PAIRE"
        report.stopped_at = "D1"
        report.notes.append("B_passif domine toutes les stratégies actives.")
        return report

    if d1.verdict == "STOP":
        report.verdict_global = "STOP_D1"
        report.stopped_at = "D1"
        report.notes.append("Pas de hiérarchie MR_pur < Signal_ref.")
        return report

    if composantes_d2:
        print("\n── PAF D2 : Attribution de performance ──")
        d2_results = {}
        for nom, (fn_avec, fn_sans) in composantes_d2.items():
            d2 = run_d2(
                prices_oos=bundle.prices_oos, r_btc_oos=bundle.r_btc_oos,
                signal_complet_fn=fn_avec, signal_sans_fn=fn_sans,
                composante_name=nom, backtester=backtester,
            )
            d2_results[nom] = d2
            print(f"  {nom}: avec={d2.cnsr_avec:.4f} sans={d2.cnsr_sans:.4f} "
                  f"delta={d2.delta:+.4f} → {d2.verdict}")
        report.d2 = list(d2_results.values())[-1]

    print("\n── PAF D3 : Source minimale ──")
    fn_to_test = signal_candidat_fn or signal_ref_fn
    d3 = run_d3(
        prices_oos=bundle.prices_oos, r_btc_oos=bundle.r_btc_oos,
        signal_fn=fn_to_test, backtester=backtester,
    )
    report.d3 = d3
    print(f"  Signal : {d3.cnsr_signal:.4f} | EMA triviale (span={d3.ema_span_used}j) : "
          f"{d3.cnsr_trivial:.4f} | delta : {d3.delta:+.4f}")
    print(f"  → D3 verdict : {d3.verdict}")

    report.verdict_global = "QUALIFIE"
    if d3.verdict == "ARTEFACT_LISSAGE":
        report.notes.append(
            f"D3 : performance = artefact de lissage (EMA {d3.ema_span_used}j). "
            f"Solution minimale = H9+EMA{d3.ema_span_used}j."
        )
        report.verdict_global = "QUALIFIE_SOURCE_MINIMALE"

    return report
