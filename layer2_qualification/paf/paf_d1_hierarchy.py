"""
paf_d1_hierarchy.py — PAF Direction 1
Layer 2 QAAF Studio 3.0

Compare MR_pur + signal_complet + benchmarks passifs.
Produit une table CNSR-USD comparative immédiate.

Verdict
-------
HIERARCHIE_CONFIRMEE : signal actif surpasse MR_pur et CNSR > 0
STOP_PASSIF_DOMINE   : benchmark passif domine tous les actifs → requalifier la paire
"""

from __future__ import annotations
from typing import Dict, Optional

import numpy as np
import pandas as pd

from layer1_engine.metrics_engine import compute_cnsr


class PAFDirection1:
    def __init__(self, bundle, split_manager, rf_annual: float = 0.04):
        self._b   = bundle
        self._sm  = split_manager
        self._rf  = rf_annual

        # Log-rendements IS pré-calculés
        r_pair_full = np.log(self._b.paxg_btc / self._b.paxg_btc.shift(1)).dropna()
        r_base_full = np.log(self._b.btc_usd  / self._b.btc_usd.shift(1)).dropna()
        self._r_pair_is, _ = split_manager.apply(r_pair_full)
        self._r_base_is, _ = split_manager.apply(r_base_full)

    def run(
        self,
        strategies: Dict[str, pd.Series],
        benchmarks: Optional[Dict[str, pd.Series]] = None,
    ) -> dict:
        """
        Paramètres
        ----------
        strategies : {nom: allocations IS ∈ [0,1]} — inclure MR_pur + candidat
        benchmarks : {nom: allocations IS} — B_5050, B_BTC, etc.
        """
        print(f"\n{'─'*55}")
        print("PAF D1 — Hiérarchie de signal (IS)")

        rows      = []
        all_alloc = {**(strategies or {}), **(benchmarks or {})}

        for name, alloc in all_alloc.items():
            r_port = self._portfolio_returns(alloc)
            r_base = self._r_base_is.reindex(r_port.index).dropna()
            r_port = r_port.reindex(r_base.index)
            cnsr   = compute_cnsr(r_port, r_base, self._rf)
            rows.append({
                "strategie":    name,
                "cnsr_usd_fed": cnsr["cnsr_usd_fed"],
                "sortino":      cnsr["sortino"],
                "max_dd_pct":   cnsr["max_dd_pct"],
                "type":         "benchmark" if (benchmarks and name in benchmarks)
                                else "active",
            })

        table = (
            pd.DataFrame(rows)
            .set_index("strategie")
            .sort_values("cnsr_usd_fed", ascending=False)
        )

        print("\nTable CNSR-USD comparative (IS) :")
        print(table[["cnsr_usd_fed", "sortino", "max_dd_pct", "type"]].round(3).to_string())

        # Verdict
        bench_names  = list(benchmarks.keys()) if benchmarks else []
        active_names = [n for n in all_alloc if n not in bench_names]
        candidate    = [n for n in active_names if n != "MR_pur"]

        best_bench  = table.loc[bench_names,  "cnsr_usd_fed"].max() if bench_names  else -np.inf
        best_active = table.loc[candidate,    "cnsr_usd_fed"].max() if candidate    else -np.inf

        if best_bench > best_active and best_active < 0.5:
            verdict = "STOP_PASSIF_DOMINE"
            notes   = (f"B_passif ({best_bench:.3f}) > actif ({best_active:.3f}). "
                       "Requalifier la paire.")
        else:
            verdict = "HIERARCHIE_CONFIRMEE"
            notes   = f"Meilleur actif CNSR-USD IS = {best_active:.3f}."

        emoji = "✅" if verdict == "HIERARCHIE_CONFIRMEE" else "🔴"
        print(f"\n{emoji} VERDICT D1 : {verdict}\n   {notes}")

        return {"verdict": verdict, "notes": notes, "table": table}

    def _portfolio_returns(self, alloc: pd.Series) -> pd.Series:
        r_pair = self._r_pair_is
        common = r_pair.index.intersection(alloc.index)
        return (alloc.reindex(common).ffill() * r_pair.loc[common]).dropna()
