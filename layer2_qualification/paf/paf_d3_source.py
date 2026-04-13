"""
paf_d3_source.py — PAF Direction 3
Layer 2 QAAF Studio 3.0

Test iso-variance : reproduire la variance d'allocation par EMA triviale,
comparer CNSR-USD à iso-variance.

Verdict
-------
H9_LISSE_SUPERIEUR  : candidat > EMA_triviale à iso-variance (> +0.10)
SIGNAL_INFORMATIF   : candidat surpasse toutes variantes dont triviale
ARTEFACT_LISSAGE    : EMA_triviale ≥ candidat → lissage explique tout
"""

from __future__ import annotations
from typing import Dict

import numpy as np
import pandas as pd

from layer1_engine.metrics_engine import compute_cnsr


class PAFDirection3:
    def __init__(self, bundle, split_manager, rf_annual: float = 0.04):
        self._b  = bundle
        self._sm = split_manager
        self._rf = rf_annual

        r_pair_full = np.log(self._b.paxg_btc / self._b.paxg_btc.shift(1)).dropna()
        r_base_full = np.log(self._b.btc_usd  / self._b.btc_usd.shift(1)).dropna()
        self._r_pair_is, _ = split_manager.apply(r_pair_full)
        self._r_base_is, _ = split_manager.apply(r_base_full)

    def run(self, variants: Dict[str, pd.Series]) -> dict:
        """
        Paramètres
        ----------
        variants : {nom: allocations IS}
                   Doit inclure :
                   - la stratégie candidate (ex. "H9+EMA60j")
                   - une EMA triviale iso-variance (ex. "EMA_triviale_isovar")
                   - la stratégie brute sans lissage (ex. "H9_brut")
        """
        print(f"\n{'─'*55}")
        print("PAF D3 — Source minimale / test iso-variance (IS)")

        rows = []
        for name, alloc in variants.items():
            r_port = self._portfolio_returns(alloc)
            r_base = self._r_base_is.reindex(r_port.index).dropna()
            r_port = r_port.reindex(r_base.index)
            cnsr   = compute_cnsr(r_port, r_base, self._rf)
            rows.append({
                "variante":     name,
                "cnsr_usd_fed": cnsr["cnsr_usd_fed"],
                "sortino":      cnsr["sortino"],
                "max_dd_pct":   cnsr["max_dd_pct"],
                "std_alloc":    float(alloc.reindex(r_port.index).std()),
            })

        table = (
            pd.DataFrame(rows)
            .set_index("variante")
            .sort_values("cnsr_usd_fed", ascending=False)
        )

        print("\nTable iso-variance (IS) :")
        print(table.round(3).to_string())

        # Identifier candidat principal vs triviale
        trivial_keys = [k for k in table.index if "trivial" in k.lower()]
        active_keys  = [k for k in table.index if "trivial" not in k.lower()
                        and "brut" not in k.lower()]

        verdict = "SIGNAL_INFORMATIF"
        notes   = "Pas de variante triviale fournie — verdict par défaut."

        if active_keys and trivial_keys:
            best_cand_key  = table.loc[active_keys,  "cnsr_usd_fed"].idxmax()
            best_triv_key  = table.loc[trivial_keys, "cnsr_usd_fed"].idxmax()
            cnsr_cand      = table.loc[best_cand_key, "cnsr_usd_fed"]
            cnsr_triv      = table.loc[best_triv_key, "cnsr_usd_fed"]
            delta          = cnsr_cand - cnsr_triv

            if delta > 0.10:
                verdict = ("H9_LISSE_SUPERIEUR"
                           if "ema" in best_cand_key.lower()
                           else "SIGNAL_INFORMATIF")
                notes = (f"'{best_cand_key}' surpasse EMA triviale de +{delta:.3f}. "
                         "Signal informatif au-delà du lissage.")
            else:
                verdict = "ARTEFACT_LISSAGE"
                notes   = (f"EMA triviale explique le gain (delta = {delta:.3f}). "
                           "Accepter la solution minimale.")

        emoji_map = {
            "H9_LISSE_SUPERIEUR": "✅",
            "SIGNAL_INFORMATIF":  "✅",
            "ARTEFACT_LISSAGE":   "🔴",
        }
        print(f"\n{emoji_map.get(verdict,'❓')} VERDICT D3 : {verdict}\n   {notes}")

        return {"verdict": verdict, "notes": notes, "table": table}

    def _portfolio_returns(self, alloc: pd.Series) -> pd.Series:
        r_pair = self._r_pair_is
        common = r_pair.index.intersection(alloc.index)
        return (alloc.reindex(common).ffill() * r_pair.loc[common]).dropna()
