"""
paf_d2_attribution.py — PAF Direction 2
Layer 2 QAAF Studio 3.0

Isole chaque couche du signal. Compare signal_complet vs signal_sans_X.

Verdict
-------
COMPOSANTE_ACTIVE    : retirer X dégrade CNSR ≥ 0.10 → X apporte de l'info
REGIMES_NEUTRES      : delta < 0.10 dans les deux sens → lissage seulement
COMPOSANTE_DEGRADANTE: retirer X améliore CNSR ≥ 0.10 → X nuit au signal
"""

from __future__ import annotations
from typing import Dict

import numpy as np
import pandas as pd

from layer1_engine.metrics_engine import compute_cnsr


class PAFDirection2:
    def __init__(self, bundle, split_manager, rf_annual: float = 0.04):
        self._b  = bundle
        self._sm = split_manager
        self._rf = rf_annual

        r_pair_full = np.log(self._b.paxg_btc / self._b.paxg_btc.shift(1)).dropna()
        r_base_full = np.log(self._b.btc_usd  / self._b.btc_usd.shift(1)).dropna()
        self._r_pair_is, _ = split_manager.apply(r_pair_full)
        self._r_base_is, _ = split_manager.apply(r_base_full)

    def run(self, layers: Dict[str, pd.Series]) -> dict:
        """
        Paramètres
        ----------
        layers : {nom: allocations IS}
                 Convention : inclure "complet" + variantes sans couche X.
                 Ex. {"complet": alloc_full, "sans_regime": alloc_no_regime}
        """
        print(f"\n{'─'*55}")
        print("PAF D2 — Attribution de performance (IS)")

        rows = []
        for name, alloc in layers.items():
            r_port = self._portfolio_returns(alloc)
            r_base = self._r_base_is.reindex(r_port.index).dropna()
            r_port = r_port.reindex(r_base.index)
            cnsr   = compute_cnsr(r_port, r_base, self._rf)
            rows.append({
                "couche":       name,
                "cnsr_usd_fed": cnsr["cnsr_usd_fed"],
                "sortino":      cnsr["sortino"],
                "std_alloc":    float(alloc.std()),
            })

        table = (
            pd.DataFrame(rows)
            .set_index("couche")
            .sort_values("cnsr_usd_fed", ascending=False)
        )

        print("\nTable comparative par couche (IS) :")
        print(table.round(3).to_string())

        # Verdict : comparer "complet" vs chaque variante sans X
        verdict = "COMPOSANTE_NEUTRE"
        notes   = "Pas de couche 'complet' fournie — verdict par défaut."

        if "complet" in layers:
            cnsr_c   = table.loc["complet", "cnsr_usd_fed"]
            variants = {k: table.loc[k, "cnsr_usd_fed"]
                        for k in table.index if k != "complet"}
            if variants:
                best_without = max(variants.values())
                delta        = best_without - cnsr_c  # positif → retirer améliore

                if delta >= 0.10:
                    worst_layer = max(variants, key=variants.get)
                    verdict = "COMPOSANTE_DEGRADANTE"
                    notes   = (f"'{worst_layer}' dégrade le signal (retirer améliore "
                               f"+{delta:.3f}). Vérifier PhaseCoherence.")
                elif delta <= -0.10:
                    verdict = "COMPOSANTE_ACTIVE"
                    notes   = f"Retirer une couche dégrade de {abs(delta):.3f}. Signal informatif."
                else:
                    verdict = "REGIMES_NEUTRES"
                    notes   = (f"Delta max = {delta:.3f} — toutes les couches sont "
                               "neutres. Source = lissage uniquement.")

        emoji_map = {
            "COMPOSANTE_ACTIVE":     "✅",
            "REGIMES_NEUTRES":       "⚠️",
            "COMPOSANTE_DEGRADANTE": "🔴",
            "COMPOSANTE_NEUTRE":     "⚠️",
        }
        print(f"\n{emoji_map.get(verdict,'❓')} VERDICT D2 : {verdict}\n   {notes}")

        return {"verdict": verdict, "notes": notes, "table": table}

    def _portfolio_returns(self, alloc: pd.Series) -> pd.Series:
        r_pair = self._r_pair_is
        common = r_pair.index.intersection(alloc.index)
        return (alloc.reindex(common).ffill() * r_pair.loc[common]).dropna()
