"""
paf_runner.py — Layer 2 QAAF Studio 3.0

Orchestre D1 → D2 → D3.
Précondition : DQF PASS sur les données (vérifié via DataLoader.dqf_reports).
Connexion des verdicts à la KB avec horodatage.

Règle d'arrêt D1 : si B_passif domine → session s'arrête,
message explicite "Requalifier la paire".
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

import pandas as pd

from layer2_qualification.paf.paf_d1_hierarchy  import PAFDirection1
from layer2_qualification.paf.paf_d2_attribution import PAFDirection2
from layer2_qualification.paf.paf_d3_source      import PAFDirection3


class PAFRunner:
    """
    Exécute le PAF (D1 → D2 → D3) pour une hypothèse.

    Usage
    -----
    runner = PAFRunner(bundle, split_manager, hypothesis="H9+EMA60j")
    result = runner.run(
        strategies  = {"MR_pur": alloc_mr, "H9+EMA60j": alloc_h9},
        benchmarks  = {"B_5050": alloc_bench},
        layers      = {"complet": alloc_full, "sans_lissage": alloc_raw},
        variants    = {"H9+EMA60j": alloc_h9, "EMA_triviale": alloc_ema_triv},
        dqf_reports = loader.dqf_reports,   # précondition
    )
    """

    def __init__(self, bundle, split_manager, hypothesis: str,
                 rf_annual: float = 0.04, config_path: str = "config.yaml"):
        self._bundle = bundle
        self._sm     = split_manager
        self._hyp    = hypothesis
        self._rf     = rf_annual
        self._config = config_path
        self._log: list[dict] = []

    def run(
        self,
        strategies:  Dict[str, pd.Series],
        benchmarks:  Optional[Dict[str, pd.Series]] = None,
        layers:      Optional[Dict[str, pd.Series]] = None,
        variants:    Optional[Dict[str, pd.Series]] = None,
        dqf_reports: Optional[dict] = None,
    ) -> dict:
        """
        Retourne un dict avec verdicts D1/D2/D3 et snippet KB.
        """
        # Précondition : DQF PASS
        if dqf_reports:
            fails = [t for t, r in dqf_reports.items()
                     if r.get("status") == "FAIL"]
            if fails:
                raise ValueError(
                    f"PAF bloqué — DQF FAIL sur : {fails}. "
                    "Corriger les données avant de qualifier le signal."
                )

        results = {}

        # D1
        d1 = PAFDirection1(self._bundle, self._sm, self._rf)
        r1 = d1.run(strategies, benchmarks)
        results["D1"] = r1
        self._log.append({"direction": "D1", "verdict": r1["verdict"],
                          "ts": datetime.now().isoformat()})

        if r1["verdict"] == "STOP_PASSIF_DOMINE":
            print("\n🛑 PAF D1 : STOP — Requalifier la paire, pas optimiser le signal.")
            return self._build_output(results)

        # D2
        if layers:
            d2 = PAFDirection2(self._bundle, self._sm, self._rf)
            r2 = d2.run(layers)
            results["D2"] = r2
            self._log.append({"direction": "D2", "verdict": r2["verdict"],
                              "ts": datetime.now().isoformat()})

        # D3
        if variants:
            d3 = PAFDirection3(self._bundle, self._sm, self._rf)
            r3 = d3.run(variants)
            results["D3"] = r3
            self._log.append({"direction": "D3", "verdict": r3["verdict"],
                              "ts": datetime.now().isoformat()})

        return self._build_output(results)

    def _build_output(self, results: dict) -> dict:
        """Construit le dict de sortie + snippet KB."""
        print(f"\n{'='*55}")
        print(f"PAF SUMMARY — {self._hyp}")
        for d, r in results.items():
            emoji = "✅" if r["verdict"] not in ("STOP_PASSIF_DOMINE",
                                                  "COMPOSANTE_DEGRADANTE",
                                                  "ARTEFACT_LISSAGE") else "🔴"
            print(f"  {emoji} {d}: {r['verdict']}")
            print(f"       {r['notes']}")

        kb_snippet = {
            "paf": {d: {"verdict": r["verdict"], "notes": r["notes"]}
                    for d, r in results.items()},
            "date_paf": datetime.now().strftime("%Y-%m-%d"),
        }
        return {"results": results, "kb_snippet": kb_snippet, "log": self._log}
