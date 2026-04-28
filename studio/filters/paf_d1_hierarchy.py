"""
studio/filters/paf_d1_hierarchy.py
PAF Direction 1 — Test de hiérarchie de signal.

Porte run_d1() vers l'interface Filter.

Rôle : vérifier que l'oracle produit une hiérarchie de performance
       cohérente (signal_ref > MR_pur) sans être dominé par le benchmark
       passif B_5050 à plus de 0.1 CNSR.

Verdict → passed :
  HIERARCHIE_CONFIRMEE → True  (oracle bat MR_pur, pas dominé par B_5050)
  PARTIELLE            → True  (oracle bat MR_pur, candidat n'améliore pas)
  B_PASSIF_DOMINE      → False (benchmark passif bat toutes stratégies actives)
  STOP                 → False (oracle pire que MR_pur)

Note KB : D1 passe quand le régime n'est pas un bull run BTC pur.
En bull run BTC, B_5050 (50% BTC) domine mécaniquement → recalibration
ou qualification sur paire dont l'alpha est indépendant de l'appréciation BTC.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from studio.interfaces import Filter, FilterConfig, FilterVerdict, FilterError, SignalData

from layer2_qualification.paf.paf_d1_hierarchy import run_d1
from layer1_engine.backtester import Backtester


class PAFD1Hierarchy(Filter):
    """
    PAF D1 — Hiérarchie de signal.

    Complexity Budget :
      - evaluate() ≤ 50 lignes
      - ≤ 3 niveaux d'imbrication
      - ≤ 5 paramètres dans FilterConfig.params

    Paramètres dans config.params :
      - window : int — fenêtre MR_pur (défaut : 60)
    """

    NAME = "paf_d1_hierarchy"

    _PASS_VERDICTS = frozenset({"HIERARCHIE_CONFIRMEE", "PARTIELLE"})

    def evaluate(
        self,
        signal: SignalData,
        config: FilterConfig,
    ) -> FilterVerdict:
        window = config.get("window", 60)

        prices_oos, r_btc_oos = self._extract_oos(signal)
        signal_ref_fn = self._make_signal_fn(signal)

        try:
            backtester = Backtester()
            result = run_d1(
                prices_oos=prices_oos,
                r_btc_oos=r_btc_oos,
                signal_ref_fn=signal_ref_fn,
                backtester=backtester,
                window=window,
            )
        except Exception as e:
            raise FilterError(
                filter_name=self.NAME,
                cause=f"Erreur lors de l'exécution de PAF D1 : {e}",
                action=(
                    "Vérifier que layer2_qualification.paf est importable "
                    "et que la période OOS contient suffisamment de données "
                    f"(≥ {window * 2} jours recommandés pour MR_pur)."
                ),
            )

        passed  = result.verdict in self._PASS_VERDICTS
        verdict = result.verdict

        if passed:
            diagnosis = (
                f"PAF D1 validée ({verdict}) : "
                f"signal_ref={result.signal_ref_cnsr:.4f} > "
                f"MR_pur={result.mr_pur_cnsr:.4f}, "
                f"B_5050={result.b_5050_cnsr:.4f}. "
                f"La hiérarchie de performance est confirmée."
            )
        elif verdict == "B_PASSIF_DOMINE":
            diagnosis = (
                f"PAF D1 échoue ({verdict}) : "
                f"B_5050={result.b_5050_cnsr:.4f} > "
                f"signal_ref={result.signal_ref_cnsr:.4f} + 0.1. "
                f"Le benchmark passif domine toutes les stratégies actives. "
                f"Envisager une paire dont l'alpha est indépendant "
                f"de l'appréciation directionnelle du marché."
            )
        else:
            diagnosis = (
                f"PAF D1 échoue ({verdict}) : "
                f"signal_ref={result.signal_ref_cnsr:.4f} < "
                f"MR_pur={result.mr_pur_cnsr:.4f} — "
                f"la règle d'allocation est moins bonne que la mean-reversion triviale. "
                f"Revoir la logique de signal."
            )

        return FilterVerdict(
            passed=passed,
            filter_name=self.NAME,
            metrics={
                "verdict":         verdict,
                "mr_pur_cnsr":     result.mr_pur_cnsr,
                "signal_ref_cnsr": result.signal_ref_cnsr,
                "b_5050_cnsr":     result.b_5050_cnsr,
                "b_btc_cnsr":      result.b_btc_cnsr,
                "delta_ref_vs_mr": result.delta_ref_vs_mr,
            },
            diagnosis=diagnosis,
        )

    def _extract_oos(self, signal: SignalData):
        oos_start = pd.Timestamp(signal.oos_start)
        oos_end   = pd.Timestamp(signal.oos_end)
        paxg_oos  = signal.prices_quote_usd.loc[oos_start:oos_end]
        btc_oos   = signal.prices_base_usd.loc[oos_start:oos_end]
        common    = paxg_oos.index.intersection(btc_oos.index)
        prices_oos = pd.DataFrame({"paxg": paxg_oos.loc[common], "btc": btc_oos.loc[common]})
        r_btc_oos  = np.log(btc_oos.loc[common] / btc_oos.loc[common].shift(1)).dropna()
        return prices_oos, r_btc_oos

    def _make_signal_fn(self, signal: SignalData):
        alloc_btc = signal.alloc_btc

        def signal_ref_fn(df):
            return (1.0 - alloc_btc.reindex(df.index).fillna(0.5)).clip(0.0, 1.0)

        return signal_ref_fn
