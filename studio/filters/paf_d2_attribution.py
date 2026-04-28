"""
studio/filters/paf_d2_attribution.py
PAF Direction 2 — Attribution de performance.

Porte run_d2() vers l'interface Filter.

Rôle : vérifier que la règle d'allocation ajoute de la performance
       par rapport à une allocation statique neutre (50/50).

Comparaison :
  signal_complet_fn = oracle (alloc_btc dynamique depuis SignalData)
  signal_sans_fn    = 50/50 statique (pas de règle d'allocation)
  composante        = "alloc_rule"

Verdict → passed :
  COMPOSANTE_ACTIVE → True  (delta > seuil_actif : la règle ajoute de la valeur)
  NEUTRE            → False (delta ≤ seuil_actif : pas d'alpha détectable)
  DEGRADANTE        → False (delta négatif : la règle détruit de la valeur)
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from studio.interfaces import Filter, FilterConfig, FilterVerdict, FilterError, SignalData

from layer2_qualification.paf.paf_d2_attribution import run_d2
from layer1_engine.backtester import Backtester


class PAFD2Attribution(Filter):
    """
    PAF D2 — Attribution de performance : règle d'allocation vs 50/50.

    Complexity Budget :
      - evaluate() ≤ 50 lignes
      - ≤ 3 niveaux d'imbrication
      - ≤ 5 paramètres dans FilterConfig.params

    Paramètres dans config.params :
      - seuil_actif  : float — delta minimum pour COMPOSANTE_ACTIVE (défaut : 0.05)
      - composante   : str   — nom de la composante testée (défaut : "alloc_rule")
    """

    NAME = "paf_d2_attribution"

    def evaluate(
        self,
        signal: SignalData,
        config: FilterConfig,
    ) -> FilterVerdict:
        seuil_actif = config.get("seuil_actif", 0.05)
        composante  = config.get("composante",  "alloc_rule")

        prices_oos, r_btc_oos  = self._extract_oos(signal)
        signal_complet_fn      = self._make_signal_fn(signal)
        signal_sans_fn         = self._make_neutral_fn()

        try:
            backtester = Backtester()
            result = run_d2(
                prices_oos=prices_oos,
                r_btc_oos=r_btc_oos,
                signal_complet_fn=signal_complet_fn,
                signal_sans_fn=signal_sans_fn,
                composante_name=composante,
                backtester=backtester,
                seuil_actif=seuil_actif,
            )
        except Exception as e:
            raise FilterError(
                filter_name=self.NAME,
                cause=f"Erreur lors de l'exécution de PAF D2 : {e}",
                action=(
                    "Vérifier que layer2_qualification.paf est importable "
                    "et que la période OOS contient suffisamment de données."
                ),
            )

        passed  = result.verdict == "COMPOSANTE_ACTIVE"
        verdict = result.verdict

        if passed:
            diagnosis = (
                f"PAF D2 validée ({verdict}) : "
                f"cnsr_avec={result.cnsr_avec:.4f}, "
                f"cnsr_sans={result.cnsr_sans:.4f}, "
                f"delta={result.delta:+.4f} > {seuil_actif}. "
                f"La composante '{composante}' ajoute de la valeur détectable."
            )
        elif verdict == "DEGRADANTE":
            diagnosis = (
                f"PAF D2 échoue ({verdict}) : "
                f"delta={result.delta:+.4f} — la composante '{composante}' "
                f"détruit de la valeur. "
                f"Envisager de supprimer ou inverser cette composante."
            )
        else:
            diagnosis = (
                f"PAF D2 échoue ({verdict}) : "
                f"delta={result.delta:+.4f} ≤ {seuil_actif} — "
                f"la composante '{composante}' n'ajoute pas de valeur détectable. "
                f"Pour passer D2, le signal doit surperformer 50/50 statique "
                f"de {seuil_actif} CNSR ou plus sur la période OOS."
            )

        return FilterVerdict(
            passed=passed,
            filter_name=self.NAME,
            metrics={
                "verdict":      verdict,
                "cnsr_avec":    result.cnsr_avec,
                "cnsr_sans":    result.cnsr_sans,
                "delta":        result.delta,
                "seuil_actif":  seuil_actif,
                "composante":   composante,
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

        def signal_complet_fn(df):
            return (1.0 - alloc_btc.reindex(df.index).fillna(0.5)).clip(0.0, 1.0)

        return signal_complet_fn

    def _make_neutral_fn(self):
        def signal_sans_fn(df):
            return pd.Series(0.5, index=df.index)

        return signal_sans_fn
