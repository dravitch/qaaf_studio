"""
studio/filters/metis_q3.py
MÉTIS Q3 — Stabilité du span EMA (sur-ajustement des paramètres).

Porte run_q3() vers l'interface Filter.

Rôle : vérifier que le span EMA cible n'est pas un optimum ponctuel
       par rapport à ses voisins sur données IS (pas d'OOS vu).

Verdict → passed :
  verdict == "PASS" → True  (pas de spike : span cible robuste)
  verdict == "FAIL" → False (spike détecté : sur-ajustement probable)

Note KB (Transition.md) : N_trials de Q3 n'est PAS comptabilisé dans le DSR.
Q3 teste la stabilité du paramètre, pas la significativité statistique.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from studio.interfaces import Filter, FilterConfig, FilterVerdict, FilterError, SignalData

from layer3_validation.metis_q3_ema_stability import run_q3
from layer1_engine.backtester import Backtester


class MetisQ3(Filter):
    """
    MÉTIS Q3 — Stabilité du span EMA sur IS uniquement.

    Complexity Budget :
      - evaluate() ≤ 50 lignes
      - ≤ 3 niveaux d'imbrication
      - ≤ 5 paramètres dans FilterConfig.params

    Paramètres acceptés dans config.params :
      - target_span : int   — span EMA cible à tester (défaut : 60)
      - spike_ratio : float — ratio de détection de spike (défaut : 1.5)
      - span_min    : int   — début de la grille EMA (défaut : 20)
      - span_max    : int   — fin de la grille EMA (défaut : 120)
      - ema_step    : int   — pas de la grille (défaut : 10)

    Grille EMA testée sur IS : [span_min, span_min+step, ..., span_max].
    Spike détecté si CNSR(target) > spike_ratio × médiane(voisins ±2*step).
    """

    NAME = "metis_q3_ema_stability"

    def evaluate(
        self,
        signal: SignalData,
        config: FilterConfig,
    ) -> FilterVerdict:
        target_span = config.get("target_span", 60)
        spike_ratio = config.get("spike_ratio", 1.5)
        span_min    = config.get("span_min", 20)
        span_max    = config.get("span_max", 120)
        ema_step    = config.get("ema_step", 10)

        prices_is, r_btc_is = self._extract_is(signal)

        try:
            backtester = Backtester()
            result = run_q3(
                prices_is=prices_is,
                r_btc_is=r_btc_is,
                target_span=target_span,
                backtester=backtester,
                span_min=span_min,
                span_max=span_max,
                ema_step=ema_step,
                spike_ratio=spike_ratio,
            )
        except Exception as e:
            raise FilterError(
                filter_name=self.NAME,
                cause=f"Erreur lors de l'exécution de MÉTIS Q3 : {e}",
                action=(
                    "Vérifier que layer3_validation est importable "
                    f"et que la période IS contient suffisamment de données "
                    f"(≥ {span_max * 2} jours pour la grille EMA)."
                ),
            )

        passed = result.verdict == "PASS"

        if passed:
            diagnosis = (
                f"MÉTIS Q3 validé ({result.verdict}) : "
                f"span={target_span}j CNSR={result.cnsr_target:.4f}, "
                f"médiane_voisins={result.median_neighbors:.4f}. "
                f"Pas de spike détecté — le paramètre span={target_span} est robuste."
            )
        else:
            diagnosis = (
                f"MÉTIS Q3 échoue ({result.verdict}) : spike détecté. "
                f"span={target_span}j CNSR={result.cnsr_target:.4f} > "
                f"{spike_ratio} × médiane_voisins={result.median_neighbors:.4f}. "
                f"Le span {target_span} est un optimum ponctuel (sur-ajustement probable). "
                f"Envisager un span plus robuste (plateau de performance identifiable) "
                f"ou valider sur IS plus diversifié."
            )

        return FilterVerdict(
            passed=passed,
            filter_name=self.NAME,
            metrics={
                "verdict":          result.verdict,
                "target_span":      result.target_span,
                "cnsr_target":      result.cnsr_target,
                "median_neighbors": result.median_neighbors,
                "is_spike":         result.is_spike,
                "spike_ratio":      spike_ratio,
                "cnsr_by_span":     result.cnsr_by_span,
            },
            diagnosis=diagnosis,
        )

    def _extract_is(self, signal: SignalData):
        is_start = pd.Timestamp(signal.is_start)
        is_end   = pd.Timestamp(signal.is_end)
        paxg     = signal.prices_quote_usd.loc[is_start:is_end]
        btc      = signal.prices_base_usd.loc[is_start:is_end]
        common   = paxg.index.intersection(btc.index)
        prices_is = pd.DataFrame({"paxg": paxg.loc[common], "btc": btc.loc[common]})
        r_btc_is  = np.log(btc.loc[common] / btc.loc[common].shift(1)).dropna()
        prices_is = prices_is.reindex(r_btc_is.index)
        return prices_is, r_btc_is
