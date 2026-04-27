"""
studio/filters/metis_q2.py
MÉTIS Q2 — Test de permutation (significativité statistique).

Porte la logique existante de layer3_validation/metis_q2_permutation.py
vers l'interface Filter. Aucune logique nouvelle — portage à l'identique.

# RECALIBRATION_PENDING
# Critère actuel : p_value < p_threshold (défaut 0.05).
# Diagnostic : le bull run 2023-2024 gonfle les CNSR de toutes les stratégies.
# Un signal trend-following réel peut obtenir p >= 0.05 si son alpha est
# indiscernable du bruit de marché sur cette période.
# Alternative recommandée (Session 2) : ajustement de régime (Alternative A) —
# exclure ou pondérer les périodes de forte tendance systémique.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from studio.interfaces import Filter, FilterConfig, FilterVerdict, FilterError, SignalData

from layer3_validation.metis_q2_permutation import run_q2
from layer1_engine.backtester import Backtester


class MetisQ2(Filter):
    """
    MÉTIS Q2 — Test de permutation sur la période OOS.

    Complexity Budget :
      - evaluate() ≤ 50 lignes
      - ≤ 3 niveaux d'imbrication
      - ≤ 5 paramètres dans FilterConfig.params

    Paramètres acceptés dans config.params :
      - n_perm       : int   — nombre de permutations (défaut : 500)
      - p_threshold  : float — seuil p-value (défaut : 0.05)
      - seed         : int   — graine aléatoire (défaut : 42)

    # RECALIBRATION_PENDING : critère p < 0.05 sur CNSR brut OOS.
    # À ajuster en Session 2 (Alternative A : ajustement de régime).
    """

    NAME = "metis_q2_permutation"

    def evaluate(
        self,
        signal: SignalData,
        config: FilterConfig,
    ) -> FilterVerdict:
        n_perm      = config.get("n_perm", 500)
        p_threshold = config.get("p_threshold", 0.05)
        seed        = config.get("seed", 42)

        prices_oos, r_btc_oos = self._extract_oos(signal)
        allocation_fn         = self._make_allocation_fn(signal)

        try:
            backtester = Backtester()
            result     = run_q2(
                prices_oos       = prices_oos,
                r_btc_oos        = r_btc_oos,
                allocation_fn    = allocation_fn,
                backtester       = backtester,
                n_perm           = n_perm,
                pvalue_threshold = p_threshold,
                seed             = seed,
            )
        except Exception as e:
            raise FilterError(
                filter_name=self.NAME,
                cause=f"Erreur lors de l'exécution de MÉTIS Q2 : {e}",
                action=(
                    "Vérifier que layer3_validation.metis_q2_permutation est importable, "
                    "que config.yaml est accessible, et que la période OOS contient "
                    "suffisamment de données (≥ 60 jours)."
                ),
            )

        passed = result.pvalue < p_threshold

        if passed:
            diagnosis = (
                f"Q2 validé : p={result.pvalue:.3f} < {p_threshold} sur "
                f"{result.n_perm} permutations. "
                f"CNSR observé ({result.cnsr_obs:.3f}) significativement "
                f"supérieur au bruit aléatoire — alpha non trivial confirmé."
            )
        else:
            diagnosis = (
                f"Q2 échoue : p={result.pvalue:.3f} ≥ {p_threshold} sur "
                f"{result.n_perm} permutations. "
                f"CNSR observé={result.cnsr_obs:.3f}, benchmark={result.cnsr_bench:.3f}, "
                f"moy. permutations={result.perm_mean:.3f}. "
                f"Pour passer Q2, le signal doit produire un CNSR OOS "
                f"significativement supérieur au bruit — envisager un filtre "
                f"de régime pour isoler l'alpha du bull run systémique."
            )

        return FilterVerdict(
            passed=passed,
            filter_name=self.NAME,
            metrics={
                "pvalue":       result.pvalue,
                "cnsr_obs":     result.cnsr_obs,
                "cnsr_bench":   result.cnsr_bench,
                "perm_mean":    result.perm_mean,
                "n_perm":       result.n_perm,
                "p_threshold":  p_threshold,
                "criterion":    "absolute",  # RECALIBRATION_PENDING
            },
            diagnosis=diagnosis,
        )

    def _extract_oos(self, signal: SignalData):
        """Extrait les données OOS depuis SignalData."""
        oos_start = pd.Timestamp(signal.oos_start)
        oos_end   = pd.Timestamp(signal.oos_end)

        paxg_oos  = signal.prices_quote_usd.loc[oos_start:oos_end]
        btc_oos   = signal.prices_base_usd.loc[oos_start:oos_end]

        common    = paxg_oos.index.intersection(btc_oos.index)
        prices_oos = pd.DataFrame(
            {"paxg": paxg_oos.loc[common], "btc": btc_oos.loc[common]}
        )
        r_btc_oos = np.log(btc_oos.loc[common] / btc_oos.loc[common].shift(1)).dropna()

        return prices_oos, r_btc_oos

    def _make_allocation_fn(self, signal: SignalData):
        """
        Retourne une allocation_fn(prices_df) -> pd.Series pour le Backtester.

        Convention Backtester : alloc = fraction PAXG ∈ [0, 1].
        signal.alloc_btc est la fraction BTC → alloc_paxg = 1 - alloc_btc.
        """
        alloc_btc = signal.alloc_btc

        def allocation_fn(prices_df):
            alloc_paxg = 1.0 - alloc_btc.reindex(prices_df.index).fillna(0.5)
            return alloc_paxg.clip(0.0, 1.0)

        return allocation_fn
