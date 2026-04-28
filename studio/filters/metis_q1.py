"""
studio/filters/metis_q1.py
MÉTIS Q1 — Walk-forward (robustesse temporelle).

Porte run_q1() vers l'interface Filter.

Rôle : vérifier que le signal atteint CNSR > seuil sur au moins
       min_windows_pass des n_windows fenêtres glissantes sur
       l'historique complet (IS + OOS).

Verdict → passed :
  verdict == "PASS" → True  (n_pass >= min_windows_pass)
  verdict == "FAIL" → False
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from studio.interfaces import Filter, FilterConfig, FilterVerdict, FilterError, SignalData

from layer3_validation.metis_q1_walkforward import run_q1
from layer1_engine.backtester import Backtester


class MetisQ1(Filter):
    """
    MÉTIS Q1 — Walk-forward sur historique complet (IS + OOS).

    Complexity Budget :
      - evaluate() ≤ 50 lignes
      - ≤ 3 niveaux d'imbrication
      - ≤ 5 paramètres dans FilterConfig.params

    Paramètres acceptés dans config.params :
      - n_windows        : int   — nombre de fenêtres (défaut : 5)
      - cnsr_threshold   : float — seuil CNSR par fenêtre (défaut : 0.5)
      - min_windows_pass : int   — minimum de fenêtres à passer (défaut : 4)

    Note : utilise l'historique complet (IS + OOS) pour les fenêtres.
    L'allocation est appliquée via reindex sur chaque fenêtre glissante.
    """

    NAME = "metis_q1_walkforward"

    def evaluate(
        self,
        signal: SignalData,
        config: FilterConfig,
    ) -> FilterVerdict:
        n_windows        = config.get("n_windows", 5)
        cnsr_threshold   = config.get("cnsr_threshold", 0.5)
        min_windows_pass = config.get("min_windows_pass", 4)

        prices_full, r_btc_full = self._extract_full(signal)
        allocation_fn = self._make_allocation_fn(signal)

        try:
            backtester = Backtester()
            result = run_q1(
                prices_full=prices_full,
                r_btc_full=r_btc_full,
                allocation_fn=allocation_fn,
                backtester=backtester,
                n_windows=n_windows,
                cnsr_threshold=cnsr_threshold,
                min_windows_pass=min_windows_pass,
            )
        except Exception as e:
            raise FilterError(
                filter_name=self.NAME,
                cause=f"Erreur lors de l'exécution de MÉTIS Q1 : {e}",
                action=(
                    "Vérifier que layer3_validation est importable "
                    "et que prices_full contient suffisamment de données "
                    f"(≥ {(n_windows + 1) * 30} jours recommandés)."
                ),
            )

        passed = result.verdict == "PASS"

        if passed:
            diagnosis = (
                f"MÉTIS Q1 validé ({result.verdict}) : "
                f"{result.n_pass}/{result.n_total} fenêtres avec CNSR > {cnsr_threshold}. "
                f"Médiane CNSR = {result.median_cnsr:.4f}. "
                f"La performance est robuste temporellement."
            )
        else:
            diagnosis = (
                f"MÉTIS Q1 échoue ({result.verdict}) : "
                f"{result.n_pass}/{result.n_total} fenêtres avec CNSR > {cnsr_threshold} "
                f"(minimum : {min_windows_pass}). "
                f"Médiane CNSR = {result.median_cnsr:.4f}. "
                f"Revoir la robustesse temporelle du signal — recalibrer sur un régime "
                f"plus représentatif ou augmenter la diversité des fenêtres d'entraînement."
            )

        return FilterVerdict(
            passed=passed,
            filter_name=self.NAME,
            metrics={
                "verdict":          result.verdict,
                "n_pass":           result.n_pass,
                "n_total":          result.n_total,
                "min_windows_pass": min_windows_pass,
                "cnsr_threshold":   cnsr_threshold,
                "median_cnsr":      result.median_cnsr,
                "windows":          result.windows,
            },
            diagnosis=diagnosis,
        )

    def _extract_full(self, signal: SignalData):
        paxg   = signal.prices_quote_usd
        btc    = signal.prices_base_usd
        common = paxg.index.intersection(btc.index)
        prices_full = pd.DataFrame({"paxg": paxg.loc[common], "btc": btc.loc[common]})
        r_btc_full  = np.log(btc.loc[common] / btc.loc[common].shift(1)).dropna()
        prices_full = prices_full.reindex(r_btc_full.index)
        return prices_full, r_btc_full

    def _make_allocation_fn(self, signal: SignalData):
        alloc_btc = signal.alloc_btc

        def allocation_fn(df):
            return (1.0 - alloc_btc.reindex(df.index).fillna(0.5)).clip(0.0, 1.0)

        return allocation_fn
