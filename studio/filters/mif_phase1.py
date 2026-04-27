"""
studio/filters/mif_phase1.py
MIF Phase 1 — Généralisation OOS (G1-G5).

Porte la logique existante de layer2_qualification/mif/phase1_oos.py
vers l'interface Filter. Aucune logique nouvelle — portage à l'identique.

# RECALIBRATION_PENDING
# Critère actuel : cnsr_strat > -1.0 (seuil absolu fixe dans run_phase1).
# Diagnostic : trop strict sur données crypto (kurtosis élevé, F2).
# Alternative recommandée : critère de dominance relative (delta > threshold).
# Voir docs/recalibration_plan.md — Filtre 1.
"""

from __future__ import annotations
import numpy as np

from studio.interfaces import Filter, FilterConfig, FilterVerdict, FilterError, SignalData

# Réutilisation de la logique existante — pas de duplication
from layer2_qualification.mif.phase1_oos import run_phase1  # nom réel (pas run_phase1_oos)


class MIFPhase1(Filter):
    """
    MIF Phase 1 — tests de généralisation sur 5 régimes synthétiques (G1-G5).

    Complexity Budget :
      - evaluate() ≤ 50 lignes
      - ≤ 3 niveaux d'imbrication
      - ≤ 5 paramètres dans FilterConfig.params

    Paramètres acceptés dans config.params :
      - min_pass   : int   — nombre minimum de régimes à passer (défaut : 3)
      - rf_annual  : float — taux sans risque annualisé (défaut : 0.04)

    # RECALIBRATION_PENDING : run_phase1() utilise cnsr_strat > -1.0 (absolu).
    # À remplacer par dominance relative (delta > threshold) en Session 2.
    """

    NAME = "mif_phase1_generalization"

    def evaluate(
        self,
        signal: SignalData,
        config: FilterConfig,
    ) -> FilterVerdict:
        min_pass  = config.get("min_pass", 3)
        rf_annual = config.get("rf_annual", 0.04)

        strategy_fn = self._make_strategy_fn(signal)

        try:
            results = run_phase1(strategy_fn, params={}, rf_annual=rf_annual)
        except Exception as e:
            raise FilterError(
                filter_name=self.NAME,
                cause=f"Erreur lors de l'exécution de MIF Phase 1 : {e}",
                action=(
                    "Vérifier que layer2_qualification.mif est importable "
                    "et que signal.alloc_btc est une pd.Series indexée par date."
                ),
            )

        n_total      = len(results)
        n_pass       = sum(1 for r in results if r.passed)
        failed_names = [r.label for r in results if not r.passed]
        passed       = n_pass >= min_pass

        if passed:
            diagnosis = (
                f"Phase 1 validée : {n_pass}/{n_total} régimes passent "
                f"le critère de généralisation (CNSR > -1.0). "
                f"Signal robuste hors-échantillon sur données synthétiques."
            )
        else:
            diagnosis = (
                f"Phase 1 échoue : seulement {n_pass}/{n_total} régimes "
                f"passent (minimum requis : {min_pass}). "
                f"Régimes en échec : {', '.join(failed_names) or 'aucun'}. "
                f"Pour passer ce filtre, le signal doit maintenir CNSR > -1.0 "
                f"sur les régimes bear et crash — envisager un filtre de régime "
                f"endogène ou réduire l'exposition en haute volatilité."
            )

        return FilterVerdict(
            passed=passed,
            filter_name=self.NAME,
            metrics={
                "n_pass":   n_pass,
                "n_total":  n_total,
                "min_pass": min_pass,
                "failed":   failed_names,
                # RECALIBRATION_PENDING : mode actuel documenté
                "delta_mode": "absolute",
                "regime_detail": {
                    r.label: {
                        "passed":     r.passed,
                        "cnsr_strat": round(r.cnsr_strat, 4) if r.cnsr_strat is not None else None,
                        "cnsr_bench": round(r.cnsr_bench, 4) if r.cnsr_bench is not None else None,
                        "delta":      round(r.delta, 4) if r.delta is not None else None,
                    }
                    for r in results
                },
            },
            diagnosis=diagnosis,
        )

    def _make_strategy_fn(self, signal: SignalData):
        """
        Adapte SignalData vers la signature attendue par run_phase1() :
          strategy_fn(r_pair: pd.Series, params: dict) -> pd.Series

        Convention run_phase1 : alloc représente la fraction PAXG.
        r_port = alloc_paxg * r_pair → r_usd = alloc_paxg * r_pair + r_btc_usd

        signal.alloc_btc est la fraction BTC → conversion alloc_paxg = 1 - alloc_btc.

        run_phase1() génère ses propres données synthétiques (dates 2020-01-01+).
        On réindexe sur ces dates ; les valeurs hors plage → 0.5 (neutre).
        """
        alloc_btc = signal.alloc_btc

        def strategy_fn(r_pair, params):
            alloc_paxg = 1.0 - alloc_btc.reindex(r_pair.index).fillna(0.5)
            return alloc_paxg

        return strategy_fn
