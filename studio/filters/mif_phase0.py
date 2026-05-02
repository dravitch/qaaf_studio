"""
studio/filters/mif_phase0.py
MIF Phase 0 — Tests d'isolation algorithmique (T1-T6).

Porte run_phase0() vers l'interface Filter.

Rôle : vérifier que l'algorithme de signal passe les 6 tests d'isolation
       sur données synthétiques avant d'être exposé aux données réelles.
       Un seul FAIL → algorithme défaillant, stopper immédiat.

Verdict → passed :
  n_pass >= min_pass (défaut 6) → True  (tous les tests passent)
  n_pass <  min_pass            → False (au moins un test a échoué)
"""

from __future__ import annotations

from studio.interfaces import Filter, FilterConfig, FilterVerdict, FilterError, SignalData

from layer2_qualification.mif.phase0_isolation import run_phase0


class MIFPhase0(Filter):
    """
    MIF Phase 0 — Isolation algorithmique sur 6 régimes synthétiques (T1-T6).

    Complexity Budget :
      - evaluate() ≤ 50 lignes
      - ≤ 3 niveaux d'imbrication
      - ≤ 5 paramètres dans FilterConfig.params

    Paramètres acceptés dans config.params :
      - min_pass  : int   — nombre minimum de tests à passer (défaut : 6 = tous)
      - rf_annual : float — taux sans risque annualisé (défaut : 0.04)

    Convention : strategy_fn(r_pair, params) → alloc_paxg ∈ [0, 1].
    L'algorithme oracle est re-calculé sur chaque série synthétique
    pour tester la généralisation de l'ALGORITHME, pas des sorties figées.
    """

    NAME = "mif_phase0_isolation"

    def evaluate(
        self,
        signal: SignalData,
        config: FilterConfig,
    ) -> FilterVerdict:
        min_pass  = config.get("min_pass", 6)
        rf_annual = config.get("rf_annual", 0.04)

        try:
            results = run_phase0(
                self._make_strategy_fn(signal),
                params={},
                rf_annual=rf_annual,
            )
        except Exception as e:
            raise FilterError(
                filter_name=self.NAME,
                cause=f"Erreur lors de l'exécution de MIF Phase 0 : {e}",
                action=(
                    "Vérifier que layer2_qualification.mif est importable "
                    "et que la stratégie ne lève pas d'exception sur données synthétiques."
                ),
            )

        tests = {r.label: {"passed": r.passed, "cnsr": r.cnsr, "notes": r.notes}
                 for r in results}
        n_pass = sum(1 for r in results if r.passed)
        failed = [r.label for r in results if not r.passed]
        passed = n_pass >= min_pass

        if passed:
            diagnosis = (
                f"MIF Phase 0 validée : {n_pass}/{len(results)} tests réussis. "
                f"L'algorithme est stable sur tous les régimes synthétiques (T1-T6)."
            )
        else:
            diagnosis = (
                f"MIF Phase 0 échoue : {n_pass}/{len(results)} tests réussis "
                f"(minimum : {min_pass}). "
                f"Tests échoués : {', '.join(failed)}. "
                f"Corriger l'algorithme avant d'exposer aux données réelles — "
                f"T3 requiert std_alloc > 0.01, T5 requiert asymétrie directionnelle."
            )

        return FilterVerdict(
            passed=passed,
            filter_name=self.NAME,
            metrics={
                "n_pass":   n_pass,
                "n_total":  len(results),
                "min_pass": min_pass,
                "failed":   failed,
                "tests":    tests,
            },
            diagnosis=diagnosis,
        )

    def _make_strategy_fn(self, signal: SignalData):
        """
        Retourne strategy_fn(r_pair, params) → alloc_paxg pour run_phase0().

        Utilise l'algorithme oracle (ORACLE_PARAMS) re-calculé dynamiquement
        sur chaque série synthétique. Voir mif_phase1._make_strategy_fn
        pour la justification du calcul dynamique vs réindexage.
        """
        import pandas as pd
        from studio.oracle import ORACLE_PARAMS

        p = ORACLE_PARAMS

        def strategy_fn(r_pair, params):
            log_ratio  = r_pair.cumsum()
            trend      = log_ratio.rolling(p["trend_window"]).mean().shift(1)
            vol        = r_pair.rolling(p["vol_window"]).std().shift(1)
            vol_median = vol.expanding().median()
            high_vol   = vol > p["vol_threshold"] * vol_median

            alloc_btc = pd.Series(p["alloc_low"], index=r_pair.index, dtype=float)
            long_btc  = trend.notna() & (log_ratio < trend)
            alloc_btc = alloc_btc.where(~long_btc, other=p["alloc_high"])
            alloc_btc = alloc_btc.where(~high_vol, other=0.5)
            alloc_btc = alloc_btc.fillna(0.5).clip(p["alloc_low"], p["alloc_high"])

            return 1.0 - alloc_btc

        return strategy_fn
