"""
studio/filters/mif_phase1.py
MIF Phase 1 — Généralisation OOS (G1-G5).

Porte la logique existante de layer2_qualification/mif/phase1_oos.py
vers l'interface Filter. Aucune logique nouvelle — portage à l'identique.

Recalibration Session 2 : critère de dominance relative (Alternative B).
Le signal passe un régime s'il domine le benchmark sur ce régime,
même quand les deux perdent. Absorbe le bruit commun du régime (kurtosis F2).
"""

from __future__ import annotations

from studio.interfaces import Filter, FilterConfig, FilterVerdict, FilterError, SignalData

from layer2_qualification.mif.phase1_oos import run_phase1


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

    Critère recalibré (Alternative B) : dominance relative.
    Un régime est passé si cnsr_strat > cnsr_bench (pas de seuil absolu).
    """

    NAME = "mif_phase1_generalization"

    def evaluate(
        self,
        signal: SignalData,
        config: FilterConfig,
    ) -> FilterVerdict:
        min_pass  = config.get("min_pass", 3)
        rf_annual = config.get("rf_annual", 0.04)

        try:
            results = run_phase1(self._make_strategy_fn(signal), params={}, rf_annual=rf_annual)
        except Exception as e:
            raise FilterError(
                filter_name=self.NAME,
                cause=f"Erreur lors de l'exécution de MIF Phase 1 : {e}",
                action=(
                    "Vérifier que layer2_qualification.mif est importable "
                    "et que signal.alloc_btc est une pd.Series indexée par date."
                ),
            )

        regimes = {}
        for r in results:
            cs = r.cnsr_strat
            cb = r.cnsr_bench
            passed_regime = (cs is not None and cb is not None and cs > cb)
            regimes[r.label] = {
                "passed": passed_regime,
                "cnsr":   round(cs, 4) if cs is not None else None,
                "bench":  round(cb, 4) if cb is not None else None,
                "delta":  round(cs - cb, 4) if (cs is not None and cb is not None) else None,
            }

        n_pass       = sum(1 for r in regimes.values() if r["passed"])
        failed_names = [k for k, r in regimes.items() if not r["passed"]]
        passed       = n_pass >= min_pass

        if passed:
            diagnosis = (
                f"Phase 1 validée : {n_pass}/{len(regimes)} régimes — "
                f"le signal domine le benchmark sur la majorité des régimes "
                f"synthétiques, y compris les régimes adverses."
            )
        else:
            diagnosis = (
                f"Phase 1 échoue : {n_pass}/{len(regimes)} régimes passent "
                f"(minimum : {min_pass}). "
                f"Régimes où le signal sous-performe le benchmark : "
                f"{', '.join(failed_names)}. "
                f"Pour passer ce filtre, le signal doit battre le benchmark "
                f"relativement sur au moins {min_pass} régimes — envisager "
                f"un filtre de régime endogène (volatilité rolling du ratio) "
                f"pour les régimes bear et latéral."
            )

        return FilterVerdict(
            passed=passed,
            filter_name=self.NAME,
            metrics={
                "n_pass":        n_pass,
                "n_total":       len(regimes),
                "min_pass":      min_pass,
                "failed":        failed_names,
                "delta_mode":    "relative",
                "regime_detail": regimes,
            },
            diagnosis=diagnosis,
        )

    def _make_strategy_fn(self, signal: SignalData):
        """
        Retourne une strategy_fn(r_pair, params) -> pd.Series pour run_phase1().

        Convention run_phase1 : alloc = fraction PAXG.
        r_port = alloc_paxg * r_pair → r_usd = r_port + r_btc_usd.

        Pourquoi calcul dynamique et non réindexage de signal.alloc_btc :
        run_phase1() génère ses propres prix synthétiques (5 régimes, seeds
        différents). Les allocations pré-calculées sur les données réelles
        sont décorrélées de ces prix → signal effectivement aléatoire.
        La logique oracle doit être RE-CALCULÉE sur chaque r_pair synthétique
        pour tester si l'ALGORITHME généralise, pas ses sorties figées.

        Paramètres oracle extraits de studio.oracle.ORACLE_PARAMS.
        alloc_btc → alloc_paxg = 1 - alloc_btc.
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

            return 1.0 - alloc_btc  # alloc_paxg pour run_phase1

        return strategy_fn
