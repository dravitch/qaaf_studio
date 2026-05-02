"""
studio/filters/mif_phase2.py
MIF Phase 2 — Transfert multi-actifs (M1-M4).

Porte run_phase2() vers l'interface Filter.

Rôle : vérifier que l'algorithme de signal produit des allocations finies
       sur 4 paires synthétiques aux caractéristiques différentes.
       Gate : gate_ratio des paires doivent passer (défaut 75 % = 3/4).

Verdict → passed :
  n_pass / n_total >= gate_ratio → True
  n_pass / n_total <  gate_ratio → False
"""

from __future__ import annotations

from studio.interfaces import Filter, FilterConfig, FilterVerdict, FilterError, SignalData

from layer2_qualification.mif.phase2_multiasset import run_phase2


class MIFPhase2(Filter):
    """
    MIF Phase 2 — Transfert multi-actifs sur 4 paires synthétiques (M1-M4).

    Complexity Budget :
      - evaluate() ≤ 50 lignes
      - ≤ 3 niveaux d'imbrication
      - ≤ 5 paramètres dans FilterConfig.params

    Paramètres acceptés dans config.params :
      - gate_ratio : float — fraction minimum de paires à passer (défaut : 0.75)
      - rf_annual  : float — taux sans risque annualisé (défaut : 0.04)

    Convention : strategy_fn(r_pair, params) → alloc_paxg ∈ [0, 1].
    Même algorithme oracle que Phase 0/1, re-calculé dynamiquement.
    """

    NAME = "mif_phase2_multiasset"

    def evaluate(
        self,
        signal: SignalData,
        config: FilterConfig,
    ) -> FilterVerdict:
        gate_ratio = config.get("gate_ratio", 0.75)
        rf_annual  = config.get("rf_annual", 0.04)

        try:
            results = run_phase2(
                self._make_strategy_fn(signal),
                params={},
                rf_annual=rf_annual,
            )
        except Exception as e:
            raise FilterError(
                filter_name=self.NAME,
                cause=f"Erreur lors de l'exécution de MIF Phase 2 : {e}",
                action=(
                    "Vérifier que layer2_qualification.mif est importable "
                    "et que la stratégie ne lève pas d'exception sur les 4 paires."
                ),
            )

        pairs = {r.label: {
            "passed":     r.passed,
            "cnsr":       round(r.cnsr, 4) if r.cnsr is not None else None,
            "cnsr_bench": round(r.cnsr_bench, 4) if r.cnsr_bench is not None else None,
            "notes":      r.notes,
        } for r in results}

        n_pass  = sum(1 for r in results if r.passed)
        n_total = len(results)
        failed  = [r.label for r in results if not r.passed]
        ratio   = n_pass / n_total if n_total > 0 else 0.0
        passed  = ratio >= gate_ratio

        if passed:
            diagnosis = (
                f"MIF Phase 2 validée : {n_pass}/{n_total} paires réussies "
                f"({ratio:.0%} ≥ {gate_ratio:.0%}). "
                f"L'algorithme généralise sur des paires aux caractéristiques variées."
            )
        else:
            diagnosis = (
                f"MIF Phase 2 échoue : {n_pass}/{n_total} paires réussies "
                f"({ratio:.0%} < {gate_ratio:.0%}). "
                f"Paires échouées : {', '.join(failed) if failed else 'gate_ratio trop élevé'}. "
                f"Vérifier que l'algorithme produit des allocations finies sur toutes les paires "
                f"et ne lève pas d'exception en conditions de marché extrêmes."
            )

        return FilterVerdict(
            passed=passed,
            filter_name=self.NAME,
            metrics={
                "n_pass":     n_pass,
                "n_total":    n_total,
                "gate_ratio": gate_ratio,
                "ratio":      round(ratio, 4),
                "failed":     failed,
                "pairs":      pairs,
            },
            diagnosis=diagnosis,
        )

    def _make_strategy_fn(self, signal: SignalData):
        """
        Retourne strategy_fn(r_pair, params) → alloc_paxg pour run_phase2().

        Même logique oracle que MIFPhase0 et MIFPhase1.
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
