"""
studio/filters/metis_q4.py
MÉTIS Q4 — DSR (Deflated Sharpe Ratio).

Porte la logique existante vers l'interface Filter.
Recalibration intégrée : N_effectif au lieu de N_trials brut.

Fondement : Bailey & López de Prado (2014), section 4.3.
"effective number of independent trials" =
  N_brut × (1 - corrélation_moyenne_inter_variantes)

Pour une famille EMA span 20j-120j (N=101) avec corrélation
inter-variantes ≈ 0.97, N_effectif ≈ 3.

Pour le signal oracle (N_trials=1), N_effectif=1.
La correction ne change rien pour un signal unique non optimisé.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from studio.interfaces import Filter, FilterConfig, FilterVerdict, FilterError, SignalData


def compute_n_effectif(
    n_trials: int,
    avg_correlation: float | None = None,
) -> int:
    """
    Calcule N_effectif depuis N_trials brut.

    Si avg_correlation est fourni :
      N_effectif = max(1, round(n_trials × (1 - avg_correlation)))

    Si avg_correlation est None (signal unique ou inconnu) :
      N_effectif = n_trials (pas de correction)

    Paramètres :
      n_trials       : nombre de variantes testées dans la famille
      avg_correlation: corrélation moyenne entre variantes adjacentes
                       (ex. 0.97 pour grille EMA 20j-120j)

    Retourne :
      int ≥ 1
    """
    if avg_correlation is None or n_trials <= 1:
        return n_trials

    return max(1, round(n_trials * (1 - avg_correlation)))


class MetisQ4(Filter):
    """
    MÉTIS Q4 — DSR avec N_effectif.

    Complexity Budget :
      - evaluate() ≤ 50 lignes
      - ≤ 3 niveaux d'imbrication
      - ≤ 5 paramètres dans FilterConfig.params

    Paramètres dans config.params (≤ 5) :
      - n_trials        : int   — variantes testées dans la famille (défaut : 1)
      - avg_correlation : float — corrélation moyenne inter-variantes
                                  None = pas de correction (défaut)
      - dsr_threshold   : float — seuil DSR (défaut : 0.95)
      - rf_annual       : float — taux sans risque annuel (défaut : 0.04)
    """

    NAME = "metis_q4_dsr"

    def evaluate(
        self,
        signal: SignalData,
        config: FilterConfig,
    ) -> FilterVerdict:
        n_trials      = config.get("n_trials", 1)
        avg_corr      = config.get("avg_correlation", None)
        dsr_threshold = config.get("dsr_threshold", 0.95)
        rf_annual     = config.get("rf_annual", 0.04)

        n_effectif = compute_n_effectif(n_trials, avg_corr)

        try:
            dsr = self._compute_dsr(signal, n_effectif, rf_annual)
        except Exception as e:
            raise FilterError(
                filter_name=self.NAME,
                cause=f"Erreur calcul DSR : {e}",
                action=(
                    "Vérifier que deflated_sharpe_ratio est importable "
                    "et que les données OOS contiennent suffisamment "
                    "de points (≥ 30 jours recommandés)."
                ),
            )

        passed = dsr >= dsr_threshold

        n_note = (
            f"N_effectif={n_effectif} (corrigé depuis N_brut={n_trials}, "
            f"corrélation inter-variantes={avg_corr:.2f})"
            if avg_corr is not None and n_trials > 1
            else f"N_trials={n_effectif}"
        )

        if passed:
            diagnosis = (
                f"Q4 validée : DSR={dsr:.4f} ≥ {dsr_threshold} avec {n_note}. "
                f"Le signal est statistiquement défendable après correction "
                f"pour le multiple testing."
            )
        else:
            diagnosis = (
                f"Q4 échoue : DSR={dsr:.4f} < {dsr_threshold} avec {n_note}. "
                f"Le signal n'est pas statistiquement défendable après correction "
                f"pour le multiple testing. "
                f"Pour passer Q4, soit réduire N_effectif (signal unique, pas "
                f"d'optimisation de paramètre), soit obtenir un CNSR OOS "
                f"significativement plus élevé sur une période plus longue "
                f"(T plus grand fait baisser le seuil sr0)."
            )

        return FilterVerdict(
            passed=passed,
            filter_name=self.NAME,
            metrics={
                "dsr":             round(dsr, 4),
                "dsr_threshold":   dsr_threshold,
                "n_trials":        n_trials,
                "n_effectif":      n_effectif,
                "avg_correlation": avg_corr,
                "rf_annual":       rf_annual,
            },
            diagnosis=diagnosis,
        )

    def _compute_dsr(
        self,
        signal: SignalData,
        n_effectif: int,
        rf_annual: float,
    ) -> float:
        """
        Calcule le DSR sur la période OOS de SignalData.

        Convention Backtester (Fix v1.1) :
          r_portfolio_usd = alloc_paxg × r_pair + r_btc_usd
          = alloc_paxg × r_paxg_usd + alloc_btc × r_btc_usd

        Note : alloc_paxg × r_pair + r_btc_usd (et NON + alloc_btc × r_btc_usd)
        car alloc_paxg + alloc_btc = 1 → identité Karnosky-Singer.
        """
        from layer1_engine.metrics_engine import deflated_sharpe_ratio

        oos_start = pd.Timestamp(signal.oos_start)
        oos_end   = pd.Timestamp(signal.oos_end)
        oos_mask  = (
            (signal.prices_pair.index >= oos_start) &
            (signal.prices_pair.index <= oos_end)
        )

        r_pair = np.log(signal.prices_pair[oos_mask]).diff().dropna()
        r_base = np.log(signal.prices_base_usd[oos_mask]).diff().dropna()
        r_pair, r_base = r_pair.align(r_base, join="inner")

        alloc_oos  = signal.alloc_btc.reindex(r_pair.index).fillna(0.5)
        alloc_paxg = 1.0 - alloc_oos

        # Formule Backtester : alloc_paxg * r_pair + r_btc_usd
        r_usd = alloc_paxg * r_pair + r_base

        return deflated_sharpe_ratio(r_usd, n_trials=n_effectif, rf_annual=rf_annual)
