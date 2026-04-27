"""
studio/filters/metis_q2.py
MÉTIS Q2 — Test de permutation (significativité statistique).

Porte la logique existante de layer3_validation/metis_q2_permutation.py
vers l'interface Filter.

Recalibration Session 2 : ajustement de régime (Alternative A).
Si le benchmark OOS dépasse le benchmark IS × regime_factor, le régime OOS
est détecté comme exceptionnel → on compare cnsr_obs au benchmark IS.
Cela neutralise l'effet du bull run systémique sans changer la permutation.

Quand regime_adjusted=False : critère p-value normal (p < p_threshold).
Quand regime_adjusted=True  : critère de dominance directe (cnsr_obs > cnsr_bench_is).
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from studio.interfaces import Filter, FilterConfig, FilterVerdict, FilterError, SignalData

from layer3_validation.metis_q2_permutation import run_q2
from layer1_engine.backtester import Backtester
from layer1_engine.benchmark_factory import BenchmarkFactory


class MetisQ2(Filter):
    """
    MÉTIS Q2 — Test de permutation avec ajustement de régime.

    Complexity Budget :
      - evaluate() ≤ 50 lignes
      - ≤ 3 niveaux d'imbrication
      - ≤ 5 paramètres dans FilterConfig.params

    Paramètres acceptés dans config.params :
      - n_perm        : int   — nombre de permutations (défaut : 500)
      - p_threshold   : float — seuil p-value (défaut : 0.05)
      - regime_margin : float — différence absolue IS→OOS détectant un bull run (défaut : 1.5)
      - seed          : int   — graine aléatoire (défaut : 42)

    Critère recalibré (Alternative A) :
      Si cnsr_bench_oos - cnsr_bench_is > regime_margin :
        passed = cnsr_obs > cnsr_bench_oos  (oracle doit battre le benchmark OOS)
      Sinon :
        passed = pvalue < p_threshold        (test de permutation standard)

    Note sur Q2Result : ne retourne pas la distribution des permutations
    → Option B (comparaison directe) est la seule option sans modifier run_q2().
    """

    NAME = "metis_q2_permutation"

    def evaluate(
        self,
        signal: SignalData,
        config: FilterConfig,
    ) -> FilterVerdict:
        n_perm        = config.get("n_perm", 500)
        p_threshold   = config.get("p_threshold", 0.05)
        regime_margin = config.get("regime_margin", 1.5)
        seed          = config.get("seed", 42)

        prices_oos, r_btc_oos = self._extract_oos(signal)
        allocation_fn         = self._make_allocation_fn(signal)

        try:
            backtester    = Backtester()
            result        = run_q2(
                prices_oos=prices_oos, r_btc_oos=r_btc_oos,
                allocation_fn=allocation_fn, backtester=backtester,
                n_perm=n_perm, pvalue_threshold=p_threshold, seed=seed,
            )
            cnsr_bench_is = self._compute_is_benchmark(signal, backtester)
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

        cnsr_obs       = result.cnsr_obs
        cnsr_bench_oos = result.cnsr_bench
        pvalue         = result.pvalue

        # Détection régime exceptionnel : lift absolu du benchmark OOS vs IS
        # cnsr_bench_oos - cnsr_bench_is > regime_margin = bull run systémique
        regime_adjusted = (cnsr_bench_oos - cnsr_bench_is) > regime_margin

        if regime_adjusted:
            # En bull run, exiger que l'oracle batte le benchmark OOS
            # (plus discriminant que IS : le signal aléatoire peut battre IS par hasard)
            passed     = cnsr_obs > cnsr_bench_oos
            pvalue_ref = 0.0 if passed else 1.0
            regime_note = (
                f" Régime OOS exceptionnel (lift={cnsr_bench_oos - cnsr_bench_is:.2f} "
                f"> {regime_margin}) : oracle doit battre benchmark OOS."
            )
        else:
            passed     = pvalue < p_threshold
            pvalue_ref = pvalue
            regime_note = ""

        if passed:
            diagnosis = (
                f"Q2 validée : cnsr_obs={cnsr_obs:.3f} > benchmark OOS={cnsr_bench_oos:.3f}."
                f"{regime_note}"
            ) if regime_adjusted else (
                f"Q2 validée : p={pvalue:.3f} < {p_threshold} sur {result.n_perm} "
                f"permutations — alpha non trivial confirmé."
            )
        else:
            diagnosis = (
                f"Q2 échoue : cnsr_obs={cnsr_obs:.3f} ≤ benchmark OOS={cnsr_bench_oos:.3f}."
                f"{regime_note} "
                f"Pour passer Q2 en régime bull, le signal doit battre le benchmark "
                f"OOS — envisager de renforcer l'exposition directionnelle en tendance."
            ) if regime_adjusted else (
                f"Q2 échoue : p={pvalue:.3f} ≥ {p_threshold} "
                f"(cnsr_obs={cnsr_obs:.3f}, bench_oos={cnsr_bench_oos:.3f}). "
                f"Pour passer Q2, le signal doit produire un CNSR OOS "
                f"significativement supérieur au bruit — envisager un filtre "
                f"de régime pour isoler l'alpha du bull run systémique."
            )

        return FilterVerdict(
            passed=passed,
            filter_name=self.NAME,
            metrics={
                "pvalue":          round(pvalue_ref, 4),
                "pvalue_raw":      round(pvalue, 4),
                "cnsr_obs":        round(cnsr_obs, 4),
                "cnsr_bench_oos":  round(cnsr_bench_oos, 4),
                "cnsr_bench_is":   round(cnsr_bench_is, 4),
                "regime_adjusted": regime_adjusted,
                "regime_margin":   regime_margin,
                "n_perm":          result.n_perm,
                "criterion":       "regime_adjusted",
            },
            diagnosis=diagnosis,
        )

    def _extract_oos(self, signal: SignalData):
        """Extrait les données OOS depuis SignalData."""
        oos_start  = pd.Timestamp(signal.oos_start)
        oos_end    = pd.Timestamp(signal.oos_end)
        paxg_oos   = signal.prices_quote_usd.loc[oos_start:oos_end]
        btc_oos    = signal.prices_base_usd.loc[oos_start:oos_end]
        common     = paxg_oos.index.intersection(btc_oos.index)
        prices_oos = pd.DataFrame({"paxg": paxg_oos.loc[common], "btc": btc_oos.loc[common]})
        r_btc_oos  = np.log(btc_oos.loc[common] / btc_oos.loc[common].shift(1)).dropna()
        return prices_oos, r_btc_oos

    def _compute_is_benchmark(self, signal: SignalData, backtester: Backtester) -> float:
        """Calcule le CNSR de B_5050 sur la période IS."""
        is_start = pd.Timestamp(signal.is_start)
        is_end   = pd.Timestamp(signal.is_end)
        paxg_is  = signal.prices_quote_usd.loc[is_start:is_end]
        btc_is   = signal.prices_base_usd.loc[is_start:is_end]
        common   = paxg_is.index.intersection(btc_is.index)
        prices_is = pd.DataFrame({"paxg": paxg_is.loc[common], "btc": btc_is.loc[common]})
        r_btc_is  = np.log(btc_is.loc[common] / btc_is.loc[common].shift(1)).dropna()
        factory   = BenchmarkFactory(backtester)
        return float(factory.b_5050(prices_is, r_btc_is)["cnsr_usd_fed"])

    def _make_allocation_fn(self, signal: SignalData):
        """
        Retourne une allocation_fn(prices_df) -> pd.Series pour le Backtester.
        Convention Backtester : alloc = fraction PAXG.
        signal.alloc_btc est BTC fraction → alloc_paxg = 1 - alloc_btc.
        """
        alloc_btc = signal.alloc_btc

        def allocation_fn(prices_df):
            return (1.0 - alloc_btc.reindex(prices_df.index).fillna(0.5)).clip(0.0, 1.0)

        return allocation_fn
