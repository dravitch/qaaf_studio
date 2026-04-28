"""
studio/filters/paf_d3_signal_informatif.py
PAF Direction 3 — Test de la source minimale.

Porte run_d3() vers l'interface Filter.

Rôle : vérifier que la performance du signal provient d'un contenu
       informationnel réel, et non d'un simple artefact de lissage
       reproductible par une EMA triviale à iso-variance.

Protocole : comparer le signal testé à un EMA H9 (IQR-normalisé)
            de même variance sur la période OOS.

Verdict → passed :
  SIGNAL_INFORMATIF → True  (delta > 0.05 : le signal apporte de l'info)
  ARTEFACT_LISSAGE  → False (delta ≤ 0.05 : reproductible par EMA triviale)
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from studio.interfaces import Filter, FilterConfig, FilterVerdict, FilterError, SignalData

from layer2_qualification.paf.paf_d3_source import run_d3
from layer1_engine.backtester import Backtester


class PAFD3SignalInformatif(Filter):
    """
    PAF D3 — Source minimale : signal informatif vs artefact de lissage.

    Complexity Budget :
      - evaluate() ≤ 50 lignes
      - ≤ 3 niveaux d'imbrication
      - ≤ 5 paramètres dans FilterConfig.params

    Paramètres dans config.params :
      - seuil_informatif : float — delta minimum pour SIGNAL_INFORMATIF (défaut : 0.05)
    """

    NAME = "paf_d3_signal_informatif"

    def evaluate(
        self,
        signal: SignalData,
        config: FilterConfig,
    ) -> FilterVerdict:
        prices_oos, r_btc_oos = self._extract_oos(signal)
        signal_fn = self._make_signal_fn(signal)

        try:
            backtester = Backtester()
            result = run_d3(
                prices_oos=prices_oos,
                r_btc_oos=r_btc_oos,
                signal_fn=signal_fn,
                backtester=backtester,
            )
        except Exception as e:
            raise FilterError(
                filter_name=self.NAME,
                cause=f"Erreur lors de l'exécution de PAF D3 : {e}",
                action=(
                    "Vérifier que layer2_qualification.paf est importable "
                    "et que la période OOS contient suffisamment de données "
                    "(≥ 60 jours recommandés pour IQR rolling)."
                ),
            )

        passed  = result.verdict == "SIGNAL_INFORMATIF"
        verdict = result.verdict

        if passed:
            diagnosis = (
                f"PAF D3 validée ({verdict}) : "
                f"cnsr_signal={result.cnsr_signal:.4f}, "
                f"cnsr_trivial={result.cnsr_trivial:.4f}, "
                f"delta={result.delta:+.4f} > 0.05. "
                f"Le signal contient un contenu informationnel non reproductible "
                f"par un EMA triviale (span={result.ema_span_used}j)."
            )
        else:
            diagnosis = (
                f"PAF D3 échoue ({verdict}) : "
                f"cnsr_signal={result.cnsr_signal:.4f}, "
                f"cnsr_trivial={result.cnsr_trivial:.4f}, "
                f"delta={result.delta:+.4f} ≤ 0.05. "
                f"La performance est reproductible par une EMA H9 triviale "
                f"(span={result.ema_span_used}j, std_alloc≈{result.std_alloc_trivial:.4f}). "
                f"Envisager d'enrichir la source du signal."
            )

        return FilterVerdict(
            passed=passed,
            filter_name=self.NAME,
            metrics={
                "verdict":           verdict,
                "cnsr_signal":       result.cnsr_signal,
                "cnsr_trivial":      result.cnsr_trivial,
                "delta":             result.delta,
                "ema_span_used":     result.ema_span_used,
                "std_alloc_signal":  result.std_alloc_signal,
                "std_alloc_trivial": result.std_alloc_trivial,
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

        def signal_fn(df):
            return (1.0 - alloc_btc.reindex(df.index).fillna(0.5)).clip(0.0, 1.0)

        return signal_fn
