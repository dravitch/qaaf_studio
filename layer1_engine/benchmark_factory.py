# benchmark_factory.py

"""
QAAF Studio 3.0 — BenchmarkFactory
Couche 1 : benchmarks passifs pré-calculés en CNSR-USD natif.

B_5050 — 50 % PAXG / 50 % BTC, rééquilibré trimestriellement
B_BTC  — 100 % BTC (buy & hold)
B_PAXG — 100 % PAXG (buy & hold)
"""

import numpy as np
import pandas as pd
from metrics_engine import compute_cnsr, compute_full_metrics


class BenchmarkFactory:
    """
    Calcule les benchmarks passifs de référence en CNSR-USD.

    Usage
    -----
    from data_loader import DataLoader
    from split_manager import SplitManager, Split

    bundle = DataLoader().load()
    sm     = SplitManager(Split(...))
    bf     = BenchmarkFactory(bundle, sm)
    table  = bf.table()   # DataFrame comparatif
    """

    def __init__(self, bundle, split_manager):
        self._b   = bundle
        self._sm  = split_manager

    # ------------------------------------------------------------------

    def b_btc(self) -> dict:
        """100 % BTC. r_USD = r_BTC/USD directement."""
        _, oos_btc = self._sm.apply(self._b.r_btc_usd)
        _, oos_paxg_btc = self._sm.apply(self._b.r_paxg_btc)

        # Pour B_BTC : r_pair = 0 (on ne trade pas), r_base_usd = r_btc_usd
        r_pair_zero = pd.Series(np.zeros(len(oos_btc)), index=oos_btc.index)
        cnsr = compute_cnsr(r_pair_zero, oos_btc)
        return {"label": "B_BTC", **{k: v for k, v in cnsr.items() if k != "r_usd"}}

    def b_paxg(self) -> dict:
        """100 % PAXG. r_USD = r_PAXG/USD directement."""
        _, oos_paxg_usd = self._sm.apply(self._b.r_paxg_usd)
        _, oos_btc_usd  = self._sm.apply(self._b.r_btc_usd)

        # r_PAXG_USD = r_PAXG_BTC + r_BTC_USD — on peut aussi utiliser r_paxg_usd direct
        r_pair_zero = pd.Series(np.zeros(len(oos_paxg_usd)), index=oos_paxg_usd.index)
        cnsr = compute_cnsr(r_pair_zero, oos_paxg_usd)
        return {"label": "B_PAXG", **{k: v for k, v in cnsr.items() if k != "r_usd"}}

    def b_5050(self, rebal_freq: str = "QE") -> dict:
        """
        50 % PAXG / 50 % BTC, rééquilibré trimestriellement.
        Lump sum, frais nuls (benchmark passif de référence).
        """
        _, oos_paxg_usd = self._sm.apply(self._b.paxg_usd)
        _, oos_btc_usd  = self._sm.apply(self._b.btc_usd)

        # Simulation lump sum 50/50
        common = oos_paxg_usd.index.intersection(oos_btc_usd.index)
        p = oos_paxg_usd.loc[common]
        b = oos_btc_usd.loc[common]

        capital = 10_000.0
        paxg_u  = (capital * 0.5) / float(p.iloc[0])
        btc_u   = (capital * 0.5) / float(b.iloc[0])

        rebal_dates = set(
            pd.date_range(start=common[0], end=common[-1], freq=rebal_freq).date
        )

        values = []
        for date in common:
            val = paxg_u * float(p.loc[date]) + btc_u * float(b.loc[date])
            if date in rebal_dates:
                paxg_u = (val * 0.5) / float(p.loc[date])
                btc_u  = (val * 0.5) / float(b.loc[date])
            values.append(val)

        value_series = pd.Series(values, index=common)
        port_returns_raw = value_series.pct_change().dropna()
        # Approximation log-rendements
        port_r_log = np.log1p(port_returns_raw)

        # CNSR : pour un portefeuille USD, r_base_usd = 0 (déjà en USD)
        r_zero = pd.Series(np.zeros(len(port_r_log)), index=port_r_log.index)
        cnsr   = compute_cnsr(r_zero, port_r_log)

        return {"label": "B_5050", **{k: v for k, v in cnsr.items() if k != "r_usd"}}

    def table(self) -> pd.DataFrame:
        """Retourne un DataFrame comparatif des trois benchmarks."""
        rows = [self.b_btc(), self.b_paxg(), self.b_5050()]
        df   = pd.DataFrame(rows).set_index("label")
        cols_order = ["cnsr_usd_fed", "cnsr_usd_usdc", "cnsr_usd_0",
                      "sortino", "calmar", "omega", "max_dd_pct"]
        return df[[c for c in cols_order if c in df.columns]].round(3)

    def print_table(self) -> None:
        print("\n=== Benchmarks passifs (CNSR-USD) ===")
        print(self.table().to_string())


if __name__ == "__main__":
    from data_loader import DataLoader
    from split_manager import SplitManager, Split

    bundle = DataLoader(cache=True).load(days=2200)
    sm     = SplitManager(Split(
        is_start="2019-01-01", is_end="2022-12-31",
        oos_start="2023-01-01", oos_end="2024-12-31",
        label="PAXG_BTC_7030"
    ))
    bf = BenchmarkFactory(bundle, sm)
    bf.print_table()