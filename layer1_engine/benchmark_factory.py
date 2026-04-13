"""
QAAF Studio 3.0 — BenchmarkFactory
Couche 1 : benchmarks passifs pré-calculés en CNSR-USD natif.

B_5050 — 50 % PAXG / 50 % BTC, allocation statique lump sum
B_BTC  — 100 % BTC (buy & hold, 0 trades)
B_PAXG — 100 % PAXG (buy & hold)

Usage
-----
    bt = Backtester(config_path="config.yaml")
    bf = BenchmarkFactory(bt)
    b  = bf.b_5050(prices_df, r_btc_usd)  # → dict avec name, cnsr_usd_fed, ...
"""

import numpy as np
import pandas as pd
from layer1_engine.metrics_engine import compute_cnsr


class BenchmarkFactory:
    """
    Calcule les benchmarks passifs en CNSR-USD via le Backtester Layer 1.

    Paramètres
    ----------
    backtester : instance de Backtester (couche 1) — assure la cohérence
                 frais + mode lump_sum avec le reste du studio.
    """

    def __init__(self, backtester):
        self._bt = backtester

    # ------------------------------------------------------------------
    # Méthodes publiques
    # ------------------------------------------------------------------

    def b_5050(self, prices_df: pd.DataFrame, r_btc_usd: pd.Series) -> dict:
        """50 % PAXG / 50 % BTC — allocation statique."""
        return self._compute(
            alloc_fn  = lambda df: pd.Series(0.5, index=df.index),
            prices_df = prices_df,
            r_btc_usd = r_btc_usd,
            name      = "B_5050",
        )

    def b_btc(self, prices_df: pd.DataFrame, r_btc_usd: pd.Series) -> dict:
        """100 % BTC — buy & hold, 0 trades."""
        return self._compute(
            alloc_fn  = lambda df: pd.Series(0.0, index=df.index),
            prices_df = prices_df,
            r_btc_usd = r_btc_usd,
            name      = "B_BTC",
        )

    def b_paxg(self, prices_df: pd.DataFrame, r_btc_usd: pd.Series) -> dict:
        """100 % PAXG — buy & hold."""
        return self._compute(
            alloc_fn  = lambda df: pd.Series(1.0, index=df.index),
            prices_df = prices_df,
            r_btc_usd = r_btc_usd,
            name      = "B_PAXG",
        )

    # ------------------------------------------------------------------
    # Helper interne
    # ------------------------------------------------------------------

    def _compute(self, alloc_fn, prices_df, r_btc_usd, name) -> dict:
        result = self._bt.run(alloc_fn, prices_df, r_btc_usd)

        r_port_usd = result["r_portfolio_usd"]
        r_base     = result["r_base_usd"].reindex(r_port_usd.index)

        # r_pair_net = weighted pair return (net of fees)
        # Identité : r_portfolio_usd = r_pair_weighted_net + r_btc
        # donc r_pair_weighted_net = r_portfolio_usd - r_btc
        r_pair_net = (r_port_usd - r_base).dropna()
        r_base_c   = r_base.reindex(r_pair_net.index)

        cnsr = compute_cnsr(r_pair_net, r_base_c)

        return {
            "name":     name,
            "n_trades": result["n_trades"],
            **cnsr,
        }
