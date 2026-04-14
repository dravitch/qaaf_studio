"""
BenchmarkFactory — Layer 1 QAAF Studio 3.0

Crée les benchmarks passifs en CNSR-USD natif.
Tous les benchmarks utilisent le même Backtester que les stratégies actives
— pas de calcul séparé, conditions identiques garanties.

Fix v1.1 : utilise r_portfolio_usd (pas r_pair) pour compute_cnsr,
cohérent avec la correction du Backtester.
"""

import pandas as pd
import numpy as np
from .backtester     import Backtester
from .metrics_engine import compute_cnsr


class BenchmarkFactory:
    def __init__(self, backtester: Backtester):
        self.backtester = backtester

    def b_5050(self, prices_df: pd.DataFrame, r_btc_usd: pd.Series) -> dict:
        """50% PAXG / 50% BTC, rééquilibré quotidiennement."""
        def alloc_fn(df):
            return pd.Series(0.5, index=df.index)
        return self._run(alloc_fn, prices_df, r_btc_usd, name="B_5050")

    def b_btc(self, prices_df: pd.DataFrame, r_btc_usd: pd.Series) -> dict:
        """100% BTC (buy & hold)."""
        def alloc_fn(df):
            return pd.Series(0.0, index=df.index)
        return self._run(alloc_fn, prices_df, r_btc_usd, name="B_BTC")

    def b_paxg(self, prices_df: pd.DataFrame, r_btc_usd: pd.Series) -> dict:
        """100% PAXG (buy & hold)."""
        def alloc_fn(df):
            return pd.Series(1.0, index=df.index)
        return self._run(alloc_fn, prices_df, r_btc_usd, name="B_PAXG")

    def _run(self, alloc_fn, prices_df, r_btc_usd, name) -> dict:
        result = self.backtester.run(alloc_fn, prices_df, r_btc_usd)

        # Utiliser r_portfolio_usd avec r_base=0 :
        # r_portfolio_usd est déjà en USD (allocation × r_pair + r_btc)
        # r_base_usd = 0 (retourné par le Backtester fix v1.1)
        metrics = compute_cnsr(result["r_portfolio_usd"], result["r_base_usd"])

        metrics["name"]          = name
        metrics["n_trades"]      = result["n_trades"]
        metrics["fees_paid"]     = result["fees_paid"]
        metrics["std_alloc"]     = result["std_alloc"]
        return metrics
