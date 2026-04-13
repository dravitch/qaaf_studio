"""
Backtester — Layer 1 QAAF Studio 3.0

Mode LUMP SUM UNIQUEMENT pour la certification.
Capital fixe au Jour 0 — pas de versements périodiques.

RÈGLE : Ne jamais modifier le mode en certification.
Le mode DCA existe pour usage analytique uniquement,
et émet un avertissement explicite s'il est activé.
"""

import pandas as pd
import numpy as np
import warnings
import yaml
from pathlib import Path
from typing import Callable, Optional


class Backtester:
    def __init__(self, config_path: str = "config.yaml"):
        cfg = yaml.safe_load(Path(config_path).read_text())
        self.fees_pct        = cfg["engine"]["fees_pct"]
        self.initial_capital = cfg["engine"]["initial_capital"]
        self.mode            = cfg["engine"]["mode"]

        if self.mode != "lump_sum":
            warnings.warn(
                "WARNING: Mode != lump_sum. "
                "Le DCA mesure l'accumulation de capital, pas la règle d'allocation. "
                "Les CNSR obtenus ne sont pas comparables entre stratégies. "
                "Utilisez mode='lump_sum' pour certifier.",
                stacklevel=2,
            )

    def run(
        self,
        allocation_fn: Callable[[pd.DataFrame], pd.Series],
        prices_df:     pd.DataFrame,
        r_btc_usd:     pd.Series,
    ) -> dict:
        """
        Exécute un backtest lump sum.

        Paramètres
        ----------
        allocation_fn : fonction qui prend prices_df et retourne
                        une Series d'allocations en PAXG [0, 1]
                        (1 - alloc = BTC)
        prices_df     : DataFrame avec colonnes 'paxg' et 'btc' (prix USD)
        r_btc_usd     : log-rendements BTC/USD (pour conversion CNSR)

        Retourne
        --------
        dict avec r_pair, r_portfolio_usd, r_base_usd, allocations,
                  n_trades, fees_paid, std_alloc
        """
        alloc = allocation_fn(prices_df).clip(0, 1)
        alloc = alloc.reindex(prices_df.index).ffill()

        # Log-rendements de la paire PAXG/BTC
        r_paxg_usd_aligned = np.log(prices_df["paxg"] / prices_df["paxg"].shift(1))
        r_btc_usd_aligned  = np.log(prices_df["btc"]  / prices_df["btc"].shift(1))
        r_pair             = r_paxg_usd_aligned - r_btc_usd_aligned

        # Rendements pondérés (en numéraire BTC)
        r_portfolio_btc = alloc.shift(1) * r_pair

        # Frais : appliquer aux changements d'allocation
        alloc_change        = alloc.diff().abs()
        fees                = alloc_change * self.fees_pct
        r_portfolio_btc_net = r_portfolio_btc - fees

        # Conversion en USD : r_USD = r_pair_net + r_BTC (identité log)
        r_portfolio_usd = (
            r_portfolio_btc_net
            + r_btc_usd.reindex(alloc.index).fillna(0)
        )

        n_trades   = int((alloc_change > 1e-6).sum())
        fees_paid  = float(fees.sum() * self.initial_capital)
        r_pair_out = r_pair.dropna()

        return {
            "r_pair":          r_pair_out,
            "r_portfolio_usd": r_portfolio_usd.dropna(),
            "r_base_usd":      r_btc_usd_aligned.reindex(
                                   r_portfolio_usd.dropna().index),
            "allocations":     alloc,
            "n_trades":        n_trades,
            "fees_paid":       fees_paid,
            "std_alloc":       float(alloc.std()),
        }
