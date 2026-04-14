"""
Backtester — Layer 1 QAAF Studio 3.0

Mode LUMP SUM UNIQUEMENT pour la certification.
Capital fixe au Jour 0 — pas de versements périodiques.

RÈGLE : Ne jamais modifier le mode en certification.
Le mode DCA existe pour usage analytique uniquement,
et émet un avertissement explicite s'il est activé.

Note CNSR (fix v1.1) :
  r_portfolio_usd est déjà en USD (r_portfolio_btc_net + r_btc).
  Pour compute_cnsr(), r_base_usd = 0 car la conversion est déjà faite.
  L'ancien bug retournait r_btc_usd_aligned comme r_base_usd, ce qui
  faisait que compute_cnsr(r_pair_brut, r_btc) calculait le Sharpe de
  la PAIRE plutôt que du PORTEFEUILLE — identique pour toutes les stratégies.
"""

import pandas as pd
import numpy as np
import warnings
import yaml
from pathlib import Path
from typing import Callable


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
        allocation_fn : fonction(prices_df) → Series d'allocations PAXG ∈ [0, 1]
        prices_df     : DataFrame colonnes 'paxg' et 'btc' (prix USD)
        r_btc_usd     : log-rendements BTC/USD (pour conversion USD)

        Retourne
        --------
        r_pair          : log-rendements bruts PAXG/BTC (indépendant de l'allocation)
        r_portfolio_usd : log-rendements du portefeuille en USD (dépend de l'allocation)
        r_base_usd      : zéros — r_portfolio_usd est déjà en USD, r_base = 0 pour CNSR
        r_btc_usd_raw   : r_btc aligné, exposé pour PAF/MIF (iso-variance, etc.)
        allocations     : série d'allocations PAXG appliquées
        n_trades        : nombre de rebalancements
        fees_paid       : frais totaux en USD
        std_alloc       : écart-type de l'allocation (mesure de turbulence du signal)
        """
        alloc = allocation_fn(prices_df).clip(0, 1)
        alloc = alloc.reindex(prices_df.index).ffill()

        # Log-rendements de la paire PAXG/BTC (brut, identique pour toutes les strats)
        r_paxg_usd_aligned = np.log(prices_df["paxg"] / prices_df["paxg"].shift(1))
        r_btc_usd_aligned  = np.log(prices_df["btc"]  / prices_df["btc"].shift(1))
        r_pair             = r_paxg_usd_aligned - r_btc_usd_aligned

        # Rendements pondérés par l'allocation (en numéraire BTC)
        # alloc.shift(1) : utiliser l'allocation de la veille (pas de lookahead)
        r_portfolio_btc = alloc.shift(1) * r_pair

        # Frais sur les changements d'allocation
        alloc_change        = alloc.diff().abs()
        fees                = alloc_change * self.fees_pct
        r_portfolio_btc_net = r_portfolio_btc - fees

        # Conversion en USD : r_USD = r_portfolio_btc_net + r_BTC
        # Identité log-rendements (Karnosky & Singer 1994)
        r_portfolio_usd = (
            r_portfolio_btc_net
            + r_btc_usd.reindex(alloc.index).fillna(0)
        )

        n_trades  = int((alloc_change > 1e-6).sum())
        fees_paid = float(fees.sum() * self.initial_capital)
        r_port    = r_portfolio_usd.dropna()

        return {
            # Série principale pour CNSR : portefeuille en USD
            # r_base_usd = 0 car la conversion USD est déjà faite dans r_portfolio_usd
            "r_pair":          r_pair.dropna(),
            "r_portfolio_usd": r_port,
            "r_base_usd":      pd.Series(0.0, index=r_port.index),

            # BTC brut exposé séparément pour PAF D3 (test iso-variance)
            "r_btc_usd_raw":   r_btc_usd_aligned.reindex(r_port.index),

            "allocations": alloc,
            "n_trades":    n_trades,
            "fees_paid":   fees_paid,
            "std_alloc":   float(alloc.std()),
        }
