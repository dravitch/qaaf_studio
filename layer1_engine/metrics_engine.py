"""
MetricsEngine — Layer 1 QAAF Studio 3.0

Métrique primaire : CNSR-USD (Common-Numeraire Sharpe Ratio)
Fondement : r_USD = r_pair + r_base_USD (identité log-rendements exacte,
Karnosky & Singer 1994).

IMPORTANT : Ce module ne contient AUCUNE logique de signal ni d'allocation.
Il calcule des métriques sur des séries de rendements fournies.
"""

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, norm
from typing import Optional
import yaml
from pathlib import Path


def compute_cnsr(
    r_pair: pd.Series,
    r_base_usd: pd.Series,
    rf_annual: float = 0.04,
) -> dict:
    """
    Calcule le CNSR-USD et les métriques complémentaires.

    Paramètres
    ----------
    r_pair     : log-rendements quotidiens de la paire (ex. PAXG/BTC)
    r_base_usd : log-rendements quotidiens de l'actif base en USD (ex. BTC/USD)
    rf_annual  : taux sans risque annuel (défaut 4% Fed)

    Retourne
    --------
    dict avec cnsr_usd_fed, cnsr_usd_usdc, cnsr_usd_0,
              sortino, calmar, omega, max_dd_pct, n_obs
    """
    # Aligner les index
    common = r_pair.index.intersection(r_base_usd.index)
    r_pair    = r_pair.loc[common].dropna()
    r_base    = r_base_usd.loc[r_pair.index].dropna()
    common2   = r_pair.index.intersection(r_base.index)
    r_pair    = r_pair.loc[common2]
    r_base    = r_base.loc[common2]

    r_usd = r_pair + r_base  # Identité exacte sur log-rendements

    rf_daily      = (1 + rf_annual) ** (1 / 252) - 1
    rf_usdc_daily = (1 + 0.03)      ** (1 / 252) - 1

    def _sharpe(r: pd.Series, rf: float) -> float:
        excess = r - rf
        std = excess.std(ddof=1)
        if std <= 0 or np.isnan(std):
            return np.nan
        return float(excess.mean() / std * np.sqrt(252))

    def _sortino(r: pd.Series, rf: float) -> float:
        excess   = r - rf
        downside = excess[excess < 0]
        if len(downside) == 0:
            return np.nan
        ds = downside.std(ddof=1)
        if ds <= 0:
            return np.nan
        return float(excess.mean() / ds * np.sqrt(252))

    def _calmar(r: pd.Series) -> float:
        cum    = (1 + r).cumprod()
        dd     = (cum - cum.cummax()) / cum.cummax()
        max_dd = abs(dd.min())
        if max_dd <= 0:
            return np.nan
        n_years = len(r) / 252
        ann_ret = cum.iloc[-1] ** (1 / max(n_years, 0.01)) - 1
        return float(ann_ret / max_dd)

    def _omega(r: pd.Series, threshold: float = 0.0) -> float:
        gains  = (r[r > threshold] - threshold).sum()
        losses = (threshold - r[r <= threshold]).sum()
        if losses <= 0:
            return np.nan
        return float(gains / losses)

    def _max_dd_pct(r: pd.Series) -> float:
        cum = (1 + r).cumprod()
        dd  = (cum - cum.cummax()) / cum.cummax()
        return float(abs(dd.min()) * 100)

    return {
        "cnsr_usd_fed":  _sharpe(r_usd, rf_daily),
        "cnsr_usd_usdc": _sharpe(r_usd, rf_usdc_daily),
        "cnsr_usd_0":    _sharpe(r_usd, 0.0),
        "sortino":       _sortino(r_usd, rf_daily),
        "calmar":        _calmar(r_usd),
        "omega":         _omega(r_usd),
        "max_dd_pct":    _max_dd_pct(r_usd),
        "n_obs":         len(r_usd),
    }


def deflated_sharpe_ratio(
    r_usd: pd.Series,
    n_trials: int,
    rf_annual: float = 0.04,
) -> float:
    """
    Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

    Probabilité que le CNSR observé ne soit pas dû au hasard,
    après correction pour le multiple testing et la non-normalité.

    Paramètres
    ----------
    r_usd    : log-rendements USD du portefeuille
    n_trials : nombre de variantes testées dans la même famille
    rf_annual: taux sans risque annuel

    Retourne
    --------
    float in [0, 1]. Seuil recommandé : 0.95.
    """
    r_usd = r_usd.dropna()
    T = len(r_usd)
    if T < 30:
        return np.nan

    rf_daily = (1 + rf_annual) ** (1 / 252) - 1
    excess   = r_usd - rf_daily
    std      = excess.std(ddof=1)
    if std <= 0:
        return np.nan

    sr_daily = float(excess.mean() / std)
    skew_    = float(skew(r_usd.dropna()))
    kurt_    = float(kurtosis(r_usd.dropna(), fisher=False))  # Pearson

    # Seuil déflaté
    sr0 = float(norm.ppf(1 - 1 / max(n_trials + 1, 2)) / np.sqrt(T))

    # Dénominateur (eq. 4 Bailey & LdP)
    inner = 1 - skew_ * sr_daily + (kurt_ - 3) / 4 * sr_daily ** 2
    denom = np.sqrt(max(inner, 1e-9))

    dsr = float(norm.cdf((sr_daily - sr0) * np.sqrt(T - 1) / denom))
    return round(dsr, 4)


class MetricsEngine:
    """
    Calcule toutes les métriques Layer 1 pour une stratégie donnée.
    Lit les taux depuis config.yaml.
    """

    def __init__(self, config_path: str = "config.yaml"):
        cfg = yaml.safe_load(Path(config_path).read_text())
        self.rf_fed  = cfg["rates"]["rf_fed"]
        self.rf_usdc = cfg["rates"]["rf_usdc"]
        self.rf_zero = cfg["rates"]["rf_zero"]

    def compute_all(
        self,
        r_pair: pd.Series,
        r_base_usd: pd.Series,
        n_trades: int = 0,
        n_trials: int = 1,
    ) -> dict:
        """
        Calcule CNSR + métriques complémentaires + DSR.

        Paramètres
        ----------
        r_pair     : log-rendements de la paire
        r_base_usd : log-rendements de la base en USD
        n_trades   : nombre de trades (fourni par le Backtester)
        n_trials   : nombre de variantes testées (pour DSR)
        """
        metrics = compute_cnsr(r_pair, r_base_usd, self.rf_fed)
        r_usd   = r_pair + r_base_usd
        metrics["dsr"]      = deflated_sharpe_ratio(r_usd, n_trials, self.rf_fed)
        metrics["n_trades"] = n_trades
        return metrics
