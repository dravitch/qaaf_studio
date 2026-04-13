# metrics_engine.py
"""  
QAAF Studio 3.0 — MetricsEngine
Couche 1 : métriques de référence, CNSR-USD natif, DSR.

Implémentations de référence conformes à QAAF_Studio_3_0_Architecture.md v1.1
"""

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, norm


def compute_cnsr(
    r_pair: pd.Series,
    r_base_usd: pd.Series,
    rf_annual: float = 0.04
) -> dict:
    """
    Common-Numeraire Sharpe Ratio (CNSR-USD).

    Fondement : r_USD = r_pair + r_base_USD (identité exacte en log-rendements,
    Karnosky & Singer 1994). Convertit les rendements d'une paire croisée
    en rendements USD avant de calculer le Sharpe.

    Paramètres
    ----------
    r_pair     : log-rendements quotidiens de la paire (ex. PAXG/BTC)
    r_base_usd : log-rendements quotidiens de l'actif de base en USD (ex. BTC/USD)
    rf_annual  : taux sans risque annuel (défaut : 4 % Fed Funds)

    Retourne
    --------
    dict avec cnsr_fed, cnsr_usdc, cnsr_0, sortino, calmar, omega, max_dd_pct
    """
    r_usd = r_pair + r_base_usd  # Identité log-rendements, sans approximation
    rf_daily_fed  = (1 + rf_annual) ** (1 / 252) - 1
    rf_daily_usdc = (1 + 0.03)    ** (1 / 252) - 1  # 3 % lending USDC (moy. 2022-2026)

    def _sharpe(r: pd.Series, rf: float) -> float:
        excess = r - rf
        std = excess.std(ddof=1)
        return float(excess.mean() / std * np.sqrt(252)) if std > 0 else np.nan

    def _sortino(r: pd.Series, rf: float) -> float:
        excess = r - rf
        downside = excess[excess < 0].std(ddof=1)
        return float(excess.mean() / downside * np.sqrt(252)) if downside > 0 else np.nan

    def _calmar(r: pd.Series) -> float:
        cum = (1 + r).cumprod()
        max_dd = abs(((cum - cum.cummax()) / cum.cummax()).min())
        ann_ret = (cum.iloc[-1]) ** (252 / len(r)) - 1
        return float(ann_ret / max_dd) if max_dd > 0 else np.nan

    def _omega(r: pd.Series, threshold: float = 0.0) -> float:
        gains  = r[r > threshold] - threshold
        losses = threshold - r[r <= threshold]
        return float(gains.sum() / losses.sum()) if losses.sum() > 0 else np.nan

    def _max_dd(r: pd.Series) -> float:
        cum = (1 + r).cumprod()
        return float(abs(((cum - cum.cummax()) / cum.cummax()).min()) * 100)

    return {
        "cnsr_usd_fed":  _sharpe(r_usd, rf_daily_fed),
        "cnsr_usd_usdc": _sharpe(r_usd, rf_daily_usdc),
        "cnsr_usd_0":    _sharpe(r_usd, 0.0),
        "sortino":       _sortino(r_usd, rf_daily_fed),
        "calmar":        _calmar(r_usd),
        "omega":         _omega(r_usd),
        "max_dd_pct":    _max_dd(r_usd),
        "r_usd":         r_usd,   # exposé pour le DSR et les tests
    }


def deflated_sharpe_ratio(
    r_usd: pd.Series,
    N_trials: int,
    rf_annual: float = 0.04
) -> float:
    """
    Deflated Sharpe Ratio (DSR) — Bailey & López de Prado (2014).

    Probabilité que le CNSR observé ne soit pas dû au hasard,
    après correction pour le multiple testing, la non-normalité
    et la longueur de l'échantillon.

    Paramètres
    ----------
    r_usd     : log-rendements USD du portefeuille
    N_trials  : nombre de variantes testées dans la même famille
    rf_annual : taux sans risque annuel

    Retourne
    --------
    float : probabilité DSR ∈ [0, 1]. Seuil d'acceptation : 0.95.
    """
    T = len(r_usd)
    rf_daily = (1 + rf_annual) ** (1 / 252) - 1
    excess = r_usd - rf_daily

    sr_daily = float(excess.mean() / excess.std(ddof=1))
    skew_    = float(skew(r_usd))
    kurt_    = float(kurtosis(r_usd, fisher=False))  # Kurtosis de Pearson

    # Seuil déflaté : niveau attendu si on avait tiré N stratégies au hasard
    sr0 = float(norm.ppf(1 - 1 / (N_trials + 1)) / np.sqrt(T))

    # Dénominateur corrigeant pour la non-normalité (Bailey & LdP, eq. 4)
    denom = np.sqrt(max(1e-9, 1 - skew_ * sr_daily + (kurt_ - 3) / 4 * sr_daily ** 2))

    dsr = float(norm.cdf((sr_daily - sr0) * np.sqrt(T - 1) / denom))
    return dsr


def compute_full_metrics(
    r_pair: pd.Series,
    r_base_usd: pd.Series,
    N_trials: int,
    rf_annual: float = 0.04,
    n_trades: int = 0,
    fees_usd: float = 0.0,
    std_alloc: float = np.nan,
) -> dict:
    """
    Calcule l'ensemble complet des métriques QAAF Studio pour une stratégie.
    Point d'entrée principal du MetricsEngine.

    Paramètres supplémentaires
    --------------------------
    N_trials  : compteur cumulatif pour DSR (alimenté par SplitManager)
    n_trades  : nombre de trades OOS
    fees_usd  : frais totaux OOS en USD
    std_alloc : écart-type de l'allocation (mesure la turbulence du signal)
    """
    cnsr = compute_cnsr(r_pair, r_base_usd, rf_annual)
    dsr  = deflated_sharpe_ratio(cnsr["r_usd"], N_trials, rf_annual)

    return {
        **{k: v for k, v in cnsr.items() if k != "r_usd"},
        "dsr":       dsr,
        "dsr_pass":  dsr >= 0.95,
        "N_trials":  N_trials,
        "n_trades":  n_trades,
        "fees_usd":  fees_usd,
        "std_alloc": std_alloc,
    }


# ---------------------------------------------------------------------------
# Tests unitaires inline — exécuter avec : python metrics_engine.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    rng = np.random.default_rng(42)
    T   = 504  # ~2 ans de données quotidiennes

    # Série synthétique : légère dérive positive, kurtosis élevé (réaliste PAXG/BTC)
    r_pair_raw   = rng.normal(0.0005, 0.02, T) + rng.standard_t(df=4, size=T) * 0.005
    r_base_raw   = rng.normal(0.0008, 0.03, T) + rng.standard_t(df=4, size=T) * 0.008
    r_pair_s     = pd.Series(r_pair_raw)
    r_base_s     = pd.Series(r_base_raw)

    print("=== Test compute_cnsr ===")
    cnsr = compute_cnsr(r_pair_s, r_base_s)
    for k, v in cnsr.items():
        if k != "r_usd":
            print(f"  {k:20s} : {v:.4f}")

    print("\n=== Test deflated_sharpe_ratio ===")
    dsr = deflated_sharpe_ratio(cnsr["r_usd"], N_trials=101)
    print(f"  DSR (N=101)          : {dsr:.4f}")
    print(f"  Seuil pass (>=0.95)  : {'PASS' if dsr >= 0.95 else 'SUSPECT_DSR'}")

    print("\n=== Test compute_full_metrics ===")
    full = compute_full_metrics(
        r_pair_s, r_base_s, N_trials=101,
        n_trades=47, fees_usd=34.0, std_alloc=0.170
    )
    for k, v in full.items():
        print(f"  {k:20s} : {v}")

    print("\n✅ Tous les tests terminés.")
    sys.exit(0)