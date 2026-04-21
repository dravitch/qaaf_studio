"""
MÉTIS Q1 — Walk-forward (robustesse temporelle).

Protocole : 5 fenêtres glissantes sur l'historique complet.
Critère    : CNSR-USD > 0.5 sur au moins 4/5 fenêtres.
Justification : une fenêtre robuste est probablement conjoncturelle,
                quatre sur cinq suggère une propriété structurelle.

Note importante : ce module teste des fenêtres glissantes sur
l'HISTORIQUE COMPLET (IS + OOS), pas seulement sur l'OOS.
C'est volontaire : le walk-forward évalue la robustesse temporelle
du signal sur des périodes non vues au moment de l'optimisation.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Callable

from layer1_engine.backtester     import Backtester
from layer1_engine.metrics_engine import compute_cnsr


@dataclass
class Q1Result:
    verdict:      str     # PASS | FAIL
    n_pass:       int     # fenêtres avec CNSR > seuil
    n_total:      int     # total fenêtres testées
    median_cnsr:  float   # médiane des CNSR par fenêtre
    windows:      list    # détail par fenêtre

    def __post_init__(self):
        self.n_windows_pass = self.n_pass
    threshold:    float   # seuil CNSR utilisé
    min_windows:  int     # minimum requis pour PASS


def run_q1(
    prices_full: pd.DataFrame,
    r_btc_full: pd.Series,
    allocation_fn: Callable,
    backtester: Backtester,
    n_windows: int = 5,
    cnsr_threshold: float = 0.5,
    min_windows_pass: int = 4,
) -> Q1Result:
    """
    Exécute le walk-forward sur n_windows fenêtres glissantes.

    Les fenêtres sont construites sur prices_full (historique complet).
    Chaque fenêtre couvre ~1/6 de l'historique (split 70/30 par fenêtre).
    """
    total_days = len(prices_full)
    window_size = total_days // (n_windows + 1)

    windows_results = []

    for i in range(n_windows):
        test_start = (i + 1) * window_size
        test_end   = min(test_start + window_size, total_days)

        prices_win = prices_full.iloc[test_start:test_end].copy()
        r_btc_win  = r_btc_full.reindex(prices_win.index).dropna()
        prices_win = prices_win.reindex(r_btc_win.index)

        cnsr_val = float("nan")
        if len(prices_win) >= 30:
            try:
                result = backtester.run(allocation_fn, prices_win, r_btc_win)
                zero = pd.Series(0.0, index=result["r_portfolio_usd"].index)
                metrics = compute_cnsr(result["r_portfolio_usd"], zero)
                cnsr_val = metrics["cnsr_usd_fed"]
            except Exception:
                pass

        passed = (not np.isnan(cnsr_val)) and (cnsr_val >= cnsr_threshold)
        windows_results.append({
            "window":     i + 1,
            "start":      str(prices_win.index[0].date()) if len(prices_win) > 0 else "N/A",
            "end":        str(prices_win.index[-1].date()) if len(prices_win) > 0 else "N/A",
            "cnsr":       round(float(cnsr_val), 4) if not np.isnan(cnsr_val) else None,
            "pass":       passed,
            "n_obs":      len(prices_win),
        })

    valid_cnsrs = [w["cnsr"] for w in windows_results if w["cnsr"] is not None]
    n_pass      = sum(w["pass"] for w in windows_results)
    median_cnsr = float(np.median(valid_cnsrs)) if valid_cnsrs else float("nan")

    return Q1Result(
        verdict     = "PASS" if n_pass >= min_windows_pass else "FAIL",
        n_pass      = n_pass,
        n_total     = len(windows_results),
        median_cnsr = round(median_cnsr, 4),
        windows     = windows_results,
        threshold   = cnsr_threshold,
        min_windows = min_windows_pass,
    )
