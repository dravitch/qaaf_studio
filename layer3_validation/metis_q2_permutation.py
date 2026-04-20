"""
MÉTIS Q2 — Test de permutation (significativité statistique).

Protocole : permuter les allocations OOS 10 000 fois, calculer le CNSR
            sur chaque permutation, calculer la p-value.
Critère   : p-value < 0.05.
Justification : le bull run 2023-2024 gonfle tous les Sharpe.
                La permutation isole ce qui vient de la règle, pas du marché.

Checkpoint : sauvegarde tous les 500 itérations (pour reprendre si interruption).
"""

import numpy as np
import pandas as pd
import shutil
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from layer1_engine.backtester     import Backtester
from layer1_engine.metrics_engine import compute_cnsr


@dataclass
class Q2Result:
    verdict:      str    # PASS | FAIL
    pvalue:       float
    cnsr_obs:     float  # CNSR observé du signal
    cnsr_bench:   float  # CNSR B_5050 (référence)
    perm_mean:    float  # CNSR moyen des permutations
    n_perm:       int    # nombre de permutations effectuées
    pvalue_threshold: float


def _atomic_save(path: Path, data: dict):
    """Sauvegarde atomique POSIX (pattern LP)."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        yaml.safe_dump(data, f)
    shutil.move(str(tmp), str(path))


def run_q2(
    prices_oos: pd.DataFrame,
    r_btc_oos: pd.Series,
    allocation_fn: Callable,
    backtester: Backtester,
    n_perm: int = 10000,
    pvalue_threshold: float = 0.05,
    checkpoint_path: Optional[Path] = None,
    checkpoint_interval: int = 500,
    seed: int = 42,
) -> Q2Result:
    """
    Exécute le test de permutation avec reprise depuis checkpoint.

    Pour n_perm=500 (mode --fast), durée ~30s.
    Pour n_perm=10000 (mode complet), durée ~10min selon machine.
    """
    np.random.seed(seed)

    # CNSR observé
    result_obs = backtester.run(allocation_fn, prices_oos, r_btc_oos)
    zero_obs   = pd.Series(0.0, index=result_obs["r_portfolio_usd"].index)
    cnsr_obs   = compute_cnsr(result_obs["r_portfolio_usd"], zero_obs)["cnsr_usd_fed"]

    # CNSR B_5050 (référence)
    from layer1_engine.benchmark_factory import BenchmarkFactory
    factory    = BenchmarkFactory(backtester)
    cnsr_bench = factory.b_5050(prices_oos, r_btc_oos)["cnsr_usd_fed"]

    # Séries pour la permutation
    r_pair_arr  = result_obs["r_portfolio_usd"].values
    r_base_arr  = pd.Series(0.0, index=result_obs["r_portfolio_usd"].index)

    # Reprendre depuis checkpoint si disponible
    start_idx   = 0
    perm_cnsrs  = []
    if checkpoint_path and checkpoint_path.exists():
        try:
            cp = yaml.safe_load(checkpoint_path.read_text())
            start_idx  = cp.get("completed", 0)
            perm_cnsrs = cp.get("perm_cnsrs", [])
        except Exception:
            pass

    # Boucle de permutation
    for i in range(start_idx, n_perm):
        perm = np.random.permutation(r_pair_arr)
        r_perm = pd.Series(perm, index=result_obs["r_portfolio_usd"].index)
        try:
            m = compute_cnsr(r_perm, r_base_arr)
            c = m["cnsr_usd_fed"]
            if not np.isnan(c):
                perm_cnsrs.append(float(c))
        except Exception:
            pass

        if checkpoint_path and (i + 1) % checkpoint_interval == 0:
            _atomic_save(checkpoint_path, {
                "completed":  i + 1,
                "total":      n_perm,
                "perm_cnsrs": perm_cnsrs,
            })
            import pandas as _pd
            _pd.Series(perm_cnsrs).to_csv(
                checkpoint_path.with_suffix(".csv"), index=False
            )

    pvalue    = float(np.mean(np.array(perm_cnsrs) >= cnsr_obs)) if perm_cnsrs else 1.0
    perm_mean = float(np.mean(perm_cnsrs)) if perm_cnsrs else float("nan")

    return Q2Result(
        verdict          = "PASS" if pvalue < pvalue_threshold else "FAIL",
        pvalue           = round(pvalue, 4),
        cnsr_obs         = round(float(cnsr_obs), 4),
        cnsr_bench       = round(float(cnsr_bench), 4),
        perm_mean        = round(perm_mean, 4) if not np.isnan(perm_mean) else None,
        n_perm           = len(perm_cnsrs),
        pvalue_threshold = pvalue_threshold,
    )
