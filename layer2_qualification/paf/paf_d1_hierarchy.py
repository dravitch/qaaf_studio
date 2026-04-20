"""
PAF Direction 1 — Test critique de hiérarchie de signal.

Question : chaque couche de sophistication apporte-t-elle quelque chose ?
Protocole : comparer MR_pur, signal_ref, signal_candidat, et benchmarks passifs
            sur le même OOS, même moteur, même split.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Callable, Optional

from layer1_engine.backtester        import Backtester
from layer1_engine.benchmark_factory import BenchmarkFactory
from layer1_engine.metrics_engine    import compute_cnsr


@dataclass
class D1Result:
    verdict: str           # HIERARCHIE_CONFIRMEE | B_PASSIF_DOMINE | STOP | PARTIELLE
    mr_pur_cnsr:   float
    signal_ref_cnsr: float
    signal_candidat_cnsr: Optional[float]
    b_5050_cnsr:   float
    b_btc_cnsr:    float
    delta_ref_vs_mr: float         # signal_ref - mr_pur
    delta_candidat_vs_ref: float   # signal_candidat - signal_ref
    details: dict


def run_d1(
    prices_oos: pd.DataFrame,
    r_btc_oos: pd.Series,
    signal_ref_fn: Callable,       # H9 ou équivalent
    backtester: Backtester,
    signal_candidat_fn: Optional[Callable] = None,
    window: int = 60,
) -> D1Result:
    """
    Exécute PAF Direction 1.

    Paramètres
    ----------
    prices_oos        : DataFrame OOS avec colonnes 'paxg' et 'btc'
    r_btc_oos         : log-rendements BTC/USD sur OOS
    signal_ref_fn     : fonction d'allocation de référence (H9 typiquement)
    backtester        : moteur unifié Layer 1
    signal_candidat_fn: fonction d'allocation du signal candidat (optionnel)
    window            : fenêtre pour MR_pur et signal_ref
    """
    factory = BenchmarkFactory(backtester)

    # ── MR_pur : mean-reversion minimal ──────────────────────────────────────
    # Allocation proportionnelle à la distance du ratio à sa moyenne rolling
    log_ratio = np.log(prices_oos["paxg"] / prices_oos["btc"])
    mean_lr   = log_ratio.rolling(window, min_periods=window // 2).mean()
    std_lr    = log_ratio.rolling(window, min_periods=window // 2).std()
    z_score   = ((log_ratio - mean_lr) / std_lr.replace(0, np.nan)).fillna(0)
    # Quand z < 0 (PAXG bas vs BTC), acheter PAXG → allocation haute
    mr_alloc  = (0.5 - z_score.clip(-1, 1) * 0.3).clip(0, 1)

    def mr_pur_fn(df): return mr_alloc.reindex(df.index).fillna(0.5)

    # ── Backtests ─────────────────────────────────────────────────────────────
    def _cnsr(alloc_fn) -> float:
        result = backtester.run(alloc_fn, prices_oos, r_btc_oos)
        zero = pd.Series(0.0, index=result["r_portfolio_usd"].index)
        return compute_cnsr(result["r_portfolio_usd"], zero)["cnsr_usd_fed"]

    mr_cnsr  = _cnsr(mr_pur_fn)
    ref_cnsr = _cnsr(signal_ref_fn)
    cand_cnsr = _cnsr(signal_candidat_fn) if signal_candidat_fn else None

    b5050 = factory.b_5050(prices_oos, r_btc_oos)["cnsr_usd_fed"]
    b_btc = factory.b_btc(prices_oos,  r_btc_oos)["cnsr_usd_fed"]

    # ── Verdict ───────────────────────────────────────────────────────────────
    delta_ref_mr   = ref_cnsr - mr_cnsr
    delta_cand_ref = (cand_cnsr - ref_cnsr) if cand_cnsr is not None else None

    # Règle d'arrêt : B_passif domine tout
    top_active = max(filter(lambda x: x is not None, [mr_cnsr, ref_cnsr, cand_cnsr]))
    if b5050 > top_active + 0.1:
        verdict = "B_PASSIF_DOMINE"
    elif delta_ref_mr < -0.05:
        verdict = "STOP"  # Signal_ref pire que MR_pur
    elif cand_cnsr is not None and delta_cand_ref is not None:
        if delta_ref_mr >= -0.05 and delta_cand_ref >= -0.05:
            verdict = "HIERARCHIE_CONFIRMEE"
        elif delta_ref_mr >= -0.05:
            verdict = "PARTIELLE"  # H9 OK mais candidat n'améliore pas
        else:
            verdict = "STOP"
    else:
        # Pas de signal candidat — tester juste MR vs ref
        verdict = "HIERARCHIE_CONFIRMEE" if delta_ref_mr >= 0 else "STOP"

    return D1Result(
        verdict=verdict,
        mr_pur_cnsr=round(mr_cnsr, 4),
        signal_ref_cnsr=round(ref_cnsr, 4),
        signal_candidat_cnsr=round(cand_cnsr, 4) if cand_cnsr else None,
        b_5050_cnsr=round(b5050, 4),
        b_btc_cnsr=round(b_btc, 4),
        delta_ref_vs_mr=round(delta_ref_mr, 4),
        delta_candidat_vs_ref=round(delta_cand_ref, 4) if delta_cand_ref else None,
        details={
            "mr_pur":   mr_cnsr,
            "signal_ref": ref_cnsr,
            "signal_candidat": cand_cnsr,
            "b_5050":   b5050,
            "b_btc":    b_btc,
        }
    )
