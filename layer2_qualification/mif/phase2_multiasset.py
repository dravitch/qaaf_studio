"""
phase2_multiasset.py — MIF Phase 2
Layer 2 QAAF Studio 3.0

Transfert multi-actifs (M1-M4) sur 4 paires synthétiques.
Gate : 3/4 paires doivent passer (75 %).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List

import numpy as np
import pandas as pd

from layer1_engine.metrics_engine import compute_cnsr

GATE_RATIO = 0.75


@dataclass
class TransferResult:
    label:      str
    pair_name:  str
    passed:     bool
    cnsr:       float | None
    cnsr_bench: float | None
    notes:      str


def run_phase2(
    strategy_fn: Callable[[pd.Series, dict], pd.Series],
    params: dict,
    rf_annual: float = 0.04,
) -> List[TransferResult]:
    """
    Exécute M1-M4 sur 4 paires synthétiques avec caractéristiques différentes.
    Gate : 3/4 ≥ 75 % doivent passer pour continuer vers Phase 3.
    """
    print("\n⚙️  MIF Phase 2 — Transfert multi-actifs (M1-M4) ...")

    # 4 paires : (label, nom, sigma_base, sigma_paire, drift, seed)
    pairs = [
        ("M1", "PAXG/BTC_replica", 0.038, 0.010,  0.00015, 111),
        ("M2", "ETH/BTC_like",     0.045, 0.040,  0.00005, 222),
        ("M3", "SOL/BTC_like",     0.060, 0.050, -0.00010, 333),
        ("M4", "BNB/BTC_like",     0.040, 0.035,  0.00000, 444),
    ]

    results = []
    for label, name, sig_b, sig_p, drift, seed in pairs:
        rng = np.random.default_rng(seed)
        T   = 600
        idx = pd.date_range("2020-01-01", periods=T, freq="D")

        r_base_raw = rng.normal(0.0003, sig_b, T)
        r_pair_raw = rng.normal(drift,  sig_p, T) + 0.3 * r_base_raw
        r_pair     = pd.Series(r_pair_raw, index=idx)
        r_base     = pd.Series(r_base_raw, index=idx)

        try:
            alloc    = strategy_fn(r_pair, params)
            common   = r_pair.index.intersection(alloc.index)
            r_port   = alloc.reindex(common).ffill() * r_pair.loc[common]
            r_base_c = r_base.reindex(common)
            cnsr_s   = compute_cnsr(r_port, r_base_c, rf_annual)["cnsr_usd_fed"]
            cnsr_b   = compute_cnsr(0.5 * r_pair.loc[common], r_base_c,
                                    rf_annual)["cnsr_usd_fed"]
            passed   = np.isfinite(cnsr_s)
            notes    = f"bench={cnsr_b:.3f} | delta={cnsr_s-cnsr_b:+.3f}"
        except Exception as e:
            cnsr_s, cnsr_b, passed, notes = None, None, False, f"Exception: {e}"

        res   = TransferResult(label, name, passed, cnsr_s, cnsr_b, notes)
        emoji = "✅" if passed else "🔴"
        c_str = f"{cnsr_s:.3f}" if cnsr_s is not None else "—"
        print(f"  {emoji} {label} {name:20s} CNSR={c_str}  {notes}")
        results.append(res)

    n_pass = sum(r.passed for r in results)
    gate_ok = n_pass / len(results) >= GATE_RATIO
    emoji   = "✅" if gate_ok else "🔴"
    print(f"  {emoji} Gate 75% : {n_pass}/{len(results)} paires — "
          f"{'PASS' if gate_ok else 'FAIL'}")

    return results
