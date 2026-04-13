"""
phase0_isolation.py — MIF Phase 0
Layer 2 QAAF Studio 3.0

Tests d'isolation (T1-T6) sur données synthétiques F1-F5.
Détecte 80 % des bugs algorithmiques AVANT d'exposer la stratégie aux données réelles.

Règle d'arrêt : un seul FAIL en Phase 0 → stopper immédiat, corriger l'algorithme.

Note sur les NaN de warmup : toute stratégie avec fenêtre glissante ou EMA produit
des NaN en début de série. Ce n'est pas un bug. Les checks de stabilité utilisent
dropna() pour ignorer le warmup et ne tester que le comportement post-initialisation.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List

import numpy as np
import pandas as pd
from scipy.stats import kurtosis

from layer1_engine.metrics_engine import compute_cnsr
from layer2_qualification.mif.synthetic_data import generate_synthetic_paxgbtc


@dataclass
class TestResult:
    label:  str
    passed: bool
    cnsr:   float | None
    notes:  str


def _stable(alloc: pd.Series) -> bool:
    """
    Vérifie la stabilité algorithmique d'une allocation.
    Ignore les NaN de warmup (début de série) — attendus pour EMA/rolling.
    Détecte les NaN pathologiques au milieu du signal.
    """
    clean = alloc.dropna()
    if len(clean) < 10:
        return False
    # NaN au milieu du signal post-warmup → bug
    first_valid = alloc.first_valid_index()
    if first_valid is None:
        return False
    after_start = alloc.loc[first_valid:]
    if after_start.isna().any():
        return False
    # L'allocation doit varier (signal non constant)
    return float(clean.std()) > 0


def run_phase0(
    strategy_fn: Callable[[pd.Series, dict], pd.Series],
    params: dict,
    rf_annual: float = 0.04,
) -> List[TestResult]:
    """
    Exécute T1-T6 sur données synthétiques.
    Retourne la liste des résultats — stopper si tout FAIL.
    """
    print("\n⚙️  MIF Phase 0 — Tests d'isolation (T1-T6) ...")
    results = []

    tests = [
        ("T1", "Dérive structurelle F1",      "standard", 101, _t1_directional),
        ("T2", "Kurtosis extrême F2",          "standard", 202, _t2_fat_tails),
        ("T3", "Corrélation oscillante F3",    "standard", 303, _t3_rolling_corr),
        ("T4", "Ratio de volatilité F4",       "standard", 404, _t4_vol_ratio),
        ("T5", "Asymétrie directionnelle F5",  "standard", 505, _t5_asymmetry),
        ("T6", "Régime crash",                 "crash",    606, _t6_crash),
    ]

    for label, name, regime, seed, test_fn in tests:
        r_pair, r_base = generate_synthetic_paxgbtc(T=600, seed=seed, regime=regime)
        try:
            alloc = strategy_fn(r_pair, params)
            passed, cnsr_val, notes = test_fn(alloc, r_pair, r_base, rf_annual)
        except Exception as e:
            passed, cnsr_val, notes = False, None, f"Exception : {e}"

        res   = TestResult(f"{label} {name}", passed, cnsr_val, notes)
        emoji = "✅" if passed else "🔴"
        c_str = f"{cnsr_val:.3f}" if cnsr_val is not None else "—"
        print(f"  {emoji} {label} {name:35s} CNSR={c_str}  {notes}")
        results.append(res)

    return results


# ── Tests T1-T6 ───────────────────────────────────────────────────────────────

def _t1_directional(alloc, r_pair, r_base, rf):
    """T1 : dérive structurelle — signal non constant, CNSR calculable."""
    passed = _stable(alloc)
    if not passed:
        return False, None, "alloc instable ou constante"
    common = r_pair.index.intersection(alloc.dropna().index)
    r_port = alloc.reindex(common).ffill() * r_pair.loc[common]
    cnsr   = compute_cnsr(r_port, r_base.reindex(common), rf)["cnsr_usd_fed"]
    passed = np.isfinite(cnsr)
    return passed, cnsr, f"std_alloc={alloc.std():.4f}"


def _t2_fat_tails(alloc, r_pair, r_base, rf):
    """T2 : queues épaisses F2 — stabilité algorithmique sous kurtosis élevé.
    CNSR peut être négatif (mauvais régime synthétique) — c'est acceptable.
    Ce test vérifie uniquement que l'algorithme ne plante pas.
    """
    kurt   = kurtosis(r_pair, fisher=False)
    passed = _stable(alloc)
    if not passed:
        return False, None, f"alloc instable, kurtosis={kurt:.1f}"
    common = r_pair.index.intersection(alloc.dropna().index)
    r_port = alloc.reindex(common).ffill() * r_pair.loc[common]
    cnsr   = compute_cnsr(r_port, r_base.reindex(common), rf)["cnsr_usd_fed"]
    passed = np.isfinite(cnsr)
    return passed, cnsr, f"kurtosis={kurt:.1f}"


def _t3_rolling_corr(alloc, r_pair, r_base, rf):
    """T3 : corrélation oscillante F3 — allocation non figée."""
    std    = float(alloc.dropna().std())
    passed = _stable(alloc) and std > 0.01
    if not passed:
        return False, None, f"std_alloc={std:.4f} trop faible"
    common = r_pair.index.intersection(alloc.dropna().index)
    r_port = alloc.reindex(common).ffill() * r_pair.loc[common]
    cnsr   = compute_cnsr(r_port, r_base.reindex(common), rf)["cnsr_usd_fed"]
    passed = np.isfinite(cnsr)
    return passed, cnsr, f"std_alloc={std:.4f}"


def _t4_vol_ratio(alloc, r_pair, r_base, rf):
    """T4 : ratio de volatilité stable F4 — métriques cohérentes."""
    vol_r  = r_base.std() / ((r_pair + r_base).std() + 1e-10)
    passed = _stable(alloc)
    if not passed:
        return False, None, "alloc instable"
    common = r_pair.index.intersection(alloc.dropna().index)
    r_port = alloc.reindex(common).ffill() * r_pair.loc[common]
    cnsr   = compute_cnsr(r_port, r_base.reindex(common), rf)["cnsr_usd_fed"]
    passed = np.isfinite(cnsr)
    return passed, cnsr, f"vol_ratio={vol_r:.2f}"


def _t5_asymmetry(alloc, r_pair, r_base, rf):
    """T5 : asymétrie directionnelle F5 — signal réagit différemment selon la position."""
    clean = alloc.dropna()
    n     = len(clean) // 2
    d     = abs(float(clean.iloc[:n].mean()) - float(clean.iloc[n:].mean()))
    passed = _stable(alloc) and d > 0.001
    if not passed:
        return False, None, f"half_delta={d:.4f} trop faible"
    common = r_pair.index.intersection(clean.index)
    r_port = alloc.reindex(common).ffill() * r_pair.loc[common]
    cnsr   = compute_cnsr(r_port, r_base.reindex(common), rf)["cnsr_usd_fed"]
    passed = np.isfinite(cnsr)
    return passed, cnsr, f"half_delta={d:.4f}"


def _t6_crash(alloc, r_pair, r_base, rf):
    """T6 : régime crash — stabilité algorithmique sous volatilité extrême.
    CNSR très négatif est normal en crash. Test de non-plantage uniquement.
    """
    passed = _stable(alloc)
    if not passed:
        return False, None, "alloc instable en régime crash"
    common = r_pair.index.intersection(alloc.dropna().index)
    r_port = alloc.reindex(common).ffill() * r_pair.loc[common]
    cnsr   = compute_cnsr(r_port, r_base.reindex(common), rf)["cnsr_usd_fed"]
    passed = np.isfinite(cnsr)
    return passed, cnsr, "crash_stability"
