"""
dsig/mapper.py — Layer 4 QAAF Studio 3.0

D-SIG v0.5 — Triple-Réduction : N métriques → 1 score + 1 label + 1 couleur + 1 trend.

Référence : D-SIG-v0.5-arXiv.md (Vision_Complete section 2).
Résout l'Observability Paradox : PAF (3 dirs × métriques) + MIF (4 phases × tests)
+ MÉTIS (4 questions × fenêtres) → ingérable sans couche de distillation.

Profil de pondération par défaut (stratégie QAAF)
------------------------------------------------
cnsr      0.35  — métrique primaire
sortino   0.20  — qualité du downside
calmar    0.15  — rendement/drawdown
drawdown  0.15  — protection capitale
stability 0.15  — robustesse walk-forward

Fail-fast
---------
- PAF verdict STOP              → score ≤ 20 (CRITICAL)
- CNSR < -0.5                   → score ≤ 20
- max_dd > 40 %                 → score ≤ 20
- DSR < 0.80                    → cap à 59 (DEGRADED max — SUSPECT_DSR)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class DSIGSignal:
    score:      int              # 0-100
    label:      str              # EXCELLENT | GOOD | DEGRADED | CRITICAL
    color:      str              # GREEN | YELLOW | ORANGE | RED
    trend:      str              # IMPROVING | STABLE | DEGRADING | CRITICAL_FALL
    source_id:  str
    dimensions: dict             # scores détaillés par dimension
    raw:        dict             # métriques brutes


def score_to_label_color(score: int) -> tuple[str, str]:
    if score >= 85: return "EXCELLENT", "GREEN"
    if score >= 60: return "GOOD",      "YELLOW"
    if score >= 35: return "DEGRADED",  "ORANGE"
    return "CRITICAL", "RED"


def strategy_to_dsig(
    metrics:     dict,
    paf_verdict: str,
    n_trials:    int,
    source_id:   str = "qaaf-studio",
    prev_score:  Optional[int] = None,
) -> DSIGSignal:
    """
    Convertit les métriques QAAF Studio en signal D-SIG v0.5.

    Paramètres
    ----------
    metrics     : dict de compute_full_metrics() — doit contenir cnsr_usd_fed,
                  sortino, calmar, max_dd_pct, dsr, et walk_forward_score (0-1)
    paf_verdict : verdict D1/D2/D3 le plus contraignant de la session
    n_trials    : N_trials courant (pour contexte, non utilisé dans le score)
    prev_score  : score de la session précédente (pour trend)
    """
    cnsr    = metrics.get("cnsr_usd_fed", np.nan)
    sortino = metrics.get("sortino",      np.nan)
    calmar  = metrics.get("calmar",       np.nan)
    max_dd  = metrics.get("max_dd_pct",   100.0)
    wf      = metrics.get("walk_forward_score", 0.0)  # fraction fenêtres OOS OK
    dsr     = metrics.get("dsr",          0.0)

    # Normalisation des dimensions → [0, 1]
    def safe(v, lo, hi):
        if v is None or np.isnan(v):
            return 0.0
        return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))

    dims = {
        "cnsr":      safe(cnsr,    -1.0, 2.0),
        "sortino":   safe(sortino, -1.0, 3.0),
        "calmar":    safe(calmar,   0.0, 2.0),
        "drawdown":  1.0 - safe(max_dd, 0.0, 50.0),
        "stability": float(np.clip(wf, 0.0, 1.0)),
    }

    weights = {"cnsr": 0.35, "sortino": 0.20, "calmar": 0.15,
               "drawdown": 0.15, "stability": 0.15}

    raw_score = sum(dims[k] * weights[k] for k in weights) * 100

    # Fail-fast
    if paf_verdict == "STOP_PASSIF_DOMINE" or (not np.isnan(cnsr) and cnsr < -0.5):
        raw_score = min(raw_score, 20.0)
    elif not np.isnan(max_dd) and max_dd > 40.0:
        raw_score = min(raw_score, 20.0)

    # DSR cap
    if not np.isnan(dsr) and dsr < 0.80:
        raw_score = min(raw_score, 59.0)

    score = round(raw_score)
    label, color = score_to_label_color(score)

    # Trend
    if prev_score is None:
        trend = "STABLE"
    else:
        delta = score - prev_score
        if delta >= 10:
            trend = "IMPROVING"
        elif delta <= -20:
            trend = "CRITICAL_FALL"
        elif delta <= -5:
            trend = "DEGRADING"
        else:
            trend = "STABLE"

    return DSIGSignal(
        score=score,
        label=label,
        color=color,
        trend=trend,
        source_id=source_id,
        dimensions={k: {"score": round(v * 100), "raw": metrics.get(k)} for k, v in dims.items()},
        raw=metrics,
    )
