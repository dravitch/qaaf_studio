from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class DSIGSignal:
    score: int
    label: str
    color: str
    trend: str
    dimensions: dict
    source_id: str
    dsr: Optional[float]


LABEL_MAP = {
    "EXCELLENT": "GREEN",
    "GOOD":      "YELLOW",
    "DEGRADED":  "ORANGE",
    "CRITICAL":  "RED",
}


def score_to_label(score: int) -> str:
    if score >= 85:
        return "EXCELLENT"
    if score >= 60:
        return "GOOD"
    if score >= 35:
        return "DEGRADED"
    return "CRITICAL"


class DSIGMapper:
    WEIGHTS = {
        "cnsr":      0.35,
        "sortino":   0.20,
        "calmar":    0.15,
        "drawdown":  0.15,
        "stability": 0.15,
    }

    def map(
        self,
        cnsr_usd_fed: float,
        sortino: float,
        calmar: float,
        max_dd_pct: float,
        walk_forward_score: float,
        paf_d1_verdict: str,
        dsr: Optional[float] = None,
        source_id: str = "",
        previous_score: Optional[int] = None,
    ) -> DSIGSignal:

        raw_dims = {
            "cnsr":      float(np.clip((cnsr_usd_fed + 1) / 3, 0, 1)),
            "sortino":   float(np.clip((sortino + 1) / 3, 0, 1)),
            "calmar":    float(np.clip(calmar / 2, 0, 1)) if calmar > 0 else 0.0,
            "drawdown":  float(1 - np.clip(max_dd_pct / 50, 0, 1)),
            "stability": float(np.clip(walk_forward_score, 0, 1)),
        }

        raw = sum(raw_dims[k] * self.WEIGHTS[k] for k in self.WEIGHTS) * 100

        # Plafonnements (préconditions D-SIG v0.5)
        if paf_d1_verdict == "STOP" or cnsr_usd_fed < -0.5:
            raw = min(raw, 20)
        elif max_dd_pct > 40:
            raw = min(raw, 20)
        elif dsr is not None and dsr < 0.80:
            raw = min(raw, 59)

        score = round(raw)
        label = score_to_label(score)

        trend = "N_A"
        if previous_score is not None:
            delta = score - previous_score
            if delta >= 10:
                trend = "IMPROVING"
            elif delta >= -5:
                trend = "STABLE"
            elif delta >= -15:
                trend = "DEGRADING"
            else:
                trend = "CRITICAL_FALL"

        dims = {
            k: {"score": round(raw_dims[k] * 100), "weight": self.WEIGHTS[k]}
            for k in self.WEIGHTS
        }

        return DSIGSignal(
            score=score,
            label=label,
            color=LABEL_MAP[label],
            trend=trend,
            dimensions=dims,
            source_id=source_id or "qaaf-studio::unknown",
            dsr=dsr,
        )


def strategy_to_dsig(
    metrics: dict,
    paf_verdict: str,
    n_trials: int = 1,
    source_id: str = "",
    previous_score: Optional[int] = None,
) -> DSIGSignal:
    """Wrapper : metrics dict + PAF verdict → DSIGSignal."""
    cnsr = float(metrics.get("cnsr_usd_fed", 0.0))
    sortino = float(metrics.get("sortino", 0.0))
    max_dd_pct = float(metrics.get("max_dd_pct", 0.0))
    walk_forward_score = float(metrics.get("walk_forward_score", 0.5))
    dsr = metrics.get("dsr")
    if dsr is not None:
        dsr = float(dsr)
    calmar_default = cnsr / (max_dd_pct / 100) if max_dd_pct > 0 else 0.0
    calmar = float(metrics.get("calmar", calmar_default))

    paf_d1 = "STOP" if paf_verdict in ("STOP", "B_PASSIF_DOMINE", "PAF_STOP") else "N_A"

    return DSIGMapper().map(
        cnsr_usd_fed=cnsr,
        sortino=sortino,
        calmar=calmar,
        max_dd_pct=max_dd_pct,
        walk_forward_score=walk_forward_score,
        paf_d1_verdict=paf_d1,
        dsr=dsr,
        source_id=source_id,
        previous_score=previous_score,
    )
