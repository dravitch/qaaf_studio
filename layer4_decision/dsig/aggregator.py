"""
aggregator.py — D-SIG Layer 4 QAAF Studio 3.0

Agrège plusieurs DSIGSignals en un signal global Studio.
Règle : le signal le plus contraignant (score le plus bas) drive le verdict global.

Usage
-----
    from layer4_decision.dsig.aggregator import aggregate
    from layer4_decision.dsig.mapper     import strategy_to_dsig

    sig_mif    = strategy_to_dsig(metrics_mif,   paf_mif,   n_trials, "mif")
    sig_metis  = strategy_to_dsig(metrics_metis, paf_metis, n_trials, "metis")
    sig_global = aggregate([sig_mif, sig_metis])
"""

from __future__ import annotations
from typing import List

from layer4_decision.dsig.mapper import DSIGSignal, score_to_label_color


def aggregate(signals: List[DSIGSignal]) -> DSIGSignal:
    """
    Agrège une liste de DSIGSignals en un signal global Studio.

    Règle principale : le signal au score le plus bas (le plus contraignant)
    drive le verdict global. Cette règle assure que toute alerte remonte.

    Paramètres
    ----------
    signals : liste de DSIGSignal (au moins 1)

    Retourne
    --------
    DSIGSignal global avec :
    - score = min des scores individuels (principe de précaution)
    - label/color recalculés sur ce score
    - trend = trend du signal le plus contraignant
    - source_id = "aggregated"
    - dimensions = union des dimensions (dernière valeur gagne si conflit)
    - raw = résumé des sources et scores individuels
    """
    if not signals:
        raise ValueError("aggregate() requires at least one signal")

    if len(signals) == 1:
        return signals[0]

    # Le signal le plus contraignant = score minimum
    most_constraining = min(signals, key=lambda s: s.score)
    min_score         = most_constraining.score

    label, color = score_to_label_color(min_score)

    # Dimensions : union de toutes les dimensions individuelles
    merged_dims: dict = {}
    for sig in signals:
        merged_dims.update(sig.dimensions)

    # Raw : résumé des contributions par source
    raw_summary = {
        "aggregation_rule": "min_score",
        "n_signals":        len(signals),
        "scores":           {s.source_id: s.score for s in signals},
        "driver":           most_constraining.source_id,
    }

    return DSIGSignal(
        score      = min_score,
        label      = label,
        color      = color,
        trend      = most_constraining.trend,
        source_id  = "aggregated",
        dimensions = merged_dims,
        raw        = raw_summary,
    )
