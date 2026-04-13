"""
n_trials_tracker.py — Layer 4 QAAF Studio 3.0

Compteur N_trials par famille de stratégies.
Alimente le calcul DSR en MÉTIS Q4.

Note architecture : le compteur est maintenu ici (Layer 4 — Décision)
et transmis au SplitManager (Layer 1) en début de session.
Cette séparation garantit que Layer 1 reste aveugle à la sémantique KB.
"""

from __future__ import annotations
from datetime import datetime
from pathlib import Path

import yaml


class NTrialsTracker:
    """
    Tracker persistant du compteur N_trials par famille.
    Sauvegarde automatique dans le YAML KB après chaque incrément.

    Usage
    -----
    tracker = NTrialsTracker("layer4_decision/kb_h9_ema60j.yaml")
    n = tracker.get("EMA_span_variants")       # 101
    n = tracker.increment("EMA_span_variants") # 102
    tracker.set("EMA_span_variants", 101)      # reprise explicite
    """

    def __init__(self, kb_path: str):
        self._path = Path(kb_path)

    def get(self, family: str) -> int:
        """Retourne le N_trials actuel pour une famille."""
        kb = self._load()
        hyp = kb.get("hypothese", {})
        if hyp.get("famille") == family:
            return int(hyp.get("N_trials_famille", 0))
        return 0

    def set(self, family: str, n: int) -> None:
        """Définit N_trials pour une famille dans la KB."""
        kb  = self._load()
        hyp = kb.setdefault("hypothese", {})
        if hyp.get("famille") == family:
            hyp["N_trials_famille"] = n
            self._save(kb)

    def increment(self, family: str, count: int = 1) -> int:
        """
        Incrémente N_trials et sauvegarde dans la KB.
        Retourne la nouvelle valeur.
        """
        current = self.get(family)
        new_n   = current + count
        self.set(family, new_n)
        return new_n

    def _load(self) -> dict:
        if not self._path.exists():
            return {}
        with open(self._path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _save(self, data: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False,
                      default_flow_style=False)
