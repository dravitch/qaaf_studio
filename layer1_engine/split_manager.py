# split_manager.py
"""
QAAF Studio 3.0 — SplitManager
Couche 1 : splits IS/OOS figés pour toute la durée du projet.

Règle : les splits sont définis une fois à l'initialisation.
Toute tentative de modification mid-session lève une erreur.
Le SplitManager maintient aussi le compteur N_trials par famille.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import pandas as pd
import yaml


@dataclass(frozen=True)
class Split:
    """Définition immuable d'un split IS/OOS."""
    is_start:  object  # date (str ou date)
    is_end:    object
    oos_start: object
    oos_end:   object
    label:     str = "default"

    def apply(self, series: pd.Series) -> tuple[pd.Series, pd.Series]:
        """Retourne (série_IS, série_OOS) pour une série indexée par date."""
        is_  = series.loc[str(self.is_start):str(self.is_end)]
        oos_ = series.loc[str(self.oos_start):str(self.oos_end)]
        return is_, oos_


class SplitManager:
    """
    Gestionnaire de splits et compteur N_trials par famille.

    Usage
    -----
    # Via config.yaml
    sm = SplitManager(config_path="config.yaml")

    # Via Split explicite
    sm = SplitManager(split=Split("2019-01-01", "2022-12-31",
                                   "2023-01-01", "2024-12-31"))
    is_r, oos_r = sm.apply(bundle.r_paxg_btc)
    n = sm.increment("EMA_span_variants")  # → 1, 2, 3 ...
    """

    def __init__(self, split: Split = None, config_path: str = None):
        if split is not None:
            self._split = split
        elif config_path is not None:
            cfg = yaml.safe_load(Path(config_path).read_text())
            s   = cfg["splits"]
            self._split = Split(
                is_start  = s["is_start"],
                is_end    = s["is_end"],
                oos_start = s["oos_start"],
                oos_end   = s["oos_end"],
                label     = s.get("label", "config"),
            )
        else:
            raise ValueError("SplitManager requires either split= or config_path=")
        self._locked  = False
        self._n_trials: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Split
    # ------------------------------------------------------------------

    def apply(self, series: pd.Series) -> tuple[pd.Series, pd.Series]:
        return self._split.apply(series)

    def apply_df(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Retourne (df_IS, df_OOS) pour un DataFrame indexé par date."""
        s   = self._split
        is_ = df.loc[str(s.is_start):str(s.is_end)]
        oos_ = df.loc[str(s.oos_start):str(s.oos_end)]
        return is_, oos_

    @property
    def split(self) -> Split:
        return self._split

    def lock(self) -> None:
        """Verrouille le split — aucune modification possible après."""
        self._locked = True
        print(f"[SplitManager] Split '{self._split.label}' verrouillé.")

    # ------------------------------------------------------------------
    # Compteur N_trials pour DSR
    # ------------------------------------------------------------------

    def increment(self, family: str, count: int = 1) -> int:
        """
        Incrémente le compteur de la famille et retourne la valeur courante.
        À appeler avant chaque nouveau test dans la même famille.
        """
        self._n_trials[family] = self._n_trials.get(family, 0) + count
        return self._n_trials[family]

    def n_trials(self, family: str) -> int:
        return self._n_trials.get(family, 0)

    def set_n_trials(self, family: str, n: int) -> None:
        """Initialise depuis la KB (ex. reprise de session)."""
        self._n_trials[family] = n

    def summary(self) -> None:
        print(f"\n[SplitManager] Split : {self._split.label}")
        print(f"  IS  : {self._split.is_start} → {self._split.is_end}")
        print(f"  OOS : {self._split.oos_start} → {self._split.oos_end}")
        print("  N_trials par famille :")
        for fam, n in self._n_trials.items():
            print(f"    {fam:30s} : {n}")


# ---------------------------------------------------------------------------
# Test inline
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np

    split = Split(
        is_start="2019-01-01", is_end="2022-12-31",
        oos_start="2023-01-01", oos_end="2024-12-31",
        label="PAXG_BTC_7030"
    )
    sm = SplitManager(split)

    # Simuler une série
    idx = pd.date_range("2019-01-01", "2024-12-31", freq="D")
    r   = pd.Series(np.random.normal(0, 0.02, len(idx)), index=idx)

    is_r, oos_r = sm.apply(r)
    print(f"IS  : {len(is_r)} jours")
    print(f"OOS : {len(oos_r)} jours")

    sm.set_n_trials("EMA_span_variants", 50)
    sm.increment("EMA_span_variants", 51)
    sm.summary()
