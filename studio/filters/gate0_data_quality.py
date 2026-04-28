"""
studio/filters/gate0_data_quality.py
Gate 0 — Qualité des données (précondition du pipeline).

Vérifie que les trois séries de prix du SignalData sont exploitables
avant toute computation statistique.

Checks (cohérents avec _dqf_stub de layer1_engine/data_loader.py) :
  - Index monotone croissant (pas de données désordonnées)
  - Pas de timestamps dupliqués
  - Ratio NaN < 5% par série (seuil bloquant C2_FAIL)
  - Minimum 30 points de données par période IS et OOS
  - Cohérence des dates : is_start < is_end ≤ oos_start < oos_end

passed=False = échec de qualité des données (récupérable avec de meilleures données).
FilterError  = échec technique inattendu (prix None, index non-datetime, etc.).
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from studio.interfaces import Filter, FilterConfig, FilterVerdict, FilterError, SignalData

_NAN_THRESHOLD = 0.05   # C2_FAIL au-dessus de 5%
_MIN_POINTS    = 30     # minimum absolu par période


class Gate0DataQuality(Filter):
    """
    Gate 0 — Qualité des données.

    Complexity Budget :
      - evaluate() ≤ 50 lignes
      - ≤ 3 niveaux d'imbrication
      - ≤ 5 paramètres dans FilterConfig.params

    Paramètres dans config.params :
      - nan_threshold : float — seuil NaN bloquant (défaut : 0.05)
      - min_points    : int   — minimum de points par période (défaut : 30)
    """

    NAME = "gate0_data_quality"

    def evaluate(
        self,
        signal: SignalData,
        config: FilterConfig,
    ) -> FilterVerdict:
        nan_threshold = config.get("nan_threshold", _NAN_THRESHOLD)
        min_points    = config.get("min_points",    _MIN_POINTS)

        try:
            issues = self._check_all(signal, nan_threshold, min_points)
        except Exception as e:
            raise FilterError(
                filter_name=self.NAME,
                cause=f"Erreur inattendue lors des checks de qualité : {e}",
                action=(
                    "Vérifier que les séries de prix du SignalData ont un "
                    "DatetimeIndex et ne sont pas None."
                ),
            )

        passed = len(issues) == 0
        n_issues = len(issues)

        if passed:
            diagnosis = (
                f"Gate 0 validée : toutes les séries de prix sont propres "
                f"(NaN < {nan_threshold:.0%}, index monotone, ≥ {min_points} points)."
            )
        else:
            diagnosis = (
                f"Gate 0 échoue : {n_issues} problème(s) détecté(s). "
                + " | ".join(issues[:3])
                + (" | ..." if n_issues > 3 else "")
                + " Corriger les données sources avant de relancer le pipeline."
            )

        return FilterVerdict(
            passed=passed,
            filter_name=self.NAME,
            metrics={
                "n_issues":     n_issues,
                "issues":       issues,
                "nan_threshold": nan_threshold,
                "min_points":   min_points,
            },
            diagnosis=diagnosis,
        )

    def _check_all(
        self,
        signal: SignalData,
        nan_threshold: float,
        min_points: int,
    ) -> list[str]:
        issues: list[str] = []

        # Cohérence des dates IS/OOS
        is_start  = pd.Timestamp(signal.is_start)
        is_end    = pd.Timestamp(signal.is_end)
        oos_start = pd.Timestamp(signal.oos_start)
        oos_end   = pd.Timestamp(signal.oos_end)

        if not (is_start < is_end):
            issues.append(f"DATE_FAIL: is_start ({is_start.date()}) ≥ is_end ({is_end.date()})")
        if not (oos_start < oos_end):
            issues.append(f"DATE_FAIL: oos_start ({oos_start.date()}) ≥ oos_end ({oos_end.date()})")
        if is_end > oos_start:
            issues.append(f"DATE_FAIL: is_end ({is_end.date()}) > oos_start ({oos_start.date()}) — chevauchement IS/OOS")

        # Checks par série
        series_map = {
            "prices_pair":      signal.prices_pair,
            "prices_base_usd":  signal.prices_base_usd,
            "prices_quote_usd": signal.prices_quote_usd,
        }
        for name, s in series_map.items():
            issues.extend(self._check_series(s, name, nan_threshold))

        # Minimum de points IS et OOS dans prices_pair
        is_mask  = (signal.prices_pair.index >= is_start)  & (signal.prices_pair.index <= is_end)
        oos_mask = (signal.prices_pair.index >= oos_start) & (signal.prices_pair.index <= oos_end)
        n_is  = int(is_mask.sum())
        n_oos = int(oos_mask.sum())

        if n_is < min_points:
            issues.append(f"DATA_FAIL: {n_is} points IS < minimum {min_points}")
        if n_oos < min_points:
            issues.append(f"DATA_FAIL: {n_oos} points OOS < minimum {min_points}")

        return issues

    def _check_series(
        self,
        s: pd.Series,
        name: str,
        nan_threshold: float,
    ) -> list[str]:
        issues = []
        if not s.index.is_monotonic_increasing:
            issues.append(f"C5_FAIL [{name}]: index non monotone")
        if s.index.duplicated().any():
            issues.append(f"C5_FAIL [{name}]: {s.index.duplicated().sum()} timestamp(s) dupliqué(s)")
        nan_ratio = float(s.isna().mean())
        if nan_ratio >= nan_threshold:
            issues.append(f"C2_FAIL [{name}]: NaN ratio {nan_ratio:.1%} ≥ {nan_threshold:.0%}")
        return issues
