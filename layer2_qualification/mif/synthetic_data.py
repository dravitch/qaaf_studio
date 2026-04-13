"""
synthetic_data.py — MIF Layer 2 QAAF Studio 3.0

Générateur de données synthétiques reproduisant les 5 faits empiriques
documentés de la paire PAXG/BTC.

Utilisé par Phase 0 (isolation) et Phase 1 (généralisation OOS) du MIF.
Règle : Layer 3 (MÉTIS) utilise les données réelles uniquement.
Ce générateur ne sort JAMAIS du périmètre Layer 2.

5 faits empiriques PAXG/BTC
----------------------------
F1 — Dérive structurelle (half-life ≈ 374j)
F2 — Kurtosis extrême (≈ 17, skew ≈ 1.08)
F3 — Corrélation rolling oscillante (min -0.51, max +0.63)
F4 — Volatilité relative stable (Vol_BTC / Vol_PAXG ≈ 3.87)
F5 — Asymétrie directionnelle selon position du ratio
"""

from __future__ import annotations
from typing import Literal, Tuple

import numpy as np
import pandas as pd

Regime = Literal["standard", "bear", "lateral", "crash"]

# Paramètres calibrés sur données réelles PAXG/BTC 2019-2024
_SIGMA_BTC_BASE  = 0.038
_SIGMA_PAXG_BASE = 0.010
_VOL_RATIO_TARGET = 3.87   # F4

_REGIME_SCALE = {
    "standard": {"sigma_scale": 1.00, "drift_btc":  0.0003, "crash_prob": 0.00},
    "bear":     {"sigma_scale": 1.45, "drift_btc": -0.0005, "crash_prob": 0.00},
    "lateral":  {"sigma_scale": 0.66, "drift_btc":  0.0000, "crash_prob": 0.00},
    "crash":    {"sigma_scale": 2.10, "drift_btc": -0.0020, "crash_prob": 0.03},
}


def generate_synthetic_paxgbtc(
    T: int = 800,
    seed: int = 42,
    regime: Regime = "standard",
    start_date: str = "2020-01-01",
) -> Tuple[pd.Series, pd.Series]:
    """
    Génère (r_pair, r_base_usd) = (r_PAXG/BTC, r_BTC/USD).

    Retourne des log-rendements quotidiens indexés par date.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start_date, periods=T, freq="D")
    p   = _REGIME_SCALE[regime]

    sigma_btc  = _SIGMA_BTC_BASE  * p["sigma_scale"]
    sigma_paxg = _SIGMA_PAXG_BASE * p["sigma_scale"]
    drift_btc  = p["drift_btc"]
    drift_paxg = 0.00015   # F1 : dérive structurelle PAXG/BTC

    # F2 : queues épaisses — mélange Gaussienne + Student(df=3)
    def _fat(mu, sigma, n):
        g = rng.normal(mu, sigma, n)
        t = rng.standard_t(df=3, size=n) * sigma * 0.45
        m = rng.uniform(0, 1, n) < 0.12
        return g + m * t

    r_btc_raw  = _fat(drift_btc,  sigma_btc,  T)
    r_paxg_raw = _fat(drift_paxg, sigma_paxg, T)

    # F3 : corrélation rolling oscillante — composante cyclique
    phase  = rng.uniform(0, 2 * np.pi)
    cycle  = 0.38 * np.sin(np.linspace(0, 3.5 * np.pi, T) + phase)
    r_paxg = r_paxg_raw + cycle * r_btc_raw * 0.35

    # F5 : asymétrie directionnelle — mean-reversion plus forte côté haut
    log_r  = np.cumsum(r_paxg - r_btc_raw)
    mu_r   = pd.Series(log_r).ewm(span=60).mean().values
    z      = (log_r - mu_r) / (np.std(log_r) + 1e-8)
    r_paxg = r_paxg + (-0.0025 * np.tanh(z * 1.5))

    # F4 : ajustement du ratio de volatilité
    actual_ratio = np.std(r_btc_raw) / (np.std(r_paxg) + 1e-10)
    if actual_ratio > 0:
        r_paxg = r_paxg * (actual_ratio / _VOL_RATIO_TARGET)

    # Crash events
    if p["crash_prob"] > 0:
        crash      = rng.uniform(0, 1, T) < p["crash_prob"]
        magnitudes = rng.choice([-0.15, -0.20, -0.25], size=int(crash.sum()))
        r_btc_raw[crash]  += magnitudes
        r_paxg[crash]     += magnitudes * 0.3

    r_pair = pd.Series(r_paxg - r_btc_raw, index=idx, name="r_paxg_btc")
    r_base = pd.Series(r_btc_raw,           index=idx, name="r_btc_usd")
    return r_pair, r_base
