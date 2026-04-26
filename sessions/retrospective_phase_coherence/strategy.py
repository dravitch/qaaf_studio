"""Signal PhaseCoherence — composante phase seule sans BTC (rétrospective Studio)."""
import numpy as np
import pandas as pd


def phase_coherence_strategy(r_pair: pd.Series, params: dict) -> pd.Series:
    T = params.get("T", 30)
    log_p = r_pair.cumsum()
    ma  = log_p.rolling(T, min_periods=max(1, T // 2)).mean()
    std = log_p.rolling(T, min_periods=max(1, T // 2)).std()
    std = std.replace(0, np.nan).ffill().fillna(1e-4)
    z   = (log_p - ma) / std
    # Vélocité de l'oscillateur — approximation de MR déguisée en géométrie (PAF D3)
    dz  = z.diff(1).fillna(0)
    dz_std = dz.rolling(T, min_periods=max(1, T // 2)).std().replace(0, np.nan).ffill().fillna(1e-4)
    phase_dir = np.tanh(dz / dz_std)
    alloc = (0.5 + 0.2 * phase_dir).clip(0.1, 0.9)
    return alloc.rename("alloc_phase_coherence")
