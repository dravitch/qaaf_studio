"""
DataLoader — Layer 1 QAAF Studio 3.0

Charge PAXG/BTC + USD prices avec validation DQF en mode diagnostique.
Cache local pour éviter les appels répétés.

Stratégie DQF :
  - Si le package `mif_dqf` est installé (PyPI) → on l'utilise en mode DIAGNOSTIC.
  - Sinon → stub minimal qui fait les 3 checks bloquants indispensables.

Le stub n'est PAS une réimplémentation de DQF. C'est un garde-fou temporaire
qui garantit que Layer 1 est testable sans dépendance externe.

Mode DIAGNOSTIC (pas CERTIFICATION) :
  - Calendar auto-détecté (CRYPTO_247 pour PAXG/BTC)
  - Pas de signature Ed25519 requise
  - Résultats annotés "DIAGNOSTIC — not eligible for MIF certification"

NE PAS répliquer ces checks en Layer 2 ou Layer 3.
"""

import pandas as pd
import numpy as np
import warnings
import yaml
from pathlib import Path
from typing import Tuple

warnings.filterwarnings("ignore")

# ── DQF availability check ───────────────────────────────────────────────────
try:
    from mif_dqf import DQFValidator, DQFConfig
    _DQF_AVAILABLE = True
except ImportError:
    _DQF_AVAILABLE = False


def _dqf_stub(prices: pd.Series, name: str) -> dict:
    """
    Stub minimal DQF pour quand le package n'est pas installé.

    Implémente uniquement les checks bloquants (C2 partiel, C5 partiel).
    Ne produit pas de MPI score ni d'enveloppe PROD.
    Annoté explicitement comme non-certifiant.
    """
    issues = []

    # C5 partiel — index
    if not prices.index.is_monotonic_increasing:
        issues.append("C5_FAIL: non-monotonic index")
    if prices.index.duplicated().any():
        issues.append("C5_FAIL: duplicate timestamps")

    # C2 partiel — NaN (prix de clôture seulement)
    nan_ratio = prices.isna().mean()
    if nan_ratio >= 0.05:
        issues.append(f"C2_FAIL: NaN ratio {nan_ratio:.1%} >= 5%")
    elif nan_ratio > 0:
        issues.append(f"C2_WARN: {prices.isna().sum()} NaN values present")

    # Minimum de données
    if len(prices) < 30:
        issues.append(f"C2_FAIL: insufficient data ({len(prices)} points)")

    # Sauts extrêmes — warning seulement
    pct        = prices.pct_change(fill_method=None).abs().dropna()
    n_extreme  = int((pct > 0.5).sum())
    if n_extreme > 5:
        issues.append(f"C2_WARN: {n_extreme} daily moves > 50%")

    has_fail = any("_FAIL" in i for i in issues)
    status   = "FAIL" if has_fail else ("WARNING" if issues else "PASS")

    return {
        "mode":       "DIAGNOSTIC_STUB",
        "annotation": (
            "DIAGNOSTIC — not eligible for MIF certification. "
            "Install mif-dqf for full DQF validation."
        ),
        "status": status,
        "issues": issues,
        "mpi":    None,  # Stub ne calcule pas le MPI
    }


def _dqf_diagnostic(prices: pd.Series, name: str) -> dict:
    """
    Validation DQF en mode DIAGNOSTIC via le package mif_dqf.
    Utilisé automatiquement si le package est installé.
    """
    config = DQFConfig(
        mode="DIAGNOSTIC",
        calendar="CRYPTO_247",
        max_consecutive_ffill=3,
    )
    validator = DQFValidator(config)
    df = pd.DataFrame({
        "open": prices, "high": prices, "low": prices,
        "close": prices, "volume": pd.Series(1.0, index=prices.index),
    })
    report = validator.validate(df)
    return {
        "mode":       "DIAGNOSTIC",
        "annotation": "DIAGNOSTIC — not eligible for MIF certification.",
        "status":     report.overall_status,
        "issues":     [str(i) for i in report.all_issues],
        "mpi":        getattr(report, "mpi", None),
    }


class DataLoader:
    def __init__(self, config_path: str = "config.yaml"):
        cfg = yaml.safe_load(Path(config_path).read_text())
        self.tickers   = cfg["data"]["tickers"]
        self.cache_dir = Path(cfg["data"]["cache_dir"])
        self.cache_dir.mkdir(exist_ok=True)
        self._dqf_reports: dict = {}

    @property
    def dqf_available(self) -> bool:
        return _DQF_AVAILABLE

    @property
    def dqf_reports(self) -> dict:
        """Rapports DQF des dernières données chargées."""
        return self._dqf_reports

    def load_prices(
        self,
        start: str = "2019-01-01",
        end:   str = "2024-12-31",
        use_cache: bool = True,
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Charge PAXG et BTC en prix USD.

        Retourne
        --------
        paxg_usd, btc_usd, r_paxg_usd, r_btc_usd (log-rendements quotidiens)
        """
        paxg_usd = self._load_ticker(self.tickers["paxg"], start, end, use_cache)
        btc_usd  = self._load_ticker(self.tickers["btc"],  start, end, use_cache)

        # Alignement des index (validation silencieuse Layer 0)
        common   = paxg_usd.index.intersection(btc_usd.index)
        paxg_usd = paxg_usd.loc[common]
        btc_usd  = btc_usd.loc[common]

        assert len(paxg_usd) == len(btc_usd), "Index mismatch after alignment"

        r_paxg = np.log(paxg_usd / paxg_usd.shift(1)).dropna()
        r_btc  = np.log(btc_usd  / btc_usd.shift(1)).dropna()
        common2 = r_paxg.index.intersection(r_btc.index)

        return (
            paxg_usd.loc[common2],
            btc_usd.loc[common2],
            r_paxg.loc[common2],
            r_btc.loc[common2],
        )

    def load_eth_btc(
        self,
        start: str = "2019-01-01",
        end:   str = "2024-12-31",
        use_cache: bool = True,
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Charge ETH et BTC en prix USD.

        Retourne
        --------
        eth_usd, btc_usd, r_eth_usd, r_btc_usd (log-rendements quotidiens)
        """
        eth_usd = self._load_ticker(self.tickers["eth"], start, end, use_cache)
        btc_usd = self._load_ticker(self.tickers["btc"], start, end, use_cache)

        common  = eth_usd.index.intersection(btc_usd.index)
        eth_usd = eth_usd.loc[common]
        btc_usd = btc_usd.loc[common]

        r_eth  = np.log(eth_usd / eth_usd.shift(1)).dropna()
        r_btc  = np.log(btc_usd / btc_usd.shift(1)).dropna()
        common2 = r_eth.index.intersection(r_btc.index)

        return (
            eth_usd.loc[common2],
            btc_usd.loc[common2],
            r_eth.loc[common2],
            r_btc.loc[common2],
        )

    def get_pair_returns(
        self,
        r_paxg_usd: pd.Series,
        r_btc_usd:  pd.Series,
    ) -> pd.Series:
        """r_pair = r_paxg_usd - r_btc_usd  (identité log-rendements exacte)."""
        return r_paxg_usd - r_btc_usd

    def _load_ticker(
        self,
        ticker:    str,
        start:     str,
        end:       str,
        use_cache: bool,
    ) -> pd.Series:
        cache_path = self.cache_dir / f"{ticker}_{start}_{end}.csv"

        if use_cache and cache_path.exists():
            df     = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            prices = df["Close"].squeeze()
        else:
            try:
                import yfinance as yf
                data   = yf.download(ticker, start=start, end=end,
                                     auto_adjust=True, progress=False,
                                     timeout=15)
                prices = data["Close"].squeeze()
            except Exception as e:
                raise RuntimeError(f"Failed to download {ticker}: {e}")
            pd.DataFrame({"Close": prices}).to_csv(cache_path)

        # Validation DQF — package ou stub
        if _DQF_AVAILABLE:
            report = _dqf_diagnostic(prices, ticker)
        else:
            report = _dqf_stub(prices, ticker)

        self._dqf_reports[ticker] = report

        if report["status"] == "FAIL":
            raise ValueError(
                f"DQF {report['mode']} FAIL for {ticker}:\n"
                + "\n".join(f"  - {i}" for i in report["issues"])
            )
        if report["status"] == "WARNING":
            warnings.warn(
                f"DQF {report['mode']} WARNING for {ticker}: "
                + "; ".join(report["issues"])
            )

        return prices
