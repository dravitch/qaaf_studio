"""
test_benchmark_calibration.py — Tests de calibrage des benchmarks
Layer 1 QAAF Studio 3.0

GATES PERMANENTES : ces tests doivent passer avant toute session comparative.
Ils vérifient que le moteur de backtest produit des résultats physiquement plausibles
sur les données réelles OOS 2023-2024.

Distinction par rapport aux tests unitaires (test_layer1_backtester.py) :
  - Tests unitaires  : le code fait ce qu'il dit (données synthétiques, logique)
  - Tests calibrage  : les résultats sont physiquement cohérents (données réelles, KPIs KB)

Exécution :
    pytest tests/test_benchmark_calibration.py -v
    python -m unittest tests/test_benchmark_calibration.py -v

Tolérances : ±0.15 CNSR (variation acceptable selon ajustements prix yfinance),
             ±3% MDD.

Si ces tests échouent : NE PAS lancer de session comparative.
"""

import sys
import math
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

# Valeurs de référence KB (session certify_h9_ema60j, Avril 2026)
KB_REFERENCE = {
    "B_5050": {"cnsr_oos": 1.343, "tolerance": 0.15},
    "B_BTC":  {"cnsr_oos": 1.244, "tolerance": 0.15},
    "oos_period": {"start": "2023-06-01", "end": "2024-12-31"},
    "oos_n_days_min": 380,
    "oos_mdd_max_b5050": 20.0,
    "oos_cnsr_min_b5050": 0.8,
}

CONFIG_PATH = str(Path(__file__).parent.parent / "config.yaml")


def _load_data():
    """Charge les données réelles. Skip si réseau indisponible."""
    from layer1_engine.data_loader  import DataLoader
    from layer1_engine.split_manager import SplitManager
    from layer1_engine.backtester   import Backtester
    from layer1_engine.benchmark_factory import BenchmarkFactory

    loader = DataLoader(config_path=CONFIG_PATH)
    try:
        paxg_usd, btc_usd, r_paxg, r_btc = loader.load_prices(
            start="2019-01-01", end="2024-12-31"
        )
    except Exception as e:
        return None, str(e)

    prices_full = pd.DataFrame({"paxg": paxg_usd, "btc": btc_usd})
    sm = SplitManager(config_path=CONFIG_PATH)
    _, prices_oos = sm.apply_df(prices_full)
    _, r_btc_oos  = sm.apply(r_btc)

    bt      = Backtester(config_path=CONFIG_PATH)
    factory = BenchmarkFactory(bt)

    return {
        "prices_full": prices_full,
        "prices_oos":  prices_oos,
        "r_btc_oos":   r_btc_oos,
        "split_manager": sm,
        "backtester":  bt,
        "factory":     factory,
    }, None


# Chargement une seule fois au niveau module
_DATA, _LOAD_ERROR = _load_data()


def _skip_if_no_data(test_fn):
    """Décorateur : skip le test si les données ne sont pas disponibles."""
    import functools
    @functools.wraps(test_fn)
    def wrapper(self):
        if _DATA is None:
            self.skipTest(f"Données non disponibles : {_LOAD_ERROR}")
        test_fn(self)
    return wrapper


class TestSplitDates(unittest.TestCase):
    """Vérifie que le split OOS couvre la bonne période."""

    @_skip_if_no_data
    def test_oos_start_correct(self):
        prices_oos = _DATA["prices_oos"]
        self.assertGreaterEqual(
            str(prices_oos.index[0].date()), "2023-06-01",
            f"OOS start trop tôt : {prices_oos.index[0].date()}"
        )

    @_skip_if_no_data
    def test_oos_end_correct(self):
        prices_oos = _DATA["prices_oos"]
        self.assertLessEqual(
            str(prices_oos.index[-1].date()), "2024-12-31",
            f"OOS end trop loin : {prices_oos.index[-1].date()}"
        )

    @_skip_if_no_data
    def test_oos_length_sufficient(self):
        prices_oos = _DATA["prices_oos"]
        self.assertGreaterEqual(
            len(prices_oos), KB_REFERENCE["oos_n_days_min"],
            f"OOS trop court : {len(prices_oos)} jours"
        )

    @_skip_if_no_data
    def test_oos_no_is_contamination(self):
        """L'OOS ne doit contenir aucun jour antérieur à 2023-06-01."""
        prices_oos = _DATA["prices_oos"]
        contaminated = (prices_oos.index < pd.Timestamp("2023-06-01")).sum()
        self.assertEqual(
            contaminated, 0,
            f"{contaminated} jours IS détectés dans l'OOS — contamination."
        )


class TestBenchmarkDifferentiation(unittest.TestCase):
    """
    GATE CRITIQUE : B_5050, B_BTC et B_PAXG doivent avoir des CNSR distincts.
    Si ce test échoue, toutes les métriques sont identiques (bug moteur).
    """

    @_skip_if_no_data
    def test_benchmarks_have_distinct_cnsr(self):
        f  = _DATA["factory"]
        p  = _DATA["prices_oos"]
        r  = _DATA["r_btc_oos"]

        b5050 = f.b_5050(p, r)["cnsr_usd_fed"]
        b_btc = f.b_btc(p,  r)["cnsr_usd_fed"]
        b_paxg = f.b_paxg(p, r)["cnsr_usd_fed"]

        self.assertNotEqual(b5050, b_btc,
            f"B_5050 == B_BTC ({b5050:.4f}) — bug dans le Backtester.")
        self.assertNotEqual(b5050, b_paxg,
            f"B_5050 == B_PAXG ({b5050:.4f}) — bug dans le Backtester.")
        self.assertNotEqual(b_btc, b_paxg,
            f"B_BTC == B_PAXG ({b_btc:.4f}) — bug dans le Backtester.")

    @_skip_if_no_data
    def test_cnsr_changes_with_allocation(self):
        """
        GATE CRITIQUE : CNSR doit changer quand l'allocation change.
        Détecte l'ancien bug où r_pair était indépendant de l'allocation.
        """
        from layer1_engine.metrics_engine import compute_cnsr

        bt = _DATA["backtester"]
        p  = _DATA["prices_oos"]
        r  = _DATA["r_btc_oos"]

        def a10(df): return pd.Series(0.10, index=df.index)
        def a90(df): return pd.Series(0.90, index=df.index)

        res10 = bt.run(a10, p, r)
        res90 = bt.run(a90, p, r)
        cnsr10 = compute_cnsr(res10["r_portfolio_usd"], res10["r_base_usd"])["cnsr_usd_fed"]
        cnsr90 = compute_cnsr(res90["r_portfolio_usd"], res90["r_base_usd"])["cnsr_usd_fed"]

        self.assertNotEqual(
            cnsr10, cnsr90,
            f"CNSR identique pour 10% et 90% PAXG ({cnsr10:.4f}) — "
            "l'allocation n'a aucun effet sur les métriques."
        )


class TestBenchmarkPhysicalPlausibility(unittest.TestCase):
    """
    Vérifie que les valeurs sont physiquement plausibles sur OOS 2023-2024.
    OOS 2023-2024 = bull market exceptionnel (BTC +150%, B_5050 CNSR ≈ 1.73 dans KB).
    """

    @_skip_if_no_data
    def test_b5050_cnsr_positive_on_bull_oos(self):
        f = _DATA["factory"]
        p = _DATA["prices_oos"]
        r = _DATA["r_btc_oos"]
        b = f.b_5050(p, r)
        self.assertGreater(
            b["cnsr_usd_fed"], KB_REFERENCE["oos_cnsr_min_b5050"],
            f"B_5050 CNSR={b['cnsr_usd_fed']:.3f} < {KB_REFERENCE['oos_cnsr_min_b5050']} "
            "— OOS probablement sur la mauvaise période."
        )

    @_skip_if_no_data
    def test_b5050_mdd_plausible_on_bull_oos(self):
        """B_5050 sur bull OOS ne peut pas avoir MDD > 20%."""
        f = _DATA["factory"]
        p = _DATA["prices_oos"]
        r = _DATA["r_btc_oos"]
        b = f.b_5050(p, r)
        self.assertLessEqual(
            b["max_dd_pct"], KB_REFERENCE["oos_mdd_max_b5050"],
            f"B_5050 MDD={b['max_dd_pct']:.1f}% > {KB_REFERENCE['oos_mdd_max_b5050']}% "
            "— données OOS probablement incorrectes (inclut 2022 bear?)."
        )

    @_skip_if_no_data
    def test_b5050_zero_trades(self):
        """B_5050 allocation fixe = 0 ou 1 trade (initialisation)."""
        f = _DATA["factory"]
        p = _DATA["prices_oos"]
        r = _DATA["r_btc_oos"]
        b = f.b_5050(p, r)
        self.assertLessEqual(b["n_trades"], 1,
            f"B_5050 a {b['n_trades']} trades — allocation fixe ne devrait pas trader.")

    @_skip_if_no_data
    def test_b_btc_positive_on_bull_oos(self):
        """B_BTC doit être positif sur un bull market BTC."""
        f = _DATA["factory"]
        p = _DATA["prices_oos"]
        r = _DATA["r_btc_oos"]
        b = f.b_btc(p, r)
        self.assertGreater(b["cnsr_usd_fed"], 0.0,
            f"B_BTC CNSR={b['cnsr_usd_fed']:.3f} ≤ 0 sur OOS — vérifier les données.")


class TestKBReferenceValues(unittest.TestCase):
    """
    Vérifie que les CNSR sont proches des valeurs de référence KB (±0.15).
    Ces valeurs viennent de la session certify_h9_ema60j (Avril 2026).
    Tolérance large pour absorber les variations de prix yfinance.
    """

    @_skip_if_no_data
    def test_b5050_cnsr_matches_kb(self):
        f   = _DATA["factory"]
        p   = _DATA["prices_oos"]
        r   = _DATA["r_btc_oos"]
        b   = f.b_5050(p, r)
        ref = KB_REFERENCE["B_5050"]["cnsr_oos"]
        tol = KB_REFERENCE["B_5050"]["tolerance"]
        self.assertAlmostEqual(
            b["cnsr_usd_fed"], ref, delta=tol,
            msg=f"B_5050 CNSR={b['cnsr_usd_fed']:.3f} hors tolérance "
                f"(ref={ref}, tol=±{tol}). Vérifier dates OOS et source prix."
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
