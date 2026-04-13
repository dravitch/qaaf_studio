"""
Tests Layer 1 — Backtester, BenchmarkFactory, DataLoader DQF stub

Fonctionne avec pytest ET unittest :
    python3 -m unittest tests/test_layer1_backtester.py -v
"""

import sys
import warnings
import unittest
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml

from tests.conftest import make_synthetic_prices, make_synthetic_returns
from layer1_engine.backtester        import Backtester
from layer1_engine.benchmark_factory import BenchmarkFactory
from layer1_engine.data_loader        import _dqf_stub


def _make_config(tmp_dir: str, mode: str = "lump_sum") -> str:
    cfg = {
        "engine": {"fees_pct": 0.001, "initial_capital": 10000.0, "mode": mode},
        "rates":  {"rf_fed": 0.04, "rf_usdc": 0.03, "rf_zero": 0.0},
        "splits": {"is_start": "2020-06-01", "is_end": "2023-05-31",
                   "oos_start": "2023-06-01", "oos_end": "2024-12-31"},
        "data":   {"cache_dir": str(Path(tmp_dir) / ".cache"),
                   "tickers": {"btc": "BTC-USD", "paxg": "PAXG-USD"}},
    }
    cfg_path = Path(tmp_dir) / "config.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.dump(cfg))
    return str(cfg_path)


class TestDQFStub(unittest.TestCase):

    def setUp(self):
        self.paxg, _ = make_synthetic_prices()

    def test_pass_on_clean_data(self):
        report = _dqf_stub(self.paxg, "PAXG-USD")
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["mode"],   "DIAGNOSTIC_STUB")
        self.assertIn("not eligible for MIF certification", report["annotation"])

    def test_fail_on_heavy_nan(self):
        dirty = self.paxg.copy()
        dirty.iloc[::5] = np.nan
        report = _dqf_stub(dirty, "PAXG-BAD")
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("C2_FAIL" in i for i in report["issues"]))

    def test_fail_on_nonmonotonic_index(self):
        shuffled = self.paxg.sample(frac=1, random_state=99)
        report   = _dqf_stub(shuffled, "PAXG-SHUFFLE")
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("C5_FAIL" in i for i in report["issues"]))

    def test_mpi_is_none(self):
        prices = pd.Series(
            [100.0 + i for i in range(50)],
            index=pd.date_range("2023-01-01", periods=50, freq="B"))
        report = _dqf_stub(prices, "SYNTHETIC")
        self.assertIsNone(report["mpi"])

    def test_fail_on_insufficient_data(self):
        tiny = pd.Series([100.0, 101.0, 99.0],
                          index=pd.date_range("2023-01-01", periods=3, freq="B"))
        report = _dqf_stub(tiny, "TINY")
        self.assertEqual(report["status"], "FAIL")


class TestBacktester(unittest.TestCase):

    def setUp(self):
        self.paxg, self.btc      = make_synthetic_prices()
        self.r_paxg, self.r_btc = make_synthetic_returns()
        self.prices_df = pd.DataFrame({"paxg": self.paxg, "btc": self.btc})
        self.tmp       = tempfile.mkdtemp()
        self.cfg_path  = _make_config(self.tmp)

    def test_run_returns_expected_keys(self):
        bt = Backtester(config_path=self.cfg_path)
        result = bt.run(lambda df: pd.Series(0.5, index=df.index),
                        self.prices_df, self.r_btc)
        for k in ("r_pair","r_portfolio_usd","r_base_usd",
                  "allocations","n_trades","fees_paid","std_alloc"):
            self.assertIn(k, result)

    def test_fixed_alloc_few_trades(self):
        bt = Backtester(config_path=self.cfg_path)
        result = bt.run(lambda df: pd.Series(0.5, index=df.index),
                        self.prices_df, self.r_btc)
        self.assertLessEqual(result["n_trades"], 1)

    def test_portfolio_returns_non_empty(self):
        bt = Backtester(config_path=self.cfg_path)
        result = bt.run(lambda df: pd.Series(0.5, index=df.index),
                        self.prices_df, self.r_btc)
        self.assertGreater(len(result["r_portfolio_usd"]), 0)

    def test_more_trades_more_fees(self):
        bt = Backtester(config_path=self.cfg_path)
        n  = len(self.prices_df)
        def volatile(df):
            arr = np.tile([0.2, 0.8], n // 2 + 1)[:n]
            return pd.Series(arr, index=df.index)
        r_fixed    = bt.run(lambda df: pd.Series(0.5, index=df.index),
                            self.prices_df, self.r_btc)
        r_volatile = bt.run(volatile, self.prices_df, self.r_btc)
        self.assertGreater(r_volatile["n_trades"], r_fixed["n_trades"])
        self.assertGreater(r_volatile["fees_paid"], r_fixed["fees_paid"])

    def test_dca_mode_emits_warning(self):
        cfg_dca = _make_config(self.tmp + "_dca", mode="dca")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Backtester(config_path=cfg_dca)
        self.assertEqual(len(w), 1)
        msg = str(w[0].message)
        self.assertTrue("lump_sum" in msg.lower() or "DCA" in msg)

    def test_alloc_clipped_to_01(self):
        bt = Backtester(config_path=self.cfg_path)
        result = bt.run(lambda df: pd.Series(2.5, index=df.index),
                        self.prices_df, self.r_btc)
        self.assertLessEqual(result["allocations"].max(), 1.0)


class TestBenchmarkFactory(unittest.TestCase):

    def setUp(self):
        self.paxg, self.btc      = make_synthetic_prices()
        self.r_paxg, self.r_btc = make_synthetic_returns()
        self.prices_df = pd.DataFrame({"paxg": self.paxg, "btc": self.btc})
        tmp            = tempfile.mkdtemp()
        self.bt        = Backtester(config_path=_make_config(tmp))
        self.factory   = BenchmarkFactory(self.bt)

    def test_b5050_keys(self):
        b = self.factory.b_5050(self.prices_df, self.r_btc)
        self.assertIn("cnsr_usd_fed", b)
        self.assertIn("max_dd_pct",   b)
        self.assertEqual(b["name"],   "B_5050")

    def test_b5050_positive_cnsr_on_bull(self):
        b = self.factory.b_5050(self.prices_df, self.r_btc)
        self.assertGreater(b["cnsr_usd_fed"], 0.0,
                           f"B_5050 CNSR={b['cnsr_usd_fed']:.3f}")

    def test_b_btc_name(self):
        self.assertEqual(self.factory.b_btc(self.prices_df, self.r_btc)["name"], "B_BTC")

    def test_b_paxg_name(self):
        self.assertEqual(self.factory.b_paxg(self.prices_df, self.r_btc)["name"], "B_PAXG")

    def test_b_btc_zero_trades(self):
        b = self.factory.b_btc(self.prices_df, self.r_btc)
        self.assertEqual(b["n_trades"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
