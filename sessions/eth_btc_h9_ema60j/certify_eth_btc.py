"""certify_eth_btc.py — H9+EMA60j généralisé sur ETH/BTC (Sprint C BLOC 2)"""
from __future__ import annotations

import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding and sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import argparse
import json
import types
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sessions.session_runner import SessionRunner
from sessions.h9_ema60j.strategy import h9_ema_strategy
from layer1_engine.data_loader import DataLoader

_KB_PATH    = Path(__file__).parent / "kb_eth_btc.yaml"
_CONFIG     = str(Path(__file__).parent.parent.parent / "config.yaml")
FAMILY      = "EMA_span_variants"
N_TRIALS    = 103


def _load_eth_btc_data():
    """Custom data loader : ETH/BTC remplace PAXG/BTC dans le bundle standard."""
    loader   = DataLoader(config_path=_CONFIG)
    eth_usd, btc_usd, r_eth_usd, r_btc_usd = loader.load_eth_btc(
        start="2019-01-01", end="2024-12-31"
    )
    bundle = types.SimpleNamespace(
        paxg_usd = eth_usd,          # alias : ETH joue le rôle de PAXG
        btc_usd  = btc_usd,
        paxg_btc = eth_usd / btc_usd,
    )
    print(f"   ETH/BTC : {len(r_eth_usd)} jours "
          f"({r_eth_usd.index[0].date()} → {r_eth_usd.index[-1].date()})")
    return bundle, r_eth_usd, r_btc_usd


def run(fast: bool = False, skip_q2: bool = False) -> dict:
    runner = SessionRunner(
        hypothesis = "H9+EMA60j_ETH-BTC",
        family     = FAMILY,
        kb_path    = _KB_PATH,
        signal_fn  = h9_ema_strategy,
        params     = {"ema_span": 60, "h9_lookback": 20},
        n_trials   = N_TRIALS,
        data_fn    = _load_eth_btc_data,
    )
    return runner.run(fast=fast, skip_q2=skip_q2, update_kb=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="H9+EMA60j sur ETH/BTC — QAAF Studio 3.0 Sprint C"
    )
    parser.add_argument("--fast",    action="store_true")
    parser.add_argument("--skip-q2", action="store_true")
    args = parser.parse_args()
    result = run(fast=args.fast, skip_q2=args.skip_q2)
    print("\n=== Log JSON ===")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    sys.exit(0)
