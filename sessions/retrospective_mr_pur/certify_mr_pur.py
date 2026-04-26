"""certify_mr_pur.py — Rétrospective Studio : MR_pur (Sprint C)"""
from __future__ import annotations

import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding and sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sessions.session_runner import SessionRunner
from sessions.retrospective_mr_pur.strategy import mr_pur_strategy

_KB_PATH = Path(__file__).parent / "kb_mr_pur.yaml"
FAMILY   = "mean_reversion"
N_TRIALS = 1


def run(fast: bool = False) -> dict:
    runner = SessionRunner(
        hypothesis = "MR_pur",
        family     = FAMILY,
        kb_path    = _KB_PATH,
        signal_fn  = mr_pur_strategy,
        params     = {"lookback": 20},
        n_trials   = N_TRIALS,
    )
    return runner.run(fast=fast, force_metis=True, update_kb=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rétrospective MR_pur — QAAF Studio 3.0")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    result = run(fast=args.fast)
    print("\n=== Log JSON ===")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    sys.exit(0)
