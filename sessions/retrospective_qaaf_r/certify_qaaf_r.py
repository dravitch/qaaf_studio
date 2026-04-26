"""certify_qaaf_r.py — Rétrospective Studio : QAAF-R (Sprint C)"""
from __future__ import annotations

import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding and sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import argparse
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sessions.session_runner import SessionRunner
from sessions.retrospective_qaaf_r.strategy import qaaf_r_strategy

_KB_PATH = Path(__file__).parent / "kb_qaaf_r.yaml"
FAMILY   = "geometric_regime"
N_TRIALS = 1


def run(fast: bool = False) -> dict:
    runner = SessionRunner(
        hypothesis  = "QAAF-R",
        family      = FAMILY,
        kb_path     = _KB_PATH,
        signal_fn   = qaaf_r_strategy,
        params      = {"T": 30},
        n_trials    = N_TRIALS,
        params_hook = lambda bundle: {
            "r_base": np.log(bundle.btc_usd / bundle.btc_usd.shift(1)).dropna()
        },
    )
    return runner.run(fast=fast, force_metis=True, update_kb=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rétrospective QAAF-R — QAAF Studio 3.0")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    result = run(fast=args.fast)
    print("\n=== Log JSON ===")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    sys.exit(0)
