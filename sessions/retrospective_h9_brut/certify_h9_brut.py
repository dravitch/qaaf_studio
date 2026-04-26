"""certify_h9_brut.py — Rétrospective Studio : H9_brut (Sprint C)"""
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
from sessions.retrospective_h9_brut.strategy import h9_brut_strategy

_KB_PATH = Path(__file__).parent / "kb_h9_brut.yaml"
FAMILY   = "EMA_span_variants"
N_TRIALS = 101


def run(fast: bool = False) -> dict:
    runner = SessionRunner(
        hypothesis = "H9_brut",
        family     = FAMILY,
        kb_path    = _KB_PATH,
        signal_fn  = h9_brut_strategy,
        params     = {"h9_lookback": 20},
        n_trials   = N_TRIALS,
    )
    return runner.run(fast=fast, force_metis=True, update_kb=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rétrospective H9_brut — QAAF Studio 3.0")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    result = run(fast=args.fast)
    print("\n=== Log JSON ===")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    sys.exit(0)
