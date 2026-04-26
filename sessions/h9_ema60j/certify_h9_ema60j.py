"""
certify_h9_ema60j.py — Session pilote H9+EMA60j
Wrapper autour de SessionRunner — logique dans signal.py
"""

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
from sessions.h9_ema60j.strategy import h9_ema_strategy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Certification H9+EMA60j — QAAF Studio 3.0"
    )
    parser.add_argument("--fast",        action="store_true",
                        help="Mode test : n_perm=500, ema_step=10")
    parser.add_argument("--force-metis", action="store_true",
                        help="Continue vers METIS même si MIF échoue (diagnostic)")
    parser.add_argument("--skip-q2",     action="store_true",
                        help="Skip MÉTIS Q2 (permutation, ~5 min)")
    args = parser.parse_args()

    runner = SessionRunner(
        hypothesis = "H9+EMA60j",
        family     = "EMA_span_variants",
        kb_path    = Path(__file__).parent / "kb_h9_ema60j.yaml",
        signal_fn  = h9_ema_strategy,
        params     = {"ema_span": 60, "h9_lookback": 20},
    )
    result = runner.run(
        fast         = args.fast,
        force_metis  = args.force_metis,
        skip_q2      = args.skip_q2,
    )
    print("\n=== Log JSON ===")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    sys.exit(0)
