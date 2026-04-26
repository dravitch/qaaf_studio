"""
certify_h9_ma200.py — Session A : H9+EMA60j+MA200 (hard & soft)
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
from sessions.h9_ma200.strategy import h9_ma200_strategy
from layer4_decision.kb_manager import KBManager

_KB_PATH = Path(__file__).parent / "kb_h9_ma200.yaml"
_KB_INV  = Path(__file__).parent.parent.parent / "layer4_decision" / "lentilles_inventory.yaml"

FAMILY   = "EMA_span_variants"
N_TRIALS = 103
VARIANTS = ["hard", "soft"]

REF = {
    "B_5050":    {"cnsr": 1.343, "mif_p1": "N/A",  "q1": "N/A",  "q2": "N/A",  "dsig": 72},
    "H9+EMA60j": {"cnsr": 1.285, "mif_p1": "FAIL", "q1": "PASS", "q2": "FAIL", "dsig": 59},
}


def run(skip_q2: bool = False, fast: bool = False) -> dict:
    results = {}

    for mode in VARIANTS:
        params = {"ema_span": 60, "h9_lookback": 20, "ma200_mode": mode}
        runner = SessionRunner(
            hypothesis   = f"H9+EMA60j+MA200_{mode}",
            family       = FAMILY,
            kb_path      = _KB_PATH,
            signal_fn    = h9_ma200_strategy,
            params       = params,
            n_trials     = N_TRIALS,
            params_hook  = lambda bundle: {"btc_prices": bundle.btc_usd},
        )
        r = runner.run(fast=fast, skip_q2=skip_q2, update_kb=False)
        if r.get("stopped"):
            results[mode] = r
        else:
            results[mode] = {
                "cnsr_oos":     r.get("cnsr_oos", 0.0),
                "dsr":          r.get("dsr", 0.0),
                "mif_p1":       "PASS" if "FAIL" not in r.get("mif", "") else "FAIL",
                "metis_q1":     r.get("metis_q1", "N/A"),
                "metis_q2":     r.get("metis_q2", "N/A"),
                "metis_verdict": r.get("verdict", "N/A"),
                "dsig_score":   r.get("dsig", {}).get("score", 0),
                "dsig_label":   r.get("dsig", {}).get("label", "N/A"),
            }

    # ── Comparative table ─────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  COMPARAISON — Famille {FAMILY}")
    print(f"{'='*72}")
    print(f"  {'Hypothèse':<28} {'CNSR':>7} {'MIF-P1':>8} {'Q1':>6} {'Q2':>6} {'D-SIG':>7}")
    print(f"  {'-'*60}")
    for name, r in REF.items():
        print(f"  {name:<28} {r['cnsr']:>7.3f} {r['mif_p1']:>8} "
              f"{r['q1']:>6} {r['q2']:>6} {r['dsig']:>7}")
    for mode, r in results.items():
        if r.get("stopped"):
            print(f"  H9+MA200_{mode:<19} {'STOP':>7} {'—':>8} {'—':>6} {'—':>6} {'—':>7}")
        else:
            print(f"  {'H9+MA200_'+mode:<28} {r['cnsr_oos']:>7.3f} {r['mif_p1']:>8} "
                  f"{r['metis_q1']:>6} {r['metis_q2']:>6} {r['dsig_score']:>7}")
    print(f"{'='*72}")

    # ── Combined KB update ───────────────────────────────────────────────────
    valid_results = {m: r for m, r in results.items() if not r.get("stopped")}
    if valid_results:
        best_mode    = max(valid_results, key=lambda m: valid_results[m]["cnsr_oos"])
        best         = valid_results[best_mode]
        overall_verdict = best["metis_verdict"]

        kb = KBManager(str(_KB_PATH), str(_KB_INV))
        kb.update_metis({
            "variants": {
                mode: {
                    "verdict":    r["metis_verdict"],
                    "cnsr_oos":   r["cnsr_oos"],
                    "dsig_score": r["dsig_score"],
                    "mif_p1":     r["mif_p1"],
                    "q1":         r["metis_q1"],
                    "q2":         r["metis_q2"],
                }
                for mode, r in valid_results.items()
            }
        })
        hard_cnsr = valid_results.get("hard", {}).get("cnsr_oos", 0.0)
        soft_cnsr = valid_results.get("soft", {}).get("cnsr_oos", 0.0)
        kb.record_verdict(
            hypothesis="H9+EMA60j+MA200",
            verdict=overall_verdict,
            metrics={
                "best_variant":  best_mode,
                "best_cnsr_oos": best["cnsr_oos"],
                "best_dsr":      best["dsr"],
                "best_dsig":     best["dsig_score"],
                "hard_cnsr":     hard_cnsr,
                "soft_cnsr":     soft_cnsr,
            },
            notes=(
                f"Variantes hard={hard_cnsr:.3f} soft={soft_cnsr:.3f} | "
                f"N={N_TRIALS} | meilleure={best_mode}"
            ),
        )

        emoji = {"CERTIFIE": "🏆", "SUSPECT_DSR": "⚠️"}.get(overall_verdict, "📁")
        print(f"\n  {emoji}  VERDICT FINAL : {overall_verdict}")
        print(f"  Meilleure variante : {best_mode} — CNSR={best['cnsr_oos']:.3f}")
        print(f"{'='*72}\n")
    else:
        overall_verdict = "STOPPED_MIF"

    return {
        "hypothesis": "H9+EMA60j+MA200",
        "verdict":    overall_verdict,
        "results":    results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Certification H9+EMA60j+MA200 — QAAF Studio 3.0"
    )
    parser.add_argument("--fast",    action="store_true",
                        help="Mode test : n_perm=500, ema_step=10")
    parser.add_argument("--skip-q2", action="store_true",
                        help="Skip MÉTIS Q2 (permutation, ~5 min)")
    args = parser.parse_args()

    result = run(skip_q2=args.skip_q2, fast=args.fast)
    print("\n=== Log JSON ===")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    sys.exit(0)
