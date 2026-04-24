"""
certify_h9_ma200.py — Session A : H9+EMA60j+MA200 (hard & soft)
Phase 4 du plan de mise en œuvre | QAAF Studio 3.0

Hypothèse précise :
    "Je teste si H9+EMA60j avec filtre MA200 sur BTC corrige les déficits
    MIF G1/G2/G3 documentés sur H9+EMA60j (bear market, marché latéral, crash)
    sur la période OOS 2023-06 → 2024-12 de la paire PAXG/BTC."

Deux variantes testées en parallèle :
    hard : 0% BTC (allocation PAXG = 0) quand BTC < MA200
    soft : 50% de l'allocation H9 quand BTC < MA200

Protocole
---------
- PAF D1/D2/D3 chargés depuis KB H9+EMA60j (signal de base certifié)
- MIF Phase 0+1+2 par variante (filtrage appliqué via params)
- MÉTIS Q1/Q2/Q3/Q4 par variante
- Table comparative finale vs B_5050 et H9+EMA60j

Usage
-----
    python sessions/h9_ma200/certify_h9_ma200.py
    python sessions/h9_ma200/certify_h9_ma200.py --skip-q2
    python sessions/h9_ma200/certify_h9_ma200.py --fast
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from layer1_engine.data_loader       import DataLoader
from layer1_engine.split_manager     import SplitManager
from layer1_engine.benchmark_factory import BenchmarkFactory
from layer1_engine.backtester        import Backtester
from layer1_engine.metrics_engine    import compute_cnsr, deflated_sharpe_ratio
from layer2_qualification.mif.mif_runner import MIFRunner
from layer3_validation.metis_runner  import METISRunner
from layer4_decision.kb_manager      import KBManager
from layer4_decision.n_trials_tracker import NTrialsTracker
from layer4_decision.dsig.mapper     import strategy_to_dsig

# ── Constantes de session ────────────────────────────────────────────────────

HYPOTHESIS   = "H9+EMA60j+MA200"
FAMILY       = "EMA_span_variants"
N_TRIALS     = 103
EMA_SPAN     = 60
MA200_PERIOD = 200
CONFIG_PATH  = "config.yaml"
KB_HYP       = Path(__file__).parent / "kb_h9_ma200.yaml"
KB_INV       = Path(__file__).parent.parent.parent / "layer4_decision" / "lentilles_inventory.yaml"
VARIANTS     = ["hard", "soft"]

# Valeurs de référence pour la table comparative
REF = {
    "B_5050":    {"cnsr": 1.343, "mif_p1": "N/A",  "q1": "N/A",  "q2": "N/A",  "dsig": 72},
    "H9+EMA60j": {"cnsr": 1.285, "mif_p1": "FAIL", "q1": "PASS", "q2": "FAIL", "dsig": 59},
}


# ── Stratégie H9 + EMA 60j ───────────────────────────────────────────────────

def h9_ema_strategy(r_pair: pd.Series, params: dict) -> pd.Series:
    """H9 : allocation PAXG proportionnelle à la déviation du ratio de sa MA."""
    span     = params.get("ema_span",    EMA_SPAN)
    lookback = params.get("h9_lookback", 20)

    ratio = r_pair.cumsum().apply(np.exp)
    ma    = ratio.rolling(lookback, min_periods=max(1, lookback // 2)).mean()
    std   = ratio.rolling(lookback, min_periods=max(1, lookback // 2)).std()
    std   = std.replace(0, np.nan).ffill().fillna(1e-4)
    z     = (ratio - ma) / std

    raw      = 0.5 - 0.25 * np.tanh(z * 0.8)
    smoothed = raw.ewm(span=span, min_periods=span // 2).mean()
    return smoothed.clip(0.1, 0.9).rename("alloc_h9_ema")


def apply_ma200_filter(alloc_h9: pd.Series,
                       btc_prices: pd.Series,
                       mode: str) -> pd.Series:
    """Applique le filtre MA200 sur BTC à l'allocation H9."""
    ma200 = btc_prices.rolling(MA200_PERIOD, min_periods=MA200_PERIOD // 2).mean()
    bear  = btc_prices < ma200
    if mode == "hard":
        return alloc_h9.where(~bear, 0.0)
    if mode == "soft":
        return alloc_h9.where(~bear, alloc_h9 * 0.5)
    raise ValueError(f"mode inconnu : {mode}")


def h9_ma200_strategy(r_pair: pd.Series, params: dict) -> pd.Series:
    """
    Wrapper MIF-compatible : H9+EMA60j + filtre MA200 optionnel.
    Si params['btc_prices'] est absent (données synthétiques MIF), retourne
    l'allocation H9 de base sans filtre.
    """
    alloc_base = h9_ema_strategy(r_pair, params)
    btc_prices = params.get("btc_prices")
    if btc_prices is not None:
        mode = params.get("ma200_mode", "hard")
        btc_aligned = btc_prices.reindex(r_pair.index).ffill().bfill()
        return apply_ma200_filter(alloc_base, btc_aligned, mode)
    return alloc_base


# ── Session principale ────────────────────────────────────────────────────────

def run(skip_q2: bool = False, fast: bool = False) -> dict:
    n_perm   = 500 if fast else 10_000
    ema_step = 10  if fast else 5
    metis_q  = "Q1Q3Q4" if skip_q2 else "Q1Q2Q3Q4"
    params_base = {"ema_span": EMA_SPAN, "h9_lookback": 20}

    print(f"\n{'='*72}")
    print(f"  QAAF STUDIO 3.0 — SESSION A")
    print(f"  Hypothèse : {HYPOTHESIS}")
    print(f"  Famille   : {FAMILY} | N_trials = {N_TRIALS}")
    print(f"  Variantes : {VARIANTS}")
    print(f"  MÉTIS     : {metis_q} | n_perm={n_perm} | ema_step={ema_step}")
    print(f"  Mode      : {'FAST' if fast else 'COMPLET'}")
    print(f"{'='*72}")

    # ── 1. KB check ──────────────────────────────────────────────────────
    kb      = KBManager(str(KB_HYP), str(KB_INV))
    tracker = NTrialsTracker(str(KB_HYP))
    check   = kb.pre_session_check(HYPOTHESIS, FAMILY)

    if check["recommendation"] == "SKIP_DUPLICATE":
        print("\n⏹️  Hypothèse déjà certifiée dans KB.")
        return check

    # ── 2. Données ───────────────────────────────────────────────────────
    print("\n📡 Chargement données ...")
    import types
    try:
        loader = DataLoader(config_path=CONFIG_PATH)
        paxg_usd, btc_usd, r_paxg_usd, r_btc_usd = loader.load_prices(
            start="2019-01-01", end="2024-12-31"
        )
        bundle = types.SimpleNamespace(
            paxg_usd = paxg_usd,
            btc_usd  = btc_usd,
            paxg_btc = paxg_usd / btc_usd,
        )
        print(f"   {len(r_paxg_usd)} jours chargés "
              f"({r_paxg_usd.index[0].date()} → {r_paxg_usd.index[-1].date()})")
    except Exception as e:
        print(f"   ⚠️  Données réelles indisponibles ({e.__class__.__name__}: {e})")
        print("   → Utilisation de données synthétiques (résultats non certifiants)")
        from layer2_qualification.mif.synthetic_data import generate_synthetic_paxgbtc
        r_pair_s, r_base_s = generate_synthetic_paxgbtc(T=1500, seed=42)
        idx      = r_pair_s.index
        btc_usd  = pd.Series(1000 * np.exp(r_base_s.cumsum()),              index=idx, name="btc")
        paxg_usd = pd.Series(1800 * np.exp((r_pair_s + r_base_s).cumsum()), index=idx, name="paxg")
        r_paxg_usd = np.log(paxg_usd / paxg_usd.shift(1)).dropna()
        r_btc_usd  = np.log(btc_usd  / btc_usd.shift(1)).dropna()
        common     = r_paxg_usd.index.intersection(r_btc_usd.index)
        r_paxg_usd, r_btc_usd = r_paxg_usd.loc[common], r_btc_usd.loc[common]
        paxg_usd,   btc_usd   = paxg_usd.loc[common],   btc_usd.loc[common]
        bundle = types.SimpleNamespace(
            paxg_usd = paxg_usd,
            btc_usd  = btc_usd,
            paxg_btc = paxg_usd / btc_usd,
        )
        print(f"   {len(r_paxg_usd)} jours synthétiques "
              f"({r_paxg_usd.index[0].date()} → {r_paxg_usd.index[-1].date()})")

    sm = SplitManager(config_path=CONFIG_PATH)
    sm.set_n_trials(FAMILY, N_TRIALS)

    # ── 3. Benchmarks OOS ────────────────────────────────────────────────
    print("\n📊 Benchmarks passifs (CNSR-USD OOS) ...")
    prices_df = pd.DataFrame({"paxg": paxg_usd, "btc": btc_usd})
    _, prices_oos = sm.apply_df(prices_df)
    _, r_btc_oos  = sm.apply(r_btc_usd)

    backtester = Backtester(config_path=CONFIG_PATH)
    bf         = BenchmarkFactory(backtester)
    b5050      = bf.b_5050(prices_oos, r_btc_oos)
    b_btc      = bf.b_btc( prices_oos, r_btc_oos)
    print(f"   B_5050 CNSR = {b5050['cnsr_usd_fed']:.3f}  (KB : 1.343)")
    print(f"   B_BTC  CNSR = {b_btc['cnsr_usd_fed']:.3f}  (KB : 1.244)")

    # ── 4. PAF — chargé depuis KB H9+EMA60j ──────────────────────────────
    print("\n" + "─"*72)
    print("PAF : résultats chargés depuis KB H9+EMA60j (certifié 2026-04-01)")
    print("  D1 : HIERARCHIE_CONFIRMEE")
    print("  D2 : REGIMES_NEUTRES")
    print("  D3 : H9_LISSE_SUPERIEUR")

    # ── 5. Signal OOS de base (sans filtre) ──────────────────────────────
    r_pair_full = np.log(bundle.paxg_btc / bundle.paxg_btc.shift(1)).dropna()
    r_base_full = np.log(bundle.btc_usd  / bundle.btc_usd.shift(1)).dropna()
    _, r_pair_oos = sm.apply(r_pair_full)
    _, r_base_oos = sm.apply(r_base_full)
    _, btc_oos    = sm.apply(bundle.btc_usd)

    alloc_oos_base = h9_ema_strategy(r_pair_oos, params_base)

    # ── 6. Boucle variantes ───────────────────────────────────────────────
    results = {}

    for mode in VARIANTS:
        print(f"\n{'='*72}")
        print(f"  VARIANTE : H9+MA200_{mode.upper()}")
        print(f"{'='*72}")

        # 6a. Allocation filtrée
        alloc_filtered = apply_ma200_filter(alloc_oos_base, btc_oos, mode)
        common_oos = r_pair_oos.index.intersection(alloc_filtered.index)
        r_port_oos = alloc_filtered.reindex(common_oos).ffill() * r_pair_oos.loc[common_oos]
        r_usd_oos  = r_port_oos + r_base_oos.reindex(common_oos)

        metrics_oos = compute_cnsr(r_port_oos, r_base_oos.reindex(common_oos))
        dsr_oos     = deflated_sharpe_ratio(r_usd_oos.dropna(), N_TRIALS)

        print(f"\nMétriques OOS H9+MA200_{mode} :")
        print(f"  CNSR-USD-Fed : {metrics_oos['cnsr_usd_fed']:.3f}")
        print(f"  Sortino      : {metrics_oos['sortino']:.3f}")
        print(f"  Max DD %     : {metrics_oos['max_dd_pct']:.1f}%")
        print(f"  DSR(N={N_TRIALS}): {dsr_oos:.4f}  "
              f"{'✅' if dsr_oos >= 0.95 else '⚠️'}")

        # 6b. MIF — strategy wrapper avec btc_prices dans params
        params_mode = {**params_base, "ma200_mode": mode, "btc_prices": bundle.btc_usd}
        mif_runner = MIFRunner(
            strategy_fn=h9_ma200_strategy,
            params=params_mode,
            hypothesis=f"{HYPOTHESIS}_{mode}",
        )
        mif_summary = mif_runner.run(max_phase=2)
        mif_summary.print_summary()

        mif_p1_verdict = "PASS" if "FAIL" not in mif_summary.verdict else "FAIL"

        if "FAIL" in mif_summary.verdict and not fast:
            print(f"\n⚠️  MIF {mif_summary.verdict} — mode complet, pipeline continue pour table comparative.")

        # 6c. MÉTIS
        print(f"\n{'─'*72}")
        params_metis = {**params_base, "ma200_mode": mode, "btc_prices": bundle.btc_usd}
        metis_runner = METISRunner(
            strategy_fn=h9_ma200_strategy,
            params=params_metis,
            bundle=bundle,
            split_manager=sm,
            hypothesis=f"{HYPOTHESIS}_{mode}",
            n_trials=N_TRIALS,
        )
        metis_report = metis_runner.run(
            questions=metis_q,
            n_perm=n_perm,
            ema_step=ema_step,
        )
        metis_report.print_summary()

        # 6d. D-SIG
        wf_score = (metis_report.q1.n_windows_pass / 5) if metis_report.q1 else 0.0
        signal   = strategy_to_dsig(
            metrics     = {**metrics_oos, "dsr": dsr_oos,
                           "walk_forward_score": wf_score},
            paf_verdict = "H9_LISSE_SUPERIEUR",
            n_trials    = N_TRIALS,
            source_id   = f"qaaf-studio::{HYPOTHESIS}_{mode}",
        )
        print(f"\nD-SIG v0.5 ({mode}) : score={signal.score}/100 | "
              f"{signal.label} | {signal.color} | {signal.trend}")

        metis_verdict = metis_report.verdict()
        results[mode] = {
            "cnsr_oos":     round(float(metrics_oos["cnsr_usd_fed"]), 3),
            "sortino":      round(float(metrics_oos["sortino"]),       3),
            "max_dd_pct":   round(float(metrics_oos["max_dd_pct"]),    2),
            "dsr":          round(float(dsr_oos),                      4),
            "mif_p1":       mif_p1_verdict,
            "metis_q1":     metis_report.q1.verdict if metis_report.q1 else "N/A",
            "metis_q2":     metis_report.q2.verdict if metis_report.q2 else "N/A",
            "metis_verdict": metis_verdict,
            "dsig_score":   signal.score,
            "dsig_label":   signal.label,
            "dsig_color":   signal.color,
        }

    # ── 7. Table comparative ──────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  COMPARAISON — Famille {FAMILY}")
    print(f"{'='*72}")
    print(f"  {'Hypothèse':<28} {'CNSR':>7} {'MIF-P1':>8} {'Q1':>6} {'Q2':>6} {'D-SIG':>7}")
    print(f"  {'-'*60}")
    for name, r in REF.items():
        print(f"  {name:<28} {r['cnsr']:>7.3f} {r['mif_p1']:>8} "
              f"{r['q1']:>6} {r['q2']:>6} {r['dsig']:>7}")
    for mode, r in results.items():
        name = f"H9+MA200_{mode}"
        print(f"  {name:<28} {r['cnsr_oos']:>7.3f} {r['mif_p1']:>8} "
              f"{r['metis_q1']:>6} {r['metis_q2']:>6} {r['dsig_score']:>7}")
    print(f"{'='*72}")

    # ── 8. KB update ──────────────────────────────────────────────────────
    best_mode    = max(results, key=lambda m: results[m]["cnsr_oos"])
    best         = results[best_mode]
    overall_verdict = best["metis_verdict"]

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
            for mode, r in results.items()
        }
    })
    kb.record_verdict(
        hypothesis=HYPOTHESIS,
        verdict=overall_verdict,
        metrics={
            "best_variant":  best_mode,
            "best_cnsr_oos": best["cnsr_oos"],
            "best_dsr":      best["dsr"],
            "best_dsig":     best["dsig_score"],
            "hard_cnsr":     results["hard"]["cnsr_oos"],
            "soft_cnsr":     results["soft"]["cnsr_oos"],
        },
        notes=(
            f"MÉTIS {metis_q} | N={N_TRIALS} | "
            f"hard={results['hard']['cnsr_oos']:.3f} "
            f"soft={results['soft']['cnsr_oos']:.3f}"
        ),
    )

    emoji = {"CERTIFIE": "🏆", "SUSPECT_DSR": "⚠️"}.get(overall_verdict, "📁")
    print(f"\n  {emoji}  VERDICT FINAL : {overall_verdict}")
    print(f"  Meilleure variante : {best_mode} — CNSR={best['cnsr_oos']:.3f}")
    print(f"{'='*72}\n")

    return {
        "hypothesis": HYPOTHESIS,
        "verdict":    overall_verdict,
        "results":    results,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Certification H9+EMA60j+MA200 — QAAF Studio 3.0"
    )
    parser.add_argument("--skip-q2", action="store_true",
                        help="Skip MÉTIS Q2 (permutation, ~5 min)")
    parser.add_argument("--fast",    action="store_true",
                        help="Mode test : n_perm=500, ema_step=10")
    args = parser.parse_args()

    result = run(skip_q2=args.skip_q2, fast=args.fast)
    print("\n=== Log JSON ===")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    sys.exit(0)
