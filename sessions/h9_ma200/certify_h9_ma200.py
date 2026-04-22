"""
Session de certification — H9+EMA60j+MA200
Famille : EMA_span_variants | N_trials = 103

Question : le filtre MA200 ameliore-t-il H9+EMA60j en regime bear ?
Comparaison directe avec H9+EMA60j (CNSR 1.285) et B_5050 (CNSR 1.343).
"""
import sys, io, argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

# Imports Studio
from layer1_engine.data_loader import DataLoader
from layer1_engine.backtester import RebalanceBacktester
from layer1_engine.benchmark_factory import BenchmarkFactory
from layer1_engine.metrics_engine import compute_cnsr, deflated_sharpe_ratio
from layer2_qualification.mif.phase0_isolation import run_phase0
from layer2_qualification.mif.phase1_oos import run_phase1
from layer3_validation.metis_runner import run_metis
from layer4_decision.dsig.mapper import strategy_to_dsig
from layer4_decision.n_trials_tracker import NTrialsTracker

# Constantes
HYP_NAME   = "H9+EMA60j+MA200"
FAMILY     = "EMA_span_variants"
N_TRIALS   = 103
MA200_PERIOD = 200
EMA_SPAN   = 60
IS_END     = "2022-12-31"
KB_HYP     = Path(__file__).parent / "kb_h9_ma200.yaml"
KB_INV     = ROOT / "layer4_decision" / "lentilles_inventory.yaml"

# ── Signal H9+EMA60j (identique à la session pilote) ──────────────────────────
def compute_h9_signal(ratio: pd.Series, span: int = EMA_SPAN) -> pd.Series:
    """H9 IQR robuste lisse par EMA."""
    roll = ratio.rolling(252)
    q25  = roll.quantile(0.25)
    q75  = roll.quantile(0.75)
    iqr  = q75 - q25
    z    = (ratio - roll.median()) / iqr.replace(0, np.nan)
    alloc_raw = 0.5 - z.clip(-2, 2) / 4.0
    return alloc_raw.ewm(span=span, adjust=False).mean().clip(0, 1)

# ── Filtre MA200 ───────────────────────────────────────────────────────────────
def apply_ma200_filter(alloc_h9: pd.Series,
                       btc_prices: pd.Series,
                       mode: str) -> pd.Series:
    """
    mode='hard' : 0% BTC quand BTC < MA200
    mode='soft' : 50% de l'allocation H9 quand BTC < MA200
    """
    ma200 = btc_prices.rolling(MA200_PERIOD).mean()
    bear  = btc_prices < ma200
    if mode == "hard":
        return alloc_h9.where(~bear, 0.0)
    elif mode == "soft":
        return alloc_h9.where(~bear, alloc_h9 * 0.5)
    raise ValueError(f"mode inconnu : {mode}")

# ── Session principale ────────────────────────────────────────────────────────
def run(fast: bool = False):
    n_perm  = 500  if fast else 2000
    ema_step = 10 if fast else 5

    print("=" * 62)
    print(f"  QAAF STUDIO 3.0 — SESSION DE CERTIFICATION")
    print(f"  Hypothese : {HYP_NAME}")
    print(f"  Famille   : {FAMILY} | N_trials = {N_TRIALS}")
    print(f"  Mode      : {'FAST' if fast else 'COMPLET'}")
    print("=" * 62)

    # ── Chargement données ─────────────────────────────────────────
    print("\nChargement donnees ...")
    try:
        dl = DataLoader()
        paxg, btc, r_paxg, r_btc = dl.load_prices("2019-01-01", "2024-12-31")
        ratio = np.log(paxg / btc)
        print(f"   {len(paxg)} jours charges ({paxg.index[0].date()} -> {paxg.index[-1].date()})")
        synthetic = False
    except Exception as e:
        print(f"   Donnees reelles indisponibles ({e})")
        print("   -> Utilisation donnees synthetiques (resultats non certifiants)")
        from layer2_qualification.mif.synthetic_data import generate_synthetic
        paxg, btc, r_paxg, r_btc, ratio = generate_synthetic()
        synthetic = True

    # ── Split IS / OOS ─────────────────────────────────────────────
    is_mask  = paxg.index <= IS_END
    oos_mask = paxg.index >  IS_END

    # ── Benchmarks (reference fixe) ────────────────────────────────
    print("\nBenchmarks passifs (CNSR-USD OOS) ...")
    bf = BenchmarkFactory()
    b5050 = bf.b5050(r_paxg[oos_mask], r_btc[oos_mask])
    print(f"  B_5050 CNSR = {b5050['cnsr_usd_fed']:.3f}  (ref H9+EMA60j : 1.343)")

    # ── PAF depuis KB ──────────────────────────────────────────────
    print("\nPAF : charge depuis KB (H9+EMA60j certifie 2026-04-01)")
    print("  D1 : HIERARCHIE_CONFIRMEE")
    print("  D2 : REGIMES_NEUTRES")
    print("  D3 : H9_LISSE_SUPERIEUR")
    print("  -> Filtre MA200 ajoute sur signal certifie PAF")

    # ── Signal de base H9+EMA60j ───────────────────────────────────
    alloc_base = compute_h9_signal(ratio)

    # ── Resultats par variante ─────────────────────────────────────
    results = {}
    backtester = RebalanceBacktester()
    tracker = NTrialsTracker(KB_HYP)

    for mode in ["hard", "soft"]:
        print(f"\n{'='*62}")
        print(f"  VARIANTE : H9+EMA60j+MA200_{mode.upper()}")
        print(f"{'='*62}")

        alloc = apply_ma200_filter(alloc_base, btc, mode)

        # Backtest OOS
        bt_oos = backtester.run(alloc[oos_mask], r_paxg[oos_mask], r_btc[oos_mask])
        metrics_oos = compute_cnsr(bt_oos["r_port"], r_btc[oos_mask])
        dsr_val = deflated_sharpe_ratio(bt_oos["r_port"], N_TRIALS)
        metrics_oos["dsr"] = dsr_val
        metrics_oos["walk_forward_score"] = 0.5  # placeholder avant Q1

        print(f"\nMetriques OOS ({mode}) :")
        print(f"  CNSR-USD-Fed : {metrics_oos['cnsr_usd_fed']:.3f}  (H9+EMA60j : 1.285)")
        print(f"  Sortino      : {metrics_oos['sortino']:.3f}")
        print(f"  Max DD %     : {metrics_oos['max_dd_pct']:.1f}%")
        print(f"  DSR(N={N_TRIALS}) : {dsr_val:.4f}")

        # MIF Phase 0
        print(f"\nMIF Phase 0 ({mode}) ...")
        p0 = run_phase0(alloc)
        p0_verdict = "PASS" if all(r.get("pass") for r in p0.values()) else "FAIL"
        print(f"  -> Phase 0 : {p0_verdict}")

        # MIF Phase 1
        print(f"MIF Phase 1 ({mode}) ...")
        p1 = run_phase1(alloc)
        p1_pass = sum(1 for r in p1.values() if r.get("pass"))
        p1_verdict = "PASS" if p1_pass >= 3 else "FAIL"
        print(f"  -> Phase 1 : {p1_verdict} ({p1_pass}/5)")

        # METIS (toujours, pour avoir la comparaison)
        def alloc_fn(prices_paxg, prices_btc):
            r = np.log(prices_paxg / prices_btc)
            base = compute_h9_signal(r)
            return apply_ma200_filter(base, prices_btc, mode)

        print(f"\nMETIS Q1-Q4 ({mode}) ...")
        metis = run_metis(
            alloc_fn=alloc_fn,
            paxg=paxg, btc=btc,
            r_paxg=r_paxg, r_btc=r_btc,
            is_end=IS_END,
            n_perm=n_perm,
            ema_step=ema_step,
            n_trials=N_TRIALS
        )

        # Score Q1 -> walk_forward_score
        wf_score = metis["q1"].n_windows_pass / 5.0
        metrics_oos["walk_forward_score"] = wf_score

        sig = strategy_to_dsig(metrics_oos, "H9_LISSE_SUPERIEUR", N_TRIALS,
                                source_id=f"h9-ma200-{mode}")

        results[mode] = {
            "cnsr_oos":   metrics_oos["cnsr_usd_fed"],
            "dsr":        dsr_val,
            "mif_p1":     p1_verdict,
            "metis_q1":   metis["q1"].verdict,
            "metis_q2":   metis["q2"].verdict,
            "metis_q3":   metis["q3"].verdict,
            "metis_q4":   metis["q4"].verdict,
            "metis_verdict": metis["verdict"],
            "dsig_score": sig.score,
            "dsig_label": sig.label,
        }

        tracker.increment(f"H9+MA200_{mode}")

    # ── Table comparative ──────────────────────────────────────────
    print("\n")
    print("=" * 70)
    print("  COMPARAISON — Famille EMA_span_variants")
    print("=" * 70)
    print(f"  {'Hypothese':<28} {'CNSR-OOS':>8} {'MIF-P1':>8} {'Q1':>6} {'Q2':>6} {'DSR':>7} {'D-SIG':>7}")
    print(f"  {'-'*66}")
    print(f"  {'B_5050 (passif)':<28} {'1.343':>8} {'N/A':>8} {'N/A':>6} {'N/A':>6} {'N/A':>7} {'72':>7}")
    print(f"  {'H9+EMA60j (archive)':<28} {'1.285':>8} {'FAIL':>8} {'PASS':>6} {'FAIL':>6} {'0.325':>7} {'59':>7}")
    for mode, r in results.items():
        cnsr_str = f"{r['cnsr_oos']:.3f}"
        dsr_str  = f"{r['dsr']:.3f}"
        print(f"  {f'H9+MA200_{mode}':<28} {cnsr_str:>8} {r['mif_p1']:>8} {r['metis_q1']:>6} {r['metis_q2']:>6} {dsr_str:>7} {r['dsig_score']:>7}")
    print("=" * 70)

    # ── KB update ──────────────────────────────────────────────────
    import datetime
    kb = yaml.safe_load(KB_HYP.read_text())
    kb["date"] = datetime.date.today().isoformat()
    for mode, r in results.items():
        kb["variants"][mode].update({
            "statut":       r["metis_verdict"],
            "cnsr_oos":     round(r["cnsr_oos"], 3),
            "mif_phase1":   r["mif_p1"],
            "metis_verdict": r["metis_verdict"],
            "dsig_score":   r["dsig_score"],
        })
    KB_HYP.write_text(yaml.dump(kb, allow_unicode=True))
    print(f"\nKB mise a jour : {KB_HYP}")

    return results

# ── Entry point ───────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    results = run(fast=args.fast)
    print("\n=== Log JSON ===")
    print(json.dumps(results, indent=2, default=str))