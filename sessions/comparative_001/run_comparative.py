"""
Session comparative_001 — QAAF Studio 3.0

Teste 8 lentilles simultanément : même moteur, mêmes données, même split.
Produit une table comparative Q1/Q2/Q4 + D-SIG par lentille.

Usage :
    python sessions/comparative_001/run_comparative.py
    python sessions/comparative_001/run_comparative.py --fast   # 50 permutations
    python sessions/comparative_001/run_comparative.py --no-q2  # skip permutation
"""

import sys
import argparse
import random
import json
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from layer1_engine.data_loader       import DataLoader
from layer1_engine.backtester        import Backtester
from layer1_engine.split_manager     import SplitManager
from layer1_engine.metrics_engine    import compute_cnsr, deflated_sharpe_ratio
from layer1_engine.benchmark_factory import BenchmarkFactory
from layer4_decision.dsig.mapper     import strategy_to_dsig
from sessions.comparative_001.signals import SIGNAL_REGISTRY

random.seed(42)
np.random.seed(42)

CONFIG_PATH  = str(ROOT / "config.yaml")
RESULTS_DIR  = Path(__file__).parent / "results"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
RESULTS_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
SESSION_ID = "comparative_001"

LENTILLES = {
    "B_5050":         {"signal": "passive",        "alloc": 0.50},
    "B_6040":         {"signal": "passive",        "alloc": 0.60},
    "B_BTC":          {"signal": "passive",        "alloc": 0.00},
    "H9_EMA60j":      {"signal": "h9_ema",         "span": 60},
    "H9_EMA30j":      {"signal": "h9_ema",         "span": 30},
    "H9_EMA90j":      {"signal": "h9_ema",         "span": 90},
    "H9_MA200":       {"signal": "h9_ma200_filter"},
    "H9_EMA60_MA200": {"signal": "h9_ema_ma200",   "span": 60},
}

Q1_MIN_WINDOWS  = 4
Q1_CNSR_THRESHOLD = 0.5
Q2_PVALUE_THRESHOLD = 0.05
Q4_DSR_THRESHOLD  = 0.95
N_TRIALS_FAMILY = 101


def atomic_save(path: Path, data: dict):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True)
    shutil.move(str(tmp), str(path))


def make_alloc_fn(signal_key: str, config: dict):
    fn = SIGNAL_REGISTRY[signal_key]
    def alloc_fn(prices_df):
        return fn(prices_df, config)
    return alloc_fn


def run_active_backtest(name, config, prices_df, r_btc, backtester):
    """Pour les stratégies actives (H9_*). Retourne les métriques OOS calculées sur r_portfolio_usd."""
    alloc_fn = make_alloc_fn(config["signal"], config)
    result = backtester.run(alloc_fn, prices_df, r_btc)
    # Correction : r_base_usd = 0 car r_portfolio_usd est déjà en USD
    metrics = compute_cnsr(result["r_portfolio_usd"], pd.Series(0, index=result["r_portfolio_usd"].index))
    metrics["name"] = name
    metrics["n_trades"] = result["n_trades"]
    metrics["fees_paid"] = result["fees_paid"]
    metrics["std_alloc"] = result["std_alloc"]
    # Conserver les séries pour Q2 et Q4
    metrics["r_pair"] = result["r_pair"]
    metrics["r_base_usd"] = result["r_base_usd"]  # série de zéros
    metrics["r_portfolio_usd"] = result["r_portfolio_usd"]
    return metrics


def run_q1(name, config, prices_full, r_btc_full, n_windows=5):
    T = len(prices_full)
    w = T // (n_windows + 1)
    results = []
    for i in range(n_windows):
        start = i * (T // (n_windows + 1))
        end   = min(start + w, T)
        p_w   = prices_full.iloc[start:end]
        r_w   = r_btc_full.reindex(p_w.index).dropna()
        bt    = Backtester(CONFIG_PATH)
        try:
            # Pour Q1 on peut utiliser la même logique corrigée
            alloc_fn = make_alloc_fn(config["signal"], config)
            res = bt.run(alloc_fn, p_w, r_w)
            cnsr = compute_cnsr(res["r_portfolio_usd"], pd.Series(0, index=res["r_portfolio_usd"].index))["cnsr_usd_fed"]
        except Exception:
            cnsr = np.nan
        passed = np.isfinite(cnsr) and cnsr >= Q1_CNSR_THRESHOLD
        results.append({
            "window": i + 1,
            "start":  str(p_w.index[0].date()),
            "end":    str(p_w.index[-1].date()),
            "cnsr":   round(float(cnsr), 4) if np.isfinite(cnsr) else None,
            "pass":   passed,
        })
    n_pass     = sum(w["pass"] for w in results)
    median_c   = float(np.nanmedian([w["cnsr"] for w in results if w["cnsr"] is not None]))
    return {
        "windows":    results,
        "n_pass":     n_pass,
        "n_total":    len(results),
        "median_cnsr": round(median_c, 4),
        "verdict":    "PASS" if n_pass >= Q1_MIN_WINDOWS else "FAIL",
    }


def run_q2(r_pair, r_base, r_pair_b5050, n_perm=500):
    cnsr_obs   = compute_cnsr(r_pair, r_base)["cnsr_usd_fed"]
    cnsr_bench = compute_cnsr(r_pair_b5050, r_base)["cnsr_usd_fed"]
    rng        = np.random.default_rng(42)
    pair_arr   = r_pair.values
    base_arr   = r_base.reindex(r_pair.index).fillna(0).values
    perm_cnsrs = []
    for _ in range(n_perm):
        perm = rng.permutation(pair_arr)
        c    = compute_cnsr(pd.Series(perm, index=r_pair.index),
                            pd.Series(base_arr, index=r_pair.index))["cnsr_usd_fed"]
        if np.isfinite(c):
            perm_cnsrs.append(c)
    if not perm_cnsrs:
        return {"verdict": "ERROR", "pvalue": None}
    pvalue  = float(np.mean(np.array(perm_cnsrs) >= cnsr_obs))
    return {
        "cnsr_obs":   round(cnsr_obs, 4),
        "cnsr_bench": round(cnsr_bench, 4),
        "perm_mean":  round(float(np.mean(perm_cnsrs)), 4),
        "pvalue":     round(pvalue, 4),
        "n_perm":     len(perm_cnsrs),
        "verdict":    "PASS" if pvalue < Q2_PVALUE_THRESHOLD else "FAIL",
    }


def run_q4(r_usd, n_trials):
    dsr = deflated_sharpe_ratio(r_usd, n_trials)
    return {
        "dsr":     round(float(dsr), 4) if np.isfinite(dsr) else None,
        "n_trials": n_trials,
        "verdict":  "PASS" if (np.isfinite(dsr) and dsr >= Q4_DSR_THRESHOLD)
                    else ("SUSPECT_DSR" if (np.isfinite(dsr) and dsr >= 0.80)
                    else "FAIL"),
    }


def compute_dsig(metrics_oos, q1, q2, q4):
    return strategy_to_dsig(
        metrics={
            "cnsr_usd_fed":       metrics_oos.get("cnsr_usd_fed", -1),
            "sortino":            metrics_oos.get("sortino", 0) or 0,
            "calmar":             metrics_oos.get("calmar", 0) or 0,
            "max_dd_pct":         metrics_oos.get("max_dd_pct", 50),
            "dsr":                q4.get("dsr") or 0,
            "walk_forward_score": q1["n_pass"] / max(q1["n_total"], 1),
        },
        paf_verdict="HIERARCHIE_CONFIRMEE",
        n_trials=N_TRIALS_FAMILY,
        source_id=f"qaaf-studio::comparative_001",
    )


def compute_verdict(q1, q2, q4):
    fails = []
    if q1["verdict"] == "FAIL":              fails.append("Q1")
    if q2.get("verdict") == "FAIL":          fails.append("Q2")
    if q4["verdict"] in ("FAIL","SUSPECT_DSR"): fails.append("Q4")
    if not fails:
        return "CERTIFIE"
    if q4["verdict"] == "SUSPECT_DSR" and fails == ["Q4"]:
        return "SUSPECT_DSR"
    return f"ARCHIVE_FAIL_{'_'.join(fails)}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast",  action="store_true", help="50 permutations")
    parser.add_argument("--no-q2", action="store_true", help="Skip permutation")
    args = parser.parse_args()
    n_perm = 50 if args.fast else 500

    print(f"\n{'='*70}")
    print(f"  SESSION COMPARATIVE — {SESSION_ID}")
    print(f"  {len(LENTILLES)} lentilles | Q1(5w) Q2({'skip' if args.no_q2 else n_perm}perm) Q4(DSR)")
    print(f"{'='*70}\n")

    print("📡 Chargement données...")
    loader = DataLoader(config_path=CONFIG_PATH)
    paxg_usd, btc_usd, r_paxg_usd, r_btc_usd = loader.load_prices(
        start="2019-01-01", end="2024-12-31"
    )
    prices_full = pd.DataFrame({"paxg": paxg_usd, "btc": btc_usd})
    print(f"   {len(prices_full)} jours ({prices_full.index[0].date()} → {prices_full.index[-1].date()})\n")

    sm = SplitManager(config_path=CONFIG_PATH)
    sm.set_n_trials("EMA_span_variants", N_TRIALS_FAMILY)
    prices_is, prices_oos = sm.apply_df(prices_full)
    r_btc_is,  r_btc_oos  = sm.apply(r_btc_usd)

    # Vérification que l'OOS est correct
    print(f"🔍 IS  : {prices_is.index[0].date()} → {prices_is.index[-1].date()} ({len(prices_is)} jours)")
    print(f"🔍 OOS : {prices_oos.index[0].date()} → {prices_oos.index[-1].date()} ({len(prices_oos)} jours)")

    bt = Backtester(config_path=CONFIG_PATH)
    factory = BenchmarkFactory(bt)

    # Benchmark B_5050 pour Q2 (pair returns) et debug
    b5050_factory = factory.b_5050(prices_oos, r_btc_oos)
    print(f"✅ Calibration B_5050 CNSR = {b5050_factory['cnsr_usd_fed']:.3f} (attendu ~1.34)\n")
    # Pour Q2 on a besoin des séries brutes de la paire de B_5050
    b5050_run = bt.run(lambda df: pd.Series(0.5, index=df.index), prices_oos, r_btc_oos)
    r_pair_b5050 = b5050_run["r_pair"]
    r_base_b5050 = b5050_run["r_base_usd"]  # série de zéros

    all_results = {}

    for name, config in LENTILLES.items():
        print(f"🔬 {name}...")

        # Construire la fonction d'allocation
        if name.startswith("B_"):
            # Benchmarks passifs : allocation fixe
            alloc_value = config.get("alloc", 0.5)  # B_5050:0.5, B_6040:0.6, B_BTC:0.0
            def alloc_fn(df):
                return pd.Series(alloc_value, index=df.index)
        else:
            # Stratégies actives
            alloc_fn = make_alloc_fn(config["signal"], config)

        # Exécuter le backtester
        res = bt.run(alloc_fn, prices_oos, r_btc_oos)

        # Calculer les métriques OOS sur r_portfolio_usd avec r_base=0
        m = compute_cnsr(res["r_portfolio_usd"], pd.Series(0, index=res["r_portfolio_usd"].index))
        m["name"] = name
        m["n_trades"] = res["n_trades"]
        m["fees_paid"] = res["fees_paid"]
        m["std_alloc"] = res["std_alloc"]
        m["r_pair"] = res["r_pair"]
        m["r_base_usd"] = res["r_base_usd"]   # série de zéros
        m["r_portfolio_usd"] = res["r_portfolio_usd"]

        # Extraire les séries pour Q2 et Q4
        r_pair_oos = m.pop("r_pair")
        r_base_oos = m.pop("r_base_usd")
        r_port_oos = m.pop("r_portfolio_usd")
        r_usd_oos = r_port_oos   # déjà en USD

        # ... reste identique

        q1 = run_q1(name, config, prices_full, r_btc_usd)
        q2 = ({"verdict":"SKIPPED"} if args.no_q2
              else run_q2(r_pair_oos, r_base_oos, r_pair_b5050, n_perm))
        q4 = run_q4(r_usd_oos.dropna(), N_TRIALS_FAMILY)

        verdict = compute_verdict(q1, q2, q4)
        dsig    = compute_dsig(m, q1, q2, q4)

        result = {
            "name": name, "config": config,
            "oos_metrics": {k: (round(v,4) if isinstance(v,float) else v)
                            for k,v in m.items() if k not in ("name",)},
            "q1": q1, "q2": q2, "q4": q4,
            "verdict": verdict,
            "dsig": {"score": dsig.score, "label": dsig.label,
                     "color": dsig.color, "trend": dsig.trend},
        }
        all_results[name] = result

        # Sauvegarde checkpoint
        atomic_save(CHECKPOINT_DIR / f"{SESSION_ID}_checkpoint.yaml",
                    {"completed": list(all_results.keys()), "results": all_results})

        q1_s  = f"Q1:{q1['n_pass']}/{q1['n_total']} med={q1['median_cnsr']:.3f}"
        q2_s  = f"p={q2.get('pvalue','?')}" if not args.no_q2 else "Q2:skip"
        q4_s  = f"DSR={q4['dsr']}"
        print(f"   CNSR={m['cnsr_usd_fed']:.3f} | {q1_s} | {q2_s} | {q4_s}")
        print(f"   → {verdict} | D-SIG:{dsig.score}/100 {dsig.label} {dsig.color}\n")

    # Table finale
    print(f"\n{'='*70}")
    print("  TABLE COMPARATIVE — résultats OOS")
    print(f"{'='*70}")
    print(f"{'Lentille':<22} {'CNSR':>7} {'MDD':>6} {'Trades':>7} "
          f"{'Q1':>7} {'Q2':>8} {'DSR':>6} {'Score':>6}  Verdict")
    print("-" * 85)
    rows = []
    for name, r in all_results.items():
        m    = r["oos_metrics"]
        cnsr = m.get("cnsr_usd_fed", 0)
        mdd  = m.get("max_dd_pct", 0)
        tr   = m.get("n_trades", 0)
        q1s  = f"{r['q1']['n_pass']}/{r['q1']['n_total']}"
        q2s  = f"p={r['q2'].get('pvalue','?')}" if r['q2']['verdict']!="SKIPPED" else "skip"
        dsrs = str(r["q4"]["dsr"])
        sc   = r["dsig"]["score"]
        print(f"{name:<22} {cnsr:>7.3f} {mdd:>6.1f} {tr:>7} "
              f"{q1s:>7} {q2s:>8} {dsrs:>6} {sc:>6}  {r['verdict']}")
        rows.append({"lentille":name,"cnsr_oos":cnsr,"mdd_pct":mdd,
                     "n_trades":tr,"q1":q1s,"q2_pvalue":r['q2'].get('pvalue'),
                     "dsr":r['q4']['dsr'],"dsig_score":sc,"verdict":r['verdict']})

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    pd.DataFrame(rows).to_csv(RESULTS_DIR / f"{SESSION_ID}_results.csv", index=False)
    atomic_save(RESULTS_DIR / f"{SESSION_ID}_results.yaml",
                {"session": SESSION_ID, "timestamp": ts, "results": all_results})
    print(f"\n📁 Résultats → {RESULTS_DIR}")

    summary = {"session": SESSION_ID, "timestamp": ts,
               "lentilles": {n: {"verdict":r["verdict"],
                                 "cnsr_oos":r["oos_metrics"].get("cnsr_usd_fed"),
                                 "dsig":r["dsig"]["score"],
                                 "q1_pass":r["q1"]["n_pass"]}
                             for n,r in all_results.items()}}
    print("\n=== Log JSON ===")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()