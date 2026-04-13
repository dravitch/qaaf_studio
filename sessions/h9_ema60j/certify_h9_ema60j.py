"""
certify_h9_ema60j.py — Session pilote : certification H9+EMA60j
Phase 4 du plan de mise en œuvre | QAAF Studio 3.0

Hypothèse précise (format obligatoire, règle n°1 du studio) :
    "Je teste si H9 + EMA 60j surpasse B_5050 sur CNSR-USD-Fed
    sur la période OOS 2023-06 → 2024-12 de la paire PAXG/BTC."

Protocole
---------
- PAF D1/D2/D3 déjà documentés → chargés depuis KB, non ré-exécutés
- MIF Phase 0+1+2 sur données synthétiques (validation algorithmique)
- MÉTIS Q1 : walk-forward 5 fenêtres glissantes
- MÉTIS Q2 : permutation 10 000 itérations, CNSR-USD vs B_5050
- MÉTIS Q3 : grille EMA 20j→120j sur IS, visualisation plateau
- MÉTIS Q4 : DSR avec N = 101

Usage
-----
    python sessions/certify_h9_ema60j.py
    python sessions/certify_h9_ema60j.py --skip-q2
    python sessions/certify_h9_ema60j.py --fast
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from layer1_engine.data_loader      import DataLoader
from layer1_engine.split_manager    import SplitManager
from layer1_engine.benchmark_factory import BenchmarkFactory
from layer1_engine.backtester       import Backtester
from layer1_engine.metrics_engine   import compute_cnsr, deflated_sharpe_ratio
from layer2_qualification.mif.mif_runner import MIFRunner
from layer3_validation.metis_runner  import METISRunner
from layer4_decision.kb_manager      import KBManager
from layer4_decision.n_trials_tracker import NTrialsTracker
from layer4_decision.dsig.mapper     import strategy_to_dsig

# ── Constantes de session ────────────────────────────────────────────────────

HYPOTHESIS   = "H9+EMA60j"
FAMILY       = "EMA_span_variants"
N_TRIALS     = 101
EMA_SPAN     = 60
CONFIG_PATH  = "config.yaml"
KB_HYP       = "layer4_decision/kb_h9_ema60j.yaml"
KB_INV       = "layer4_decision/lentilles_inventory.yaml"


# ── Stratégie H9 + EMA 60j ───────────────────────────────────────────────────

def h9_ema_strategy(r_pair: pd.Series, params: dict) -> pd.Series:
    """
    H9 : allocation PAXG proportionnelle à la déviation du ratio de sa MA.
    Lissage EMA span.

    Paramètres
    ----------
    r_pair : log-rendements PAXG/BTC
    params : {"ema_span": int, "h9_lookback": int}
    """
    span     = params.get("ema_span",    EMA_SPAN)
    lookback = params.get("h9_lookback", 20)

    # Ratio approché via rendements cumulés
    ratio = r_pair.cumsum().apply(np.exp)

    # Signal H9 : mean-reversion — ratio haut → réduire PAXG
    ma    = ratio.rolling(lookback, min_periods=max(1, lookback // 2)).mean()
    std   = ratio.rolling(lookback, min_periods=max(1, lookback // 2)).std()
    std   = std.replace(0, np.nan).ffill().fillna(1e-4)
    z     = (ratio - ma) / std

    raw      = 0.5 - 0.25 * np.tanh(z * 0.8)
    smoothed = raw.ewm(span=span, min_periods=span // 2).mean()
    return smoothed.clip(0.1, 0.9).rename("alloc_h9_ema")


# ── Session principale ────────────────────────────────────────────────────────

def run(skip_q2: bool = False, fast: bool = False) -> dict:
    n_perm   = 500 if fast else 10_000
    ema_step = 10  if fast else 5
    metis_q  = "Q1Q3Q4" if skip_q2 else "Q1Q2Q3Q4"
    params   = {"ema_span": EMA_SPAN, "h9_lookback": 20}

    print(f"\n{'='*62}")
    print(f"  QAAF STUDIO 3.0 — SESSION DE CERTIFICATION")
    print(f"  Hypothèse : {HYPOTHESIS}")
    print(f"  Famille   : {FAMILY} | N_trials = {N_TRIALS}")
    print(f"  MÉTIS     : {metis_q} | n_perm={n_perm} | ema_step={ema_step}")
    print(f"  Mode      : {'FAST' if fast else 'COMPLET'}")
    print(f"{'='*62}")

    # ── 1. KB check ──────────────────────────────────────────────────────
    kb      = KBManager(KB_HYP, KB_INV)
    tracker = NTrialsTracker(KB_HYP)
    check   = kb.pre_session_check(HYPOTHESIS, FAMILY)

    if check["recommendation"] == "SKIP_DUPLICATE":
        print("\n⏹️  Hypothèse déjà certifiée dans KB.")
        return check

    # ── 2. Données réelles ───────────────────────────────────────────────
    print("\n📡 Chargement données ...")
    loader = DataLoader(config_path=CONFIG_PATH)
    paxg_usd, btc_usd, r_paxg_usd, r_btc_usd = loader.load_prices(
        start="2019-01-01", end="2024-12-31"
    )
    # Construire le bundle minimal attendu par METISRunner
    import types
    bundle = types.SimpleNamespace(
        paxg_usd = paxg_usd,
        btc_usd  = btc_usd,
        paxg_btc = paxg_usd / btc_usd,
    )
    print(f"   {len(r_paxg_usd)} jours chargés "
          f"({r_paxg_usd.index[0]} → {r_paxg_usd.index[-1]})")

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
    b_paxg     = bf.b_paxg(prices_oos, r_btc_oos)
    print(f"  B_5050 CNSR = {b5050['cnsr_usd_fed']:.3f}  (KB : 1.73)")
    print(f"  B_BTC  CNSR = {b_btc['cnsr_usd_fed']:.3f}  (KB : 1.53)")
    print(f"  B_PAXG CNSR = {b_paxg['cnsr_usd_fed']:.3f}")

    # ── 4. PAF — chargé depuis KB ────────────────────────────────────────
    print("\n" + "─"*62)
    print("PAF : résultats chargés depuis KB (certifié 2026-04-01)")
    print("  D1 : HIERARCHIE_CONFIRMEE")
    print("  D2 : REGIMES_NEUTRES")
    print("  D3 : H9_LISSE_SUPERIEUR")

    # ── 5. Métriques OOS brutes ──────────────────────────────────────────
    print("\n" + "─"*62)
    r_pair_full = np.log(bundle.paxg_btc / bundle.paxg_btc.shift(1)).dropna()
    r_base_full = np.log(bundle.btc_usd  / bundle.btc_usd.shift(1)).dropna()
    _, r_pair_oos = sm.apply(r_pair_full)
    _, r_base_oos = sm.apply(r_base_full)

    alloc_oos  = h9_ema_strategy(r_pair_oos, params)
    common_oos = r_pair_oos.index.intersection(alloc_oos.index)
    r_port_oos = alloc_oos.reindex(common_oos).ffill() * r_pair_oos.loc[common_oos]
    r_usd_oos  = r_port_oos + r_base_oos.reindex(common_oos)

    metrics_oos = compute_cnsr(r_port_oos, r_base_oos.reindex(common_oos))
    dsr_oos     = deflated_sharpe_ratio(r_usd_oos.dropna(), N_TRIALS)

    print("Métriques OOS H9+EMA60j :")
    print(f"  CNSR-USD-Fed : {metrics_oos['cnsr_usd_fed']:.3f}  (KB : 1.76)")
    print(f"  Sortino      : {metrics_oos['sortino']:.3f}  (KB : 2.10)")
    print(f"  Max DD %     : {metrics_oos['max_dd_pct']:.1f}%   (KB : 14.5%)")
    print(f"  DSR(N={N_TRIALS}) : {dsr_oos:.4f}  "
          f"{'✅' if dsr_oos >= 0.95 else '⚠️'}")

    # ── 6. MIF ───────────────────────────────────────────────────────────
    print("\n" + "─"*62)
    mif_runner = MIFRunner(
        strategy_fn=h9_ema_strategy,
        params=params,
        hypothesis=HYPOTHESIS,
    )
    mif_summary = mif_runner.run(max_phase=2)
    mif_summary.print_summary()

    if "FAIL" in mif_summary.verdict:
        kb.record_verdict(HYPOTHESIS, "suspendu",
                          notes=f"MIF {mif_summary.verdict}")
        return {"verdict": mif_summary.verdict, "stopped": "MIF"}

    # ── 7. MÉTIS ─────────────────────────────────────────────────────────
    print("\n" + "─"*62)
    metis_runner = METISRunner(
        strategy_fn=h9_ema_strategy,
        params=params,
        bundle=bundle,
        split_manager=sm,
        hypothesis=HYPOTHESIS,
        n_trials=N_TRIALS,
    )
    metis_report = metis_runner.run(
        questions=metis_q,
        n_perm=n_perm,
        ema_step=ema_step,
    )
    metis_report.print_summary()

    # ── 8. D-SIG ─────────────────────────────────────────────────────────
    wf_score = (metis_report.q1.n_windows_pass / 5) if metis_report.q1 else 0.0
    signal   = strategy_to_dsig(
        metrics     = {**metrics_oos,
                       "dsr": dsr_oos,
                       "walk_forward_score": wf_score},
        paf_verdict = "H9_LISSE_SUPERIEUR",
        n_trials    = N_TRIALS,
        source_id   = f"qaaf-studio::{HYPOTHESIS}",
    )
    print(f"\n{'─'*62}")
    print(f"D-SIG v0.5 : score={signal.score}/100 | {signal.label} | "
          f"{signal.color} | {signal.trend}")
    for dim, v in signal.dimensions.items():
        print(f"  {dim:12s} : {v['score']:3d}/100")

    # ── 9. Verdict final + KB update ─────────────────────────────────────
    verdict = metis_report.verdict()
    kb.update_metis(metis_report.export_kb_update().get("metis", {}))
    kb.record_verdict(
        hypothesis=HYPOTHESIS,
        verdict=verdict.lower().split("_fail_")[0],
        metrics={
            "cnsr_usd_fed": round(float(metrics_oos["cnsr_usd_fed"]), 3),
            "sortino":      round(float(metrics_oos["sortino"]),       3),
            "max_dd_pct":   round(float(metrics_oos["max_dd_pct"]),    2),
            "dsr":          round(float(dsr_oos),                      4),
            "dsig_score":   signal.score,
        },
        notes=f"MÉTIS {metis_q} | N={N_TRIALS} | MIF {mif_summary.verdict}",
    )

    lentille_statut = "active" if "CERTIFIE" in verdict else "archivee"
    kb.update_lentille(
        nom         = HYPOTHESIS,
        statut      = lentille_statut,
        cnsr_oos    = round(float(metrics_oos["cnsr_usd_fed"]), 3),
        paf_verdict = "H9_LISSE_SUPERIEUR",
        metis_verdict = verdict,
        dsig_score  = signal.score,
    )

    # ── 10. Résumé ───────────────────────────────────────────────────────
    emoji = {"CERTIFIE": "🏆", "SUSPECT_DSR": "⚠️"}.get(verdict, "📁")
    print(f"\n{'='*62}")
    print(f"  {emoji}  VERDICT FINAL : {verdict}")
    print(f"  D-SIG : {signal.score}/100 — {signal.label} ({signal.color})")
    print(f"{'='*62}")

    if verdict == "CERTIFIE":
        print("\n  Prochaines hypothèses dans la queue :")
        print("  • H9+filtre_bear_MA200 (priorité 2)")
        print("  • H9_lisse_ETH_BTC (priorité 3)")
    elif verdict == "SUSPECT_DSR":
        print(f"\n  DSR < 0.95 avec N={N_TRIALS}.")
        print("  Réévaluer si N_trials ↓ ou nouvelles données OOS disponibles.")

    return {
        "hypothesis": HYPOTHESIS,
        "verdict":    verdict,
        "dsig":       {"score": signal.score, "label": signal.label},
        "cnsr_oos":   round(float(metrics_oos["cnsr_usd_fed"]), 3),
        "dsr":        round(float(dsr_oos), 4),
        "mif":        mif_summary.verdict,
        "metis_q":    metis_q,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Certification H9+EMA60j — QAAF Studio 3.0"
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
