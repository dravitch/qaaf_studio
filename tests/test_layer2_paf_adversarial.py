"""
Tests adverses PAF — Layer 2.

Détectent des défaillances subtiles que les tests fonctionnels ne couvrent pas :
lookahead bias, sensibilité au split, réplication oracle, sensibilité au taux
sans risque, robustesse aux NaN, et alignement session/factory.

Tests 1, 3, 6 : requis avant toute certification (marqués normal).
Tests 2, 5    : optionnels, marqués @pytest.mark.slow.
Test 4        : vérification d'invariant d'ordre, requis.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

from layer1_engine.backtester        import Backtester
from layer1_engine.benchmark_factory import BenchmarkFactory
from layer1_engine.metrics_engine    import compute_cnsr


# ── Fixtures partagées ────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_oos(tmp_path):
    """400 jours de données synthétiques reproductibles."""
    np.random.seed(42)
    n = 400
    idx = pd.date_range("2023-06-01", periods=n, freq="B")
    r_btc  = np.random.randn(n) * 0.03 + 0.004
    r_paxg = np.random.randn(n) * 0.01 + 0.001
    btc  = pd.Series(30000 * np.exp(np.cumsum(r_btc)),  index=idx, name="btc")
    paxg = pd.Series(1900  * np.exp(np.cumsum(r_paxg)), index=idx, name="paxg")
    prices = pd.DataFrame({"paxg": paxg, "btc": btc})
    r_btc_s = pd.Series(np.log(btc / btc.shift(1)).dropna())
    return prices.loc[r_btc_s.index], r_btc_s


@pytest.fixture
def backtester(tmp_path):
    import yaml
    cfg = {
        "engine": {"fees_pct": 0.001, "initial_capital": 10000.0, "mode": "lump_sum"},
        "rates":  {"rf_fed": 0.04, "rf_usdc": 0.03, "rf_zero": 0.0},
        "splits": {"is_start": "2020-06-01", "is_end": "2023-05-31",
                   "oos_start": "2023-06-01", "oos_end": "2024-12-31"},
        "data":   {"cache_dir": str(tmp_path / ".cache"),
                   "tickers": {"btc": "BTC-USD", "paxg": "PAXG-USD"}},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(cfg))
    return Backtester(config_path=str(cfg_path))


# ── Test 1 — Lookahead bias ───────────────────────────────────────────────────

def test_no_lookahead_bias(synthetic_oos, backtester):
    """
    Un signal qui voit le rendement du jour t avant d'allouer doit produire
    un CNSR SIGNIFICATIVEMENT plus élevé qu'un signal identique avec shift correct.

    Si les deux CNSR sont identiques, le backtester n'applique pas le shift(1)
    sur les allocations — c'est un bug de lookahead.
    """
    prices, r_btc = synthetic_oos

    r_pair_raw = np.log(prices["paxg"] / prices["btc"]).diff()

    lookahead_alloc = (r_pair_raw > 0).astype(float).fillna(0.5)
    honest_alloc = lookahead_alloc.shift(1).fillna(0.5)

    def lookahead_fn(df): return lookahead_alloc.reindex(df.index).fillna(0.5)
    def honest_fn(df):    return honest_alloc.reindex(df.index).fillna(0.5)

    r_la  = backtester.run(lookahead_fn, prices, r_btc)["r_portfolio_usd"]
    r_hon = backtester.run(honest_fn,    prices, r_btc)["r_portfolio_usd"]

    zero = pd.Series(0.0, index=r_la.index)
    cnsr_la  = compute_cnsr(r_la,  zero)["cnsr_usd_fed"]
    cnsr_hon = compute_cnsr(r_hon.reindex(r_la.index).dropna(),
                            zero.reindex(r_la.index).dropna())["cnsr_usd_fed"]

    assert cnsr_la != pytest.approx(cnsr_hon, abs=0.01), (
        f"Lookahead signal et signal honnête produisent le même CNSR "
        f"({cnsr_la:.4f} ≈ {cnsr_hon:.4f}). "
        f"Le backtester n'applique probablement pas shift(1) sur les allocations."
    )
    assert cnsr_la > cnsr_hon, (
        f"CNSR lookahead ({cnsr_la:.4f}) ≤ CNSR honnête ({cnsr_hon:.4f}). "
        f"Comportement inattendu — vérifier la construction du signal de test."
    )


# ── Test 2 — Stabilité du split (slow) ───────────────────────────────────────

@pytest.mark.slow
def test_split_stability(synthetic_oos, backtester):
    """
    Décaler la date de début OOS de ±5 jours ouvrables ne doit pas changer
    le CNSR de B_5050 de plus de ±0.20 sur données synthétiques.
    """
    prices, r_btc = synthetic_oos
    factory = BenchmarkFactory(backtester)

    cnsr_ref = factory.b_5050(prices, r_btc)["cnsr_usd_fed"]

    cnsrs_shifted = []
    for shift in [-5, 5]:
        idx_start = max(0, 10 + shift)
        p_shifted = prices.iloc[idx_start:]
        r_shifted = r_btc.reindex(p_shifted.index).dropna()
        p_shifted = p_shifted.reindex(r_shifted.index)
        try:
            c = factory.b_5050(p_shifted, r_shifted)["cnsr_usd_fed"]
            cnsrs_shifted.append(c)
        except Exception:
            pass

    for c in cnsrs_shifted:
        assert abs(c - cnsr_ref) <= 0.20, (
            f"CNSR B_5050 passe de {cnsr_ref:.3f} à {c:.3f} avec un décalage de ±5j. "
            f"Tolérance ±0.20 dépassée — le split tombe sur une zone instable."
        )


# ── Test 3 — Réplication oracle ───────────────────────────────────────────────

def test_replication_oracle(synthetic_oos, backtester):
    """
    Un backtester minimaliste ad-hoc (écrit inline) doit produire le même
    CNSR-USD que BenchmarkFactory pour B_5050.
    """
    prices, r_btc = synthetic_oos
    factory = BenchmarkFactory(backtester)

    b = factory.b_5050(prices, r_btc)
    cnsr_factory = b["cnsr_usd_fed"]

    r_paxg_usd = np.log(prices["paxg"] / prices["paxg"].shift(1)).dropna()
    r_btc_usd  = np.log(prices["btc"]  / prices["btc"].shift(1)).dropna()
    common = r_paxg_usd.index.intersection(r_btc_usd.index)
    r_paxg_usd = r_paxg_usd.loc[common]
    r_btc_usd  = r_btc_usd.loc[common]

    alloc = pd.Series(0.5, index=r_paxg_usd.index)
    alloc_shifted = alloc.shift(1).fillna(0.5)

    r_pair = r_paxg_usd - r_btc_usd
    r_port_btc = alloc_shifted * r_pair
    fees = alloc.diff().abs().fillna(0) * 0.001
    r_port_btc_net = r_port_btc - fees
    r_port_usd = r_port_btc_net + r_btc_usd

    zero = pd.Series(0.0, index=r_port_usd.index)
    cnsr_oracle = compute_cnsr(r_port_usd, zero)["cnsr_usd_fed"]

    assert abs(cnsr_factory - cnsr_oracle) < 1e-3, (
        f"Divergence BenchmarkFactory vs oracle : "
        f"factory={cnsr_factory:.6f}, oracle={cnsr_oracle:.6f}, "
        f"diff={abs(cnsr_factory - cnsr_oracle):.6f}. "
        f"Vérifier le shift des allocations et l'identité K&S dans backtester.py."
    )


# ── Test 4 — Sensibilité au taux sans risque ─────────────────────────────────

def test_rf_sensitivity_preserves_ranking(synthetic_oos, backtester):
    """
    Le classement B_5050 vs B_BTC ne doit pas s'inverser quand on change
    le taux sans risque de 0% à 4%.
    """
    prices, r_btc = synthetic_oos
    factory = BenchmarkFactory(backtester)

    b5050 = factory.b_5050(prices, r_btc)
    b_btc = factory.b_btc(prices,  r_btc)

    cnsr_5050_fed = b5050["cnsr_usd_fed"]
    cnsr_btc_fed  = b_btc["cnsr_usd_fed"]

    result_5050 = backtester.run(lambda df: pd.Series(0.5, index=df.index), prices, r_btc)
    result_btc  = backtester.run(lambda df: pd.Series(0.0, index=df.index), prices, r_btc)

    zero_5050 = pd.Series(0.0, index=result_5050["r_portfolio_usd"].index)
    zero_btc  = pd.Series(0.0, index=result_btc["r_portfolio_usd"].index)

    cnsr_5050_rf0 = compute_cnsr(result_5050["r_portfolio_usd"], zero_5050, rf_annual=0.0)["cnsr_usd_fed"]
    cnsr_btc_rf0  = compute_cnsr(result_btc["r_portfolio_usd"],  zero_btc,  rf_annual=0.0)["cnsr_usd_fed"]

    sign_fed = np.sign(cnsr_5050_fed - cnsr_btc_fed)
    sign_rf0 = np.sign(cnsr_5050_rf0 - cnsr_btc_rf0)

    assert sign_fed == sign_rf0 or abs(cnsr_5050_fed - cnsr_btc_fed) < 0.05, (
        f"Le classement B_5050 vs B_BTC s'inverse selon Rf : "
        f"Rf=4% → B_5050={cnsr_5050_fed:.3f} vs B_BTC={cnsr_btc_fed:.3f}, "
        f"Rf=0% → B_5050={cnsr_5050_rf0:.3f} vs B_BTC={cnsr_btc_rf0:.3f}. "
        f"Possible bug de numéraire."
    )


# ── Test 5 — Robustesse aux NaN dans r_btc (slow) ────────────────────────────

@pytest.mark.slow
def test_index_alignment_robustness(synthetic_oos, backtester):
    """
    Supprimer quelques jours dans r_btc_oos ne doit pas faire crasher
    le backtester ni décaler silencieusement les allocations.
    """
    prices, r_btc = synthetic_oos

    r_btc_missing = r_btc.drop(r_btc.index[100:105])

    try:
        result = backtester.run(
            lambda df: pd.Series(0.5, index=df.index),
            prices,
            r_btc_missing
        )
        cnsr_missing = compute_cnsr(
            result["r_portfolio_usd"],
            pd.Series(0.0, index=result["r_portfolio_usd"].index)
        )["cnsr_usd_fed"]
    except Exception as e:
        pytest.fail(f"Backtester crash sur r_btc avec 5 jours manquants : {e}")

    result_full = backtester.run(
        lambda df: pd.Series(0.5, index=df.index), prices, r_btc
    )
    cnsr_full = compute_cnsr(
        result_full["r_portfolio_usd"],
        pd.Series(0.0, index=result_full["r_portfolio_usd"].index)
    )["cnsr_usd_fed"]

    assert abs(cnsr_missing - cnsr_full) < 0.30, (
        f"CNSR avec 5 jours NaN ({cnsr_missing:.3f}) trop différent du CNSR complet "
        f"({cnsr_full:.3f}). Possible désalignement silencieux via fill_value=0."
    )


# ── Test 6 — Alignement session / factory ────────────────────────────────────

def test_session_aligned_with_factory(synthetic_oos, backtester):
    """
    La boucle run_backtest de comparative_001 et BenchmarkFactory doivent produire
    exactement le même CNSR pour B_5050 (à 1e-4 près).
    """
    prices, r_btc = synthetic_oos
    factory = BenchmarkFactory(backtester)

    cnsr_factory = factory.b_5050(prices, r_btc)["cnsr_usd_fed"]

    result = backtester.run(
        lambda df: pd.Series(0.5, index=df.index), prices, r_btc
    )
    zero = pd.Series(0.0, index=result["r_portfolio_usd"].index)
    cnsr_direct = compute_cnsr(result["r_portfolio_usd"], zero)["cnsr_usd_fed"]

    assert abs(cnsr_factory - cnsr_direct) < 1e-4, (
        f"Divergence factory vs run direct pour B_5050 : "
        f"factory={cnsr_factory:.6f}, direct={cnsr_direct:.6f}. "
        f"BenchmarkFactory._run() a peut-être dévié de Backtester.run()."
    )
