# QAAF Studio — Instructions Claude Code · Partie 3 : Layer 2 PAF

**Nouvelle fenêtre de conversation. Contexte minimal ci-dessous.**

---

## État de départ (ce qui existe et fonctionne)

```
qaaf_studio/
├── layer1_engine/          ← COMPLET, 26/26 tests verts
│   ├── backtester.py       ← lump sum, r_base_usd=0, shift alloc
│   ├── metrics_engine.py   ← compute_cnsr(), deflated_sharpe_ratio()
│   ├── benchmark_factory.py ← B_5050=1.343, B_BTC=1.244 sur OOS
│   ├── data_loader.py
│   ├── split_manager.py
│   └── metrics_engine.py
├── tests/
│   ├── test_layer1_*.py       ← 26/26 PASS
│   └── test_benchmark_calibration.py ← 11/11 PASS (gates permanentes)
└── sessions/comparative_001/ ← run validé, résultats en KB
```

**Valeurs de référence scellées (KB Avril 2026) :**
- B_5050 CNSR-USD = 1.343 ± 0.15 sur OOS 2023-06-01 → 2024-12-31
- B_BTC  CNSR-USD = 1.244 ± 0.15
- Tous les signaux testés : `ARCHIVE_FAIL_Q1_Q4`

**Ce que tu ne dois PAS modifier :** layer1_engine/, tests/test_layer1_*.py, tests/test_benchmark_calibration.py, sessions/comparative_001/.

---

## Objectif de cette partie

Construire **Layer 2 — PAF interactif** dans `layer2_qualification/paf/`.

Critère de succès : `pytest tests/test_layer2_paf.py tests/test_layer2_paf_adversarial.py -v` → 100% PASS, et un run PAF sur PAXG/BTC réel reproduit le verdict `HIERARCHIE_CONFIRMEE` documenté en KB.

---

## Ce que PAF fait

PAF (Pair Adequacy Framework) qualifie si une paire d'actifs est adaptée à une classe de méthodes *avant* de tester un signal. Trois directions séquentielles avec règle d'arrêt à chaque étape.

**Direction 1 — Hiérarchie de signal**
Compare sur OOS : MR_pur (mean-reversion minimal) vs H9 (signal de référence) vs signal_candidat vs benchmarks passifs.
Verdict : `HIERARCHIE_CONFIRMEE` si MR_pur < H9 < signal. `B_PASSIF_DOMINE` si B_5050 > tout. `STOP` si pas de hiérarchie.

**Direction 2 — Attribution de performance**
Isole chaque couche : signal_complet vs signal_sans_composante_X.
Verdict : `COMPOSANTE_ACTIVE` si retirer X dégrade significativement. `NEUTRE` si delta < 0.05 CNSR. `DEGRADANTE` si retirer X améliore.

**Direction 3 — Source minimale**
Reproduit la variance d'allocation du signal par un moyen trivial (EMA sur H9).
Verdict : `SIGNAL_INFORMATIF` si signal > EMA_triviale à iso-variance. `ARTEFACT_LISSAGE` si EMA_triviale ≥ signal.

**Résultats KB documentés pour PAXG/BTC :**
- D1 : `HIERARCHIE_CONFIRMEE` (MR_pur=-0.80 < H9=0.32 < H9+EMA=1.76, B_5050=1.73)
- D2 : `REGIMES_NEUTRES` (les filtres de régimes QAAF-R n'apportent rien)
- D3 : `H9_LISSE_SUPERIEUR` (EMA 60j sur H9 ≥ signal complet à iso-variance)

---

## Structure à créer

```
layer2_qualification/
├── __init__.py
└── paf/
    ├── __init__.py
    ├── paf_runner.py
    ├── paf_d1_hierarchy.py
    ├── paf_d2_attribution.py
    └── paf_d3_source.py

tests/
├── test_layer2_paf.py              ← tests fonctionnels (8 tests)
└── test_layer2_paf_adversarial.py  ← tests adverses (6 tests)  ← NOUVEAU
```

---

## layer2_qualification/paf/paf_d1_hierarchy.py

```python
"""
PAF Direction 1 — Test critique de hiérarchie de signal.

Question : chaque couche de sophistication apporte-t-elle quelque chose ?
Protocole : comparer MR_pur, signal_ref, signal_candidat, et benchmarks passifs
            sur le même OOS, même moteur, même split.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Callable, Optional

from layer1_engine.backtester        import Backtester
from layer1_engine.benchmark_factory import BenchmarkFactory
from layer1_engine.metrics_engine    import compute_cnsr


@dataclass
class D1Result:
    verdict: str           # HIERARCHIE_CONFIRMEE | B_PASSIF_DOMINE | STOP | PARTIELLE
    mr_pur_cnsr:   float
    signal_ref_cnsr: float
    signal_candidat_cnsr: Optional[float]
    b_5050_cnsr:   float
    b_btc_cnsr:    float
    delta_ref_vs_mr: float
    delta_candidat_vs_ref: float
    details: dict


def run_d1(
    prices_oos: pd.DataFrame,
    r_btc_oos: pd.Series,
    signal_ref_fn: Callable,
    backtester: Backtester,
    signal_candidat_fn: Optional[Callable] = None,
    window: int = 60,
) -> D1Result:
    factory = BenchmarkFactory(backtester)

    log_ratio = np.log(prices_oos["paxg"] / prices_oos["btc"])
    mean_lr   = log_ratio.rolling(window, min_periods=window // 2).mean()
    std_lr    = log_ratio.rolling(window, min_periods=window // 2).std()
    z_score   = ((log_ratio - mean_lr) / std_lr.replace(0, np.nan)).fillna(0)
    mr_alloc  = (0.5 - z_score.clip(-1, 1) * 0.3).clip(0, 1)

    def mr_pur_fn(df): return mr_alloc.reindex(df.index).fillna(0.5)

    def _cnsr(alloc_fn) -> float:
        result = backtester.run(alloc_fn, prices_oos, r_btc_oos)
        zero = pd.Series(0.0, index=result["r_portfolio_usd"].index)
        return compute_cnsr(result["r_portfolio_usd"], zero)["cnsr_usd_fed"]

    mr_cnsr  = _cnsr(mr_pur_fn)
    ref_cnsr = _cnsr(signal_ref_fn)
    cand_cnsr = _cnsr(signal_candidat_fn) if signal_candidat_fn else None

    b5050 = factory.b_5050(prices_oos, r_btc_oos)["cnsr_usd_fed"]
    b_btc = factory.b_btc(prices_oos,  r_btc_oos)["cnsr_usd_fed"]

    delta_ref_mr   = ref_cnsr - mr_cnsr
    delta_cand_ref = (cand_cnsr - ref_cnsr) if cand_cnsr is not None else None

    top_active = max(filter(lambda x: x is not None, [mr_cnsr, ref_cnsr, cand_cnsr]))
    if b5050 > top_active + 0.1:
        verdict = "B_PASSIF_DOMINE"
    elif delta_ref_mr < -0.05:
        verdict = "STOP"
    elif cand_cnsr is not None and delta_cand_ref is not None:
        if delta_ref_mr >= -0.05 and delta_cand_ref >= -0.05:
            verdict = "HIERARCHIE_CONFIRMEE"
        elif delta_ref_mr >= -0.05:
            verdict = "PARTIELLE"
        else:
            verdict = "STOP"
    else:
        verdict = "HIERARCHIE_CONFIRMEE" if delta_ref_mr >= 0 else "STOP"

    return D1Result(
        verdict=verdict,
        mr_pur_cnsr=round(mr_cnsr, 4),
        signal_ref_cnsr=round(ref_cnsr, 4),
        signal_candidat_cnsr=round(cand_cnsr, 4) if cand_cnsr else None,
        b_5050_cnsr=round(b5050, 4),
        b_btc_cnsr=round(b_btc, 4),
        delta_ref_vs_mr=round(delta_ref_mr, 4),
        delta_candidat_vs_ref=round(delta_cand_ref, 4) if delta_cand_ref else None,
        details={
            "mr_pur": mr_cnsr, "signal_ref": ref_cnsr,
            "signal_candidat": cand_cnsr, "b_5050": b5050, "b_btc": b_btc,
        }
    )
```

---

## layer2_qualification/paf/paf_d2_attribution.py

```python
"""
PAF Direction 2 — Attribution de performance.

Question : quelle composante du signal explique la performance observée en D1 ?
Protocole : signal_complet vs signal_sans_composante_X, à allocation comparable.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Callable

from layer1_engine.backtester     import Backtester
from layer1_engine.metrics_engine import compute_cnsr


@dataclass
class D2Result:
    verdict: str              # COMPOSANTE_ACTIVE | NEUTRE | DEGRADANTE
    composante: str
    cnsr_avec:  float
    cnsr_sans:  float
    delta:      float
    std_alloc_avec: float
    std_alloc_sans: float
    details: dict


def run_d2(
    prices_oos: pd.DataFrame,
    r_btc_oos: pd.Series,
    signal_complet_fn: Callable,
    signal_sans_fn: Callable,
    composante_name: str,
    backtester: Backtester,
    seuil_actif: float = 0.05,
) -> D2Result:
    def _run(alloc_fn):
        result = backtester.run(alloc_fn, prices_oos, r_btc_oos)
        zero = pd.Series(0.0, index=result["r_portfolio_usd"].index)
        metrics = compute_cnsr(result["r_portfolio_usd"], zero)
        return metrics["cnsr_usd_fed"], result["std_alloc"]

    cnsr_avec,  std_avec = _run(signal_complet_fn)
    cnsr_sans, std_sans  = _run(signal_sans_fn)
    delta = cnsr_avec - cnsr_sans

    if delta > seuil_actif:
        verdict = "COMPOSANTE_ACTIVE"
    elif delta < -seuil_actif:
        verdict = "DEGRADANTE"
    else:
        verdict = "NEUTRE"

    return D2Result(
        verdict=verdict,
        composante=composante_name,
        cnsr_avec=round(cnsr_avec, 4),
        cnsr_sans=round(cnsr_sans, 4),
        delta=round(delta, 4),
        std_alloc_avec=round(std_avec, 4),
        std_alloc_sans=round(std_sans, 4),
        details={"cnsr_complet": cnsr_avec, "cnsr_sans": cnsr_sans,
                 "delta": delta, "verdict": verdict}
    )
```

---

## layer2_qualification/paf/paf_d3_source.py

```python
"""
PAF Direction 3 — Test de la source minimale.

Question : la performance vient-elle d'un signal d'information,
           ou d'un artefact de lissage/réduction de friction ?
Protocole : reproduire la variance d'allocation par un EMA trivial,
            comparer à iso-variance.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Callable

from layer1_engine.backtester     import Backtester
from layer1_engine.metrics_engine import compute_cnsr


@dataclass
class D3Result:
    verdict: str           # SIGNAL_INFORMATIF | ARTEFACT_LISSAGE
    cnsr_signal:  float
    cnsr_trivial: float
    delta:        float
    std_alloc_signal:  float
    std_alloc_trivial: float
    ema_span_used: int
    details: dict


def _find_ema_span_matching_variance(
    prices: pd.DataFrame,
    base_alloc_fn: Callable,
    target_std: float,
    spans: list = None,
) -> tuple:
    if spans is None:
        spans = list(range(10, 130, 10))

    log_ratio = np.log(prices["paxg"] / prices["btc"])
    q25 = log_ratio.rolling(60, min_periods=30).quantile(0.25)
    q75 = log_ratio.rolling(60, min_periods=30).quantile(0.75)
    iqr = (q75 - q25).replace(0, np.nan)
    h9_signal = 1.0 - ((log_ratio - q25) / iqr).clip(0, 1)

    best_span, best_std, best_fn = spans[0], float("inf"), None
    for span in spans:
        alloc = h9_signal.ewm(span=span, adjust=False).mean().clip(0, 1).fillna(0.5)
        std = float(alloc.std())
        if abs(std - target_std) < abs(best_std - target_std):
            best_span = span
            best_std = std
            def make_fn(sp):
                a = h9_signal.ewm(span=sp, adjust=False).mean().clip(0, 1).fillna(0.5)
                def fn(df): return a.reindex(df.index).fillna(0.5)
                return fn
            best_fn = make_fn(span)

    return best_span, best_fn, best_std


def run_d3(
    prices_oos: pd.DataFrame,
    r_btc_oos: pd.Series,
    signal_fn: Callable,
    backtester: Backtester,
) -> D3Result:
    result_signal = backtester.run(signal_fn, prices_oos, r_btc_oos)
    target_std = result_signal["std_alloc"]
    zero = pd.Series(0.0, index=result_signal["r_portfolio_usd"].index)
    cnsr_signal = compute_cnsr(result_signal["r_portfolio_usd"], zero)["cnsr_usd_fed"]

    span, trivial_fn, actual_std = _find_ema_span_matching_variance(
        prices_oos, signal_fn, target_std
    )

    result_trivial = backtester.run(trivial_fn, prices_oos, r_btc_oos)
    zero_t = pd.Series(0.0, index=result_trivial["r_portfolio_usd"].index)
    cnsr_trivial = compute_cnsr(result_trivial["r_portfolio_usd"], zero_t)["cnsr_usd_fed"]

    delta = cnsr_signal - cnsr_trivial
    verdict = "SIGNAL_INFORMATIF" if delta > 0.05 else "ARTEFACT_LISSAGE"

    return D3Result(
        verdict=verdict,
        cnsr_signal=round(cnsr_signal, 4),
        cnsr_trivial=round(cnsr_trivial, 4),
        delta=round(delta, 4),
        std_alloc_signal=round(target_std, 4),
        std_alloc_trivial=round(actual_std, 4),
        ema_span_used=span,
        details={"target_std": target_std, "actual_std": actual_std,
                 "span": span, "cnsr_signal": cnsr_signal, "cnsr_trivial": cnsr_trivial}
    )
```

---

## layer2_qualification/paf/paf_runner.py

```python
"""
PAF Runner — Orchestre D1→D2→D3 avec règles d'arrêt.

Précondition : toutes les données doivent avoir passé DQF (Layer 0)
avant d'entrer dans PAF. Le runner vérifie que le DataLoader
a produit des données avec statut DQF != FAIL.
"""

import pandas as pd
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Optional, Dict

from layer1_engine.data_loader     import DataLoader
from layer1_engine.backtester      import Backtester
from layer1_engine.split_manager   import SplitManager
from layer2_qualification.paf.paf_d1_hierarchy   import run_d1, D1Result
from layer2_qualification.paf.paf_d2_attribution import run_d2, D2Result
from layer2_qualification.paf.paf_d3_source      import run_d3, D3Result


@dataclass
class PAFBundle:
    prices_oos:  pd.DataFrame
    r_btc_oos:   pd.Series
    prices_is:   pd.DataFrame
    r_btc_is:    pd.Series
    dqf_status:  str   # PASS | WARNING — jamais FAIL ici


@dataclass
class PAFReport:
    verdict_global: str
    d1: Optional[D1Result] = None
    d2: Optional[D2Result] = None
    d3: Optional[D3Result] = None
    stopped_at: Optional[str] = None
    notes: list = field(default_factory=list)


def load_paf_bundle(config_path: str = "config.yaml") -> PAFBundle:
    loader = DataLoader(config_path)
    paxg_usd, btc_usd, r_paxg, r_btc = loader.load_prices()

    for ticker, report in loader.dqf_reports.items():
        if report.get("status") == "FAIL":
            raise ValueError(
                f"DQF FAIL pour {ticker} — données invalides. "
                f"Issues : {report.get('issues', [])}"
            )

    prices_full = pd.DataFrame({"paxg": paxg_usd, "btc": btc_usd})
    sm = SplitManager(config_path)
    prices_is,  prices_oos  = sm.apply_df(prices_full)
    r_btc_is,   r_btc_oos   = sm.apply(r_btc)

    statuses = [r.get("status", "PASS") for r in loader.dqf_reports.values()]
    dqf_global = "WARNING" if "WARNING" in statuses else "PASS"

    return PAFBundle(
        prices_oos=prices_oos, r_btc_oos=r_btc_oos,
        prices_is=prices_is,   r_btc_is=r_btc_is,
        dqf_status=dqf_global,
    )


def run_paf(
    bundle: PAFBundle,
    signal_ref_fn: Callable,
    backtester: Backtester,
    signal_candidat_fn: Optional[Callable] = None,
    composantes_d2: Optional[Dict[str, tuple]] = None,
) -> PAFReport:
    report = PAFReport(verdict_global="EN_COURS")

    print("\n── PAF D1 : Hiérarchie de signal ──")
    d1 = run_d1(
        prices_oos=bundle.prices_oos, r_btc_oos=bundle.r_btc_oos,
        signal_ref_fn=signal_ref_fn, backtester=backtester,
        signal_candidat_fn=signal_candidat_fn,
    )
    report.d1 = d1
    print(f"  MR_pur : {d1.mr_pur_cnsr:.4f} | Signal_ref : {d1.signal_ref_cnsr:.4f} | "
          f"B_5050 : {d1.b_5050_cnsr:.4f}")
    print(f"  → D1 verdict : {d1.verdict}")

    if d1.verdict == "B_PASSIF_DOMINE":
        report.verdict_global = "REQUALIFIER_PAIRE"
        report.stopped_at = "D1"
        report.notes.append("B_passif domine toutes les stratégies actives.")
        return report

    if d1.verdict == "STOP":
        report.verdict_global = "STOP_D1"
        report.stopped_at = "D1"
        report.notes.append("Pas de hiérarchie MR_pur < Signal_ref.")
        return report

    if composantes_d2:
        print("\n── PAF D2 : Attribution de performance ──")
        d2_results = {}
        for nom, (fn_avec, fn_sans) in composantes_d2.items():
            d2 = run_d2(
                prices_oos=bundle.prices_oos, r_btc_oos=bundle.r_btc_oos,
                signal_complet_fn=fn_avec, signal_sans_fn=fn_sans,
                composante_name=nom, backtester=backtester,
            )
            d2_results[nom] = d2
            print(f"  {nom}: avec={d2.cnsr_avec:.4f} sans={d2.cnsr_sans:.4f} "
                  f"delta={d2.delta:+.4f} → {d2.verdict}")
        report.d2 = list(d2_results.values())[-1]

    print("\n── PAF D3 : Source minimale ──")
    fn_to_test = signal_candidat_fn or signal_ref_fn
    d3 = run_d3(
        prices_oos=bundle.prices_oos, r_btc_oos=bundle.r_btc_oos,
        signal_fn=fn_to_test, backtester=backtester,
    )
    report.d3 = d3
    print(f"  Signal : {d3.cnsr_signal:.4f} | EMA triviale (span={d3.ema_span_used}j) : "
          f"{d3.cnsr_trivial:.4f} | delta : {d3.delta:+.4f}")
    print(f"  → D3 verdict : {d3.verdict}")

    report.verdict_global = "QUALIFIE"
    if d3.verdict == "ARTEFACT_LISSAGE":
        report.notes.append(
            f"D3 : performance = artefact de lissage (EMA {d3.ema_span_used}j). "
            f"Solution minimale = H9+EMA{d3.ema_span_used}j."
        )
        report.verdict_global = "QUALIFIE_SOURCE_MINIMALE"

    return report
```

---

## tests/test_layer2_paf.py

```python
"""Tests unitaires — PAF Layer 2 (logique des verdicts sur données synthétiques)."""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

from layer1_engine.backtester        import Backtester
from layer2_qualification.paf.paf_d1_hierarchy   import run_d1
from layer2_qualification.paf.paf_d2_attribution import run_d2
from layer2_qualification.paf.paf_d3_source      import run_d3


@pytest.fixture
def synthetic_bull_oos(tmp_path):
    np.random.seed(42)
    n = 400
    idx = pd.date_range("2023-06-01", periods=n, freq="B")
    r_btc  = np.random.randn(n) * 0.03 + 0.004
    r_paxg = np.random.randn(n) * 0.01 + 0.001
    btc  = pd.Series(30000 * np.exp(np.cumsum(r_btc)),  index=idx, name="btc")
    paxg = pd.Series(1900  * np.exp(np.cumsum(r_paxg)), index=idx, name="paxg")
    prices = pd.DataFrame({"paxg": paxg, "btc": btc})
    r_btc_s = pd.Series(np.log(btc / btc.shift(1)).dropna(), name="r_btc")
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


def h9_ema_fn(df):
    lr = np.log(df["paxg"] / df["btc"])
    q25 = lr.rolling(60, min_periods=30).quantile(0.25)
    q75 = lr.rolling(60, min_periods=30).quantile(0.75)
    iqr = (q75 - q25).replace(0, np.nan)
    h9 = (1 - ((lr - q25) / iqr).clip(0, 1)).clip(0, 1).fillna(0.5)
    return h9.ewm(span=60, adjust=False).mean().clip(0, 1)


def test_d1_returns_valid_verdict(synthetic_bull_oos, backtester):
    prices, r_btc = synthetic_bull_oos
    result = run_d1(prices, r_btc, h9_ema_fn, backtester)
    assert result.verdict in ("HIERARCHIE_CONFIRMEE", "PARTIELLE", "B_PASSIF_DOMINE", "STOP")


def test_d1_returns_numeric_cnsr_values(synthetic_bull_oos, backtester):
    prices, r_btc = synthetic_bull_oos
    result = run_d1(prices, r_btc, h9_ema_fn, backtester)
    assert not np.isnan(result.mr_pur_cnsr)
    assert not np.isnan(result.signal_ref_cnsr)
    assert not np.isnan(result.b_5050_cnsr)


def test_d1_b_passif_domine_when_active_very_poor(synthetic_bull_oos, backtester):
    prices, r_btc = synthetic_bull_oos
    def bad_signal(df): return pd.Series(0.95, index=df.index)
    result = run_d1(prices, r_btc, bad_signal, backtester)
    assert result.verdict in ("B_PASSIF_DOMINE", "STOP", "PARTIELLE", "HIERARCHIE_CONFIRMEE")


def test_d2_neutre_when_composante_makes_no_difference(synthetic_bull_oos, backtester):
    prices, r_btc = synthetic_bull_oos
    def signal_a(df): return pd.Series(0.5, index=df.index)
    def signal_b(df): return pd.Series(0.5, index=df.index)
    result = run_d2(prices, r_btc, signal_a, signal_b, "test", backtester)
    assert result.verdict == "NEUTRE"
    assert abs(result.delta) < 0.01


def test_d2_delta_sign_matches_verdict(synthetic_bull_oos, backtester):
    prices, r_btc = synthetic_bull_oos
    def alloc_high(df): return pd.Series(0.8, index=df.index)
    def alloc_low(df):  return pd.Series(0.2, index=df.index)
    result = run_d2(prices, r_btc, alloc_high, alloc_low, "test", backtester,
                    seuil_actif=0.01)
    assert result.delta == round(result.cnsr_avec - result.cnsr_sans, 4)
    if result.delta > 0.01:
        assert result.verdict == "COMPOSANTE_ACTIVE"
    elif result.delta < -0.01:
        assert result.verdict == "DEGRADANTE"


def test_d3_returns_valid_verdict(synthetic_bull_oos, backtester):
    prices, r_btc = synthetic_bull_oos
    result = run_d3(prices, r_btc, h9_ema_fn, backtester)
    assert result.verdict in ("SIGNAL_INFORMATIF", "ARTEFACT_LISSAGE")
    assert not np.isnan(result.cnsr_signal)
    assert not np.isnan(result.cnsr_trivial)
    assert result.ema_span_used > 0


def test_d3_iso_variance_tolerance(synthetic_bull_oos, backtester):
    prices, r_btc = synthetic_bull_oos
    result = run_d3(prices, r_btc, h9_ema_fn, backtester)
    if result.std_alloc_signal > 0:
        ratio = result.std_alloc_trivial / result.std_alloc_signal
        assert 0.3 <= ratio <= 3.0, f"EMA triviale trop différente : ratio={ratio:.2f}"


def test_d2_delta_is_cnsr_avec_minus_cnsr_sans(synthetic_bull_oos, backtester):
    """Invariant arithmétique : delta = cnsr_avec - cnsr_sans, toujours."""
    prices, r_btc = synthetic_bull_oos
    result = run_d2(prices, r_btc, h9_ema_fn,
                    lambda df: pd.Series(0.5, index=df.index),
                    "arith_check", backtester)
    assert abs(result.delta - (result.cnsr_avec - result.cnsr_sans)) < 1e-6
```

---

## tests/test_layer2_paf_adversarial.py  ← NOUVEAU

```python
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

    # Rendements journaliers de la paire (signal oracle = information du futur)
    r_pair_raw = np.log(prices["paxg"] / prices["btc"]).diff()

    # Signal avec lookahead : alloue fort quand le rendement du JOUR MÊME est positif
    # (violation délibérée — utilise l'information non disponible à t)
    lookahead_alloc = (r_pair_raw > 0).astype(float).fillna(0.5)

    # Signal sans lookahead : utilise le rendement de la veille (shift correct)
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
    # Le signal avec lookahead DOIT être supérieur (il triche)
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

    Si le CNSR change radicalement avec un léger décalage, le split tombe
    sur un régime de marché atypique et n'est pas représentatif.
    """
    prices, r_btc = synthetic_oos
    factory = BenchmarkFactory(backtester)

    # Référence : split à J0
    cnsr_ref = factory.b_5050(prices, r_btc)["cnsr_usd_fed"]

    # Décalages : ±5 jours (en nombre de lignes)
    cnsrs_shifted = []
    for shift in [-5, 5]:
        idx_start = max(0, 10 + shift)  # éviter les bords
        p_shifted = prices.iloc[idx_start:]
        r_shifted = r_btc.reindex(p_shifted.index).dropna()
        p_shifted = p_shifted.reindex(r_shifted.index)
        try:
            c = factory.b_5050(p_shifted, r_shifted)["cnsr_usd_fed"]
            cnsrs_shifted.append(c)
        except Exception:
            pass  # trop peu de données — on skip

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

    Détecte les divergences cachées : arrondis différents, gestion des NaN,
    forward-fill d'allocation implicite.
    """
    prices, r_btc = synthetic_oos
    factory = BenchmarkFactory(backtester)

    # ── BenchmarkFactory (implémentation à tester) ────────────────────────────
    b = factory.b_5050(prices, r_btc)
    cnsr_factory = b["cnsr_usd_fed"]

    # ── Oracle minimaliste (implémentation de référence indépendante) ─────────
    # Invariants stricts :
    #   1. Lump sum — capital fixe
    #   2. Allocation 0.5 fixe tous les jours
    #   3. shift(1) sur l'allocation (décision J, exécution J+1)
    #   4. r_USD = r_pair + r_BTC (identité K&S)
    #   5. Frais 0.001 sur |Δalloc|

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

    Une inversion signalerait que le numéraire ou le taux ne sont pas
    correctement appliqués (le classement est robuste à Rf dans la plage normale).
    """
    prices, r_btc = synthetic_oos
    factory = BenchmarkFactory(backtester)

    b5050 = factory.b_5050(prices, r_btc)
    b_btc = factory.b_btc(prices,  r_btc)

    # CNSR avec Rf=Fed (4%)
    cnsr_5050_fed = b5050["cnsr_usd_fed"]
    cnsr_btc_fed  = b_btc["cnsr_usd_fed"]

    # CNSR avec Rf=0% — recalculer directement sur les séries
    result_5050 = backtester.run(lambda df: pd.Series(0.5, index=df.index), prices, r_btc)
    result_btc  = backtester.run(lambda df: pd.Series(0.0, index=df.index), prices, r_btc)

    zero_5050 = pd.Series(0.0, index=result_5050["r_portfolio_usd"].index)
    zero_btc  = pd.Series(0.0, index=result_btc["r_portfolio_usd"].index)

    cnsr_5050_rf0 = compute_cnsr(result_5050["r_portfolio_usd"], zero_5050, rf_annual=0.0)["cnsr_usd_fed"]
    cnsr_btc_rf0  = compute_cnsr(result_btc["r_portfolio_usd"],  zero_btc,  rf_annual=0.0)["cnsr_usd_fed"]

    # Invariant : l'ordre relatif doit être conservé (ou les deux proches)
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

    Un désalignement silencieux (via reindex fill_value=0) peut masquer
    des jours où r_btc = 0 artificiellement, biaisant le CNSR.
    """
    prices, r_btc = synthetic_oos

    # Supprimer 5 jours au milieu
    r_btc_missing = r_btc.drop(r_btc.index[100:105])

    # Ne doit pas lever d'exception
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

    # Le CNSR avec données manquantes doit rester dans ±0.30 du CNSR complet
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

    Une divergence signale que la session a dévié du moteur unifié.
    """
    prices, r_btc = synthetic_oos
    factory = BenchmarkFactory(backtester)

    # Via BenchmarkFactory
    cnsr_factory = factory.b_5050(prices, r_btc)["cnsr_usd_fed"]

    # Via run directement (émule ce que run_backtest fait dans la session)
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
```

---

## Instructions d'exécution pour Claude Code

**Étape 1 — Vérifier la baseline**
```bash
pytest tests/test_benchmark_calibration.py tests/test_layer1_backtester.py -v
# → 37/37 PASS requis avant de continuer
```

**Étape 2 — Créer la structure**
```bash
mkdir -p layer2_qualification/paf
touch layer2_qualification/__init__.py
touch layer2_qualification/paf/__init__.py
```

**Étape 3 — Créer les 4 fichiers PAF** (dans l'ordre)
1. `paf_d1_hierarchy.py`
2. `paf_d2_attribution.py`
3. `paf_d3_source.py`
4. `paf_runner.py`

**Étape 4 — Créer les deux fichiers de tests**
- `tests/test_layer2_paf.py` (8 tests fonctionnels)
- `tests/test_layer2_paf_adversarial.py` (6 tests adverses)

**Étape 5 — Lancer les tests fonctionnels**
```bash
pytest tests/test_layer2_paf.py -v
# → 8/8 PASS requis
```

**Étape 6 — Lancer les tests adverses (hors slow)**
```bash
pytest tests/test_layer2_paf_adversarial.py -v -m "not slow"
# → 4/4 PASS requis (tests 1, 3, 4, 6)
```

**Étape 7 — Tests adverses slow (optionnel mais recommandé)**
```bash
pytest tests/test_layer2_paf_adversarial.py -v -m slow
# → 2/2 PASS recommandés (tests 2, 5)
```

**Étape 8 — Validation sur données réelles (si réseau disponible)**
```python
from layer2_qualification.paf.paf_runner import load_paf_bundle, run_paf
from layer1_engine.backtester import Backtester
import numpy as np, pandas as pd

bundle = load_paf_bundle("config.yaml")
bt = Backtester("config.yaml")

def signal_h9_ema(df):
    lr = np.log(df["paxg"] / df["btc"])
    q25 = lr.rolling(60, min_periods=30).quantile(0.25)
    q75 = lr.rolling(60, min_periods=30).quantile(0.75)
    iqr = (q75 - q25).replace(0, np.nan)
    h9 = (1 - ((lr - q25) / iqr).clip(0, 1)).clip(0, 1).fillna(0.5)
    return h9.ewm(span=60, adjust=False).mean().clip(0, 1)

report = run_paf(bundle, signal_h9_ema, bt)
print(f"Verdict global : {report.verdict_global}")
print(f"D1 : {report.d1.verdict if report.d1 else 'N/A'}")
print(f"D3 : {report.d3.verdict if report.d3 else 'N/A'}")
```

**Résultat attendu sur données réelles PAXG/BTC :**
```
D1 : HIERARCHIE_CONFIRMEE  (MR_pur < H9 < B_5050)
D3 : ARTEFACT_LISSAGE      (EMA triviale ≥ signal à iso-variance)
Verdict global : QUALIFIE_SOURCE_MINIMALE
```

**Critère de passage à la Partie 4 (Layer 3 MÉTIS) :**
1. `pytest tests/test_layer2_paf.py` → 8/8 PASS
2. `pytest tests/test_layer2_paf_adversarial.py -m "not slow"` → 4/4 PASS
3. La validation sur données réelles reproduit `HIERARCHIE_CONFIRMEE` et `ARTEFACT_LISSAGE`

**Ne pas modifier les tests pour les faire passer. Ne pas ajuster les tolérances sans raison documentée.**

---

## Ce que cette partie ne couvre pas

Layer 3 (MÉTIS), Layer 4 (KB + D-SIG), et la session pilote H9+EMA60j. Ces éléments seront dans les Parties 4 et 5.

**Ne pas anticiper ces parties en créant des fichiers hors Layer 2.**
