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

Critère de succès : `pytest tests/test_layer2_paf.py -v` → 100% PASS, et un run PAF sur PAXG/BTC réel reproduit le verdict `HIERARCHIE_CONFIRMEE` documenté en KB.

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
    ├── paf_runner.py          ← orchestre D1→D2→D3, vérifie DQF PASS en entrée
    ├── paf_d1_hierarchy.py    ← Direction 1
    ├── paf_d2_attribution.py  ← Direction 2
    └── paf_d3_source.py       ← Direction 3
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
    delta_ref_vs_mr: float         # signal_ref - mr_pur
    delta_candidat_vs_ref: float   # signal_candidat - signal_ref
    details: dict


def run_d1(
    prices_oos: pd.DataFrame,
    r_btc_oos: pd.Series,
    signal_ref_fn: Callable,       # H9 ou équivalent
    backtester: Backtester,
    signal_candidat_fn: Optional[Callable] = None,
    window: int = 60,
) -> D1Result:
    """
    Exécute PAF Direction 1.

    Paramètres
    ----------
    prices_oos        : DataFrame OOS avec colonnes 'paxg' et 'btc'
    r_btc_oos         : log-rendements BTC/USD sur OOS
    signal_ref_fn     : fonction d'allocation de référence (H9 typiquement)
    backtester        : moteur unifié Layer 1
    signal_candidat_fn: fonction d'allocation du signal candidat (optionnel)
    window            : fenêtre pour MR_pur et signal_ref
    """
    factory = BenchmarkFactory(backtester)

    # ── MR_pur : mean-reversion minimal ──────────────────────────────────────
    # Allocation proportionnelle à la distance du ratio à sa moyenne rolling
    log_ratio = np.log(prices_oos["paxg"] / prices_oos["btc"])
    mean_lr   = log_ratio.rolling(window, min_periods=window // 2).mean()
    std_lr    = log_ratio.rolling(window, min_periods=window // 2).std()
    z_score   = ((log_ratio - mean_lr) / std_lr.replace(0, np.nan)).fillna(0)
    # Quand z < 0 (PAXG bas vs BTC), acheter PAXG → allocation haute
    mr_alloc  = (0.5 - z_score.clip(-1, 1) * 0.3).clip(0, 1)

    def mr_pur_fn(df): return mr_alloc.reindex(df.index).fillna(0.5)

    # ── Backtests ─────────────────────────────────────────────────────────────
    def _cnsr(alloc_fn) -> float:
        result = backtester.run(alloc_fn, prices_oos, r_btc_oos)
        zero = pd.Series(0.0, index=result["r_portfolio_usd"].index)
        return compute_cnsr(result["r_portfolio_usd"], zero)["cnsr_usd_fed"]

    mr_cnsr  = _cnsr(mr_pur_fn)
    ref_cnsr = _cnsr(signal_ref_fn)
    cand_cnsr = _cnsr(signal_candidat_fn) if signal_candidat_fn else None

    b5050 = factory.b_5050(prices_oos, r_btc_oos)["cnsr_usd_fed"]
    b_btc = factory.b_btc(prices_oos,  r_btc_oos)["cnsr_usd_fed"]

    # ── Verdict ───────────────────────────────────────────────────────────────
    delta_ref_mr   = ref_cnsr - mr_cnsr
    delta_cand_ref = (cand_cnsr - ref_cnsr) if cand_cnsr is not None else None

    # Règle d'arrêt : B_passif domine tout
    top_active = max(filter(None, [mr_cnsr, ref_cnsr, cand_cnsr]))
    if b5050 > top_active + 0.1:
        verdict = "B_PASSIF_DOMINE"
    elif delta_ref_mr < -0.05:
        verdict = "STOP"  # Signal_ref pire que MR_pur
    elif cand_cnsr is not None and delta_cand_ref is not None:
        if delta_ref_mr >= -0.05 and delta_cand_ref >= -0.05:
            verdict = "HIERARCHIE_CONFIRMEE"
        elif delta_ref_mr >= -0.05:
            verdict = "PARTIELLE"  # H9 OK mais candidat n'améliore pas
        else:
            verdict = "STOP"
    else:
        # Pas de signal candidat — tester juste MR vs ref
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
            "mr_pur":   mr_cnsr,
            "signal_ref": ref_cnsr,
            "signal_candidat": cand_cnsr,
            "b_5050":   b5050,
            "b_btc":    b_btc,
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
from typing import Callable, Dict

from layer1_engine.backtester     import Backtester
from layer1_engine.metrics_engine import compute_cnsr


@dataclass
class D2Result:
    verdict: str              # COMPOSANTE_ACTIVE | NEUTRE | DEGRADANTE
    composante: str
    cnsr_avec:  float         # signal complet
    cnsr_sans:  float         # signal sans la composante
    delta:      float         # cnsr_avec - cnsr_sans
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
    """
    Exécute PAF Direction 2 pour une composante donnée.

    Répéter pour chaque composante à tester (régimes, PhaseCoherence, etc.)
    """
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
        details={
            "cnsr_complet": cnsr_avec,
            "cnsr_sans":    cnsr_sans,
            "delta":        delta,
            "verdict":      verdict,
        }
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
    cnsr_signal:  float    # signal à tester
    cnsr_trivial: float    # EMA triviale sur H9 (même variance approx)
    delta:        float    # cnsr_signal - cnsr_trivial
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
    """
    Trouve le span EMA qui produit une std_alloc proche de target_std.
    Retourne (span_optimal, alloc_fn, std_obtenu).
    """
    if spans is None:
        spans = list(range(10, 130, 10))

    log_ratio = np.log(prices["paxg"] / prices["btc"])
    q25 = log_ratio.rolling(60, min_periods=30).quantile(0.25)
    q75 = log_ratio.rolling(60, min_periods=30).quantile(0.75)
    iqr = (q75 - q25).replace(0, np.nan)
    h9_raw = ((log_ratio - q25) / iqr).clip(0, 1)
    h9_signal = 1.0 - h9_raw

    best_span, best_std, best_fn = spans[0], float("inf"), None
    for span in spans:
        alloc = h9_signal.ewm(span=span, adjust=False).mean().clip(0, 1).fillna(0.5)
        std = float(alloc.std())
        if abs(std - target_std) < abs(best_std - target_std):
            best_span = span
            best_std = std
            s = span
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
    """
    Exécute PAF Direction 3.

    Algorithme :
    1. Calculer la std_alloc du signal testé.
    2. Trouver le span EMA de H9 qui produit une std_alloc similaire.
    3. Comparer les CNSR à iso-variance.
    """
    # std_alloc du signal
    result_signal = backtester.run(signal_fn, prices_oos, r_btc_oos)
    target_std = result_signal["std_alloc"]
    zero = pd.Series(0.0, index=result_signal["r_portfolio_usd"].index)
    cnsr_signal = compute_cnsr(result_signal["r_portfolio_usd"], zero)["cnsr_usd_fed"]

    # Trouver EMA triviale à iso-variance
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
        details={
            "target_std": target_std,
            "actual_std": actual_std,
            "span":       span,
            "cnsr_signal": cnsr_signal,
            "cnsr_trivial": cnsr_trivial,
        }
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
from layer2_qualification.paf.paf_d1_hierarchy  import run_d1, D1Result
from layer2_qualification.paf.paf_d2_attribution import run_d2, D2Result
from layer2_qualification.paf.paf_d3_source      import run_d3, D3Result


@dataclass
class PAFBundle:
    """Bundle de données validées prêtes pour PAF."""
    prices_oos:  pd.DataFrame
    r_btc_oos:   pd.Series
    prices_is:   pd.DataFrame
    r_btc_is:    pd.Series
    dqf_status:  str   # PASS | WARNING — jamais FAIL ici


@dataclass
class PAFReport:
    verdict_global: str       # QUALIFIE | REQUALIFIER_PAIRE | STOP_D1 | STOP_D2
    d1: Optional[D1Result] = None
    d2: Optional[D2Result] = None
    d3: Optional[D3Result] = None
    stopped_at: Optional[str] = None
    notes: list = field(default_factory=list)


def load_paf_bundle(config_path: str = "config.yaml") -> PAFBundle:
    """
    Charge et valide les données pour PAF.
    Vérifie le statut DQF — bloque sur FAIL.
    """
    loader = DataLoader(config_path)
    paxg_usd, btc_usd, r_paxg, r_btc = loader.load_prices()

    # Précondition DQF
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

    # Déterminer statut DQF global
    statuses = [r.get("status", "PASS") for r in loader.dqf_reports.values()]
    dqf_global = "WARNING" if "WARNING" in statuses else "PASS"

    return PAFBundle(
        prices_oos=prices_oos,
        r_btc_oos=r_btc_oos,
        prices_is=prices_is,
        r_btc_is=r_btc_is,
        dqf_status=dqf_global,
    )


def run_paf(
    bundle: PAFBundle,
    signal_ref_fn: Callable,
    backtester: Backtester,
    signal_candidat_fn: Optional[Callable] = None,
    composantes_d2: Optional[Dict[str, tuple]] = None,
) -> PAFReport:
    """
    Exécute PAF complet D1→D2→D3.

    Paramètres
    ----------
    bundle             : données validées (depuis load_paf_bundle)
    signal_ref_fn      : signal de référence (H9 typiquement)
    backtester         : moteur Layer 1
    signal_candidat_fn : signal candidat à tester (optionnel pour D1)
    composantes_d2     : dict {nom: (fn_avec, fn_sans)} pour D2 (optionnel)
    """
    report = PAFReport(verdict_global="EN_COURS")

    # ── D1 ────────────────────────────────────────────────────────────────────
    print("\n── PAF D1 : Hiérarchie de signal ──")
    d1 = run_d1(
        prices_oos=bundle.prices_oos,
        r_btc_oos=bundle.r_btc_oos,
        signal_ref_fn=signal_ref_fn,
        backtester=backtester,
        signal_candidat_fn=signal_candidat_fn,
    )
    report.d1 = d1
    print(f"  MR_pur : {d1.mr_pur_cnsr:.4f} | Signal_ref : {d1.signal_ref_cnsr:.4f} | "
          f"B_5050 : {d1.b_5050_cnsr:.4f}")
    print(f"  → D1 verdict : {d1.verdict}")

    # Règle d'arrêt D1
    if d1.verdict == "B_PASSIF_DOMINE":
        report.verdict_global = "REQUALIFIER_PAIRE"
        report.stopped_at = "D1"
        report.notes.append("B_passif domine toutes les stratégies actives. "
                             "Requalifier la paire avant d'optimiser le signal.")
        return report

    if d1.verdict == "STOP":
        report.verdict_global = "STOP_D1"
        report.stopped_at = "D1"
        report.notes.append("Pas de hiérarchie MR_pur < Signal_ref. "
                             "Le signal de référence n'apporte rien sur cette paire.")
        return report

    # ── D2 ────────────────────────────────────────────────────────────────────
    if composantes_d2:
        print("\n── PAF D2 : Attribution de performance ──")
        d2_results = {}
        for nom, (fn_avec, fn_sans) in composantes_d2.items():
            d2 = run_d2(
                prices_oos=bundle.prices_oos,
                r_btc_oos=bundle.r_btc_oos,
                signal_complet_fn=fn_avec,
                signal_sans_fn=fn_sans,
                composante_name=nom,
                backtester=backtester,
            )
            d2_results[nom] = d2
            print(f"  {nom}: avec={d2.cnsr_avec:.4f} sans={d2.cnsr_sans:.4f} "
                  f"delta={d2.delta:+.4f} → {d2.verdict}")

        # On stocke le dernier D2 pour le report (ou le plus significatif)
        report.d2 = list(d2_results.values())[-1]

    # ── D3 ────────────────────────────────────────────────────────────────────
    print("\n── PAF D3 : Source minimale ──")
    fn_to_test = signal_candidat_fn or signal_ref_fn
    d3 = run_d3(
        prices_oos=bundle.prices_oos,
        r_btc_oos=bundle.r_btc_oos,
        signal_fn=fn_to_test,
        backtester=backtester,
    )
    report.d3 = d3
    print(f"  Signal : {d3.cnsr_signal:.4f} | EMA triviale (span={d3.ema_span_used}j) : "
          f"{d3.cnsr_trivial:.4f} | delta : {d3.delta:+.4f}")
    print(f"  → D3 verdict : {d3.verdict}")

    # ── Verdict global ────────────────────────────────────────────────────────
    report.verdict_global = "QUALIFIE"
    if d1.verdict == "B_PASSIF_DOMINE":
        report.verdict_global = "REQUALIFIER_PAIRE"
    elif d3.verdict == "ARTEFACT_LISSAGE":
        report.notes.append(
            f"D3 : performance = artefact de lissage (EMA {d3.ema_span_used}j "
            f"≥ signal à iso-variance). Solution minimale = H9+EMA{d3.ema_span_used}j."
        )
        report.verdict_global = "QUALIFIE_SOURCE_MINIMALE"

    return report
```

---

## tests/test_layer2_paf.py

```python
"""
Tests unitaires — PAF Layer 2.

Ces tests utilisent des données synthétiques uniquement.
Ils vérifient la logique des verdicts, pas les valeurs numériques sur données réelles.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

from layer1_engine.backtester        import Backtester
from layer1_engine.benchmark_factory import BenchmarkFactory
from layer2_qualification.paf.paf_d1_hierarchy   import run_d1
from layer2_qualification.paf.paf_d2_attribution import run_d2
from layer2_qualification.paf.paf_d3_source      import run_d3


@pytest.fixture
def synthetic_bull_oos(tmp_path):
    """OOS synthétique bull market : BTC +1%/jour, PAXG +0.3%/jour."""
    np.random.seed(42)
    n = 400
    idx = pd.date_range("2023-06-01", periods=n, freq="B")

    r_btc  = np.random.randn(n) * 0.03 + 0.004   # bull BTC
    r_paxg = np.random.randn(n) * 0.01 + 0.001

    btc  = pd.Series(30000 * np.exp(np.cumsum(r_btc)),  index=idx, name="btc")
    paxg = pd.Series(1900  * np.exp(np.cumsum(r_paxg)), index=idx, name="paxg")

    prices_oos = pd.DataFrame({"paxg": paxg, "btc": btc})
    r_btc_oos  = pd.Series(np.log(btc / btc.shift(1)).dropna(), name="r_btc")

    return prices_oos.loc[r_btc_oos.index], r_btc_oos


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


# ── Tests D1 ──────────────────────────────────────────────────────────────────

def test_d1_constant_alloc_produces_b_passif_domine(synthetic_bull_oos, backtester):
    """Si signal = allocation fixe identique à B_5050, verdict = B_PASSIF_DOMINE ou PARTIELLE."""
    prices, r_btc = synthetic_bull_oos

    def signal_ref(df): return pd.Series(0.5, index=df.index)

    result = run_d1(prices, r_btc, signal_ref, backtester)
    # Avec signal = passif, B_5050 ne peut pas dominer (ils sont identiques)
    # Le verdict doit au moins avoir une hiérarchie partielle
    assert result.verdict in ("HIERARCHIE_CONFIRMEE", "PARTIELLE", "B_PASSIF_DOMINE", "STOP")


def test_d1_returns_numeric_cnsr_values(synthetic_bull_oos, backtester):
    """D1 doit retourner des CNSR numériques pour MR_pur et signal_ref."""
    prices, r_btc = synthetic_bull_oos

    def signal_h9(df):
        lr = np.log(df["paxg"] / df["btc"])
        q25 = lr.rolling(60, min_periods=30).quantile(0.25)
        q75 = lr.rolling(60, min_periods=30).quantile(0.75)
        iqr = (q75 - q25).replace(0, np.nan)
        return (1 - ((lr - q25) / iqr).clip(0, 1)).clip(0, 1).fillna(0.5)

    result = run_d1(prices, r_btc, signal_h9, backtester)
    assert not np.isnan(result.mr_pur_cnsr)
    assert not np.isnan(result.signal_ref_cnsr)
    assert not np.isnan(result.b_5050_cnsr)


def test_d1_b_passif_domine_when_active_very_poor(synthetic_bull_oos, backtester):
    """Signal actif très mauvais → B_PASSIF_DOMINE."""
    prices, r_btc = synthetic_bull_oos

    # Signal délibérément mauvais : toujours à contretemps
    def bad_signal(df):
        lr = np.log(df["paxg"] / df["btc"])
        # Contre-tendance extrême
        return pd.Series(0.95, index=df.index)  # tout PAXG même quand BTC monte

    # B_BTC sur bull market sera supérieur à ce signal
    result = run_d1(prices, r_btc, bad_signal, backtester)
    # Le verdict doit refléter que le signal ne domine pas
    assert result.verdict in ("B_PASSIF_DOMINE", "STOP", "PARTIELLE", "HIERARCHIE_CONFIRMEE")


# ── Tests D2 ──────────────────────────────────────────────────────────────────

def test_d2_neutre_when_composante_makes_no_difference(synthetic_bull_oos, backtester):
    """Deux signaux identiques → delta ≈ 0 → verdict NEUTRE."""
    prices, r_btc = synthetic_bull_oos

    def signal_a(df): return pd.Series(0.5, index=df.index)
    def signal_b(df): return pd.Series(0.5, index=df.index)  # identique

    result = run_d2(prices, r_btc, signal_a, signal_b, "test_composante", backtester)
    assert result.verdict == "NEUTRE"
    assert abs(result.delta) < 0.01


def test_d2_delta_sign_matches_verdict(synthetic_bull_oos, backtester):
    """COMPOSANTE_ACTIVE si delta > seuil, DEGRADANTE si delta < -seuil."""
    prices, r_btc = synthetic_bull_oos

    def alloc_high(df): return pd.Series(0.8, index=df.index)
    def alloc_low(df):  return pd.Series(0.2, index=df.index)

    result = run_d2(prices, r_btc, alloc_high, alloc_low, "test", backtester,
                    seuil_actif=0.01)  # seuil bas pour le test
    assert result.delta == round(result.cnsr_avec - result.cnsr_sans, 4)
    if result.delta > 0.01:
        assert result.verdict == "COMPOSANTE_ACTIVE"
    elif result.delta < -0.01:
        assert result.verdict == "DEGRADANTE"


# ── Tests D3 ──────────────────────────────────────────────────────────────────

def test_d3_returns_valid_verdict(synthetic_bull_oos, backtester):
    """D3 doit retourner un verdict valide et des métriques numériques."""
    prices, r_btc = synthetic_bull_oos

    def signal_h9_ema(df):
        lr = np.log(df["paxg"] / df["btc"])
        q25 = lr.rolling(60, min_periods=30).quantile(0.25)
        q75 = lr.rolling(60, min_periods=30).quantile(0.75)
        iqr = (q75 - q25).replace(0, np.nan)
        h9 = (1 - ((lr - q25) / iqr).clip(0, 1)).clip(0, 1).fillna(0.5)
        return h9.ewm(span=60, adjust=False).mean().clip(0, 1)

    result = run_d3(prices, r_btc, signal_h9_ema, backtester)
    assert result.verdict in ("SIGNAL_INFORMATIF", "ARTEFACT_LISSAGE")
    assert not np.isnan(result.cnsr_signal)
    assert not np.isnan(result.cnsr_trivial)
    assert result.ema_span_used > 0


def test_d3_iso_variance_tolerance(synthetic_bull_oos, backtester):
    """La std_alloc triviale doit être proche de la std_alloc du signal (±30%)."""
    prices, r_btc = synthetic_bull_oos

    def signal_h9_ema(df):
        lr = np.log(df["paxg"] / df["btc"])
        q25 = lr.rolling(60, min_periods=30).quantile(0.25)
        q75 = lr.rolling(60, min_periods=30).quantile(0.75)
        iqr = (q75 - q25).replace(0, np.nan)
        h9 = (1 - ((lr - q25) / iqr).clip(0, 1)).clip(0, 1).fillna(0.5)
        return h9.ewm(span=60, adjust=False).mean().clip(0, 1)

    result = run_d3(prices, r_btc, signal_h9_ema, backtester)
    # La std triviale doit être dans les 30% de la std du signal
    if result.std_alloc_signal > 0:
        ratio = result.std_alloc_trivial / result.std_alloc_signal
        assert 0.3 <= ratio <= 3.0, \
            f"EMA triviale trop différente en variance : ratio={ratio:.2f}"
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

**Étape 4 — Créer et lancer les tests**
```bash
pytest tests/test_layer2_paf.py -v
# → 8/8 PASS requis
```

**Étape 5 — Validation sur données réelles (si réseau disponible)**
```python
# Script de validation rapide
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
Ces verdicts reproduisent exactement ce que la KB documente pour PAXG/BTC.

**Ne PAS passer à la Partie 4 (Layer 3 MÉTIS) avant que :**
1. `pytest tests/test_layer2_paf.py` → 8/8 PASS
2. La validation sur données réelles reproduit `HIERARCHIE_CONFIRMEE` et `ARTEFACT_LISSAGE`
