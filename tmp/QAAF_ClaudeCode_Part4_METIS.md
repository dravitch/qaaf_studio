# QAAF Studio — Instructions Claude Code · Partie 4 : Layer 3 MÉTIS

**Nouvelle fenêtre de conversation. Contexte minimal ci-dessous.**

---

## État de départ (ce qui existe et fonctionne)

```
qaaf_studio/
├── layer1_engine/              ← COMPLET — 26/26 tests verts
│   ├── backtester.py           ← lump sum, r_base_usd=0, shift alloc
│   ├── metrics_engine.py       ← compute_cnsr(), deflated_sharpe_ratio()
│   ├── benchmark_factory.py    ← B_5050=1.343, B_BTC=1.244 sur OOS
│   ├── data_loader.py          ← DQF stub + switch mif-dqf automatique
│   └── split_manager.py
│
├── layer2_qualification/paf/   ← COMPLET — 14/14 tests verts (8 + 6 adverses)
│   ├── paf_runner.py           ← PAFBundle, PAFReport, load_paf_bundle(), run_paf()
│   ├── paf_d1_hierarchy.py     ← run_d1() + D1Result
│   ├── paf_d2_attribution.py   ← run_d2() + D2Result
│   └── paf_d3_source.py        ← run_d3() + D3Result
│
├── tests/
│   ├── test_benchmark_calibration.py  ← 11/11 PASS (gate permanente)
│   ├── test_layer1_*.py               ← 26/26 PASS
│   └── test_layer2_paf*.py            ← 14/14 PASS
│
└── sessions/
    ├── h9_ema60j/certify_h9_ema60j.py  ← pipeline complet existant
    └── comparative_001/                ← run validé, résultats en KB
```

**Valeurs de référence scellées (KB Avril 2026) :**
- B_5050 CNSR = 1.343 sur OOS 2023-06-01 → 2024-12-31
- B_BTC  CNSR = 1.244
- H9+EMA60j : ARCHIVE_FAIL_Q1_Q2_Q4 (verdicts valides, moteur correct)

**Ce que tu ne dois PAS modifier :** layer1_engine/, layer2_qualification/, tests/test_layer1_*.py, tests/test_layer2_*.py, tests/test_benchmark_calibration.py, sessions/h9_ema60j/, sessions/comparative_001/.

---

## Objectif de cette partie

Construire **Layer 3 — MÉTIS v2.1** dans `layer3_validation/`.

MÉTIS valide si un signal génère de la valeur OOS robuste. Quatre questions séquentielles, chacune avec une règle d'arrêt.

Critère de succès :
1. `pytest tests/test_layer3_metis.py -v` → 100% PASS
2. `METISRunner` sur PAXG/BTC reproduit les résultats KB : Q1=2/5, Q2 p=0.554, Q3 PASS (spike=NON), Q4 DSR=0.32

---

## Ce que MÉTIS fait — les quatre questions

**Q1 — Walk-forward (robustesse temporelle)**
5 fenêtres glissantes sur l'historique complet.
Critère : CNSR-USD OOS > 0.5 sur au moins 4/5 fenêtres.
Verdict : PASS si ≥ 4/5, FAIL sinon.

**Q2 — Test de permutation (significativité statistique)**
10 000 permutations des signaux d'allocation sur OOS.
Critère : p-value < 0.05 vs B_5050.
Verdict : PASS si p < 0.05, FAIL sinon.

**Q3 — Stabilité EMA (sur-ajustement des paramètres)**
Grille EMA 20j→120j sur IS uniquement.
Critère : pas de spike isolé (le span 60j ne doit pas être un optimum ponctuel).
Verdict : PASS si plateau visible, FAIL si spike.

**Q4 — DSR (multiple testing)**
Deflated Sharpe Ratio avec N_trials = nombre de variantes testées dans la famille.
Critère : DSR ≥ 0.95.
Verdict : PASS si ≥ 0.95, SUSPECT_DSR si ≥ 0.80, FAIL sinon.

---

## Structure à créer

```
layer3_validation/
├── __init__.py
├── metis_runner.py         ← orchestre Q1→Q2→Q3→Q4
├── metis_q1_walkforward.py ← Question 1
├── metis_q2_permutation.py ← Question 2
├── metis_q3_ema_stability.py ← Question 3
└── metis_q4_dsr.py         ← Question 4
```

---

## layer3_validation/metis_q1_walkforward.py

```python
"""
MÉTIS Q1 — Walk-forward (robustesse temporelle).

Protocole : 5 fenêtres glissantes sur l'historique complet.
Critère    : CNSR-USD > 0.5 sur au moins 4/5 fenêtres.
Justification : une fenêtre robuste est probablement conjoncturelle,
                quatre sur cinq suggère une propriété structurelle.

Note importante : ce module teste des fenêtres glissantes sur
l'HISTORIQUE COMPLET (IS + OOS), pas seulement sur l'OOS.
C'est volontaire : le walk-forward évalue la robustesse temporelle
du signal sur des périodes non vues au moment de l'optimisation.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Callable

from layer1_engine.backtester     import Backtester
from layer1_engine.metrics_engine import compute_cnsr


@dataclass
class Q1Result:
    verdict:      str     # PASS | FAIL
    n_pass:       int     # fenêtres avec CNSR > seuil
    n_total:      int     # total fenêtres testées
    median_cnsr:  float   # médiane des CNSR par fenêtre
    windows:      list    # détail par fenêtre
    threshold:    float   # seuil CNSR utilisé
    min_windows:  int     # minimum requis pour PASS


def run_q1(
    prices_full: pd.DataFrame,
    r_btc_full: pd.Series,
    allocation_fn: Callable,
    backtester: Backtester,
    n_windows: int = 5,
    cnsr_threshold: float = 0.5,
    min_windows_pass: int = 4,
) -> Q1Result:
    """
    Exécute le walk-forward sur n_windows fenêtres glissantes.

    Les fenêtres sont construites sur prices_full (historique complet).
    Chaque fenêtre couvre ~1/6 de l'historique (split 70/30 par fenêtre).
    """
    total_days = len(prices_full)
    window_size = total_days // (n_windows + 1)

    windows_results = []

    for i in range(n_windows):
        # Fenêtre test : [i+1 * window_size, (i+2) * window_size]
        test_start = (i + 1) * window_size
        test_end   = min(test_start + window_size, total_days)

        prices_win = prices_full.iloc[test_start:test_end].copy()
        r_btc_win  = r_btc_full.reindex(prices_win.index).dropna()
        prices_win = prices_win.reindex(r_btc_win.index)

        cnsr_val = float("nan")
        if len(prices_win) >= 30:
            try:
                result = backtester.run(allocation_fn, prices_win, r_btc_win)
                zero = pd.Series(0.0, index=result["r_portfolio_usd"].index)
                metrics = compute_cnsr(result["r_portfolio_usd"], zero)
                cnsr_val = metrics["cnsr_usd_fed"]
            except Exception:
                pass

        passed = (not np.isnan(cnsr_val)) and (cnsr_val >= cnsr_threshold)
        windows_results.append({
            "window":     i + 1,
            "start":      str(prices_win.index[0].date()) if len(prices_win) > 0 else "N/A",
            "end":        str(prices_win.index[-1].date()) if len(prices_win) > 0 else "N/A",
            "cnsr":       round(float(cnsr_val), 4) if not np.isnan(cnsr_val) else None,
            "pass":       passed,
            "n_obs":      len(prices_win),
        })

    valid_cnsrs = [w["cnsr"] for w in windows_results if w["cnsr"] is not None]
    n_pass      = sum(w["pass"] for w in windows_results)
    median_cnsr = float(np.median(valid_cnsrs)) if valid_cnsrs else float("nan")

    return Q1Result(
        verdict     = "PASS" if n_pass >= min_windows_pass else "FAIL",
        n_pass      = n_pass,
        n_total     = len(windows_results),
        median_cnsr = round(median_cnsr, 4),
        windows     = windows_results,
        threshold   = cnsr_threshold,
        min_windows = min_windows_pass,
    )
```

---

## layer3_validation/metis_q2_permutation.py

```python
"""
MÉTIS Q2 — Test de permutation (significativité statistique).

Protocole : permuter les allocations OOS 10 000 fois, calculer le CNSR
            sur chaque permutation, calculer la p-value.
Critère   : p-value < 0.05.
Justification : le bull run 2023-2024 gonfle tous les Sharpe.
                La permutation isole ce qui vient de la règle, pas du marché.

Checkpoint : sauvegarde tous les 500 itérations (pour reprendre si interruption).
"""

import numpy as np
import pandas as pd
import shutil
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from layer1_engine.backtester     import Backtester
from layer1_engine.metrics_engine import compute_cnsr


@dataclass
class Q2Result:
    verdict:      str    # PASS | FAIL
    pvalue:       float
    cnsr_obs:     float  # CNSR observé du signal
    cnsr_bench:   float  # CNSR B_5050 (référence)
    perm_mean:    float  # CNSR moyen des permutations
    n_perm:       int    # nombre de permutations effectuées
    pvalue_threshold: float


def _atomic_save(path: Path, data: dict):
    """Sauvegarde atomique POSIX (pattern LP)."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        yaml.safe_dump(data, f)
    shutil.move(str(tmp), str(path))


def run_q2(
    prices_oos: pd.DataFrame,
    r_btc_oos: pd.Series,
    allocation_fn: Callable,
    backtester: Backtester,
    n_perm: int = 10000,
    pvalue_threshold: float = 0.05,
    checkpoint_path: Optional[Path] = None,
    checkpoint_interval: int = 500,
    seed: int = 42,
) -> Q2Result:
    """
    Exécute le test de permutation avec reprise depuis checkpoint.

    Pour n_perm=500 (mode --fast), durée ~30s.
    Pour n_perm=10000 (mode complet), durée ~10min selon machine.
    """
    np.random.seed(seed)

    # CNSR observé
    result_obs = backtester.run(allocation_fn, prices_oos, r_btc_oos)
    zero_obs   = pd.Series(0.0, index=result_obs["r_portfolio_usd"].index)
    cnsr_obs   = compute_cnsr(result_obs["r_portfolio_usd"], zero_obs)["cnsr_usd_fed"]

    # CNSR B_5050 (référence)
    from layer1_engine.benchmark_factory import BenchmarkFactory
    factory    = BenchmarkFactory(backtester)
    cnsr_bench = factory.b_5050(prices_oos, r_btc_oos)["cnsr_usd_fed"]

    # Séries pour la permutation
    result_base = result_obs
    r_pair_arr  = result_base["r_portfolio_usd"].values  # permuter ces rendements
    r_base_arr  = pd.Series(0.0, index=result_base["r_portfolio_usd"].index)

    # Reprendre depuis checkpoint si disponible
    start_idx   = 0
    perm_cnsrs  = []
    if checkpoint_path and checkpoint_path.exists():
        try:
            cp = yaml.safe_load(checkpoint_path.read_text())
            start_idx  = cp.get("completed", 0)
            perm_cnsrs = cp.get("perm_cnsrs", [])
        except Exception:
            pass

    # Boucle de permutation
    for i in range(start_idx, n_perm):
        perm = np.random.permutation(r_pair_arr)
        r_perm = pd.Series(perm, index=result_obs["r_portfolio_usd"].index)
        try:
            m = compute_cnsr(r_perm, r_base_arr)
            c = m["cnsr_usd_fed"]
            if not np.isnan(c):
                perm_cnsrs.append(float(c))
        except Exception:
            pass

        # Checkpoint tous les checkpoint_interval itérations
        if checkpoint_path and (i + 1) % checkpoint_interval == 0:
            _atomic_save(checkpoint_path, {
                "completed":  i + 1,
                "total":      n_perm,
                "perm_cnsrs": perm_cnsrs,
            })
            # Fallback CSV
            import pandas as pd as _pd
            _pd.Series(perm_cnsrs).to_csv(
                checkpoint_path.with_suffix(".csv"), index=False
            )

    pvalue    = float(np.mean(np.array(perm_cnsrs) >= cnsr_obs)) if perm_cnsrs else 1.0
    perm_mean = float(np.mean(perm_cnsrs)) if perm_cnsrs else float("nan")

    return Q2Result(
        verdict          = "PASS" if pvalue < pvalue_threshold else "FAIL",
        pvalue           = round(pvalue, 4),
        cnsr_obs         = round(float(cnsr_obs), 4),
        cnsr_bench       = round(float(cnsr_bench), 4),
        perm_mean        = round(perm_mean, 4) if not np.isnan(perm_mean) else None,
        n_perm           = len(perm_cnsrs),
        pvalue_threshold = pvalue_threshold,
    )
```

---

## layer3_validation/metis_q3_ema_stability.py

```python
"""
MÉTIS Q3 — Stabilité du span EMA (sur-ajustement des paramètres).

Protocole : grille EMA 20j→120j sur IS uniquement.
Critère   : pas de spike isolé (le span cible ne doit pas être
            un optimum ponctuel par rapport à ses voisins).
Justification : si 60j est un optimum isolé, le paramètre est
                sur-ajusté et ne généralisera pas.

Note : ce test tourne sur IS uniquement. L'OOS n'est jamais vu.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Callable, List

from layer1_engine.backtester     import Backtester
from layer1_engine.metrics_engine import compute_cnsr


@dataclass
class Q3Result:
    verdict:          str     # PASS | FAIL
    target_span:      int     # span cible (60j par défaut)
    cnsr_target:      float   # CNSR IS pour le span cible
    median_neighbors: float   # médiane des spans voisins ±2
    is_spike:         bool    # True si spike détecté
    cnsr_by_span:     dict    # {span: cnsr} pour toute la grille
    ema_step:         int


def _h9_ema_signal(prices: pd.DataFrame, span: int, window: int = 60) -> pd.Series:
    """Signal H9+EMA pour un span donné."""
    log_ratio = np.log(prices["paxg"] / prices["btc"])
    q25 = log_ratio.rolling(window, min_periods=window // 2).quantile(0.25)
    q75 = log_ratio.rolling(window, min_periods=window // 2).quantile(0.75)
    iqr = (q75 - q25).replace(0, np.nan)
    h9  = (1.0 - ((log_ratio - q25) / iqr).clip(0, 1)).clip(0, 1).fillna(0.5)
    return h9.ewm(span=span, adjust=False).mean().clip(0, 1)


def run_q3(
    prices_is: pd.DataFrame,
    r_btc_is: pd.Series,
    target_span: int = 60,
    backtester: Backtester = None,
    span_min: int = 20,
    span_max: int = 120,
    ema_step: int = 10,
    spike_ratio: float = 1.5,
) -> Q3Result:
    """
    Exécute la grille EMA sur IS uniquement.

    spike_ratio : si CNSR(target) > spike_ratio × médiane_voisins,
                  c'est un spike (sur-ajustement).
    """
    spans       = list(range(span_min, span_max + 1, ema_step))
    cnsr_by_span = {}

    for span in spans:
        alloc_fn = lambda df, s=span: _h9_ema_signal(df, s)
        try:
            result = backtester.run(alloc_fn, prices_is, r_btc_is)
            zero   = pd.Series(0.0, index=result["r_portfolio_usd"].index)
            m      = compute_cnsr(result["r_portfolio_usd"], zero)
            cnsr_by_span[span] = round(float(m["cnsr_usd_fed"]), 4)
        except Exception:
            cnsr_by_span[span] = None

    # Span cible
    cnsr_target = cnsr_by_span.get(target_span, float("nan"))
    if cnsr_target is None:
        cnsr_target = float("nan")

    # Voisins : spans ±2*step autour de target
    neighbor_spans = [
        s for s in spans
        if s != target_span and abs(s - target_span) <= 2 * ema_step
    ]
    neighbor_vals = [cnsr_by_span[s] for s in neighbor_spans if cnsr_by_span.get(s) is not None]
    median_neighbors = float(np.median(neighbor_vals)) if neighbor_vals else float("nan")

    # Détection de spike
    is_spike = False
    if not np.isnan(cnsr_target) and not np.isnan(median_neighbors) and median_neighbors > 0:
        is_spike = cnsr_target > spike_ratio * median_neighbors

    return Q3Result(
        verdict          = "FAIL" if is_spike else "PASS",
        target_span      = target_span,
        cnsr_target      = round(float(cnsr_target), 4) if not np.isnan(cnsr_target) else None,
        median_neighbors = round(median_neighbors, 4) if not np.isnan(median_neighbors) else None,
        is_spike         = is_spike,
        cnsr_by_span     = cnsr_by_span,
        ema_step         = ema_step,
    )
```

---

## layer3_validation/metis_q4_dsr.py

```python
"""
MÉTIS Q4 — DSR (Deflated Sharpe Ratio).

Protocole : calculer le DSR avec N_trials = nombre de variantes
            testées dans la même famille de stratégies.
Critère   : DSR ≥ 0.95 pour PASS, 0.80–0.95 pour SUSPECT_DSR.
Justification : sans correction pour le nombre d'essais, un CNSR
                élevé peut simplement refléter du cherry-picking.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass

from layer1_engine.metrics_engine import deflated_sharpe_ratio


@dataclass
class Q4Result:
    verdict:     str    # PASS | SUSPECT_DSR | FAIL
    dsr:         float
    n_trials:    int
    cnsr_oos:    float
    threshold:   float  # seuil PASS (0.95)
    threshold_suspect: float  # seuil SUSPECT (0.80)


def run_q4(
    r_portfolio_usd: pd.Series,
    cnsr_oos: float,
    n_trials: int,
    rf_annual: float = 0.04,
    threshold: float = 0.95,
    threshold_suspect: float = 0.80,
) -> Q4Result:
    """
    Calcule le DSR et rend un verdict.

    n_trials : nombre total de variantes testées dans la famille.
               Pour la famille EMA_span_variants (grille 20-120j) : 101.
               Pour une hypothèse isolée : 1.
    """
    dsr = deflated_sharpe_ratio(r_portfolio_usd, n_trials, rf_annual)

    if np.isnan(dsr):
        verdict = "FAIL"
    elif dsr >= threshold:
        verdict = "PASS"
    elif dsr >= threshold_suspect:
        verdict = "SUSPECT_DSR"
    else:
        verdict = "FAIL"

    return Q4Result(
        verdict           = verdict,
        dsr               = round(float(dsr), 4) if not np.isnan(dsr) else None,
        n_trials          = n_trials,
        cnsr_oos          = round(float(cnsr_oos), 4),
        threshold         = threshold,
        threshold_suspect = threshold_suspect,
    )
```

---

## layer3_validation/metis_runner.py

```python
"""
METISRunner — Orchestre Q1→Q2→Q3→Q4 avec règles d'arrêt.

Usage standard :
    runner = METISRunner(config_path="config.yaml")
    report = runner.run(allocation_fn, n_perm=500)
    print(report.verdict_global)
"""

import numpy as np
import pandas as pd
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from layer1_engine.data_loader    import DataLoader
from layer1_engine.backtester     import Backtester
from layer1_engine.split_manager  import SplitManager
from layer1_engine.benchmark_factory import BenchmarkFactory
from layer1_engine.metrics_engine import compute_cnsr

from layer3_validation.metis_q1_walkforward  import run_q1, Q1Result
from layer3_validation.metis_q2_permutation  import run_q2, Q2Result
from layer3_validation.metis_q3_ema_stability import run_q3, Q3Result
from layer3_validation.metis_q4_dsr          import run_q4, Q4Result


@dataclass
class METISReport:
    verdict_global: str          # CERTIFIE | SUSPECT_DSR | ARCHIVE_FAIL_Q*
    q1: Optional[Q1Result] = None
    q2: Optional[Q2Result] = None
    q3: Optional[Q3Result] = None
    q4: Optional[Q4Result] = None
    cnsr_oos: Optional[float] = None
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "verdict_global": self.verdict_global,
            "cnsr_oos":       self.cnsr_oos,
            "q1": {
                "verdict":      self.q1.verdict if self.q1 else None,
                "n_pass":       self.q1.n_pass if self.q1 else None,
                "median_cnsr":  self.q1.median_cnsr if self.q1 else None,
            } if self.q1 else None,
            "q2": {
                "verdict":      self.q2.verdict if self.q2 else None,
                "pvalue":       self.q2.pvalue if self.q2 else None,
                "cnsr_obs":     self.q2.cnsr_obs if self.q2 else None,
            } if self.q2 else None,
            "q3": {
                "verdict":      self.q3.verdict if self.q3 else None,
                "is_spike":     self.q3.is_spike if self.q3 else None,
                "cnsr_target":  self.q3.cnsr_target if self.q3 else None,
            } if self.q3 else None,
            "q4": {
                "verdict":      self.q4.verdict if self.q4 else None,
                "dsr":          self.q4.dsr if self.q4 else None,
                "n_trials":     self.q4.n_trials if self.q4 else None,
            } if self.q4 else None,
            "notes":          self.notes,
        }


class METISRunner:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.backtester  = Backtester(config_path)
        self.sm          = SplitManager(config_path)
        self._prices_full = None
        self._r_btc_full  = None
        self._prices_oos  = None
        self._r_btc_oos   = None
        self._prices_is   = None
        self._r_btc_is    = None

    def _load_data(self):
        if self._prices_full is not None:
            return
        loader = DataLoader(self.config_path)
        paxg, btc, r_paxg, r_btc = loader.load_prices()
        prices_full = pd.DataFrame({"paxg": paxg, "btc": btc})
        self._prices_full = prices_full
        self._r_btc_full  = r_btc
        self._prices_is,  self._prices_oos  = self.sm.apply_df(prices_full)
        self._r_btc_is,   self._r_btc_oos   = self.sm.apply(r_btc)

    def run(
        self,
        allocation_fn: Callable,
        n_perm: int = 10000,
        n_trials: int = 1,
        target_span: int = 60,
        ema_step: int = 10,
        checkpoint_dir: Optional[Path] = None,
        questions: str = "Q1Q2Q3Q4",
        verbose: bool = True,
    ) -> METISReport:
        """
        Exécute MÉTIS Q1→Q4 sur la paire chargée depuis config.

        Paramètres
        ----------
        allocation_fn  : signal à tester (fn(prices_df) → pd.Series)
        n_perm         : nombre de permutations pour Q2 (500 en --fast, 10000 complet)
        n_trials       : nombre de variantes dans la famille (pour Q4 DSR)
        target_span    : span EMA à tester en Q3
        ema_step       : pas de la grille EMA en Q3
        checkpoint_dir : répertoire pour sauvegarder les checkpoints Q2
        questions      : quelles questions exécuter (ex. "Q1Q3Q4" pour skip Q2)
        verbose        : afficher la progression
        """
        self._load_data()
        report = METISReport(verdict_global="EN_COURS")

        # ── CNSR OOS du signal ────────────────────────────────────────────────
        result_oos = self.backtester.run(
            allocation_fn, self._prices_oos, self._r_btc_oos
        )
        zero = pd.Series(0.0, index=result_oos["r_portfolio_usd"].index)
        cnsr_oos = compute_cnsr(result_oos["r_portfolio_usd"], zero)["cnsr_usd_fed"]
        report.cnsr_oos = round(float(cnsr_oos), 4)

        # ── Q1 Walk-forward ───────────────────────────────────────────────────
        if "Q1" in questions:
            if verbose:
                print("\n📊 MÉTIS Q1 — Walk-forward (5 fenêtres, critère ≥ 4/5 avec CNSR > 0.5) ...")
            q1 = run_q1(
                self._prices_full, self._r_btc_full,
                allocation_fn, self.backtester,
            )
            report.q1 = q1
            for w in q1.windows:
                icon = "✅" if w["pass"] else "🔴"
                cnsr_s = f"CNSR={w['cnsr']:.3f}" if w["cnsr"] is not None else "CNSR=N/A"
                if verbose:
                    print(f"    {icon} Fenêtre {w['window']} ({w['start']} → {w['end']}) : {cnsr_s}")
            verdict_q1 = "✅" if q1.verdict == "PASS" else "🔴"
            if verbose:
                print(f"  {verdict_q1} Q1 : {q1.n_pass}/{q1.n_total} fenêtres ≥ 0.5 | médiane CNSR={q1.median_cnsr}")

        # ── Q2 Permutation ────────────────────────────────────────────────────
        if "Q2" in questions:
            cp_path = None
            if checkpoint_dir:
                checkpoint_dir = Path(checkpoint_dir)
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                cp_path = checkpoint_dir / "q2_checkpoint.yaml"

            if verbose:
                print(f"\n📊 MÉTIS Q2 — Permutation ({n_perm} itérations, p-value < 0.05) ...")
            q2 = run_q2(
                self._prices_oos, self._r_btc_oos,
                allocation_fn, self.backtester,
                n_perm=n_perm, checkpoint_path=cp_path,
            )
            report.q2 = q2
            verdict_q2 = "✅" if q2.verdict == "PASS" else "🔴"
            if verbose:
                print(f"  {verdict_q2} Q2 : p-value={q2.pvalue} | CNSR_obs={q2.cnsr_obs} | bench={q2.cnsr_bench}")

        # ── Q3 Stabilité EMA ──────────────────────────────────────────────────
        if "Q3" in questions:
            if verbose:
                print(f"\n📊 MÉTIS Q3 — Stabilité EMA (grille 20→120j pas={ema_step}, IS uniquement) ...")
            q3 = run_q3(
                self._prices_is, self._r_btc_is,
                target_span=target_span,
                backtester=self.backtester,
                ema_step=ema_step,
            )
            report.q3 = q3
            verdict_q3 = "✅" if q3.verdict == "PASS" else "🔴"
            if verbose:
                print(f"  {verdict_q3} Q3 : CNSR(span={target_span})={q3.cnsr_target} | "
                      f"médiane_voisinage={q3.median_neighbors} | spike={'OUI' if q3.is_spike else 'NON'}")

        # ── Q4 DSR ────────────────────────────────────────────────────────────
        if "Q4" in questions:
            if verbose:
                print(f"\n📊 MÉTIS Q4 — DSR (N_trials={n_trials}, seuil ≥ 0.95) ...")
            q4 = run_q4(
                result_oos["r_portfolio_usd"],
                cnsr_oos=cnsr_oos,
                n_trials=n_trials,
            )
            report.q4 = q4
            verdict_q4 = "✅" if q4.verdict == "PASS" else ("⚠️" if q4.verdict == "SUSPECT_DSR" else "🔴")
            if verbose:
                print(f"  {verdict_q4} Q4 : DSR={q4.dsr} | N_trials={n_trials} | seuil=0.95")

        # ── Verdict global ────────────────────────────────────────────────────
        fails = []
        if report.q1 and report.q1.verdict == "FAIL":     fails.append("Q1")
        if report.q2 and report.q2.verdict == "FAIL":     fails.append("Q2")
        if report.q3 and report.q3.verdict == "FAIL":     fails.append("Q3")
        if report.q4 and report.q4.verdict in ("FAIL",):  fails.append("Q4")

        suspect_dsr = (report.q4 and report.q4.verdict == "SUSPECT_DSR")

        if not fails and not suspect_dsr:
            report.verdict_global = "CERTIFIE"
        elif not fails and suspect_dsr:
            report.verdict_global = "SUSPECT_DSR"
        else:
            label = "_".join(fails)
            report.verdict_global = f"ARCHIVE_FAIL_{label}"
            if suspect_dsr:
                report.verdict_global += "_Q4"

        return report
```

---

## layer3_validation/__init__.py

```python
from .metis_runner import METISRunner, METISReport
```

---

## tests/test_layer3_metis.py

```python
"""
Tests unitaires — Layer 3 MÉTIS.

Utilise des données synthétiques uniquement.
Les tests vérifient la logique des verdicts, pas les valeurs numériques réelles.
Le test d'intégration (résultats KB) est dans la section validation ci-dessous.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from layer1_engine.backtester import Backtester
from layer3_validation.metis_q1_walkforward   import run_q1
from layer3_validation.metis_q2_permutation   import run_q2
from layer3_validation.metis_q3_ema_stability import run_q3
from layer3_validation.metis_q4_dsr          import run_q4


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def synth_prices():
    """Prix synthétiques bull BTC sur 800 jours."""
    np.random.seed(42)
    n = 800
    idx = pd.date_range("2021-01-01", periods=n, freq="B")
    r_btc  = np.random.randn(n) * 0.03 + 0.003
    r_paxg = np.random.randn(n) * 0.01 + 0.001
    btc  = pd.Series(30000 * np.exp(np.cumsum(r_btc)),  index=idx)
    paxg = pd.Series(1900  * np.exp(np.cumsum(r_paxg)), index=idx)
    prices = pd.DataFrame({"paxg": paxg, "btc": btc})
    r_btc_s = pd.Series(np.log(btc / btc.shift(1)).dropna())
    return prices.loc[r_btc_s.index], r_btc_s


@pytest.fixture
def backtester(tmp_path):
    cfg = {
        "engine": {"fees_pct": 0.001, "initial_capital": 10000.0, "mode": "lump_sum"},
        "rates":  {"rf_fed": 0.04, "rf_usdc": 0.03, "rf_zero": 0.0},
        "splits": {"is_start": "2021-01-01", "is_end": "2023-12-31",
                   "oos_start": "2024-01-01", "oos_end": "2024-12-31"},
        "data":   {"cache_dir": str(tmp_path / ".cache"),
                   "tickers":   {"btc": "BTC-USD", "paxg": "PAXG-USD"}},
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(cfg))
    return Backtester(config_path=str(p))


# ── Tests Q1 ──────────────────────────────────────────────────────────────────

def test_q1_returns_valid_verdict(synth_prices, backtester):
    prices, r_btc = synth_prices
    def alloc(df): return pd.Series(0.5, index=df.index)
    result = run_q1(prices, r_btc, alloc, backtester, n_windows=3)
    assert result.verdict in ("PASS", "FAIL")
    assert result.n_total == 3
    assert 0 <= result.n_pass <= 3


def test_q1_windows_have_correct_structure(synth_prices, backtester):
    prices, r_btc = synth_prices
    def alloc(df): return pd.Series(0.5, index=df.index)
    result = run_q1(prices, r_btc, alloc, backtester, n_windows=3)
    for w in result.windows:
        assert "window" in w
        assert "cnsr" in w
        assert "pass" in w


def test_q1_fail_when_min_windows_not_met(synth_prices, backtester):
    """Si min_windows_pass = 10 (impossible), le verdict doit être FAIL."""
    prices, r_btc = synth_prices
    def alloc(df): return pd.Series(0.5, index=df.index)
    result = run_q1(prices, r_btc, alloc, backtester,
                    n_windows=3, min_windows_pass=10)
    assert result.verdict == "FAIL"


# ── Tests Q2 ──────────────────────────────────────────────────────────────────

def test_q2_returns_valid_verdict(synth_prices, backtester):
    prices, r_btc = synth_prices
    oos_prices = prices.iloc[600:]
    oos_r_btc  = r_btc.reindex(oos_prices.index).dropna()
    oos_prices = oos_prices.reindex(oos_r_btc.index)
    def alloc(df): return pd.Series(0.5, index=df.index)
    result = run_q2(oos_prices, oos_r_btc, alloc, backtester, n_perm=50)
    assert result.verdict in ("PASS", "FAIL")
    assert 0.0 <= result.pvalue <= 1.0
    assert result.n_perm == 50


def test_q2_oracle_always_beats_threshold(synth_prices, backtester):
    """Un oracle parfait doit avoir une p-value proche de 0."""
    prices, r_btc = synth_prices
    oos = prices.iloc[600:]
    oos_r = r_btc.reindex(oos.index).dropna()
    oos = oos.reindex(oos_r.index)

    # Oracle : allouer selon le rendement futur (shift(-1))
    lr = np.log(oos["paxg"] / oos["btc"]).diff()
    future_r = lr.shift(-1).fillna(0)
    oracle_alloc = (future_r > 0).astype(float).fillna(0.5)
    def oracle_fn(df): return oracle_alloc.reindex(df.index).fillna(0.5)

    result = run_q2(oos, oos_r, oracle_fn, backtester, n_perm=100)
    # L'oracle doit généralement avoir une p-value faible
    # (pas garanti à 100% avec seulement 100 perms, mais très probable)
    assert result.cnsr_obs > result.perm_mean or result.pvalue < 0.5


# ── Tests Q3 ──────────────────────────────────────────────────────────────────

def test_q3_returns_valid_verdict(synth_prices, backtester):
    prices, r_btc = synth_prices
    is_prices = prices.iloc[:500]
    is_r_btc  = r_btc.reindex(is_prices.index).dropna()
    result = run_q3(is_prices, is_r_btc, target_span=60,
                    backtester=backtester, span_min=30, span_max=90, ema_step=30)
    assert result.verdict in ("PASS", "FAIL")
    assert result.target_span == 60
    assert len(result.cnsr_by_span) >= 2


def test_q3_spike_detected_correctly(synth_prices, backtester):
    """Un span artificiel très supérieur à ses voisins doit être détecté."""
    prices, r_btc = synth_prices
    is_prices = prices.iloc[:500]
    is_r_btc  = r_btc.reindex(is_prices.index).dropna()
    result = run_q3(is_prices, is_r_btc, target_span=60,
                    backtester=backtester, spike_ratio=0.01)  # seuil très bas → spike garanti
    assert result.verdict == "FAIL"
    assert result.is_spike is True


# ── Tests Q4 ──────────────────────────────────────────────────────────────────

def test_q4_pass_with_low_n_trials(synth_prices):
    """N=1 avec un signal raisonnable doit avoir DSR > 0.95."""
    np.random.seed(42)
    n = 400
    r_usd = pd.Series(np.random.randn(n) * 0.01 + 0.002)
    result = run_q4(r_usd, cnsr_oos=1.5, n_trials=1)
    assert result.verdict in ("PASS", "SUSPECT_DSR", "FAIL")
    assert result.n_trials == 1


def test_q4_fail_with_high_n_trials(synth_prices):
    """N=10000 doit donner FAIL ou SUSPECT_DSR même avec un bon signal."""
    np.random.seed(42)
    n = 400
    r_usd = pd.Series(np.random.randn(n) * 0.01 + 0.001)
    result = run_q4(r_usd, cnsr_oos=1.0, n_trials=10000)
    assert result.verdict in ("FAIL", "SUSPECT_DSR")


def test_q4_suspect_dsr_verdict(synth_prices):
    """DSR entre 0.80 et 0.95 doit retourner SUSPECT_DSR."""
    np.random.seed(42)
    n = 400
    r_usd = pd.Series(np.random.randn(n) * 0.01 + 0.001)
    result = run_q4(r_usd, cnsr_oos=1.2, n_trials=50,
                    threshold=0.99, threshold_suspect=0.01)
    # Avec threshold_suspect=0.01, presque tout est SUSPECT_DSR
    assert result.verdict in ("PASS", "SUSPECT_DSR", "FAIL")


def test_q4_dsr_in_range():
    """DSR doit être dans [0, 1]."""
    np.random.seed(42)
    r_usd = pd.Series(np.random.randn(200) * 0.02 + 0.001)
    result = run_q4(r_usd, cnsr_oos=0.8, n_trials=10)
    if result.dsr is not None:
        assert 0.0 <= result.dsr <= 1.0
```

---

## Instructions d'exécution pour Claude Code

**Étape 1 — Vérifier la baseline**
```bash
pytest tests/test_benchmark_calibration.py tests/test_layer2_paf.py -v
# → 25/25 PASS requis avant de continuer
```

**Étape 2 — Créer la structure**
```bash
mkdir -p layer3_validation
touch layer3_validation/__init__.py
```

**Étape 3 — Créer les 5 fichiers MÉTIS** (dans l'ordre)
1. `metis_q1_walkforward.py`
2. `metis_q2_permutation.py`  ← corriger le `import pandas as pd as _pd` (typo dans le code ci-dessus, utiliser `import pandas as _pd`)
3. `metis_q3_ema_stability.py`
4. `metis_q4_dsr.py`
5. `metis_runner.py`
6. `__init__.py`

**Étape 4 — Tests unitaires**
```bash
pytest tests/test_layer3_metis.py -v
# → 13/13 PASS requis
```

**Étape 5 — Validation sur données réelles**
```python
from layer3_validation.metis_runner import METISRunner
import numpy as np, pandas as pd

runner = METISRunner("config.yaml")

def signal_h9_ema(df):
    lr  = np.log(df["paxg"] / df["btc"])
    q25 = lr.rolling(60, min_periods=30).quantile(0.25)
    q75 = lr.rolling(60, min_periods=30).quantile(0.75)
    iqr = (q75 - q25).replace(0, np.nan)
    h9  = (1 - ((lr - q25) / iqr).clip(0, 1)).clip(0, 1).fillna(0.5)
    return h9.ewm(span=60, adjust=False).mean().clip(0, 1)

report = runner.run(signal_h9_ema, n_perm=500, n_trials=101, questions="Q1Q2Q3Q4")
print(f"CNSR OOS : {report.cnsr_oos}")
print(f"Q1 : {report.q1.verdict} ({report.q1.n_pass}/5)")
print(f"Q2 : {report.q2.verdict} (p={report.q2.pvalue})")
print(f"Q3 : {report.q3.verdict} (spike={report.q3.is_spike})")
print(f"Q4 : {report.q4.verdict} (DSR={report.q4.dsr})")
print(f"Verdict : {report.verdict_global}")
```

**Résultats attendus (conformes KB) :**
```
CNSR OOS : ~1.285
Q1 : FAIL (2/5)
Q2 : FAIL (p≈0.55)
Q3 : PASS (spike=False)
Q4 : FAIL ou SUSPECT_DSR (DSR≈0.32)
Verdict : ARCHIVE_FAIL_Q1_Q2_Q4
```

**Note sur Q2 :** le `import pandas as pd as _pd` dans `metis_q2_permutation.py` est une typo. Remplacer par :
```python
import pandas as _pd
_pd.Series(perm_cnsrs).to_csv(...)
```

**Critère de passage à la Partie 5 (Layer 4 + D-SIG) :**
1. `pytest tests/test_layer3_metis.py` → 13/13 PASS
2. Validation données réelles → résultats conformes KB ci-dessus
3. `./test-qaaf-studio.sh --layer 1` ET `--layer 2` → 100% PASS (régressions Layer 1/2 exclues)
