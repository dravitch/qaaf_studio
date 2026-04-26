"""
session_template.py — Template de session QAAF Studio 3.0

Pattern LP (Long-running Process) avec checkpoints atomiques.
Référence : QAAF_Studio_Vision_Complete.md section 3.

Usage
-----
    from sessions.session_template import SessionTemplate

    class MySession(SessionTemplate):
        def run_iteration(self, i):
            # logique de l'itération i
            return {"iteration": i, "result": ...}

    sess = MySession(
        hypothesis="H10+EMA45j",
        family="EMA_span_variants",
        output_path="sessions/h10_ema45j/results.json",
    )
    sess.run(n_iterations=200)
"""

from __future__ import annotations

import sys, io
# Standard obligatoire QAAF Studio : encodage UTF-8 universel (Windows cp1252 safe)
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding and sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np


# ── Outils fondamentaux ───────────────────────────────────────────────────────

def atomic_save(path: str | Path, data: Any) -> None:
    """
    Sauvegarde atomique via write-to-tmp + rename POSIX.
    Garantit qu'un crash mid-write ne corrompt pas le fichier de sortie.

    Paramètres
    ----------
    path : chemin final du fichier (sera créé ou écrasé atomiquement)
    data : données sérialisables en JSON
    """
    path    = Path(path)
    tmp     = path.with_suffix(".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(data, indent=2, ensure_ascii=False, default=str)
    tmp.write_text(content, encoding="utf-8")
    os.replace(str(tmp), str(path))   # POSIX rename : atomique


CHECKPOINT_INTERVAL = 50  # sauvegarder toutes les N itérations


# ── Template de session ───────────────────────────────────────────────────────

class SessionTemplate:
    """
    Template de session avec :
    - Seeds déterministes (random + numpy)
    - Checkpoints atomiques toutes les CHECKPOINT_INTERVAL itérations
    - Reprise depuis le dernier checkpoint si interrupted

    Sous-classer et implémenter `run_iteration(i)`.
    """

    def __init__(
        self,
        hypothesis:   str,
        family:       str,
        output_path:  str | Path,
        seed:         int = 42,
    ):
        self.hypothesis  = hypothesis
        self.family      = family
        self.output_path = Path(output_path)
        self.seed        = seed

        # Seeds déterministes
        random.seed(seed)
        np.random.seed(seed)

        self._results:    list = []
        self._start_iter: int  = 0
        self._elapsed:    float = 0.0

    # ------------------------------------------------------------------
    # Interface à implémenter
    # ------------------------------------------------------------------

    def run_iteration(self, i: int) -> dict:
        """
        À surcharger dans les sous-classes.

        Paramètres
        ----------
        i : indice de l'itération courante (commence à 0)

        Retourne
        --------
        dict avec les résultats de l'itération (sérialisable JSON)
        """
        raise NotImplementedError(
            "Implémenter run_iteration(i) dans la sous-classe."
        )

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self, n_iterations: int = 200) -> list:
        """
        Exécute n_iterations itérations avec checkpoints atomiques.

        Tente de reprendre depuis un checkpoint existant si disponible.
        """
        # Tentative de reprise
        self._load_checkpoint()

        print(f"\n{'='*60}")
        print(f"  SESSION : {self.hypothesis}")
        print(f"  Famille : {self.family}")
        print(f"  Itérations : {n_iterations} | Checkpoint : /{CHECKPOINT_INTERVAL}")
        print(f"  Reprise depuis : itération {self._start_iter}")
        print(f"{'='*60}")

        t0 = time.perf_counter()

        for i in range(self._start_iter, n_iterations):
            result = self.run_iteration(i)
            self._results.append(result)

            # Checkpoint atomique
            if (i + 1) % CHECKPOINT_INTERVAL == 0:
                elapsed = time.perf_counter() - t0 + self._elapsed
                self._save_checkpoint(i + 1, elapsed)
                print(f"  ✅ Checkpoint {i+1}/{n_iterations} "
                      f"({elapsed:.1f}s)")

        # Sauvegarde finale
        elapsed = time.perf_counter() - t0 + self._elapsed
        self._save_final(n_iterations, elapsed)
        print(f"\n  Session terminée : {n_iterations} itérations en {elapsed:.1f}s")

        return self._results

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def _checkpoint_path(self) -> Path:
        return self.output_path.with_suffix(".checkpoint.json")

    def _load_checkpoint(self) -> None:
        cp = self._checkpoint_path()
        if not cp.exists():
            return
        try:
            data = json.loads(cp.read_text(encoding="utf-8"))
            self._results    = data.get("results",    [])
            self._start_iter = data.get("n_done",     0)
            self._elapsed    = data.get("elapsed_s",  0.0)
            print(f"  [checkpoint] Reprise depuis itération {self._start_iter}")
        except Exception as e:
            print(f"  [checkpoint] Erreur lecture — repartir de zéro ({e})")

    def _save_checkpoint(self, n_done: int, elapsed: float) -> None:
        atomic_save(self._checkpoint_path(), {
            "hypothesis": self.hypothesis,
            "family":     self.family,
            "n_done":     n_done,
            "elapsed_s":  elapsed,
            "results":    self._results,
        })

    def _save_final(self, n_done: int, elapsed: float) -> None:
        atomic_save(self.output_path, {
            "hypothesis":   self.hypothesis,
            "family":       self.family,
            "n_iterations": n_done,
            "elapsed_s":    elapsed,
            "seed":         self.seed,
            "results":      self._results,
        })
        # Supprimer le checkpoint après succès
        cp = self._checkpoint_path()
        if cp.exists():
            cp.unlink()


# ── Exemple d'utilisation ─────────────────────────────────────────────────────

if __name__ == "__main__":
    class DemoSession(SessionTemplate):
        def run_iteration(self, i):
            return {"i": i, "val": np.random.randn()}

    sess = DemoSession(
        hypothesis  = "DEMO",
        family      = "demo_family",
        output_path = "/tmp/qaaf_demo_session.json",
    )
    results = sess.run(n_iterations=100)
    print(f"Done: {len(results)} results")
