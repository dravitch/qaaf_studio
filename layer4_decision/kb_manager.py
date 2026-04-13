"""
kb_manager.py — Layer 4 QAAF Studio 3.0

Mémoire active du studio.
Empêche de retester ce qu'on sait déjà.
Applique les règles d'arrêt globales (table architecture v1.1).
Alimente N_trials via NTrialsTracker.

Vérification KB avant tout test (4 questions architecture)
----------------------------------------------------------
1. Cette hypothèse a-t-elle déjà été testée ?
2. Une hypothèse structurellement similaire a-t-elle échoué ?
3. Quels artefacts connus cette hypothèse pourrait-elle déclencher ?
4. Quel est le N_trials actuel pour cette famille ?
"""

from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml


VALID_STATUTS = {"en_cours", "certifie", "archive", "suspect_dsr", "suspendu"}

# Règles d'arrêt globales (architecture v1.1)
STOP_RULES = {
    "PAF_D1_PASSIF_DOMINE": (
        "B_passif domine en PAF D1 — hiérarchie absente.",
        "Requalifier la paire, pas optimiser le signal.",
    ),
    "PAF_D2_TRANSPARENT": (
        "Composante transparente en PAF D2.",
        "Chercher la source de la performance ailleurs.",
    ),
    "PAF_D3_ARTEFACT": (
        "Artefact de lissage confirmé en PAF D3.",
        "Accepter la solution minimale (EMA triviale).",
    ),
    "METIS_Q1_CONJONCTUREL": (
        "Walk-forward < 3/5 fenêtres en MÉTIS Q1.",
        "Hypothèse conjoncturelle — archiver.",
    ),
    "METIS_Q2_NON_SIG": (
        "p-value ≥ 0.05 en MÉTIS Q2.",
        "Gain non significatif vs benchmark passif — archiver.",
    ),
    "METIS_Q3_INSTABLE": (
        "Span EMA instable (spike isolé) en MÉTIS Q3.",
        "Sur-ajustement — revenir à paramètre plus robuste.",
    ),
    "METIS_Q4_SUSPECT": (
        "DSR < 0.95 en MÉTIS Q4.",
        "Marquer SUSPECT_DSR. Non déployable. Réévaluer si N_trials ↓ ou T ↑.",
    ),
}


class KBManager:
    """
    Gestionnaire de la Knowledge Base.

    Usage
    -----
    kb = KBManager("layer4_decision/kb_h9_ema60j.yaml",
                   "layer4_decision/lentilles_inventory.yaml")
    check = kb.pre_session_check("H9+EMA60j", "EMA_span_variants")
    kb.record_verdict("H9+EMA60j", "certifie", metrics={...})
    kb.update_metis({...})
    kb.print_inventory()
    """

    def __init__(
        self,
        kb_hyp_path: str = "layer4_decision/kb_h9_ema60j.yaml",
        kb_inv_path: str = "layer4_decision/lentilles_inventory.yaml",
    ):
        self._hyp_path = Path(kb_hyp_path)
        self._inv_path = Path(kb_inv_path)

    # ------------------------------------------------------------------
    # Vérification pré-session
    # ------------------------------------------------------------------

    def pre_session_check(self, hypothesis: str, family: str) -> dict:
        print(f"\n{'='*55}")
        print(f"KB CHECK (Couche 4) — {hypothesis}")
        print(f"{'='*55}")

        report = {
            "hypothesis":      hypothesis,
            "family":          family,
            "q1_duplicate":    False,
            "q1_statut":       None,
            "q2_similar_fail": [],
            "q3_artifacts":    [],
            "q4_n_trials":     0,
            "recommendation":  "PROCEED",
        }

        hyp_data = self._load_hyp()
        hyp      = hyp_data.get("hypothese", {})

        # Q1
        nom, statut = hyp.get("nom", ""), hyp.get("statut", "")
        if nom == hypothesis and statut in ("certifie", "archive", "suspect_dsr"):
            report["q1_duplicate"]   = True
            report["q1_statut"]      = statut
            report["recommendation"] = "SKIP_DUPLICATE"
            print(f"  ⚠️  Q1 : '{hypothesis}' déjà dans KB — statut '{statut}'.")
        elif nom == hypothesis:
            print(f"  ✅ Q1 : '{hypothesis}' en cours dans KB — continuer.")
        else:
            print(f"  ✅ Q1 : '{hypothesis}' nouvelle hypothèse.")

        # Q2
        inv      = self._load_inv()
        archived = inv.get("lentilles", {}).get("archivees", []) or []
        for e in archived:
            if self._similar(hypothesis, e.get("nom", "")):
                report["q2_similar_fail"].append(
                    {"nom": e["nom"], "raison": e.get("raison", "")}
                )
                print(f"  ⚠️  Q2 : Similaire archivée : '{e['nom']}'.")
        if not report["q2_similar_fail"]:
            print(f"  ✅ Q2 : Pas d'hypothèse similaire archivée.")

        # Q3
        artifacts = []
        if "dca" in hypothesis.lower():
            artifacts.append("Biais DCA — utiliser lump sum.")
        if "/" in hypothesis:
            artifacts.append("Biais de numéraire — CNSR-USD obligatoire.")
        n_cur = self._n_trials(family)
        if n_cur > 30:
            artifacts.append(f"Multiple testing (N={n_cur}) — DSR exigeant.")
        report["q3_artifacts"] = artifacts
        for a in artifacts:
            print(f"  ⚠️  Q3 : {a}")
        if not artifacts:
            print(f"  ✅ Q3 : Pas d'artefact identifié.")

        # Q4
        report["q4_n_trials"] = n_cur
        print(f"  ℹ️  Q4 : N_trials famille '{family}' = {n_cur}")
        print(f"\n  → Recommandation : {report['recommendation']}")
        return report

    # ------------------------------------------------------------------
    # Verdicts et mises à jour
    # ------------------------------------------------------------------

    def record_verdict(
        self,
        hypothesis: str,
        verdict:    str,
        metrics:    Optional[dict] = None,
        notes:      str = "",
    ) -> None:
        if verdict not in VALID_STATUTS:
            raise ValueError(f"Verdict inconnu : '{verdict}'. Valides : {VALID_STATUTS}")

        hyp_data = self._load_hyp()
        hyp      = hyp_data.get("hypothese", {})
        hyp["statut"]        = verdict
        hyp["verdict_final"] = verdict.upper()
        if metrics:
            hyp.setdefault("metriques_oos", {}).update(metrics)
        hyp.setdefault("historique", []).append({
            "date":   datetime.now().strftime("%Y-%m-%d"),
            "action": f"Verdict final : {verdict.upper()}",
            "notes":  notes,
            "auteur": "KBManager",
        })
        self._save_hyp(hyp_data)
        print(f"\n[KBManager] '{hypothesis}' → {verdict.upper()} ✅")

    def update_metis(self, metis_dict: dict) -> None:
        """Met à jour le nœud 'metis' dans la KB hypothèse."""
        hyp_data = self._load_hyp()
        hyp_data.get("hypothese", {}).setdefault("metis", {}).update(metis_dict)
        self._save_hyp(hyp_data)

    def update_lentille(
        self,
        nom:         str,
        statut:      str,   # "active" | "archivee" | "queue"
        cnsr_oos:    Optional[float] = None,
        paf_verdict: Optional[str]   = None,
        raison:      Optional[str]   = None,
        **kwargs: Any,
    ) -> None:
        inv      = self._load_inv()
        lentilles = inv.setdefault("lentilles", {})
        entry: dict = {"nom": nom}
        if cnsr_oos   is not None: entry["cnsr_oos"]    = cnsr_oos
        if paf_verdict:             entry["paf_verdict"] = paf_verdict
        if raison:                  entry["raison"]      = raison
        entry.update(kwargs)

        for key in ("archivees", "active", "queue"):
            lentilles.setdefault(key, [])
            lentilles[key] = [e for e in lentilles[key] if e.get("nom") != nom]

        bucket = "archivees" if statut == "archivee" else statut
        lentilles.setdefault(bucket, []).append(entry)
        self._save_inv(inv)
        print(f"[KBManager] Lentille '{nom}' → {statut}.")

    def apply_stop_rule(self, rule_key: str) -> str:
        if rule_key not in STOP_RULES:
            return ""
        diag, action = STOP_RULES[rule_key]
        print(f"\n🛑 RÈGLE D'ARRÊT : {rule_key}")
        print(f"   Diagnostic : {diag}")
        print(f"   Action     : {action}")
        return action

    def print_inventory(self) -> None:
        inv       = self._load_inv()
        lentilles = inv.get("lentilles", {})
        print(f"\n{'='*55}\nINVENTAIRE DES LENTILLES\n{'='*55}")
        for bucket, label in [("active",    "🟢 ACTIVE"),
                               ("archivees", "📁 ARCHIVÉES"),
                               ("queue",     "⏳ QUEUE")]:
            entries = lentilles.get(bucket, [])
            if not entries:
                continue
            print(f"\n{label}")
            for e in entries:
                c = f"  CNSR={e['cnsr_oos']:.2f}" if e.get("cnsr_oos") is not None else ""
                print(f"  • {e['nom']:30s}{c}")
                if e.get("raison"):
                    print(f"    {e['raison']}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _n_trials(self, family: str) -> int:
        hyp = self._load_hyp().get("hypothese", {})
        return int(hyp.get("N_trials_famille", 0)) if hyp.get("famille") == family else 0

    def _similar(self, h1: str, h2: str) -> bool:
        stop = {"le", "la", "et", "ou", "de", "du", "un", "une", "a"}
        t1   = {t for t in h1.lower().replace("+", " ").split() if t not in stop}
        t2   = {t for t in h2.lower().replace("+", " ").split() if t not in stop}
        return len(t1 & t2) >= 2

    def _load_hyp(self) -> dict:
        if not self._hyp_path.exists():
            return {}
        with open(self._hyp_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _save_hyp(self, data: dict) -> None:
        self._hyp_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._hyp_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False,
                      default_flow_style=False)

    def _load_inv(self) -> dict:
        if not self._inv_path.exists():
            return {}
        with open(self._inv_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _save_inv(self, data: dict) -> None:
        self._inv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._inv_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False,
                      default_flow_style=False)
