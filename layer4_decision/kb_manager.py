from pathlib import Path
from dataclasses import dataclass
from datetime import date
from typing import Optional
import yaml
import shutil


@dataclass
class LentilleStatus:
    nom: str
    id: str
    status: str
    label: str
    color: str
    trend: str
    score: Optional[float]
    is_stale: bool
    cnsr_usd_fed: Optional[float]
    N_trials_famille: int
    metis_complete: bool


class KBManager:
    def __init__(self, hyp_or_inv_path, inventory_path=None):
        if inventory_path is None:
            # Backward-compat: single path = inventory only
            self.path = Path(hyp_or_inv_path)
            self.hyp_path = None
        else:
            # Two-path mode: hypothesis YAML + inventory YAML
            self.hyp_path = Path(hyp_or_inv_path)
            self.path = Path(inventory_path)
        self._data = None

    # ── Inventory ────────────────────────────────────────────────────────────

    def load(self) -> dict:
        if self._data is None:
            self._data = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        return self._data

    def get_active(self) -> list:
        return self.load().get("lentilles", {}).get("active", [])

    def get_archived(self) -> list:
        return self.load().get("lentilles", {}).get("archivees", [])

    def get_by_nom(self, nom: str) -> Optional[dict]:
        data = self.load()
        for section in ["active", "archivees"]:
            for l in data.get("lentilles", {}).get(section, []):
                if l.get("nom") == nom:
                    return l
        return None

    def update_metis_verdicts(self, nom: str, verdicts: dict) -> None:
        data = self.load()
        for l in data.get("lentilles", {}).get("active", []):
            if l.get("nom") == nom:
                l.update(verdicts)
                l["last_verified"] = str(date.today())
                break
        self._atomic_write(data)
        self._data = None

    def update_dsig_signal(self, nom: str, signal: dict) -> None:
        data = self.load()
        for l in data.get("lentilles", {}).get("active", []):
            if l.get("nom") == nom:
                l.update(signal)
                l["last_verified"] = str(date.today())
                break
        self._atomic_write(data)
        self._data = None

    def certify(self, nom: str) -> None:
        self._set_status(nom, "CERTIFIE", "active")

    def archive(self, nom: str, raison: str) -> None:
        data = self.load()
        lentilles = data.get("lentilles", {})
        active = lentilles.get("active", [])
        target = next((l for l in active if l.get("nom") == nom), None)
        if target:
            target.update({"status": "ARCHIVE", "raison_archive": raison, "ttl_days": None})
            active.remove(target)
            lentilles.setdefault("archivees", []).append(target)
            self._atomic_write(data)
            self._data = None

    def is_stale(self, lentille: dict) -> bool:
        ttl = lentille.get("ttl_days")
        if ttl is None:
            return False
        last = lentille.get("last_verified")
        if not last:
            return True
        return (date.today() - date.fromisoformat(str(last))).days > ttl

    def get_status(self, nom: str) -> Optional[LentilleStatus]:
        l = self.get_by_nom(nom)
        if not l:
            return None
        metis_done = all(
            l.get(f"metis_q{i}") not in ["en_cours", None]
            for i in [1, 2, 3, 4]
        )
        return LentilleStatus(
            nom=l["nom"],
            id=l.get("id", ""),
            status=l.get("status", ""),
            label=l.get("label", ""),
            color=l.get("color", ""),
            trend=l.get("trend", "N_A"),
            score=l.get("score"),
            is_stale=self.is_stale(l),
            cnsr_usd_fed=l.get("cnsr_usd_fed"),
            N_trials_famille=l.get("N_trials_famille", 0),
            metis_complete=metis_done,
        )

    def update_lentille(
        self,
        nom: str,
        statut: str,
        cnsr_oos: float,
        paf_verdict: str,
        metis_verdict: str,
        dsig_score: int,
    ) -> None:
        data = self.load()
        for l in data.get("lentilles", {}).get("active", []):
            if l.get("nom") == nom:
                l.update({
                    "status": "CERTIFIE" if statut == "active" else "ARCHIVE",
                    "cnsr_usd_fed": cnsr_oos,
                    "paf_verdict": paf_verdict,
                    "metis_verdict": metis_verdict,
                    "dsig_score": dsig_score,
                    "last_verified": str(date.today()),
                })
                break
        self._atomic_write(data)
        self._data = None

    # ── Hypothesis YAML ──────────────────────────────────────────────────────

    def pre_session_check(self, hypothesis: str, family: str) -> dict:
        """Returns {"recommendation": "PROCEED"} or {"recommendation": "SKIP_DUPLICATE"}."""
        if self.hyp_path and self.hyp_path.exists():
            data = self._load_hyp()
            if data.get("verdict") in ("certifie", "certified", "CERTIFIE"):
                return {
                    "recommendation": "SKIP_DUPLICATE",
                    "reason": f"{hypothesis} already certified",
                }
        return {"recommendation": "PROCEED"}

    def record_verdict(
        self,
        hypothesis: str,
        verdict: str,
        metrics: dict = None,
        notes: str = None,
    ) -> None:
        """
        Met à jour le fichier KB hypothèse avec le verdict final.
        Merge dans le dict existant — ne jamais appender.
        """
        if self.hyp_path is None:
            return
        data = self._load_hyp()
        data["date"]    = str(date.today())
        data["verdict"] = verdict
        if metrics:
            if not isinstance(data.get("metrics"), dict):
                data["metrics"] = {}
            data["metrics"].update(metrics)
        if notes:
            existing = data.get("notes") or ""
            data["notes"] = notes if not existing else existing
        self._save_hyp(data)

    def update_metis(self, metis_dict: dict) -> None:
        """Met à jour la section metis dans le dict existant."""
        if self.hyp_path is None:
            return
        data = self._load_hyp()
        if not isinstance(data.get("metis"), dict):
            data["metis"] = {}
        data["metis"].update(metis_dict)
        self._save_hyp(data)

    def _load_hyp(self) -> dict:
        if not (self.hyp_path and self.hyp_path.exists()):
            return {}
        content = self.hyp_path.read_text(encoding="utf-8")
        docs = list(yaml.safe_load_all(content))
        non_empty = [d for d in docs if d]
        if len(non_empty) > 1:
            import warnings
            warnings.warn(
                f"KB {self.hyp_path} est multi-blocs ({len(non_empty)} documents YAML). "
                f"Seul le premier est chargé. Exécuter tools/fix_kb_multiblock.py pour nettoyer.",
                stacklevel=2,
            )
        return non_empty[0] if non_empty else {}

    def _save_hyp(self, data: dict) -> None:
        self.hyp_path.write_text(
            yaml.dump(data, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    # ── Private ──────────────────────────────────────────────────────────────

    def _set_status(self, nom: str, new_status: str, section: str) -> None:
        data = self.load()
        for l in data.get("lentilles", {}).get(section, []):
            if l.get("nom") == nom:
                l["status"] = new_status
                l["last_verified"] = str(date.today())
                break
        self._atomic_write(data)
        self._data = None

    def _atomic_write(self, data: dict) -> None:
        self._atomic_write_path(self.path, data)

    def _atomic_write_path(self, path: Path, data: dict) -> None:
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, default_flow_style=False)
        shutil.move(str(tmp), str(path))
