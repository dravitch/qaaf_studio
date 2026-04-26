from pathlib import Path
import yaml
import shutil


class NTrialsTracker:
    def __init__(self, state_path):
        self.path = Path(state_path)
        self._state: dict = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            data = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
            return data.get("n_trials", {})
        return {}

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            yaml.safe_dump({"n_trials": self._state}, f, allow_unicode=True)
        shutil.move(str(tmp), str(self.path))

    def register(self, famille: str) -> int:
        """Incrémente N_trials et retourne la nouvelle valeur."""
        self._state[famille] = self._state.get(famille, 0) + 1
        self._save()
        return self._state[famille]

    def get(self, famille: str) -> int:
        return self._state.get(famille, 0)

    def get_all(self) -> dict:
        return dict(self._state)

    def get_family_n_trials(self, family: str) -> int:
        """
        Reads N_trials_famille from the KB file.
        Checks top-level and hypothese.N_trials_famille (KB YAML convention).
        Fallback: returns the count stored in the tracker for this family.
        """
        if self.path.exists():
            raw = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
            # Top-level (standalone tracker file)
            if "N_trials_famille" in raw:
                return int(raw["N_trials_famille"])
            # Nested under hypothese (KB YAML convention)
            hyp = raw.get("hypothese", {})
            if "N_trials_famille" in hyp:
                return int(hyp["N_trials_famille"])
        return self._state.get(family, 1)

    def sync_from_inventory(self, inventory: dict) -> None:
        """Initialise depuis lentilles_inventory.yaml au démarrage du projet."""
        lentilles = inventory.get("lentilles", {})
        for section in ["active", "archivees"]:
            for l in lentilles.get(section, []):
                famille = l.get("famille", "")
                n = l.get("N_trials_famille", 0)
                if famille and n > self._state.get(famille, 0):
                    self._state[famille] = n
        self._save()
