from pathlib import Path
import yaml
import shutil


class NTrialsTracker:
    def __init__(self, state_path: Path):
        self.path = state_path
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
