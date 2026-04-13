#!/usr/bin/env python3
"""
cleanup_v2.py - Nettoyage Intelligent QAAF v2.0

Nettoie les fichiers temporaires et backups de l'architecture v2.0

Usage:
    python tools/cleanup_v2.py              # Dry-run (voir ce qui sera supprime)
    python tools/cleanup_v2.py --execute    # Vraiment supprimer
    python tools/cleanup_v2.py --aggressive # Inclure __pycache__

Auteur: QAAF Team
Version: 2.0.0
"""

import os
import sys
import shutil
import logging
from pathlib import Path
from datetime import datetime

# ============================================================================
# SETUP GLOBAL
# ============================================================================

# PROJECT_ROOT
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Logging
os.makedirs(str(PROJECT_ROOT / "gitlog"), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(str(PROJECT_ROOT / "gitlog" / "cleanup.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Couleurs
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
RESET = "\033[0m"


def log(msg, color=RESET, prefix=""):
    """Affiche un message avec logging"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    if prefix:
        print(f"{color}[{timestamp}] {prefix} {msg}{RESET}")
    else:
        print(f"{color}{msg}{RESET}")
    logger.info(f"{prefix} {msg}")


# ============================================================================
# CLEANER
# ============================================================================


class QAAFv2Cleaner:
    """Nettoyeur intelligent QAAF v2.0"""

    TEMP_PATTERNS = [
        "*.backup",
        "*.bak",
        "*.txt.backup",
        "*.py.backup",
        "*~",
        "*.swp",
        "*.swo",
        ".DS_Store",
        "Thumbs.db",
    ]

    def __init__(self, dry_run=True):
        self.root = PROJECT_ROOT #/ "qaaf_v2.0"
        self.dry_run = dry_run
        self.total_size = 0
        self.file_count = 0
        logger.info(f"Cleaner init: dry_run={dry_run}, root={self.root}")

    def get_file_size(self, filepath):
        """Retourne taille en KB"""
        try:
            return os.path.getsize(filepath) / 1024
        except Exception as e:
            logger.error(f"Cannot get size of {filepath}: {e}")
            return 0

    def clean_temp_files(self):
        """Nettoie fichiers temporaires"""

        log("", YELLOW, "")
        log("Fichiers temporaires", YELLOW, "")

        count = 0
        for pattern in self.TEMP_PATTERNS:
            for filepath in self.root.rglob(pattern):
                if filepath.is_file():
                    size = self.get_file_size(filepath)
                    self.total_size += size
                    self.file_count += 1
                    count += 1

                    try:
                        rel_path = filepath.relative_to(self.root)
                    except ValueError:
                        rel_path = filepath

                    log(f"   {rel_path} ({size:.1f} KB)", CYAN)

                    if not self.dry_run:
                        try:
                            filepath.unlink()
                            logger.info(f"Deleted: {filepath}")
                        except Exception as e:
                            log(f"      Erreur: {e}", RED)
                            logger.error(f"Delete error: {filepath}: {e}")

        if count == 0:
            log("   (Aucun fichier trouve)", CYAN)

    def clean_pycache(self):
        """Nettoie __pycache__"""

        log("", YELLOW, "")
        log("Dossiers __pycache__", YELLOW, "")

        count = 0
        for pycache in self.root.rglob("__pycache__"):
            if pycache.is_dir():
                try:
                    size = (
                        sum(
                            f.stat().st_size
                            for f in pycache.rglob("*")
                            if f.is_file()
                        )
                        / 1024
                    )
                    self.total_size += size
                    count += 1

                    try:
                        rel_path = pycache.relative_to(self.root)
                    except ValueError:
                        rel_path = pycache

                    log(f"   {rel_path} ({size:.1f} KB)", CYAN)

                    if not self.dry_run:
                        shutil.rmtree(pycache)
                        logger.info(f"Deleted pycache: {pycache}")
                except Exception as e:
                    log(f"      Erreur: {e}", RED)
                    logger.error(f"Pycache error: {pycache}: {e}")

        if count == 0:
            log("   (Aucun dossier trouve)", CYAN)

    def clean_pytest_cache(self):
        """Nettoie .pytest_cache"""

        log("", YELLOW, "")
        log("Dossiers .pytest_cache", YELLOW, "")

        count = 0
        for pytest_cache in self.root.rglob(".pytest_cache"):
            if pytest_cache.is_dir():
                try:
                    size = (
                        sum(
                            f.stat().st_size
                            for f in pytest_cache.rglob("*")
                            if f.is_file()
                        )
                        / 1024
                    )
                    self.total_size += size
                    count += 1

                    try:
                        rel_path = pytest_cache.relative_to(self.root)
                    except ValueError:
                        rel_path = pytest_cache

                    log(f"   {rel_path} ({size:.1f} KB)", CYAN)

                    if not self.dry_run:
                        shutil.rmtree(pytest_cache)
                        logger.info(f"Deleted pytest_cache: {pytest_cache}")
                except Exception as e:
                    log(f"      Erreur: {e}", RED)
                    logger.error(f"Pytest cache error: {pytest_cache}: {e}")

        if count == 0:
            log("   (Aucun dossier trouve)", CYAN)

    def run(self):
        """Execute le nettoyage"""

        log("", BLUE)
        log("=" * 70, BLUE)
        if self.dry_run:
            log("Nettoyage QAAF v2.0 (DRY-RUN)", BLUE)
        else:
            log("Nettoyage QAAF v2.0 (EXECUTION)", BLUE)
        log("=" * 70, BLUE)

        if not self.root.exists():
            log(f"Dossier {self.root} n'existe pas", RED, "Erreur:")
            logger.error(f"Root not found: {self.root}")
            return

        self.clean_temp_files()
        self.clean_pycache()
        self.clean_pytest_cache()

        # Resume
        log("", BLUE)
        log("=" * 70, BLUE)
        log("Resume", BLUE)
        log("=" * 70, BLUE)

        log(f"", CYAN)
        log(f"Fichiers trouves: {self.file_count}", CYAN)
        log(
            f"Espace liberrable: {self.total_size:.1f} KB ({self.total_size/1024:.2f} MB)",
            CYAN,
        )

        if self.dry_run:
            log(f"", YELLOW)
            log("Pour nettoyer vraiment:", YELLOW)
            log("   python tools/cleanup_v2.py --execute", YELLOW)
        else:
            log(f"", GREEN)
            log("Nettoyage termine!", GREEN)

        logger.info(
            f"Cleanup complete: dry_run={self.dry_run}, "
            f"files={self.file_count}, size={self.total_size}KB"
        )


# ============================================================================
# MAIN
# ============================================================================


def main():
    """Point d'entree"""
    import argparse

    parser = argparse.ArgumentParser(description="QAAF v2.0 Cleaner")
    parser.add_argument("--execute", action="store_true", help="Nettoyer vraiment")
    parser.add_argument(
        "--aggressive",
        action="store_true",
        help="Inclure __pycache__ (automatic avec --execute)",
    )
    args = parser.parse_args()

    cleaner = QAAFv2Cleaner(dry_run=not args.execute)
    cleaner.run()


if __name__ == "__main__":
    main()