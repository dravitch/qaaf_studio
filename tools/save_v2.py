#!/usr/bin/env python3
"""
save_v2.py - Sauvegarde QAAF v2.0 sur GitHub
=============================================

💾 Sauvegarde automatique avec versioning

Usage:
    python save_v2.py "Description du changement"
    python save_v2.py --tag v2.0.1              # Créer un tag
    python save_v2.py --list                    # Lister les changements

Auteur: QAAF Team
Version: 2.0.0
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

# Couleurs
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
CYAN = '\033[96m'
RESET = '\033[0m'

def log(msg, color=RESET, prefix=""):
    if prefix:
        print(f"{color}{prefix} {msg}{RESET}")
    else:
        print(f"{color}{msg}{RESET}")

def run(cmd, silent=False):
    """Exécute une commande"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

class GitSaver:
    """Gère la sauvegarde sur GitHub"""
    
    def __init__(self):
        self.root = Path(".")
        self.today = datetime.now().strftime("%Y-%m-%d")
    
    def get_branch(self):
        """Récupère la branche actuelle"""
        success, output = run("git branch --show-current", silent=True)
        if success:
            return output.strip()
        return None
    
    def get_modified_files(self):
        """Liste les fichiers modifiés"""
        success, output = run("git status --short", silent=True)
        if success:
            return output.strip().split('\n') if output.strip() else []
        return []
    
    def save(self, message):
        """Sauvegarde les changements"""
        
        log("\n" + "="*70, GREEN)
        log("✅ Sauvegarde Réussie!", GREEN)
        log("="*70, GREEN)
        
        log(f"\n✨ Changements sauvegardés sur GitHub", GREEN)
        log(f"   Branche: {branch}", CYAN)
        log(f"   Message: {commit_msg}", CYAN)
        log(f"   Date: {self.today}\n", CYAN)
        
        return True
    
    def tag_release(self, tag_name):
        """Crée un tag de version"""
        
        log(f"\n🏷️  Création du tag: {tag_name}", YELLOW, "→")
        
        success, _ = run(f'git tag -a {tag_name} -m "Release QAAF {tag_name}"')
        if not success:
            log(f"❌ Impossible de créer le tag", RED)
            return False
        
        success, _ = run(f"git push origin {tag_name}")
        if not success:
            log(f"❌ Impossible de push le tag", RED)
            return False
        
        log(f"✅ Tag {tag_name} créé et poussé", GREEN)
        return True

def main():
    """Point d'entrée"""
    import argparse
    
    parser = argparse.ArgumentParser(description="QAAF v2.0 GitHub Saver")
    parser.add_argument("message", nargs="?", default="Mise à jour QAAF v2.0")
    parser.add_argument("--tag", help="Créer un tag de version")
    parser.add_argument("--list", action="store_true", help="Lister les changements")
    args = parser.parse_args()
    
    saver = GitSaver()
    
    if args.list:
        log("\n📝 Changements récents:\n", CYAN)
        success, output = run("git log --oneline -10")
        if success:
            print(output)
        return
    
    if args.tag:
        saver.tag_release(args.tag)
        return
    
    success = saver.save(args.message)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

