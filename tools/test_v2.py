#!/usr/bin/env python3
"""
test_v2.py - Suite de Tests QAAF v2.0
======================================

🧪 Teste l'intégrité et la structure de v2.0

Usage:
    python test_v2.py              # Tests complets
    python test_v2.py --quick      # Tests rapides (structure seulement)
    python test_v2.py --import     # Tests d'imports
    python test_v2.py --pairs      # Tests multi-paires

Auteur: QAAF Team
Version: 2.0.0
"""

import sys
import json
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
    timestamp = datetime.now().strftime("%H:%M:%S")
    if prefix:
        print(f"{color}[{timestamp}] {prefix} {msg}{RESET}")
    else:
        print(f"{color}{msg}{RESET}")

class QAAFv2Tester:
    """Suite de tests pour QAAF v2.0"""
    
    def __init__(self):
        self.root = Path("qaaf_v2.0")
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "version": "2.0.0",
            "tests": {},
            "passed": 0,
            "failed": 0
        }
    
    def test_structure_exists(self):
        """Test: Structure v2.0 existe"""
        log("Test: Structure exists", CYAN, "🧪")
        
        if not self.root.exists():
            log("  ❌ Dossier qaaf_v2.0 n'existe pas", RED)
            self.results["failed"] += 1
            return False
        
        log("  ✅ Dossier qaaf_v2.0 trouvé", GREEN)
        self.results["passed"] += 1
        return True
    
    def test_layers_complete(self):
        """Test: 6 couches présentes"""
        log("Test: Layers complete", CYAN, "🧪")
        
        layers = [
            "metrics", "decision_engine", "integration",
            "backtesting", "data", "monitoring", "tools"
        ]
        
        missing = []
        for layer in layers:
            layer_path = self.root / layer
            if not layer_path.exists():
                missing.append(layer)
        
        if missing:
            log(f"  ❌ Couches manquantes: {', '.join(missing)}", RED)
            self.results["failed"] += 1
            return False
        
        log(f"  ✅ Toutes les {len(layers)} couches présentes", GREEN)
        self.results["passed"] += 1
        return True
    
    def test_metrics_complete(self):
        """Test: 4 métriques présentes"""
        log("Test: Metrics complete", CYAN, "🧪")
        
        metrics = ["vol_ratio", "bound_coherence", "alpha_stability", "spectral_score"]
        metrics_dir = self.root / "metrics"
        
        missing = []
        for metric in metrics:
            metric_path = metrics_dir / metric / "v1.0"
            if not metric_path.exists():
                missing.append(metric)
        
        if missing:
            log(f"  ❌ Métriques manquantes: {', '.join(missing)}", RED)
            self.results["failed"] += 1
            return False
        
        log(f"  ✅ Les 4 métriques présentes", GREEN)
        self.results["passed"] += 1
        return True
    
    def test_metric_files(self):
        """Test: Chaque métrique a les fichiers essentiels"""
        log("Test: Metric files structure", CYAN, "🧪")
        
        metrics = ["vol_ratio", "bound_coherence", "alpha_stability", "spectral_score"]
        metrics_dir = self.root / "metrics"
        
        all_ok = True
        for metric in metrics:
            metric_path = metrics_dir / metric / "v1.0"
            
            required_files = [
                "implementation.py",
                "certification.yaml",
                "tests/__init__.py",
                "tests/test_isolated.py",
                "tests/test_oos.py",
                "tests/test_orthogonality.py"
            ]
            
            missing = []
            for req_file in required_files:
                file_path = metric_path / req_file
                if not file_path.exists():
                    missing.append(req_file)
            
            if missing:
                log(f"  ⚠️  {metric}: manquent {len(missing)} fichiers", YELLOW)
                all_ok = False
            else:
                log(f"  ✅ {metric}: structure complète", GREEN)
        
        if all_ok:
            self.results["passed"] += 1
        else:
            self.results["failed"] += 1
        
        return all_ok
    
    def test_pairs_configured(self):
        """Test: 4 paires configurées"""
        log("Test: Pairs configured", CYAN, "🧪")
        
        pairs = ["btc_paxg", "spy_tlt", "spy_gld", "qqq_ief"]
        pairs_dir = self.root / "data" / "pairs"
        
        missing = []
        for pair in pairs:
            pair_file = pairs_dir / f"{pair}.yaml"
            if not pair_file.exists():
                missing.append(pair)
        
        if missing:
            log(f"  ❌ Paires manquantes: {', '.join(missing)}", RED)
            self.results["failed"] += 1
            return False
        
        log(f"  ✅ Les 4 paires configurées", GREEN)
        self.results["passed"] += 1
        return True
    
    def test_registry_valid(self):
        """Test: Registry JSON valide"""
        log("Test: Registry valid JSON", CYAN, "🧪")
        
        registry_path = self.root / "metrics" / "metrics_registry.json"
        
        if not registry_path.exists():
            log(f"  ❌ Registry n'existe pas", RED)
            self.results["failed"] += 1
            return False
        
        try:
            with open(registry_path) as f:
                registry = json.load(f)
            
            if "metrics" not in registry:
                log(f"  ❌ Registry invalide (pas de 'metrics')", RED)
                self.results["failed"] += 1
                return False
            
            if len(registry["metrics"]) != 4:
                log(f"  ⚠️  Registry a {len(registry['metrics'])} métriques (attendu 4)", YELLOW)
            
            log(f"  ✅ Registry JSON valide ({len(registry['metrics'])} métriques)", GREEN)
            self.results["passed"] += 1
            return True
        
        except json.JSONDecodeError as e:
            log(f"  ❌ Registry JSON invalide: {e}", RED)
            self.results["failed"] += 1
            return False
    
    def test_documentation_complete(self):
        """Test: Documentation présente"""
        log("Test: Documentation complete", CYAN, "🧪")
        
        docs = [
            "QAAF_v2.0_ARCHITECTURE.md",
            "METRICS_CERTIFICATION_PLAN.md",
            "PROGRESS_v2.0.md"
        ]
        
        missing = []
        for doc in docs:
            doc_path = self.root / doc
            if not doc_path.exists():
                missing.append(doc)
        
        if missing:
            log(f"  ⚠️  Documents manquants: {', '.join(missing)}", YELLOW)
            self.results["failed"] += 1
            return False
        
        log(f"  ✅ Documentation complète ({len(docs)} fichiers)", GREEN)
        self.results["passed"] += 1
        return True
    
    def test_init_files(self):
        """Test: __init__.py présents"""
        log("Test: __init__.py files", CYAN, "🧪")
        
        init_count = len(list(self.root.rglob("__init__.py")))
        
        if init_count < 15:
            log(f"  ⚠️  Seulement {init_count} __init__.py trouvés", YELLOW)
            self.results["failed"] += 1
            return False
        
        log(f"  ✅ {init_count} fichiers __init__.py trouvés", GREEN)
        self.results["passed"] += 1
        return True
    
    def test_python_imports(self):
        """Test: Imports Python valides (syntax check)"""
        log("Test: Python imports syntax", CYAN, "🧪")
        
        py_files = list(self.root.rglob("*.py"))
        
        errors = []
        for py_file in py_files:
            try:
                with open(py_file) as f:
                    compile(f.read(), py_file, 'exec')
            except SyntaxError as e:
                errors.append((py_file, str(e)))
        
        if errors:
            log(f"  ❌ {len(errors)} fichiers avec erreurs de syntaxe", RED)
            for file, error in errors[:3]:
                log(f"     {file}: {error[:50]}", RED)
            self.results["failed"] += 1
            return False
        
        log(f"  ✅ {len(py_files)} fichiers Python - syntaxe OK", GREEN)
        self.results["passed"] += 1
        return True
    
    def run_quick_tests(self):
        """Exécute tests rapides (structure seulement)"""
        log("\n" + "="*70, BLUE)
        log("⚡ Tests Rapides (Structure)", BLUE)
        log("="*70, BLUE)
        
        self.test_structure_exists()
        self.test_layers_complete()
        self.test_metrics_complete()
        self.test_pairs_configured()
    
    def run_all_tests(self):
        """Exécute tous les tests"""
        log("\n" + "="*70, BLUE)
        log("🧪 Suite Complète de Tests QAAF v2.0", BLUE)
        log("="*70, BLUE)
        
        # Tests de structure
        self.test_structure_exists()
        self.test_layers_complete()
        self.test_metrics_complete()
        self.test_metric_files()
        self.test_pairs_configured()
        
        # Tests de configuration
        self.test_registry_valid()
        self.test_documentation_complete()
        self.test_init_files()
        
        # Tests de syntaxe
        self.test_python_imports()
    
    def print_summary(self):
        """Affiche le résumé"""
        log("\n" + "="*70, BLUE)
        log("📊 RÉSUMÉ DES TESTS", BLUE)
        log("="*70, BLUE)
        
        total = self.results["passed"] + self.results["failed"]
        success_rate = (self.results["passed"] / total * 100) if total > 0 else 0
        
        log(f"\n✅ Réussis: {self.results['passed']}/{total} ({success_rate:.1f}%)", GREEN)
        log(f"❌ Échoués: {self.results['failed']}/{total}", RED if self.results["failed"] > 0 else GREEN)
        
        if self.results["failed"] == 0:
            log("\n🎉 Tous les tests passent! Architecture v2.0 opérationnelle.", GREEN)
            log("   Prochaine étape: python tools/check.py", CYAN)
            return True
        else:
            log(f"\n⚠️  {self.results['failed']} test(s) échoué(s)", YELLOW)
            return False

def main():
    """Point d'entrée"""
    import argparse
    
    parser = argparse.ArgumentParser(description="QAAF v2.0 Test Suite")
    parser.add_argument("--quick", action="store_true", help="Tests rapides")
    parser.add_argument("--import", action="store_true", help="Tests d'imports seulement")
    parser.add_argument("--pairs", action="store_true", help="Tests paires seulement")
    args = parser.parse_args()
    
    tester = QAAFv2Tester()
    
    if args.quick:
        tester.run_quick_tests()
    else:
        tester.run_all_tests()
    
    success = tester.print_summary()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()