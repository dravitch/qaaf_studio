"""
Script de promotion d'une métrique de cooking → production
"""
from pathlib import Path
import shutil
import yaml
import json

class MetricPromoter:
    def __init__(self, metric_name: str, version: str):
        self.metric_name = metric_name
        self.version = version
        self.root = Path("qaaf_v2.0")
        
    def validate_readiness(self) -> bool:
        """Vérifie que la métrique est prête pour production"""
        checks = []
        
        # 1. Vérifier certification
        cert_path = self.root / "tests/outputs/reports" / f"{self.metric_name}_v{self.version}_certification.md"
        checks.append(("Certification exists", cert_path.exists()))
        
        # 2. Vérifier résultats d'expériences
        exp_dirs = list((self.root / "tests/outputs" / self.metric_name).glob("exp_*"))
        checks.append(("At least 3 experiments", len(exp_dirs) >= 3))
        
        # 3. Vérifier implémentation
        impl_path = self.root / "tests/cooking" / f"{self.metric_name}_tuning/scripts/implementation.py"
        checks.append(("Implementation exists", impl_path.exists()))
        
        # 4. Vérifier tests
        test_path = self.root / "tests/cooking" / f"{self.metric_name}_tuning/scripts/tests"
        checks.append(("Tests exist", test_path.exists()))
        
        print("\n🔍 Validation de la préparation:")
        for check_name, passed in checks:
            status = "✅" if passed else "❌"
            print(f"  {status} {check_name}")
        
        return all(check[1] for check in checks)
    
    def promote(self):
        """Promeut la métrique vers production"""
        if not self.validate_readiness():
            print("\n❌ La métrique n'est pas prête pour promotion")
            return False
        
        print(f"\n🚀 Promotion de {self.metric_name} v{self.version} vers production...")
        
        # Créer la structure de destination
        dest = self.root / "metrics" / self.metric_name / f"v{self.version}"
        dest.mkdir(parents=True, exist_ok=True)
        
        # Copier l'implémentation
        src_impl = self.root / "tests/cooking" / f"{self.metric_name}_tuning/scripts/implementation.py"
        shutil.copy(src_impl, dest / "implementation.py")
        
        # Copier les tests
        src_tests = self.root / "tests/cooking" / f"{self.metric_name}_tuning/scripts/tests"
        dest_tests = dest / "tests"
        if dest_tests.exists():
            shutil.rmtree(dest_tests)
        shutil.copytree(src_tests, dest_tests)
        
        # Créer certification.yaml
        self.create_certification_yaml(dest)
        
        # Créer config.yaml avec paramètres optimaux
        self.create_config_yaml(dest)
        
        # Créer __init__.py
        self.create_init_py(dest)
        
        # Mettre à jour le symlink latest
        latest_link = self.root / "metrics" / self.metric_name / "latest"
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(f"v{self.version}")
        
        # Mettre à jour metrics_registry.json
        self.update_registry()
        
        print(f"✅ Promotion terminée: metrics/{self.metric_name}/v{self.version}/")
        return True
    
    def create_certification_yaml(self, dest: Path):
        """Crée le fichier de certification"""
        # Charger les résultats d'expériences
        exp_results = self.load_experiment_results()
        
        cert = {
            'metric_name': self.metric_name,
            'version': self.version,
            'status': 'certified',
            'certification_date': '2025-10-14',
            'parameters': {
                'window': exp_results['best_params']['window'],
                'lambda': exp_results['best_params']['lambda'],
            },
            'performance': {
                'sharpe_ratio': exp_results['best_params']['metrics']['sharpe_ratio'],
                'sortino_ratio': exp_results['best_params']['metrics']['sortino_ratio'],
                'max_drawdown': exp_results['best_params']['metrics']['max_drawdown'],
            },
            'validation': {
                'experiments': len(list((self.root / "tests/outputs" / self.metric_name).glob("exp_*"))),
                'data_points': 'multi-pair',
                'test_period': '2023-2024',
            },
            'dependencies': ['numpy', 'pandas'],
            'supports_multi_pair': True,
        }
        
        with open(dest / "certification.yaml", 'w') as f:
            yaml.dump(cert, f, default_flow_style=False)
    
    def create_config_yaml(self, dest: Path):
        """Crée le fichier de configuration avec paramètres optimaux"""
        exp_results = self.load_experiment_results()
        
        config = {
            'default_parameters': {
                'window': exp_results['best_params']['window'],
                'lambda_param': exp_results['best_params']['lambda'],
            },
            'thresholds': {
                'alert_low': 0.50,
                'alert_critical': 0.30,
            },
            'validation': {
                'min_periods': 30,
                'max_nan_rate': 0.10,
            }
        }
        
        with open(dest / "config.yaml", 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def create_init_py(self, dest: Path):
        """Crée __init__.py avec imports propres"""
        init_content = f'''"""
{self.metric_name} v{self.version}
Métrique certifiée pour QAAF
"""

from .implementation import BoundCoherence

__version__ = "{self.version}"
__all__ = ["BoundCoherence"]
'''
        with open(dest / "__init__.py", 'w') as f:
            f.write(init_content)
    
    def load_experiment_results(self):
        """Charge les résultats consolidés des expériences"""
        results_file = self.root / "tests/outputs" / self.metric_name / "consolidated_results.json"
        
        if not results_file.exists():
            # Fallback: charger depuis la dernière expérience
            exp_dirs = sorted((self.root / "tests/outputs" / self.metric_name).glob("exp_*"))
            if exp_dirs:
                results_file = exp_dirs[-1] / "tuning_analysis.json"
        
        with open(results_file) as f:
            return json.load(f)
    
    def update_registry(self):
        """Met à jour metrics_registry.json"""
        registry_path = self.root / "metrics/metrics_registry.json"
        
        if registry_path.exists():
            with open(registry_path) as f:
                registry = json.load(f)
        else:
            registry = {"metrics": {}}
        
        registry['metrics'][self.metric_name] = {
            'latest_version': self.version,
            'status': 'production',
            'supports_multi_pair': True,
        }
        
        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=2)

# Usage
if __name__ == "__main__":
    promoter = MetricPromoter("bound_coherence", "2.0")
    promoter.promote()

