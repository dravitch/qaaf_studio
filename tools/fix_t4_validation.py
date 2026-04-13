#!/usr/bin/env python3
"""
Fix T4 Phase 3 Validation Error
================================

Problème: ValueError: The truth value of a Series is ambiguous
Cause: yfinance retourne DataFrame → nan_ratio devient Series
Solution: Forcer conversion scalar avant comparaison

Usage:
    cd ~/Projects/01_QAAF/qaaf_v2.0/
    python tools/fix_t4_validation.py
    
Backups créés automatiquement dans: metrics/*/v1_0/*.backup
"""

import re
from pathlib import Path
import shutil
from datetime import datetime


def backup_file(filepath: Path) -> Path:
    """Crée backup avec timestamp."""
    backup_path = filepath.with_suffix(f".backup_{datetime.now():%Y%m%d_%H%M%S}")
    shutil.copy2(filepath, backup_path)
    print(f"✅ Backup créé: {backup_path.name}")
    return backup_path


def fix_vol_ratio():
    """Fix vol_ratio/v1_0/implementation.py"""
    
    filepath = Path("metrics/vol_ratio/v1_0/implementation.py")
    
    if not filepath.exists():
        print(f"❌ Fichier introuvable: {filepath}")
        return False
    
    # Backup
    backup_file(filepath)
    
    # Lire contenu
    content = filepath.read_text()
    
    # Pattern à remplacer
    old_pattern = r"""nan_ratio_1 = asset1_prices\.isna\(\)\.sum\(\) / len\(asset1_prices\)
        nan_ratio_2 = asset2_prices\.isna\(\)\.sum\(\) / len\(asset2_prices\)
        
        if nan_ratio_1 > 0\.05 or nan_ratio_2 > 0\.05:"""
    
    # Nouveau code
    new_code = """nan_ratio_1 = asset1_prices.isna().sum() / len(asset1_prices)
        nan_ratio_2 = asset2_prices.isna().sum() / len(asset2_prices)

        # Forcer conversion scalar (fix yfinance DataFrame)
        if isinstance(nan_ratio_1, pd.Series):
            nan_ratio_1 = float(nan_ratio_1.iloc[0])
        if isinstance(nan_ratio_2, pd.Series):
            nan_ratio_2 = float(nan_ratio_2.iloc[0])

        if nan_ratio_1 > 0.05 or nan_ratio_2 > 0.05:"""
    
    # Remplacer
    content_fixed = re.sub(old_pattern, new_code, content, flags=re.MULTILINE)
    
    if content == content_fixed:
        print("⚠️  vol_ratio: Pattern non trouvé (déjà fixé?)")
        return False
    
    # Écrire
    filepath.write_text(content_fixed)
    print(f"✅ vol_ratio fixé: {filepath}")
    return True


def fix_bound_coherence():
    """Fix bound_coherence/v1_0/implementation.py"""
    
    filepath = Path("metrics/bound_coherence/v1_0/implementation.py")
    
    if not filepath.exists():
        print(f"❌ Fichier introuvable: {filepath}")
        return False
    
    # Backup
    backup_file(filepath)
    
    # Lire contenu
    content = filepath.read_text()
    
    # Pattern à remplacer
    old_pattern = r"""# Check 1: NaN detection
        nan1 = asset1_prices\.isna\(\)\.sum\(\)
        nan2 = asset2_prices\.isna\(\)\.sum\(\)
        if nan1 > 0 or nan2 > 0:"""
    
    # Nouveau code
    new_code = """# Check 1: NaN detection
        nan1 = asset1_prices.isna().sum()
        nan2 = asset2_prices.isna().sum()
        
        # Forcer conversion scalar (fix yfinance DataFrame)
        if isinstance(nan1, pd.Series):
            nan1 = int(nan1.iloc[0])
        if isinstance(nan2, pd.Series):
            nan2 = int(nan2.iloc[0])
        
        if nan1 > 0 or nan2 > 0:"""
    
    # Remplacer
    content_fixed = re.sub(old_pattern, new_code, content, flags=re.MULTILINE)
    
    if content == content_fixed:
        print("⚠️  bound_coherence: Pattern non trouvé (déjà fixé?)")
        return False
    
    # Écrire
    filepath.write_text(content_fixed)
    print(f"✅ bound_coherence fixé: {filepath}")
    return True


def verify_fixes():
    """Vérifie que les fixes sont appliqués."""
    
    checks = []
    
    # Vérifier vol_ratio
    vol_ratio_path = Path("metrics/vol_ratio/v1_0/implementation.py")
    if vol_ratio_path.exists():
        content = vol_ratio_path.read_text()
        has_fix = "isinstance(nan_ratio_1, pd.Series)" in content
        checks.append(("vol_ratio", has_fix))
    
    # Vérifier bound_coherence
    bound_path = Path("metrics/bound_coherence/v1_0/implementation.py")
    if bound_path.exists():
        content = bound_path.read_text()
        has_fix = "isinstance(nan1, pd.Series)" in content
        checks.append(("bound_coherence", has_fix))
    
    return checks


def main():
    """Point d'entrée principal."""
    
    print("\n" + "="*70)
    print("FIX T4 PHASE 3 VALIDATION ERROR")
    print("="*70 + "\n")
    
    # Fix vol_ratio
    print("📝 Fixing vol_ratio...")
    vol_fixed = fix_vol_ratio()
    
    print()
    
    # Fix bound_coherence
    print("📝 Fixing bound_coherence...")
    bound_fixed = fix_bound_coherence()
    
    print("\n" + "="*70)
    
    # Vérification
    print("\n🔍 Vérification des fixes...")
    checks = verify_fixes()
    
    for metric, has_fix in checks:
        status = "✅" if has_fix else "❌"
        print(f"  {status} {metric}: {'OK' if has_fix else 'FAIL'}")
    
    # Instructions finales
    print("\n" + "="*70)
    print("PROCHAINES ÉTAPES")
    print("="*70)
    
    if all(has_fix for _, has_fix in checks):
        print("""
✅ Tous les fixes appliqués!

Lancer les tests:
    cd ~/Projects/01_QAAF/qaaf_v2.0/
    
    # Test rapide (2 métriques)
    pytest tests/test_integration_2metrics.py -v
    
    # Test complet (3 métriques)
    pytest tests/test_metrics_integration.py -v
    
Si succès:
    - Documenter dans certification.yaml
    - Passer à Phase 4 (BOME integration)
        """)
    else:
        print("""
⚠️  Certains fixes n'ont pas été appliqués.

Options:
    1. Vérifier manuellement les fichiers
    2. Appliquer le fix manuellement (voir phase3_diagnostic_summary.md)
    3. Restaurer depuis backup si erreur
        """)
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
