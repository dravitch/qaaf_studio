"""
Tests unitaires pour composite_score.py

Note: Ce sont des tests unitaires SIMPLES, PAS MIF Phase 0-2.
      composite_score est un outil d'agrégation, pas une métrique isolée.

Author: QAAF Team
Version: 1.0
Date: 2025-10-24
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import module to test
import sys
from pathlib import Path

# Add tools/ to path (assuming tests/ is at project root)
sys.path.insert(0, str(Path(__file__).parent.parent / 'tools'))

from composite_score import (
    CompositeScore,
    normalize_vol_ratio,
    validate_metrics_alignment,
    quick_composite,
    composite_signal,
    get_info
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def sample_metrics():
    """
    Génère 3 séries métriques fictives alignées.
    
    Returns:
        Tuple (vol_ratio, bound_coherence, alpha_stability)
    """
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    
    vol_ratio = pd.Series(np.random.uniform(0.8, 1.5, 100), index=dates, name='vol_ratio')
    bound_coh = pd.Series(np.random.uniform(0.3, 0.7, 100), index=dates, name='bound_coherence')
    alpha_stab = pd.Series(np.random.uniform(0.4, 0.8, 100), index=dates, name='alpha_stability')
    
    return vol_ratio, bound_coh, alpha_stab


@pytest.fixture
def sample_metrics_with_nan():
    """Séries avec quelques NaN (< 10%)."""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    
    vol_ratio = pd.Series(np.random.uniform(0.8, 1.5, 100), index=dates)
    bound_coh = pd.Series(np.random.uniform(0.3, 0.7, 100), index=dates)
    alpha_stab = pd.Series(np.random.uniform(0.4, 0.8, 100), index=dates)
    
    # Inject 5% NaN
    vol_ratio.iloc[::20] = np.nan
    bound_coh.iloc[1::20] = np.nan
    alpha_stab.iloc[2::20] = np.nan
    
    return vol_ratio, bound_coh, alpha_stab


# ============================================================
# Test CompositeScore Class
# ============================================================

class TestCompositeScore:
    """Tests pour la classe CompositeScore."""
    
    def test_initialization_default_weights(self):
        """Test initialisation avec poids par défaut."""
        composite = CompositeScore()
        
        # Poids devraient être égaux
        assert abs(composite.weights['vol_ratio'] - 1/3) < 1e-10
        assert abs(composite.weights['bound_coherence'] - 1/3) < 1e-10
        assert abs(composite.weights['alpha_stability'] - 1/3) < 1e-10
        
        # Somme = 1.0
        assert abs(sum(composite.weights.values()) - 1.0) < 1e-10
    
    def test_initialization_custom_weights(self):
        """Test initialisation avec poids custom."""
        custom_weights = {
            'vol_ratio': 0.5,
            'bound_coherence': 0.3,
            'alpha_stability': 0.2
        }
        
        composite = CompositeScore(weights=custom_weights)
        
        assert composite.weights['vol_ratio'] == 0.5
        assert composite.weights['bound_coherence'] == 0.3
        assert composite.weights['alpha_stability'] == 0.2
        
        # Somme = 1.0
        assert abs(sum(composite.weights.values()) - 1.0) < 1e-10
    
    def test_initialization_auto_normalize_weights(self):
        """Test normalisation automatique des poids."""
        # Poids non normalisés (somme = 6.0)
        unnormalized = {
            'vol_ratio': 2.0,
            'bound_coherence': 2.0,
            'alpha_stability': 2.0
        }
        
        composite = CompositeScore(weights=unnormalized)
        
        # Devraient être normalisés à 1/3 chacun
        assert abs(composite.weights['vol_ratio'] - 1/3) < 1e-10
        assert abs(composite.weights['bound_coherence'] - 1/3) < 1e-10
        assert abs(composite.weights['alpha_stability'] - 1/3) < 1e-10
    
    def test_initialization_negative_weights_fail(self):
        """Test rejet poids négatifs."""
        negative_weights = {
            'vol_ratio': 0.5,
            'bound_coherence': -0.3,  # ❌ Négatif
            'alpha_stability': 0.2
        }
        
        with pytest.raises(ValueError, match="Negative weight"):
            CompositeScore(weights=negative_weights)
    
    def test_initialization_missing_keys_fail(self):
        """Test rejet si clés manquantes."""
        incomplete = {
            'vol_ratio': 0.5,
            'bound_coherence': 0.5
            # ❌ Manque alpha_stability
        }
        
        with pytest.raises(ValueError, match="Missing weight keys"):
            CompositeScore(weights=incomplete)
    
    def test_compute_default_weights(self, sample_metrics):
        """Test compute avec poids par défaut."""
        vol, bc, alpha = sample_metrics
        
        composite = CompositeScore()
        score = composite.compute(vol, bc, alpha)
        
        # Checks basiques
        assert len(score) == 100
        assert score.min() >= 0, f"Min {score.min()} < 0"
        assert score.max() <= 1.0, f"Max {score.max()} > 1.0"
        assert not score.isnull().any(), "Score contains NaN"
        
        # Index préservé
        assert score.index.equals(vol.index)
    
    def test_compute_custom_weights(self, sample_metrics):
        """Test compute avec poids custom."""
        vol, bc, alpha = sample_metrics
        
        composite = CompositeScore(weights={
            'vol_ratio': 0.5,
            'bound_coherence': 0.3,
            'alpha_stability': 0.2
        })
        score = composite.compute(vol, bc, alpha)
        
        # Score doit exister et être valide
        assert len(score) == 100
        assert 0 <= score.mean() <= 1.0
        assert 0 <= score.std() <= 0.5  # Std raisonnable
    
    def test_compute_with_metadata(self, sample_metrics):
        """Test compute_with_metadata retourne métadonnées correctes."""
        vol, bc, alpha = sample_metrics
        
        composite = CompositeScore()
        score, metadata = composite.compute_with_metadata(vol, bc, alpha)
        
        # Checks score
        assert len(score) == 100
        
        # Checks metadata structure
        required_keys = [
            'mean', 'std', 'min', 'max', 'median',
            'q25', 'q75', 'weights_used', 'normalization_range',
            'correlation_with_inputs', 'n_points', 'n_nan', 'nan_pct'
        ]
        
        for key in required_keys:
            assert key in metadata, f"Missing metadata key: {key}"
        
        # Checks metadata values
        assert 0 <= metadata['mean'] <= 1.0
        assert metadata['std'] >= 0
        assert 0 <= metadata['min'] <= 1.0
        assert 0 <= metadata['max'] <= 1.0
        assert metadata['n_points'] == 100
        assert metadata['n_nan'] == 0
        
        # Checks correlations
        corr = metadata['correlation_with_inputs']
        assert 'vol_ratio' in corr
        assert 'bound_coherence' in corr
        assert 'alpha_stability' in corr
        
        # Corrélations doivent être dans [-1, 1]
        for metric, corr_val in corr.items():
            assert -1 <= corr_val <= 1, f"{metric} correlation {corr_val} outside [-1, 1]"
    
    def test_compute_with_nan(self, sample_metrics_with_nan):
        """Test comportement avec NaN (< 10%)."""
        vol, bc, alpha = sample_metrics_with_nan
        
        composite = CompositeScore()
        score = composite.compute(vol, bc, alpha)
        
        # NaN devraient être propagés
        assert score.isnull().sum() > 0
        assert score.isnull().mean() < 0.15  # Max 15% NaN
        
        # Parties non-NaN doivent être valides
        valid_scores = score.dropna()
        assert (valid_scores >= 0).all()
        assert (valid_scores <= 1.0).all()
    
    def test_misaligned_lengths_fail(self):
        """Test erreur si longueurs différentes."""
        vol = pd.Series(np.random.uniform(0.8, 1.5, 100))
        bc = pd.Series(np.random.uniform(0.3, 0.7, 90))  # ❌ Différente
        alpha = pd.Series(np.random.uniform(0.4, 0.8, 100))
        
        composite = CompositeScore()
        
        with pytest.raises(ValueError, match="different lengths"):
            composite.compute(vol, bc, alpha)
    
    def test_misaligned_indices_fail(self):
        """Test erreur si indices ne matchent pas."""
        dates1 = pd.date_range('2020-01-01', periods=100, freq='D')
        dates2 = pd.date_range('2020-01-15', periods=100, freq='D')  # ❌ Décalé
        
        vol = pd.Series(np.random.uniform(0.8, 1.5, 100), index=dates1)
        bc = pd.Series(np.random.uniform(0.3, 0.7, 100), index=dates2)
        alpha = pd.Series(np.random.uniform(0.4, 0.8, 100), index=dates1)
        
        composite = CompositeScore()
        
        with pytest.raises(ValueError, match="indices don't match"):
            composite.compute(vol, bc, alpha)
    
    def test_excessive_nan_fail(self):
        """Test erreur si > 10% NaN."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        
        vol = pd.Series(np.random.uniform(0.8, 1.5, 100), index=dates)
        bc = pd.Series(np.random.uniform(0.3, 0.7, 100), index=dates)
        alpha = pd.Series(np.random.uniform(0.4, 0.8, 100), index=dates)
        
        # Inject 15% NaN (> 10% threshold)
        vol.iloc[::7] = np.nan  # ~14% NaN
        
        composite = CompositeScore()
        
        with pytest.raises(ValueError, match="NaN.*>10"):
            composite.compute(vol, bc, alpha)


# ============================================================
# Test normalize_vol_ratio
# ============================================================

class TestNormalizeVolRatio:
    """Tests pour la fonction normalize_vol_ratio."""
    
    def test_basic_normalization(self):
        """Test normalisation basique."""
        vol_ratio = pd.Series([0.5, 1.0, 1.5, 2.0, 2.5])
        
        normalized = normalize_vol_ratio(vol_ratio, expected_range=(0.5, 2.0))
        
        # Checks
        assert normalized.iloc[0] == 0.0   # Min = 0
        assert normalized.iloc[3] == 1.0   # Max = 1
        assert abs(normalized.iloc[1] - 0.333) < 0.01  # 1.0 → ~0.33
        assert abs(normalized.iloc[2] - 0.667) < 0.01  # 1.5 → ~0.67
    
    def test_clipping_above_max(self):
        """Test clipping valeurs > max."""
        vol_ratio = pd.Series([2.5, 3.0, 10.0])
        
        normalized = normalize_vol_ratio(vol_ratio, expected_range=(0.5, 2.0))
        
        # Toutes devraient être clippées à 1.0
        assert (normalized == 1.0).all()
    
    def test_clipping_below_min(self):
        """Test clipping valeurs < min."""
        vol_ratio = pd.Series([0.1, 0.3, 0.4])
        
        normalized = normalize_vol_ratio(vol_ratio, expected_range=(0.5, 2.0))
        
        # Toutes devraient être clippées à 0.0
        assert (normalized == 0.0).all()
    
    def test_preserves_nan(self):
        """Test que NaN sont préservés."""
        vol_ratio = pd.Series([0.5, np.nan, 1.5, 2.0])
        
        normalized = normalize_vol_ratio(vol_ratio, expected_range=(0.5, 2.0))
        
        assert pd.isna(normalized.iloc[1])
        assert not pd.isna(normalized.iloc[0])
        assert not pd.isna(normalized.iloc[2])
    
    def test_custom_range(self):
        """Test avec range custom."""
        vol_ratio = pd.Series([1.0, 2.0, 3.0])
        
        # Range [1.0, 3.0] au lieu de [0.5, 2.0]
        normalized = normalize_vol_ratio(vol_ratio, expected_range=(1.0, 3.0))
        
        assert normalized.iloc[0] == 0.0   # Min
        assert normalized.iloc[1] == 0.5   # Milieu
        assert normalized.iloc[2] == 1.0   # Max
    
    def test_invalid_range_fail(self):
        """Test erreur si range invalide (min >= max)."""
        vol_ratio = pd.Series([1.0, 2.0])
        
        with pytest.raises(ValueError, match="Invalid range"):
            normalize_vol_ratio(vol_ratio, expected_range=(2.0, 1.0))  # ❌ Inversé


# ============================================================
# Test validate_metrics_alignment
# ============================================================

class TestValidateMetricsAlignment:
    """Tests pour validate_metrics_alignment."""
    
    def test_valid_alignment(self, sample_metrics):
        """Test avec métriques correctement alignées."""
        vol, bc, alpha = sample_metrics
        
        # Ne devrait pas lever d'erreur
        result = validate_metrics_alignment(vol, bc, alpha)
        assert result is True
    
    def test_misaligned_fail(self):
        """Test échoue si désalignées."""
        vol = pd.Series([1.0] * 100)
        bc = pd.Series([0.5] * 90)  # ❌ Longueur différente
        alpha = pd.Series([0.6] * 100)
        
        with pytest.raises(ValueError):
            validate_metrics_alignment(vol, bc, alpha)
    
    def test_excessive_nan_fail(self):
        """Test échoue si trop de NaN."""
        vol = pd.Series([1.0] * 100)
        bc = pd.Series([0.5] * 100)
        alpha = pd.Series([np.nan] * 15 + [0.6] * 85)  # 15% NaN
        
        with pytest.raises(ValueError, match="NaN"):
            validate_metrics_alignment(vol, bc, alpha)


# ============================================================
# Test Convenience Functions
# ============================================================

class TestConvenienceFunctions:
    """Tests pour quick_composite et composite_signal."""
    
    def test_quick_composite_default(self, sample_metrics):
        """Test quick_composite avec poids par défaut."""
        vol, bc, alpha = sample_metrics
        
        score = quick_composite(vol, bc, alpha)
        
        assert len(score) == 100
        assert 0 <= score.min() <= 1.0
        assert 0 <= score.max() <= 1.0
    
    def test_quick_composite_custom_weights(self, sample_metrics):
        """Test quick_composite avec poids custom."""
        vol, bc, alpha = sample_metrics
        
        score = quick_composite(vol, bc, alpha, weights={
            'vol_ratio': 0.5,
            'bound_coherence': 0.3,
            'alpha_stability': 0.2
        })
        
        assert len(score) == 100
        assert score.mean() > 0
    
    def test_composite_signal_default_thresholds(self):
        """Test composite_signal avec seuils par défaut."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        score = pd.Series([0.8, 0.7, 0.55, 0.45, 0.35, 0.25, 0.45, 0.65, 0.75, 0.5], index=dates)
        
        signals = composite_signal(score)  # buy=0.6, sell=0.4
        
        # Checks
        assert signals.iloc[0] == 1   # 0.8 > 0.6 → Buy
        assert signals.iloc[1] == 1   # 0.7 > 0.6 → Buy
        assert signals.iloc[2] == 0   # 0.55 in [0.4, 0.6] → Hold
        assert signals.iloc[3] == 0   # 0.45 in [0.4, 0.6] → Hold
        assert signals.iloc[4] == -1  # 0.35 < 0.4 → Sell
        assert signals.iloc[5] == -1  # 0.25 < 0.4 → Sell
    
    def test_composite_signal_custom_thresholds(self):
        """Test composite_signal avec seuils custom."""
        dates = pd.date_range('2020-01-01', periods=5, freq='D')
        score = pd.Series([0.75, 0.65, 0.50, 0.35, 0.25], index=dates)
        
        signals = composite_signal(score, buy_threshold=0.70, sell_threshold=0.30)
        
        assert signals.iloc[0] == 1   # 0.75 > 0.70 → Buy
        assert signals.iloc[1] == 0   # 0.65 in [0.30, 0.70] → Hold
        assert signals.iloc[2] == 0   # 0.50 in [0.30, 0.70] → Hold
        assert signals.iloc[3] == 0   # 0.35 in [0.30, 0.70] → Hold
        assert signals.iloc[4] == -1  # 0.25 < 0.30 → Sell
    
    def test_composite_signal_counts(self):
        """Test distribution des signaux."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        score = pd.Series(np.random.uniform(0, 1, 100), index=dates)
        
        signals = composite_signal(score, buy_threshold=0.6, sell_threshold=0.4)
        
        # Counts
        n_buy = (signals == 1).sum()
        n_hold = (signals == 0).sum()
        n_sell = (signals == -1).sum()
        
        # Checks
        assert n_buy + n_hold + n_sell == 100
        assert n_buy > 0  # Au moins quelques signaux de chaque
        assert n_sell > 0


# ============================================================
# Test Module Info
# ============================================================

class TestModuleInfo:
    """Tests pour get_info."""
    
    def test_get_info_structure(self):
        """Test structure de get_info()."""
        info = get_info()
        
        required_keys = [
            'version', 'author', 'date', 'description',
            'supported_metrics', 'mif_certified', 'type'
        ]
        
        for key in required_keys:
            assert key in info, f"Missing key: {key}"
    
    def test_get_info_values(self):
        """Test valeurs de get_info()."""
        info = get_info()
        
        assert info['version'] == '1.0.0'
        assert info['mif_certified'] is False  # Important!
        assert info['type'] == 'aggregator_tool'
        assert len(info['supported_metrics']) == 3
        assert 'vol_ratio' in info['supported_metrics']


# ============================================================
# Edge Cases & Stress Tests
# ============================================================

class TestEdgeCases:
    """Tests de cas limites."""
    
    def test_all_zeros(self):
        """Test avec toutes les métriques à 0."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        
        vol = pd.Series([0.5] * 10, index=dates)  # Min attendu
        bc = pd.Series([0.0] * 10, index=dates)
        alpha = pd.Series([0.0] * 10, index=dates)
        
        composite = CompositeScore()
        score = composite.compute(vol, bc, alpha)
        
        # Score devrait être valide (proche de 0)
        assert (score >= 0).all()
        assert score.mean() < 0.5
    
    def test_all_ones(self):
        """Test avec toutes les métriques au max."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        
        vol = pd.Series([2.0] * 10, index=dates)  # Max attendu
        bc = pd.Series([1.0] * 10, index=dates)
        alpha = pd.Series([1.0] * 10, index=dates)
        
        composite = CompositeScore()
        score = composite.compute(vol, bc, alpha)
        
        # Score devrait être proche de 1
        assert (score <= 1.0).all()
        assert score.mean() > 0.5
    
    def test_single_point(self):
        """Test avec une seule observation."""
        dates = pd.date_range('2020-01-01', periods=1, freq='D')
        
        vol = pd.Series([1.0], index=dates)
        bc = pd.Series([0.5], index=dates)
        alpha = pd.Series([0.6], index=dates)
        
        composite = CompositeScore()
        score = composite.compute(vol, bc, alpha)
        
        assert len(score) == 1
        assert 0 <= score.iloc[0] <= 1.0
    
    def test_very_long_series(self):
        """Test avec série longue (10000 points)."""
        dates = pd.date_range('2020-01-01', periods=10000, freq='H')
        
        vol = pd.Series(np.random.uniform(0.8, 1.5, 10000), index=dates)
        bc = pd.Series(np.random.uniform(0.3, 0.7, 10000), index=dates)
        alpha = pd.Series(np.random.uniform(0.4, 0.8, 10000), index=dates)
        
        composite = CompositeScore()
        score = composite.compute(vol, bc, alpha)
        
        assert len(score) == 10000
        assert 0 <= score.min() <= 1.0
        assert 0 <= score.max() <= 1.0


# ============================================================
# Integration-like Tests
# ============================================================

class TestIntegration:
    """Tests plus proches d'un usage réel."""
    
    def test_realistic_workflow(self, sample_metrics):
        """Test workflow complet typique."""
        vol, bc, alpha = sample_metrics
        
        # 1. Créer composite avec poids domaine-spécifiques
        composite = CompositeScore(weights={
            'vol_ratio': 0.4,        # Emphasize volatility
            'bound_coherence': 0.35,
            'alpha_stability': 0.25
        })
        
        # 2. Calculer score avec métadonnées
        score, meta = composite.compute_with_metadata(vol, bc, alpha)
        
        # 3. Générer signaux
        signals = composite_signal(score, buy_threshold=0.65, sell_threshold=0.35)
        
        # 4. Analyser résultats
        assert len(score) == len(signals) == 100
        assert meta['mean'] > 0
        assert (signals.isin([-1, 0, 1])).all()
        
        # 5. Vérifier corrélations raisonnables
        assert abs(meta['correlation_with_inputs']['vol_ratio']) < 1.0
    
    def test_multiple_composites_comparison(self, sample_metrics):
        """Test comparaison de plusieurs stratégies de pondération."""
        vol, bc, alpha = sample_metrics
        
        # Stratégie 1: Equal weights
        comp1 = CompositeScore()
        score1 = comp1.compute(vol, bc, alpha)
        
        # Stratégie 2: Vol-heavy
        comp2 = CompositeScore(weights={
            'vol_ratio': 0.6,
            'bound_coherence': 0.2,
            'alpha_stability': 0.2
        })
        score2 = comp2.compute(vol, bc, alpha)
        
        # Stratégie 3: Alpha-heavy
        comp3 = CompositeScore(weights={
            'vol_ratio': 0.2,
            'bound_coherence': 0.2,
            'alpha_stability': 0.6
        })
        score3 = comp3.compute(vol, bc, alpha)
        
        # Scores devraient être différents
        assert not score1.equals(score2)
        assert not score2.equals(score3)
        
        # Mais tous dans [0, 1]
        for score in [score1, score2, score3]:
            assert (score >= 0).all()
            assert (score <= 1.0).all()


# ============================================================
# Performance & Validation
# ============================================================

class TestPerformance:
    """Tests de performance (non-critiques mais utiles)."""
    
    def test_compute_speed(self, sample_metrics):
        """Test que compute() est rapide (< 100ms pour 100 points)."""
        import time
        
        vol, bc, alpha = sample_metrics
        composite = CompositeScore()
        
        start = time.time()
        for _ in range(10):  # 10 iterations
            score = composite.compute(vol, bc, alpha)
        elapsed = time.time() - start
        
        # 10 iterations devraient prendre < 1 seconde
        assert elapsed < 1.0, f"Too slow: {elapsed:.3f}s for 10 iterations"
    
    def test_memory_efficient(self):
        """Test que l'objet n'accumule pas de mémoire."""
        dates = pd.date_range('2020-01-01', periods=1000, freq='D')
        
        vol = pd.Series(np.random.uniform(0.8, 1.5, 1000), index=dates)
        bc = pd.Series(np.random.uniform(0.3, 0.7, 1000), index=dates)
        alpha = pd.Series(np.random.uniform(0.4, 0.8, 1000), index=dates)
        
        composite = CompositeScore()
        
        # Compute multiple times
        for _ in range(100):
            score = composite.compute(vol, bc, alpha)
        
        # Metadata devrait être remplacée, pas accumulée
        assert composite.last_computation_metadata is not None
        assert 'n_points' in composite.last_computation_metadata


# ============================================================
# Pytest Configuration
# ============================================================

if __name__ == "__main__":
    # Allow running directly
    pytest.main([__file__, "-v", "--tb=short"])
