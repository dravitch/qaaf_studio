"""
Composite Score - Agrégateur de métriques QAAF v2.0

Combine 3 métriques certifiées:
  - vol_ratio (risque volatilité)
  - bound_coherence (corrélation dynamique)
  - alpha_stability (persistance alpha)

Note: Ceci N'EST PAS une métrique isolée.
      C'est un OUTIL pour créer signaux composites.
      Ne passe PAS par MIF certification.

Author: QAAF Team
Version: 1.0
Date: 2025-10-24
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import warnings


class CompositeScore:
    """
    Agrégateur de métriques QAAF v2.0.
    
    Combine 3 métriques certifiées avec pondération configurable
    pour produire un score composite normalisé [0, 1].
    
    Attributes:
        weights: Dict des poids appliqués (somme = 1.0)
        normalization_range: Tuple (min, max) pour vol_ratio
    
    Example:
        >>> composite = CompositeScore(weights={
        ...     'vol_ratio': 0.5,
        ...     'bound_coherence': 0.3,
        ...     'alpha_stability': 0.2
        ... })
        >>> score = composite.compute(vol_series, bc_series, alpha_series)
        >>> print(f"Mean score: {score.mean():.2f}")
    """
    
    def __init__(self, 
                 weights: Optional[Dict[str, float]] = None,
                 normalization_range: Tuple[float, float] = (0.5, 2.0)):
        """
        Initialise l'agrégateur avec poids et range de normalisation.
        
        Args:
            weights: Dict avec poids pour chaque métrique.
                    Par défaut: égal (0.333, 0.333, 0.334)
                    Clés attendues: 'vol_ratio', 'bound_coherence', 'alpha_stability'
            normalization_range: Tuple (min, max) pour normaliser vol_ratio
                                Défaut: (0.5, 2.0) basé sur observations empiriques
        
        Raises:
            ValueError: Si poids invalides ou négatifs
        """
        self.weights = self._validate_and_normalize_weights(weights)
        self.normalization_range = normalization_range
        
        # Metadata pour tracking
        self.last_computation_metadata = None
    
    def _validate_and_normalize_weights(self, 
                                       weights: Optional[Dict[str, float]]) -> Dict[str, float]:
        """
        Valide et normalise les poids pour somme = 1.0.
        
        Args:
            weights: Dict des poids bruts ou None
        
        Returns:
            Dict des poids normalisés (somme = 1.0)
        
        Raises:
            ValueError: Si poids négatifs ou clés manquantes
        """
        expected_keys = {'vol_ratio', 'bound_coherence', 'alpha_stability'}
        
        # Poids par défaut si None
        if weights is None:
            return {
                'vol_ratio': 1/3,
                'bound_coherence': 1/3,
                'alpha_stability': 1/3
            }
        
        # Vérifier clés
        if not expected_keys.issubset(weights.keys()):
            missing = expected_keys - weights.keys()
            raise ValueError(
                f"Missing weight keys: {missing}. "
                f"Expected: {expected_keys}"
            )
        
        # Vérifier positivité
        for key, value in weights.items():
            if value < 0:
                raise ValueError(f"Negative weight for {key}: {value}")
        
        # Normaliser (somme = 1.0)
        total = sum(weights[k] for k in expected_keys)
        
        if total == 0:
            raise ValueError("Sum of weights is zero")
        
        normalized = {k: weights[k] / total for k in expected_keys}
        
        return normalized
    
    def compute(self, 
                vol_ratio: pd.Series,
                bound_coherence: pd.Series,
                alpha_stability: pd.Series) -> pd.Series:
        """
        Calcule score composite normalisé [0, 1].
        
        Pipeline:
          1. Valider alignement des séries
          2. Normaliser vol_ratio vers [0, 1]
          3. Combiner avec poids
          4. Retourner série composite [0, 1]
        
        Args:
            vol_ratio: Série vol_ratio (0 à +∞, typiquement 0.5-2.0)
            bound_coherence: Série bound_coherence (0 à 1.0)
            alpha_stability: Série alpha_stability (0 à 1.0)
        
        Returns:
            Série composite normalisée (0 à 1.0)
            Index identique aux inputs
        
        Raises:
            ValueError: Si séries incompatibles (longueurs, NaN > 10%, misalignment)
        
        Example:
            >>> score = composite.compute(vol, bc, alpha)
            >>> signal = (score > 0.6).astype(int)  # Buy si score > 0.6
        """
        # 1. Validation
        self._validate_metrics_alignment(vol_ratio, bound_coherence, alpha_stability)
        
        # 2. Normalisation vol_ratio
        vol_norm = normalize_vol_ratio(
            vol_ratio, 
            expected_range=self.normalization_range
        )
        
        # 3. Weighted sum
        composite = (
            self.weights['vol_ratio'] * vol_norm +
            self.weights['bound_coherence'] * bound_coherence +
            self.weights['alpha_stability'] * alpha_stability
        )
        
        # 4. Ensure [0, 1] (defensive)
        composite = composite.clip(0, 1)
        
        # Store basic metadata
        self.last_computation_metadata = {
            'n_points': len(composite),
            'n_nan': composite.isnull().sum()
        }
        
        return composite
    
    def compute_with_metadata(self,
                             vol_ratio: pd.Series,
                             bound_coherence: pd.Series,
                             alpha_stability: pd.Series) -> Tuple[pd.Series, Dict]:
        """
        Calcule score composite + métadonnées détaillées.
        
        Metadata inclut:
            - mean: Moyenne du score
            - std: Écart-type
            - min/max: Bornes observées
            - weights_used: Poids appliqués
            - correlation_with_inputs: Corrélation avec chaque métrique
            - n_points: Nombre de points
            - n_nan: Nombre de NaN
        
        Args:
            vol_ratio: Série vol_ratio
            bound_coherence: Série bound_coherence
            alpha_stability: Série alpha_stability
        
        Returns:
            Tuple (composite_series, metadata_dict)
        
        Example:
            >>> score, meta = composite.compute_with_metadata(vol, bc, alpha)
            >>> print(f"Correlation with vol_ratio: {meta['correlation_with_inputs']['vol_ratio']:.3f}")
        """
        # Compute composite
        composite = self.compute(vol_ratio, bound_coherence, alpha_stability)
        
        # Normalize vol_ratio for correlation
        vol_norm = normalize_vol_ratio(
            vol_ratio,
            expected_range=self.normalization_range
        )
        
        # Build metadata
        metadata = {
            'mean': float(composite.mean()),
            'std': float(composite.std()),
            'min': float(composite.min()),
            'max': float(composite.max()),
            'median': float(composite.median()),
            'q25': float(composite.quantile(0.25)),
            'q75': float(composite.quantile(0.75)),
            'weights_used': self.weights.copy(),
            'normalization_range': self.normalization_range,
            'correlation_with_inputs': {
                'vol_ratio': float(composite.corr(vol_norm)),
                'bound_coherence': float(composite.corr(bound_coherence)),
                'alpha_stability': float(composite.corr(alpha_stability))
            },
            'n_points': len(composite),
            'n_nan': int(composite.isnull().sum()),
            'nan_pct': float(composite.isnull().mean())
        }
        
        return composite, metadata
    
    def _validate_metrics_alignment(self,
                                   vol_ratio: pd.Series,
                                   bound_coherence: pd.Series,
                                   alpha_stability: pd.Series) -> None:
        """
        Valide que les 3 métriques sont alignées et valides.
        
        Checks:
          - Même longueur
          - Index identique (si pandas Series)
          - NaN < 10% pour chaque série
          - Ranges attendus respectés
        
        Args:
            vol_ratio: Série vol_ratio
            bound_coherence: Série bound_coherence
            alpha_stability: Série alpha_stability
        
        Raises:
            ValueError: Si validation échoue
        """
        # Check 1: Lengths
        lengths = {
            'vol_ratio': len(vol_ratio),
            'bound_coherence': len(bound_coherence),
            'alpha_stability': len(alpha_stability)
        }
        
        if len(set(lengths.values())) > 1:
            raise ValueError(
                f"Series have different lengths: {lengths}"
            )
        
        # Check 2: Index alignment (si pandas Series)
        if hasattr(vol_ratio, 'index'):
            if not vol_ratio.index.equals(bound_coherence.index):
                raise ValueError(
                    "vol_ratio and bound_coherence indices don't match. "
                    f"Intersection: {len(vol_ratio.index.intersection(bound_coherence.index))} / {len(vol_ratio)}"
                )
            if not vol_ratio.index.equals(alpha_stability.index):
                raise ValueError(
                    "vol_ratio and alpha_stability indices don't match. "
                    f"Intersection: {len(vol_ratio.index.intersection(alpha_stability.index))} / {len(vol_ratio)}"
                )
        
        # Check 3: NaN percentage
        nan_threshold = 0.10
        for name, series in [
            ('vol_ratio', vol_ratio),
            ('bound_coherence', bound_coherence),
            ('alpha_stability', alpha_stability)
        ]:
            nan_pct = series.isnull().mean()
            if nan_pct > nan_threshold:
                raise ValueError(
                    f"{name} has {nan_pct:.1%} NaN (>{nan_threshold:.0%} threshold). "
                    f"This exceeds acceptable data quality."
                )
        
        # Check 4: Ranges (warnings, pas erreurs)
        # bound_coherence et alpha_stability devraient être [0, 1]
        for name, series, expected_min, expected_max in [
            ('bound_coherence', bound_coherence, 0.0, 1.0),
            ('alpha_stability', alpha_stability, 0.0, 1.0)
        ]:
            actual_min = series.min()
            actual_max = series.max()
            
            if actual_min < expected_min - 0.1 or actual_max > expected_max + 0.1:
                warnings.warn(
                    f"{name} range [{actual_min:.2f}, {actual_max:.2f}] "
                    f"outside expected [{expected_min}, {expected_max}]. "
                    f"Normalization may clip values."
                )
        
        # vol_ratio: juste vérifier > 0
        if (vol_ratio <= 0).any():
            raise ValueError(
                f"vol_ratio contains non-positive values. "
                f"Min: {vol_ratio.min():.4f}"
            )


def normalize_vol_ratio(vol_ratio: pd.Series, 
                       expected_range: Tuple[float, float] = (0.5, 2.0)) -> pd.Series:
    """
    Normalise vol_ratio de [0, +∞] vers [0, 1].
    
    Méthode: Min-max scaling avec expected_range.
    Valeurs hors range sont clippées vers [0, 1].
    
    Formule:
        normalized = (x - min) / (max - min)
        clipped = clip(normalized, 0, 1)
    
    Args:
        vol_ratio: Série brute (0 à +∞)
        expected_range: Tuple (min, max) attendu
                       Défaut: (0.5, 2.0) basé sur BTC/PAXG, SPY/TLT
    
    Returns:
        Série normalisée [0, 1]
        - 0.0 = vol_ratio au minimum attendu (ou en dessous)
        - 1.0 = vol_ratio au maximum attendu (ou au dessus)
        - 0.5 = vol_ratio au milieu du range
    
    Example:
        >>> vol = pd.Series([0.5, 1.0, 1.5, 2.0, 2.5])
        >>> norm = normalize_vol_ratio(vol, expected_range=(0.5, 2.0))
        >>> # norm = [0.0, 0.333, 0.667, 1.0, 1.0]  (2.5 clipped à 1.0)
    
    Notes:
        - Préserve NaN
        - Valeurs < min → 0
        - Valeurs > max → 1
        - Linéaire entre min et max
    """
    min_val, max_val = expected_range
    
    if min_val >= max_val:
        raise ValueError(
            f"Invalid range: min={min_val} >= max={max_val}"
        )
    
    # Min-max normalization
    normalized = (vol_ratio - min_val) / (max_val - min_val)
    
    # Clip to [0, 1]
    normalized = normalized.clip(0, 1)
    
    return normalized


def validate_metrics_alignment(vol_ratio: pd.Series,
                               bound_coherence: pd.Series,
                               alpha_stability: pd.Series) -> bool:
    """
    Fonction utilitaire standalone pour valider alignement.
    
    Identique à CompositeScore._validate_metrics_alignment()
    mais utilisable sans instancier la classe.
    
    Args:
        vol_ratio: Série vol_ratio
        bound_coherence: Série bound_coherence  
        alpha_stability: Série alpha_stability
    
    Returns:
        True si valide
    
    Raises:
        ValueError: Si validation échoue
    
    Example:
        >>> try:
        ...     validate_metrics_alignment(vol, bc, alpha)
        ...     print("✅ Metrics aligned")
        ... except ValueError as e:
        ...     print(f"❌ Alignment failed: {e}")
    """
    # Créer instance temporaire juste pour validation
    temp_composite = CompositeScore()
    temp_composite._validate_metrics_alignment(
        vol_ratio, 
        bound_coherence, 
        alpha_stability
    )
    return True


# ============================================================
# Convenience Functions
# ============================================================

def quick_composite(vol_ratio: pd.Series,
                   bound_coherence: pd.Series,
                   alpha_stability: pd.Series,
                   weights: Optional[Dict[str, float]] = None) -> pd.Series:
    """
    Fonction rapide pour calculer composite avec poids par défaut.
    
    Args:
        vol_ratio: Série vol_ratio
        bound_coherence: Série bound_coherence
        alpha_stability: Série alpha_stability
        weights: Poids optionnels
    
    Returns:
        Série composite [0, 1]
    
    Example:
        >>> score = quick_composite(vol, bc, alpha)
        >>> signal = (score > 0.6).astype(int)
    """
    composite = CompositeScore(weights=weights)
    return composite.compute(vol_ratio, bound_coherence, alpha_stability)


def composite_signal(composite_score: pd.Series,
                    buy_threshold: float = 0.6,
                    sell_threshold: float = 0.4) -> pd.Series:
    """
    Convertit score composite en signaux trading.
    
    Args:
        composite_score: Série composite [0, 1]
        buy_threshold: Seuil achat (défaut: 0.6)
        sell_threshold: Seuil vente (défaut: 0.4)
    
    Returns:
        Série de signaux:
            +1: Buy (score > buy_threshold)
             0: Hold (between thresholds)
            -1: Sell (score < sell_threshold)
    
    Example:
        >>> signals = composite_signal(score, buy_threshold=0.65, sell_threshold=0.35)
        >>> # signals = [1, 1, 0, 0, -1, -1, 0, 1, ...]
    """
    signals = pd.Series(0, index=composite_score.index)
    signals[composite_score > buy_threshold] = 1
    signals[composite_score < sell_threshold] = -1
    return signals


# ============================================================
# Version & Info
# ============================================================

__version__ = "1.0.0"
__author__ = "QAAF Team"
__date__ = "2025-10-24"

def get_info() -> Dict:
    """
    Retourne informations sur le module.
    
    Returns:
        Dict avec version, auteur, métriques supportées
    """
    return {
        'version': __version__,
        'author': __author__,
        'date': __date__,
        'description': 'QAAF v2.0 Composite Score Aggregator',
        'supported_metrics': ['vol_ratio', 'bound_coherence', 'alpha_stability'],
        'mif_certified': False,  # Important: NOT a certified metric
        'type': 'aggregator_tool'
    }


if __name__ == "__main__":
    # Demo / Smoke test
    print("=" * 60)
    print("Composite Score - Demo")
    print("=" * 60)
    
    # Generate synthetic data
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    
    vol = pd.Series(np.random.uniform(0.8, 1.5, 100), index=dates)
    bc = pd.Series(np.random.uniform(0.3, 0.7, 100), index=dates)
    alpha = pd.Series(np.random.uniform(0.4, 0.8, 100), index=dates)
    
    # Test 1: Default weights
    print("\n1. Default weights (equal)")
    composite = CompositeScore()
    score = composite.compute(vol, bc, alpha)
    print(f"   Mean: {score.mean():.3f}")
    print(f"   Std:  {score.std():.3f}")
    print(f"   Range: [{score.min():.3f}, {score.max():.3f}]")
    
    # Test 2: Custom weights
    print("\n2. Custom weights (vol=0.5, bc=0.3, alpha=0.2)")
    composite2 = CompositeScore(weights={
        'vol_ratio': 0.5,
        'bound_coherence': 0.3,
        'alpha_stability': 0.2
    })
    score2 = composite2.compute(vol, bc, alpha)
    print(f"   Mean: {score2.mean():.3f}")
    print(f"   Std:  {score2.std():.3f}")
    
    # Test 3: With metadata
    print("\n3. With metadata")
    score3, meta = composite.compute_with_metadata(vol, bc, alpha)
    print(f"   Correlation with vol_ratio: {meta['correlation_with_inputs']['vol_ratio']:.3f}")
    print(f"   Correlation with bound_coherence: {meta['correlation_with_inputs']['bound_coherence']:.3f}")
    print(f"   Correlation with alpha_stability: {meta['correlation_with_inputs']['alpha_stability']:.3f}")
    
    # Test 4: Quick composite
    print("\n4. Quick composite (convenience)")
    quick_score = quick_composite(vol, bc, alpha)
    print(f"   Mean: {quick_score.mean():.3f}")
    
    # Test 5: Signals
    print("\n5. Trading signals")
    signals = composite_signal(score, buy_threshold=0.6, sell_threshold=0.4)
    print(f"   Buy signals:  {(signals == 1).sum()}")
    print(f"   Hold signals: {(signals == 0).sum()}")
    print(f"   Sell signals: {(signals == -1).sum()}")
    
    print("\n" + "=" * 60)
    print("✅ Demo complete - All functions working")
    print("=" * 60)
