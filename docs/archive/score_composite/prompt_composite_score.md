# 📋 PROMPT POUR CLAUDE - Création composite_score

**Context**: QAAF v2.0 a 3 métriques certifiées MIF (vol_ratio, bound_coherence, alpha_stability). Nous avons besoin d'un **agrégateur** qui les combine en un score unique.

**Important**: `composite_score` N'EST PAS une métrique isolée. C'est un **outil d'agrégation**. Il ne passe PAS par MIF certification (Phase 0-2).

---

## 🎯 OBJECTIF

Créer `tools/composite_score.py` qui:
1. Prend les 3 métriques certifiées en input
2. Les combine avec poids configurables
3. Produit un score composite normalisé [0, 1]
4. Inclut validation de cohérence (pas de NaN, range correct)

---

## 📐 SPECIFICATIONS TECHNIQUES

### Localisation
```
qaaf_v2.0/
├── metrics/                    # 3 métriques certifiées (ne pas toucher)
│   ├── vol_ratio/
│   ├── bound_coherence/
│   └── alpha_stability/
│
└── tools/                      # Utilitaires
    ├── composite_score.py      ← À CRÉER
    └── certify_metric.py       # Existant
```

### Structure de fichier

```python
# tools/composite_score.py

"""
Composite Score - Agrégateur de métriques QAAF v2.0

Combine 3 métriques certifiées:
  - vol_ratio (risque volatilité)
  - bound_coherence (corrélation dynamique)
  - alpha_stability (persistance alpha)

Note: Ceci N'EST PAS une métrique isolée.
      C'est un OUTIL pour créer un signal composite.
      Ne passe PAS par MIF certification.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


class CompositeScore:
    """
    Agrégateur de métriques QAAF v2.0.
    
    Usage:
        composite = CompositeScore(weights={'vol_ratio': 0.5, 'bound_coherence': 0.3, 'alpha_stability': 0.2})
        score = composite.compute(vol_ratio_series, bound_coherence_series, alpha_stability_series)
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialise l'agrégateur avec poids.
        
        Args:
            weights: Dict avec poids pour chaque métrique.
                     Par défaut: égal (0.33, 0.33, 0.34)
        """
        # TODO: Implémenter validation poids
        # TODO: Normaliser poids (somme = 1.0)
        pass
    
    def compute(self, 
                vol_ratio: pd.Series,
                bound_coherence: pd.Series,
                alpha_stability: pd.Series) -> pd.Series:
        """
        Calcule score composite.
        
        Args:
            vol_ratio: Série vol_ratio (0 à +∞, typiquement 0.5-2.0)
            bound_coherence: Série bound_coherence (0 à 1.0)
            alpha_stability: Série alpha_stability (0 à 1.0)
        
        Returns:
            Série composite normalisée (0 à 1.0)
        
        Raises:
            ValueError: Si séries incompatibles (longueurs, NaN excessifs)
        """
        # TODO: Valider inputs (alignement, NaN, ranges)
        # TODO: Normaliser vol_ratio (scale to 0-1)
        # TODO: Combiner avec poids
        # TODO: Retourner série composite
        pass
    
    def compute_with_metadata(self,
                               vol_ratio: pd.Series,
                               bound_coherence: pd.Series,
                               alpha_stability: pd.Series) -> tuple:
        """
        Calcule score composite + métadonnées.
        
        Returns:
            Tuple (composite_series, metadata_dict)
            
        Metadata contient:
            - mean: Moyenne du score
            - std: Écart-type
            - weights_used: Poids appliqués
            - correlation_with_inputs: Corrélation avec chaque métrique
        """
        # TODO: Implémenter calcul + metadata
        pass


def normalize_vol_ratio(vol_ratio: pd.Series, 
                        expected_range=(0.5, 2.0)) -> pd.Series:
    """
    Normalise vol_ratio de [0, +∞] vers [0, 1].
    
    Méthode: Min-max scaling avec expected_range
    
    Args:
        vol_ratio: Série brute
        expected_range: Tuple (min, max) attendu
    
    Returns:
        Série normalisée [0, 1]
    """
    # TODO: Implémenter normalisation
    # Hint: (x - min) / (max - min), clamped to [0, 1]
    pass


def validate_metrics_alignment(vol_ratio: pd.Series,
                                 bound_coherence: pd.Series,
                                 alpha_stability: pd.Series) -> bool:
    """
    Valide que les 3 métriques sont alignées (même index, pas trop de NaN).
    
    Checks:
      - Même longueur
      - Index identique (ou intersection > 80%)
      - NaN < 10% pour chaque série
    
    Returns:
        True si valide, sinon raise ValueError
    """
    # TODO: Implémenter validation
    pass
```

---

## 📋 TESTS REQUIS

### Tests unitaires (PAS MIF Phase 0-2)

Créer `tests/test_composite_score.py`:

```python
import pytest
import pandas as pd
import numpy as np
from tools.composite_score import CompositeScore, normalize_vol_ratio


class TestCompositeScore:
    """Tests unitaires pour composite_score."""
    
    @pytest.fixture
    def sample_metrics(self):
        """Génère 3 séries métriques fictives."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        
        vol_ratio = pd.Series(np.random.uniform(0.8, 1.5, 100), index=dates)
        bound_coh = pd.Series(np.random.uniform(0.3, 0.7, 100), index=dates)
        alpha_stab = pd.Series(np.random.uniform(0.4, 0.8, 100), index=dates)
        
        return vol_ratio, bound_coh, alpha_stab
    
    def test_default_weights(self, sample_metrics):
        """Test avec poids par défaut (égaux)."""
        vol, bc, alpha = sample_metrics
        
        composite = CompositeScore()
        score = composite.compute(vol, bc, alpha)
        
        # Checks
        assert len(score) == 100
        assert score.min() >= 0
        assert score.max() <= 1.0
        assert not score.isnull().any()
    
    def test_custom_weights(self, sample_metrics):
        """Test avec poids custom."""
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
    
    def test_normalize_vol_ratio(self):
        """Test normalisation vol_ratio."""
        vol_ratio = pd.Series([0.5, 1.0, 1.5, 2.0, 2.5])
        
        normalized = normalize_vol_ratio(vol_ratio, expected_range=(0.5, 2.0))
        
        # Checks
        assert normalized.min() >= 0
        assert normalized.max() <= 1.0
        assert normalized.iloc[0] == 0.0   # Min = 0
        assert normalized.iloc[3] == 1.0   # Max = 1
    
    def test_metadata(self, sample_metrics):
        """Test métadonnées."""
        vol, bc, alpha = sample_metrics
        
        composite = CompositeScore()
        score, metadata = composite.compute_with_metadata(vol, bc, alpha)
        
        # Checks metadata
        assert 'mean' in metadata
        assert 'std' in metadata
        assert 'weights_used' in metadata
        assert 'correlation_with_inputs' in metadata
    
    def test_misaligned_series(self):
        """Test erreur si séries désalignées."""
        vol = pd.Series([1.0] * 100)
        bc = pd.Series([0.5] * 90)  # Différente longueur
        alpha = pd.Series([0.6] * 100)
        
        composite = CompositeScore()
        
        with pytest.raises(ValueError):
            composite.compute(vol, bc, alpha)
```

---

## 🎯 CRITÈRES DE SUCCÈS

### Fonctionnalités
- [x] `CompositeScore` class implémentée
- [x] `compute()` retourne série valide [0, 1]
- [x] `compute_with_metadata()` retourne tuple (série, dict)
- [x] `normalize_vol_ratio()` fonction helper
- [x] `validate_metrics_alignment()` validation inputs

### Validation
- [x] Tests unitaires passent (5+ tests)
- [x] Pas de NaN dans output (sauf si inputs ont NaN alignés)
- [x] Range [0, 1] respecté
- [x] Poids custom fonctionnent correctement

### Documentation
- [x] Docstrings complètes (Google style)
- [x] README.md dans tools/ expliquant usage
- [x] Exemple d'utilisation inclus

---

## 📊 EXEMPLE D'UTILISATION ATTENDU

```python
# Exemple: Utiliser composite_score dans une stratégie

from metrics.vol_ratio.v1_1.implementation import VolRatio
from metrics.bound_coherence.v1_0.implementation import BoundCoherence
from metrics.alpha_stability.v1_0.implementation import AlphaStability
from tools.composite_score import CompositeScore

# Charger données
btc = load_prices('BTC-USD')
paxg = load_prices('PAXG-USD')

# Calculer métriques (certifiées MIF)
vol = VolRatio(window=20).compute(btc, paxg)
bc = BoundCoherence(window=30).compute(btc, paxg)
alpha = AlphaStability(window=60).compute(btc, paxg)

# Créer score composite
composite = CompositeScore(weights={
    'vol_ratio': 0.4,
    'bound_coherence': 0.3,
    'alpha_stability': 0.3
})

score = composite.compute(vol, bc, alpha)

# Utiliser dans stratégie
signals = (score > 0.6).astype(int)  # Buy si score > 0.6
```

---

## ⚠️ CONTRAINTES IMPORTANTES

1. **Ne PAS créer dans metrics/**
   - composite_score va dans `tools/` (pas certifié MIF)

2. **Ne PAS essayer de certifier MIF**
   - Pas de Phase 0-2 tests
   - Juste tests unitaires basiques

3. **Normalisation vol_ratio obligatoire**
   - vol_ratio range: [0, +∞]
   - Doit être normalisé vers [0, 1] avant agrégation

4. **Validation inputs stricte**
   - Vérifier alignement séries
   - Rejeter si > 10% NaN
   - Rejeter si longueurs incompatibles

5. **Métadonnées utiles**
   - Corrélation du composite avec chaque input
   - Poids effectivement appliqués
   - Stats descriptives (mean, std, min, max)

---

## 📁 FICHIERS À CRÉER

```
qaaf_v2.0/
├── tools/
│   ├── composite_score.py         ← Implémentation principale
│   └── README_composite.md        ← Documentation usage
│
└── tests/
    └── test_composite_score.py    ← Tests unitaires
```

---

## ✅ CHECKLIST FINALE

Avant de considérer terminé:

- [ ] Code implémenté dans `tools/composite_score.py`
- [ ] Tests unitaires dans `tests/test_composite_score.py`
- [ ] Tous les tests passent (`pytest tests/test_composite_score.py -v`)
- [ ] Documentation README_composite.md créée
- [ ] Exemple d'utilisation testé et fonctionnel
- [ ] Validation stricte des inputs implémentée
- [ ] Métadonnées complètes retournées
- [ ] Code commenté et docstrings complets

---

## 🚫 CE QU'IL NE FAUT PAS FAIRE

### ❌ NE PAS créer comme métrique isolée
```python
# ❌ MAUVAIS
class CompositeScore(MetricBase):
    def compute(self, asset1_prices, asset2_prices):
        # Calcule vol_ratio, bc, alpha en interne
        pass
```

### ✅ À FAIRE: Agrégateur de métriques existantes
```python
# ✅ BON
class CompositeScore:
    def compute(self, vol_ratio_series, bound_coherence_series, alpha_stability_series):
        # Combine 3 séries déjà calculées
        pass
```

### ❌ NE PAS essayer de passer MIF Phase 0-2
- Pas de `tests/test_phase0.py`
- Pas de `tests/test_phase1.py`
- Pas de `tests/test_phase2.py`
- Pas de `certification.yaml`

### ✅ À FAIRE: Tests unitaires simples seulement
- `tests/test_composite_score.py` (tests basiques)
- Validation inputs/outputs
- Edge cases (NaN, misalignment, etc.)

---

## 📊 STRUCTURE ATTENDUE DU CODE

```python
# tools/composite_score.py

class CompositeScore:
    """Agrégateur QAAF v2.0."""
    
    def __init__(self, weights=None):
        """Initialize with weights."""
        self.weights = self._validate_weights(weights)
    
    def _validate_weights(self, weights):
        """Valide et normalise poids."""
        if weights is None:
            return {'vol_ratio': 1/3, 'bound_coherence': 1/3, 'alpha_stability': 1/3}
        
        # Check somme = 1.0 (ou normaliser)
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}
    
    def compute(self, vol_ratio, bound_coherence, alpha_stability):
        """
        Combine 3 métriques.
        
        Steps:
          1. Validate alignment
          2. Normalize vol_ratio (0-1)
          3. Weighted sum
          4. Return normalized [0, 1]
        """
        # 1. Validation
        self._validate_alignment(vol_ratio, bound_coherence, alpha_stability)
        
        # 2. Normalisation vol_ratio
        vol_norm = normalize_vol_ratio(vol_ratio)
        
        # 3. Weighted sum
        composite = (
            self.weights['vol_ratio'] * vol_norm +
            self.weights['bound_coherence'] * bound_coherence +
            self.weights['alpha_stability'] * alpha_stability
        )
        
        # 4. Ensure [0, 1]
        return composite.clip(0, 1)
    
    def compute_with_metadata(self, vol_ratio, bound_coherence, alpha_stability):
        """Compute + metadata."""
        composite = self.compute(vol_ratio, bound_coherence, alpha_stability)
        
        metadata = {
            'mean': composite.mean(),
            'std': composite.std(),
            'min': composite.min(),
            'max': composite.max(),
            'weights_used': self.weights,
            'correlation_with_inputs': {
                'vol_ratio': composite.corr(normalize_vol_ratio(vol_ratio)),
                'bound_coherence': composite.corr(bound_coherence),
                'alpha_stability': composite.corr(alpha_stability)
            },
            'n_points': len(composite),
            'n_nan': composite.isnull().sum()
        }
        
        return composite, metadata
    
    def _validate_alignment(self, vol_ratio, bound_coherence, alpha_stability):
        """Validate 3 series are aligned."""
        # Check lengths
        lengths = [len(vol_ratio), len(bound_coherence), len(alpha_stability)]
        if len(set(lengths)) > 1:
            raise ValueError(f"Series have different lengths: {lengths}")
        
        # Check index alignment (if pandas Series)
        if hasattr(vol_ratio, 'index'):
            if not vol_ratio.index.equals(bound_coherence.index):
                raise ValueError("vol_ratio and bound_coherence indices don't match")
            if not vol_ratio.index.equals(alpha_stability.index):
                raise ValueError("vol_ratio and alpha_stability indices don't match")
        
        # Check NaN percentage
        for name, series in [('vol_ratio', vol_ratio), 
                             ('bound_coherence', bound_coherence),
                             ('alpha_stability', alpha_stability)]:
            nan_pct = series.isnull().mean()
            if nan_pct > 0.10:
                raise ValueError(f"{name} has {nan_pct:.1%} NaN (>10% threshold)")


def normalize_vol_ratio(vol_ratio, expected_range=(0.5, 2.0)):
    """
    Normalize vol_ratio from [0, +∞] to [0, 1].
    
    Method: Min-max scaling with expected range.
    Values outside expected_range are clipped.
    
    Args:
        vol_ratio: Series with vol_ratio values
        expected_range: Tuple (min, max) for normalization
    
    Returns:
        Normalized series [0, 1]
    """
    min_val, max_val = expected_range
    
    # Min-max normalization
    normalized = (vol_ratio - min_val) / (max_val - min_val)
    
    # Clip to [0, 1]
    return normalized.clip(0, 1)
```

---

## 📖 EXEMPLE README_composite.md

```markdown
# Composite Score - QAAF v2.0 Tool

## Overview

`composite_score` is an **aggregator tool** that combines 3 certified QAAF v2.0 metrics:
- `vol_ratio` (volatility risk)
- `bound_coherence` (dynamic correlation)
- `alpha_stability` (alpha persistence)

**Important**: This is NOT an isolated metric. It's a tool for creating composite signals.

## Usage

### Basic Example

```python
from tools.composite_score import CompositeScore

# Assume you have 3 metric series already computed
vol = vol_ratio_series
bc = bound_coherence_series
alpha = alpha_stability_series

# Create composite with default weights (equal)
composite = CompositeScore()
score = composite.compute(vol, bc, alpha)

print(f"Composite score: {score.mean():.2f}")
```

### Custom Weights

```python
# Emphasize vol_ratio (40%), moderate others (30% each)
composite = CompositeScore(weights={
    'vol_ratio': 0.4,
    'bound_coherence': 0.3,
    'alpha_stability': 0.3
})

score = composite.compute(vol, bc, alpha)
```

### With Metadata

```python
score, metadata = composite.compute_with_metadata(vol, bc, alpha)

print(f"Mean: {metadata['mean']:.2f}")
print(f"Std: {metadata['std']:.2f}")
print(f"Correlation with inputs:")
for metric, corr in metadata['correlation_with_inputs'].items():
    print(f"  {metric}: {corr:.3f}")
```

## Strategy Integration

```python
# Example: Use composite score for trading signals

# Calculate composite
composite = CompositeScore()
score = composite.compute(vol, bc, alpha)

# Generate signals
buy_threshold = 0.6
sell_threshold = 0.4

signals = pd.Series(0, index=score.index)
signals[score > buy_threshold] = 1   # Buy
signals[score < sell_threshold] = -1  # Sell

# Backtest with signals
```

## Input Requirements

- All 3 inputs must be pandas Series
- Same length and aligned indices
- NaN < 10% for each series
- `vol_ratio` will be normalized to [0, 1] internally

## Output

- pandas Series with values in [0, 1]
- Same index as inputs
- Higher score = more favorable conditions

## Not MIF Certified

This tool does NOT go through MIF certification (Phase 0-2).

Why? Because it aggregates already-certified metrics, not an isolated metric.

## Testing

```bash
pytest tests/test_composite_score.py -v
```
```

---

## 🎯 RÉSUMÉ POUR CLAUDE

**Tâche**: Créer `tools/composite_score.py` + tests

**Type**: Outil d'agrégation (PAS métrique isolée)

**Inputs**: 3 séries pandas (vol_ratio, bound_coherence, alpha_stability)

**Output**: Série composite normalisée [0, 1]

**Tests**: Unitaires seulement (PAS MIF Phase 0-2)

**Localisation**: `tools/` (PAS `metrics/`)

**Validation stricte**: Alignement, NaN < 10%, normalisation vol_ratio

**Documentation**: Docstrings + README_composite.md + exemples

---

## 📞 SI BLOQUÉ

**Question fréquente 1**: "Dois-je certifier composite_score via MIF?"
- **Non**. C'est un outil, pas une métrique isolée.

**Question fréquente 2**: "Où mettre le code?"
- **`tools/composite_score.py`** (pas dans `metrics/`)

**Question fréquente 3**: "Quels tests faire?"
- Tests unitaires simples: validation inputs, normalisation, poids

**Question fréquente 4**: "Comment normaliser vol_ratio?"
- Min-max scaling: `(x - min) / (max - min)`, clip [0, 1]

**Question fréquente 5**: "Metadata à inclure?"
- Mean, std, poids utilisés, corrélation avec inputs

---

## ✅ LIVRABLE FINAL ATTENDU

```
qaaf_v2.0/
├── tools/
│   ├── composite_score.py          # ~200 lignes
│   └── README_composite.md         # Documentation
│
└── tests/
    └── test_composite_score.py     # ~150 lignes, 5+ tests
```

**Temps estimé**: 2-3h d'implémentation

**Critère succès**: Tous les tests passent, exemple fonctionne

---

**BON COURAGE! 🎯**