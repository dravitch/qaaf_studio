# Composite Score - QAAF v2.0 Aggregator Tool

**Version:** 1.0.0  
**Date:** 2025-10-24  
**Type:** Aggregator Tool (NOT a certified metric)  
**Status:** ✅ Production Ready

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Installation](#installation)
4. [Core Concepts](#core-concepts)
5. [API Reference](#api-reference)
6. [Usage Examples](#usage-examples)
7. [Strategy Integration](#strategy-integration)
8. [Input Requirements](#input-requirements)
9. [Output Specification](#output-specification)
10. [Best Practices](#best-practices)
11. [Testing](#testing)
12. [Troubleshooting](#troubleshooting)
13. [FAQ](#faq)

---

## 🎯 Overview

`composite_score` is an **aggregator tool** that combines 3 MIF-certified QAAF v2.0 metrics into a single normalized score for decision-making:

| Metric | Domain | Range | Role |
|--------|--------|-------|------|
| **vol_ratio** | Risk | [0, +∞] | Volatility asymmetry |
| **bound_coherence** | Risk | [0, 1] | Dynamic correlation |
| **alpha_stability** | Performance | [0, 1] | Alpha persistence |

### Key Features

✅ **Flexible Weighting** - Configure weights by trading style  
✅ **Automatic Normalization** - vol_ratio scaled to [0, 1]  
✅ **Validation Built-in** - Strict alignment & quality checks  
✅ **Metadata Rich** - Correlations, stats, diagnostics  
✅ **Zero Dependencies** - Uses only pandas/numpy  
✅ **Production Tested** - 35+ unit tests

### ⚠️ Important Note

> **This is NOT a MIF-certified metric.**  
> It's an aggregation tool. The 3 input metrics are certified, but the composite itself bypasses Phase 0-2 certification.

---

## 🚀 Quick Start

### Basic Usage (Equal Weights)

```python
from tools.composite_score import CompositeScore

# Assume you have 3 pre-computed metric series
vol = vol_ratio_series       # Série [0, +∞]
bc = bound_coherence_series  # Série [0, 1]
alpha = alpha_stability_series  # Série [0, 1]

# Create composite with default weights (1/3 each)
composite = CompositeScore()
score = composite.compute(vol, bc, alpha)

print(f"Composite score: {score.mean():.2f}")
# Output: Composite score: 0.57
```

### Custom Weights (Volatility-Heavy Strategy)

```python
# Emphasize vol_ratio (50%), moderate others
composite = CompositeScore(weights={
    'vol_ratio': 0.5,
    'bound_coherence': 0.3,
    'alpha_stability': 0.2
})

score = composite.compute(vol, bc, alpha)
```

### With Diagnostic Metadata

```python
score, meta = composite.compute_with_metadata(vol, bc, alpha)

print(f"Mean: {meta['mean']:.3f}")
print(f"Std: {meta['std']:.3f}")
print(f"Correlations:")
for metric, corr in meta['correlation_with_inputs'].items():
    print(f"  {metric}: {corr:.3f}")
```

**Output:**
```
Mean: 0.573
Std: 0.124
Correlations:
  vol_ratio: 0.482
  bound_coherence: 0.391
  alpha_stability: 0.356
```

---

## 📦 Installation

### Option 1: Copy to Project

```bash
# Copy composite_score.py to your tools/ directory
cp composite_score.py /path/to/qaaf_v2.0/tools/

# Verify
python -c "from tools.composite_score import CompositeScore; print('✅ Import OK')"
```

### Option 2: Add to Python Path

```python
import sys
sys.path.insert(0, '/path/to/qaaf_v2.0/tools')

from composite_score import CompositeScore
```

### Dependencies

- **Python:** 3.8+
- **pandas:** any version
- **numpy:** any version
- **No external libraries required** ✅

---

## 🧠 Core Concepts

### 1. Why Composite Scores?

Individual metrics capture specific aspects:
- **vol_ratio** → Regime volatility shifts
- **bound_coherence** → Mean-reversion strength
- **alpha_stability** → Trend persistence

Combining them creates a **multi-dimensional signal** more robust than any single metric.

### 2. Weight Configuration Strategies

| Strategy | Weights | Use Case |
|----------|---------|----------|
| **Equal** | (0.33, 0.33, 0.34) | Balanced, neutral |
| **Vol-Heavy** | (0.5, 0.3, 0.2) | Risk-parity focus |
| **Alpha-Heavy** | (0.2, 0.3, 0.5) | Trend-following |
| **BC-Heavy** | (0.3, 0.5, 0.2) | Mean-reversion |

### 3. Normalization Pipeline

```
vol_ratio [0, +∞] → normalize_vol_ratio() → [0, 1]
                                              ↓
bound_coherence [0, 1] ─────────────────────→ weighted_sum()
                                              ↓
alpha_stability [0, 1] ──────────────────────→ [0, 1]
```

**Default normalization range:** (0.5, 2.0)  
Based on empirical observations from BTC/PAXG, SPY/TLT, SPY/GLD, QQQ/IEF.

---

## 📚 API Reference

### Class: `CompositeScore`

```python
class CompositeScore(weights=None, normalization_range=(0.5, 2.0))
```

**Parameters:**
- `weights` (dict, optional): Custom weights. Keys: `vol_ratio`, `bound_coherence`, `alpha_stability`. Auto-normalized to sum=1.0.
- `normalization_range` (tuple): (min, max) for vol_ratio normalization. Default: (0.5, 2.0).

**Attributes:**
- `weights`: Normalized weights dict
- `normalization_range`: Tuple (min, max)
- `last_computation_metadata`: Metadata from last compute

---

### Method: `compute()`

```python
score = composite.compute(vol_ratio, bound_coherence, alpha_stability)
```

**Parameters:**
- `vol_ratio` (pd.Series): vol_ratio values [0, +∞]
- `bound_coherence` (pd.Series): bound_coherence values [0, 1]
- `alpha_stability` (pd.Series): alpha_stability values [0, 1]

**Returns:**
- `pd.Series`: Composite score [0, 1], same index as inputs

**Raises:**
- `ValueError`: If series misaligned, NaN > 10%, or invalid ranges

**Example:**
```python
score = composite.compute(vol, bc, alpha)
assert (score >= 0).all() and (score <= 1.0).all()
```

---

### Method: `compute_with_metadata()`

```python
score, metadata = composite.compute_with_metadata(vol_ratio, bound_coherence, alpha_stability)
```

**Returns:**
- `tuple`: (score: pd.Series, metadata: dict)

**Metadata Keys:**
- `mean`, `std`, `min`, `max`, `median`, `q25`, `q75`: Statistics
- `weights_used`: Weights applied
- `normalization_range`: Range used for vol_ratio
- `correlation_with_inputs`: Dict of correlations with each input
- `n_points`, `n_nan`, `nan_pct`: Data quality

**Example:**
```python
score, meta = composite.compute_with_metadata(vol, bc, alpha)
print(f"Vol_ratio contributes {meta['correlation_with_inputs']['vol_ratio']:.1%}")
```

---

### Function: `normalize_vol_ratio()`

```python
from tools.composite_score import normalize_vol_ratio

normalized = normalize_vol_ratio(vol_ratio, expected_range=(0.5, 2.0))
```

**Parameters:**
- `vol_ratio` (pd.Series): Raw vol_ratio [0, +∞]
- `expected_range` (tuple): (min, max) for scaling

**Returns:**
- `pd.Series`: Normalized [0, 1]

**Behavior:**
- Values < min → 0.0
- Values > max → 1.0
- Linear interpolation between min and max
- NaN preserved

**Example:**
```python
vol = pd.Series([0.3, 0.5, 1.0, 1.5, 2.0, 2.5])
norm = normalize_vol_ratio(vol, expected_range=(0.5, 2.0))
# Result: [0.0, 0.0, 0.333, 0.667, 1.0, 1.0]
```

---

### Function: `quick_composite()`

```python
from tools.composite_score import quick_composite

score = quick_composite(vol_ratio, bound_coherence, alpha_stability, weights=None)
```

**Convenience function** - Creates CompositeScore, computes, returns score.

**Example:**
```python
# One-liner for default weights
score = quick_composite(vol, bc, alpha)
```

---

### Function: `composite_signal()`

```python
from tools.composite_score import composite_signal

signals = composite_signal(composite_score, buy_threshold=0.6, sell_threshold=0.4)
```

**Parameters:**
- `composite_score` (pd.Series): Composite score [0, 1]
- `buy_threshold` (float): Score > this → Buy (1)
- `sell_threshold` (float): Score < this → Sell (-1)

**Returns:**
- `pd.Series`: Signals (-1, 0, +1)

**Example:**
```python
signals = composite_signal(score, buy_threshold=0.65, sell_threshold=0.35)
print(f"Buy signals: {(signals == 1