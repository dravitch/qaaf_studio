# tests/test_metrics_orthogonality.py
"""
UNIQUE objectif: Prouver que les 4 métriques sont indépendantes
Pas de suppositions. Données brutes seulement.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from metrics_validation_protocol_2 import MetricsFromScratch, generate_synthetic_prices, SyntheticScenario

# Génère un scénario synthétique neutre
scenario = SyntheticScenario(
    name='Orthogonality Test',
    btc_volatility=0.30,
    paxg_volatility=0.15,
    correlation=0.25,
    trend_btc=0.10,
    trend_paxg=0.05,
    regime='stable',
    days=300
)

df = generate_synthetic_prices(scenario)
calc = MetricsFromScratch(volatility_window=30, spectral_window=60)
metrics = calc.calculate_all(df)
df_metrics = pd.DataFrame(metrics).dropna()

class MetricsOrthogonalityTest:
    def test_all_correlations(self):
        """Test linéaire : Pearson |ρ| < 0.4 pour toutes les paires"""
        corr = df_metrics.corr(method='pearson')
        pairs = [
            ('vol_ratio', 'bound_coherence'),
            ('vol_ratio', 'alpha_stability'),
            ('vol_ratio', 'spectral_score'),
            ('bound_coherence', 'alpha_stability'),
            ('bound_coherence', 'spectral_score'),
            ('alpha_stability', 'spectral_score')
        ]
        for m1, m2 in pairs:
            rho = abs(corr.loc[m1, m2])
            print(f"✅ Pearson |ρ| {m1} vs {m2} = {rho:.3f}")
            assert rho < 0.4, f"❌ Corrélation trop forte entre {m1} et {m2} (ρ={rho:.3f})"

    def test_mutual_information(self):
        """Test non-linéaire : MI(X,Y) < 0.25 pour toutes les paires"""
        mi_matrix = {}
        for m1 in df_metrics.columns:
            for m2 in df_metrics.columns:
                if m1 != m2:
                    mi = mutual_info_regression(
                        df_metrics[[m1]].values,
                        df_metrics[m2].values,
                        random_state=42
                    )[0]
                    mi_matrix[(m1, m2)] = mi
                    print(f"✅ MI {m1} → {m2} = {mi:.3f}")
                    assert mi < 0.25, f"❌ Mutual Information trop élevée entre {m1} et {m2} (MI={mi:.3f})"

    def test_predictivity_r2(self):
        """Test de prédictivité : R² > 0.01 = utile, R² < 0.001 = métrique morte"""
        target = df['btc_close'].pct_change().shift(-1).dropna()
        for name, series in metrics.items():
            aligned = pd.DataFrame({
                'metric': series,
                'target': target
            }).dropna()
            X = aligned[['metric']].values
            y = aligned['target'].values
            model = LinearRegression().fit(X, y)
            r2 = model.score(X, y)
            print(f"✅ R² {name} → returns = {r2:.4f}")
            assert r2 > 0.001, f"❌ {name} n'explique rien (R²={r2:.4f})"
