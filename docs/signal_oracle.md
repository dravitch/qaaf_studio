# Signal Oracle — QAAF Studio 3.1

**Rôle :** Stratégie de référence construite pour être certifiable
par un pipeline bien calibré. Pas un signal de production — un
étalon de mesure pour valider la recalibration des filtres.

## Logique

Suiveur de tendance asymétrique avec filtre de volatilité :

1. **Position longue PAXG** (alloc_btc bas) quand le ratio
   PAXG/BTC est au-dessus de sa moyenne 120 jours.
2. **Position longue BTC** (alloc_btc haut) quand le ratio est
   en dessous de sa moyenne 120 jours.
3. **Position neutre** (alloc_btc = 0.5) quand la volatilité
   réalisée du ratio (30j) dépasse 2× sa médiane historique.

## Paramètres

```python
ORACLE_PARAMS = {
    "trend_window":  120,  # Fenêtre de tendance (jours)
    "vol_window":     30,  # Fenêtre de volatilité réalisée (jours)
    "vol_threshold":  2.0, # Multiplicateur médiane vol → neutre
    "alloc_high":    0.75, # Allocation BTC max (longue BTC)
    "alloc_low":     0.25, # Allocation BTC min (longue PAXG)
}
```

## Pourquoi ces paramètres

- `trend_window=120` : fenêtre longue, résistante au sur-ajustement.
  Pas un spike sur une valeur précise → Q3 EMA passe.
- `vol_threshold=2.0` : seuil conservateur. Coupe l'exposition
  dans les régimes crash et bear → MIF G1/G3 passe.
- `alloc_high/low = 0.75/0.25` : exposition modérée. Pas de
  position extrême → drawdown maîtrisé, Q4 DSR robuste.

## Invariants

- Jamais de lookahead : la moyenne 120j est calculée avec
  `.shift(1)` (valeur d'hier).
- Toujours dans [alloc_low, alloc_high].
- Signal stationnaire sur la durée complète des données.

## Critère de succès

Le signal oracle est certifié quand `tests/test_signal_oracle_certified.py`
est vert après la recalibration de MIF G1/G2/G3, MÉTIS Q2,
et DSR N_effectif (Livrable 4).

Avant la recalibration, les tests de certification doivent être
**rouges ou skipped**. Ce contraste est le critère de succès de la Phase 2.
