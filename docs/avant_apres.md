# Avant / Après — Démonstration de valeur QAAF Studio

**Hypothèse** : H9+EMA60j (PAXG/BTC, rebalancement proportionnel lissé EMA 60j)  
**Période OOS** : 2023-06 → 2024-12

---

## Tableau comparatif

| Dimension | QAAF-R (avant) | QAAF Studio 3.0 (après) |
|---|---|---|
| Sharpe affiché | **1.77** (paire brute, non corrigé) | **CNSR-USD-Fed 1.285** OOS |
| Correction numéraire | Absente | CNSR-USD appliqué : BTC converti en USD au taux Fed Funds |
| Benchmark de référence | Aucun (ou BTC naïf) | B_5050 : CNSR 1.343, Max DD 13% |
| Position vs benchmark | Supérieure (affirmée) | Inférieure à B_5050 (1.285 < 1.343) |
| Test de significativité | Absent | Q2 permutation : p = 0.487 sur 10 000 permutations |
| Multiple testing | Absent | DSR = 0.325 avec N_trials = 101 (seuil ≥ 0.95) |
| Robustesse de régime | Non testée | MIF Phase 1 : FAIL sur bear, lateral, crash |
| Walk-forward | Non testé | Q1 : PASS (4/5 fenêtres, CNSR > 0.5) |
| **Verdict final** | **"Valide"** | **ARCHIVE_FAIL_Q2_Q4** |

---

## Pourquoi le Sharpe 1.77 de QAAF-R était trompeur

### 1. Problème de numéraire

QAAF-R calculait le Sharpe sur les rendements PAXG/BTC — c'est-à-dire dans l'unité **BTC**, pas USD. Cette paire peut afficher un Sharpe élevé simplement parce que PAXG est moins volatil que BTC en termes de BTC (il "tient" relativement bien les bear markets BTC), sans que le portefeuille génère de valeur réelle en USD.

QAAF Studio applique **CNSR-USD** : tous les rendements sont convertis en USD via le taux Fed Funds comme taux sans risque. Le Sharpe passe de 1.77 (paire brute) à **1.285** (CNSR-USD corrigé) — une différence de **-27%** qui disparaissait dans l'ancienne formule.

### 2. Absence de benchmark équitable

QAAF-R comparait à "faire rien" ou à BTC pur (B_BTC). Or le benchmark naturel d'une stratégie de rebalancement PAXG/BTC est **B_5050** (50% PAXG, 50% BTC, rééquilibré annuellement). B_5050 obtient un CNSR-USD de **1.343** — supérieur à H9+EMA60j (1.285). La stratégie ne bat pas son benchmark naturel.

### 3. Absence de test de significativité

Avec **N_trials = 101** variantes testées dans la famille EMA_span_variants, le risque de sur-ajustement est élevé. Le Deflated Sharpe Ratio (Bailey & López de Prado 2014) en tient compte :

```
DSR(N=101) = 0.325
```

Un DSR de 0.325 signifie que la probabilité que ce Sharpe soit dû au hasard (sélection parmi 101 variantes) est de **67%**. Le seuil minimal acceptable est DSR ≥ 0.95.

### 4. Test de permutation

La p-value de **0.487** (Q2, 10 000 permutations) confirme : les rendements réordonnés aléatoirement obtiennent un CNSR aussi élevé que la stratégie dans ~49% des cas. La stratégie n'est statistiquement pas distinguable du bruit sur la période OOS.

---

## Ce que QAAF Studio détecte que QAAF-R manquait

```
QAAF-R → "Sharpe 1.77 → Valide"
              ↓
         CNSR-USD : 1.285 (numéraire corrigé)
         B_5050   : 1.343 (benchmark équitable)
                          ↓ inférieur au benchmark
         p-value  : 0.487 (test de permutation)
                          ↓ pas distinguable du bruit
         DSR(N=101): 0.325 (multiple testing)
                          ↓ sur-ajustement probable
         MIF Phase 1: FAIL bear/lateral/crash
                          ↓ fragile hors bull market
              ↓
QAAF Studio → ARCHIVE_FAIL_Q2_Q4
```

---

## Valeur du diagnostic

Le verdict ARCHIVE_FAIL_Q2_Q4 n'est pas un échec — c'est une **connaissance précise** :

- Le signal H9 a une hiérarchie réelle (PAF D1-D3 confirmés)
- Il échoue spécifiquement sur les régimes bear/crash (MIF G1/G2/G3)
- Le filtre MA200 (Sprint A) est une hypothèse naturelle de correction ciblée
- N_trials trop élevé relativement à la qualité du signal : revoir la stratégie d'exploration de la famille

Sans QAAF Studio, cette stratégie aurait pu être déployée en production avec un Sharpe "validé" de 1.77.

---

*QAAF Studio 3.0 — Démonstration de valeur | Avril 2026*  
*Co-signé : Andrei, Claude Sonnet 4.6*
