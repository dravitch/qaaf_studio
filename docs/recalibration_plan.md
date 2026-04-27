# Recalibration Plan — QAAF Studio 3.1

## Principe directeur : Ground Truth or Silence (Mindset #1)

**Ordre correct de calibration d'un paramètre :**
1. Consulter la KB pour la valeur empirique documentée
2. Fixer le paramètre sur cette valeur (avec marge de sécurité)
3. Vérifier que les tests synthétiques tiennent — et non l'inverse

**Anti-pattern à éviter :** calibrer d'abord sur données synthétiques, puis
découvrir que le paramètre ne se déclenche jamais sur les données réelles.

---

## Cas documenté : MetisQ2 `regime_margin`

**Session 1 :** `regime_factor=1.3` (ratio, calibré sur données synthétiques)
→ `cnsr_bench_is=-0.052` (période IS négative) bloquait la détection

**Session 2a :** `regime_margin=1.5` (différence absolue, calibré sur données synthétiques)
→ diff réel = 1.04 < 1.5 → ajustement ne se déclenchait jamais en production

**Session 2b (correct) :** `regime_margin=0.8`
→ Calibration : B_5050_OOS=1.343 (KB), B_5050_IS=0.303 → diff=1.04
→ margin=0.8 = 23% sous le diff empirique, marge de sécurité documentée
→ Déclenchement : 1.04 > 0.8 ✓, données synthétiques : 2.32 > 0.8 ✓

**Leçon :** La KB contenait B_5050_OOS=1.343 dès le début. Consulter la KB
en premier aurait évité deux recalibrages successifs.

---

## Template pour les futurs paramètres de seuil

```
Paramètre       : <nom>
Valeur KB       : <valeur empirique documentée en KB>
Valeur retenue  : <valeur choisie> (= KB × <facteur> pour marge)
Justification   : <diff empirique X, margin = X × 0.8 par exemple>
Test synthétique: <valeur sur données synthétiques> > seuil ✓
```

---

## Paramètres calibrés

| Filtre    | Paramètre       | Valeur KB      | Valeur retenue | Marge |
|-----------|-----------------|----------------|----------------|-------|
| MetisQ2   | regime_margin   | diff=1.04      | 0.8            | 23%   |
| MetisQ4   | dsr_threshold   | Bailey-LdP 0.95| 0.95           | —     |
| MetisQ4   | n_trials oracle | 1 (non optimisé)| 1             | —     |

---

## Note sur N_effectif (MetisQ4)

Pour une famille EMA span 20j–120j (N=101), corrélation inter-variantes ≈ 0.97 :
```
N_effectif = max(1, round(101 × (1 - 0.97))) = max(1, round(3.03)) = 3
```

Cette valeur doit être mesurée empiriquement sur les variantes réelles
avant d'être utilisée en production (corrélation entre EMA_20, EMA_21, ..., EMA_120).
La valeur 0.97 est une estimation conservatrice — à valider sur données réelles.
