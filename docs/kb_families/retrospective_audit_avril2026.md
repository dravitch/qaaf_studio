# Audit Rétrospectif — Sprint C — Avril 2026

**Date d'exécution :** 2026-04-26  
**Pipeline :** QAAF Studio 3.0 (session_runner générique, mode `--fast`)  
**OOS :** 2023-06-01 → 2024-12-31 (bull market BTC exceptionnel)  
**Objectif :** Recertifier les 4 lentilles pré-Studio avec CNSR-USD normalisé + tester la généralisation ETH/BTC

---

## 1. Contexte

Les 4 premières lentilles (MR_pur, H9_brut, QAAF-R, PhaseCoherence) ont été développées avant QAAF Studio 3.0.
Leurs métriques d'origine utilisaient un Sharpe dénominé en BTC ou en PAXG, sans correction de numéraire.  
Ce sprint exécute chaque stratégie dans le pipeline Studio complet pour obtenir des CNSR-USD comparables.

**Référence :** H9+EMA60j (certifié Avril 2026, N=101)

---

## 2. Tableau Comparatif Complet

| Hypothèse             | Famille           | N  | CNSR-USD | DSR    | MIF         | Q1   | Q2 (p-val) | Verdict                  | D-SIG |
|-----------------------|-------------------|----|----------|--------|-------------|------|------------|--------------------------|-------|
| **B_5050** *(ref)*    | —                 | —  | 1.343    | —      | —           | —    | —          | Benchmark passif         | —     |
| MR_pur                | mean_reversion    | 1  | -0.544   | 0.2053 | FAIL_P1     | FAIL | FAIL (0.79)| ARCHIVE_FAIL_Q1_Q2       | 20    |
| H9_brut               | EMA_span_variants | 101| -0.541   | 0.0008 | FAIL_P1     | FAIL | FAIL (0.33)| ARCHIVE_FAIL_Q1_Q2_Q4    | 20    |
| QAAF-R                | geometric_regime  | 1  | 1.680    | 0.9944 | FAIL_P1 (G1)| FAIL | FAIL (0.98)| ARCHIVE_FAIL_Q1_Q2       | 86    |
| PhaseCoherence        | geometric_regime  | 1  | 3.801    | 1.0000 | FAIL_P0 (T5)| FAIL | FAIL (0.99)| ARCHIVE_FAIL_Q1_Q2       | 92    |
| **H9+EMA60j** *(ref)*| EMA_span_variants | 101| 1.285    | 0.3246 | FAIL_P1     | PASS | FAIL (0.49)| ARCHIVE_FAIL_Q2_Q4       | 59    |
| H9+EMA60j_ETH-BTC     | EMA_span_variants | 103| 0.821    | 0.1257 | FAIL_P1 (G1-3)| FAIL | FAIL (0.96)| ARCHIVE_FAIL_Q1_Q2_Q4   | 20    |

> **Note :** Les CNSR pré-Studio pour MR_pur (-0.8), H9_brut (0.32) et QAAF-R (0.99) étaient calculés sur
> des périodes IS différentes et sans correction de numéraire USD. Les valeurs Studio sont OOS 2023-2024.

---

## 3. Analyse par Lentille

### 3.1 MR_pur — Mean-Reversion Simple

**Verdict Studio :** ARCHIVE_FAIL_Q1_Q2 | CNSR=-0.544 | D-SIG 20/100 CRITICAL

La stratégie de mean-reversion pure sur PAXG/BTC est fortement négative en OOS bull market.
Le mécanisme (acheter quand ratio bas, vendre quand ratio haut) est incompatible avec la dérive structurelle
haussière de BTC en 2023-2024. Q1 3/5 (réussite uniquement sur fenêtres haussières), Q2 p=0.79.

**Enseignement :** MR pur sous dérive = destruction de valeur. Diagnostic PAF D1=STOP confirmé.

---

### 3.2 H9_brut — H9 sans EMA

**Verdict Studio :** ARCHIVE_FAIL_Q1_Q2_Q4 | CNSR=-0.541 | D-SIG 20/100 CRITICAL

Quasiment identique à MR_pur en termes de CNSR (-0.541 vs -0.544). Le lissage EMA n'affecte pas les
grandes tendances OOS. Mais avec N=101 (famille EMA_span_variants actuelle), le DSR chute à 0.0008 —
confirmation que sur 101 tests, aucune chance que ce signal soit le "bon".

**Enseignement :** Le EMA60j n'améliore pas le signal de base en OOS — il réduit uniquement la variance
d'allocation (0.259 → 0.170) ce qui réduit les frais. La hiérarchie D1=HIERARCHIE_CONFIRMEE était correcte
(H9 > MR_pur en IS) mais aucun des deux ne performe en OOS bear.

---

### 3.3 QAAF-R — Phase-Cohérence Géométrique

**Verdict Studio :** ARCHIVE_FAIL_Q1_Q2 | CNSR=1.680 | D-SIG 86/100 EXCELLENT

Résultat paradoxal : **excellentes métriques brutes mais tests statistiques catastrophiques** (Q2 p=0.98).
Le D-SIG=86 GREEN trompe — il repose sur les métriques OOS brutes, pas sur la significativité.

**Pourquoi CNSR=1.680 mais p=0.98 ?** La stratégie profite du bull market 2023-2024 mais de façon
identique aux benchmarks passifs. La permutation montre que N'IMPORTE quel arrangement aléatoire du
signal produit le même CNSR — il n'y a pas d'alpha propre.

**Avant correction de numéraire :** CNSR reporté à 0.99 (pré-Studio). Avec correction USD, on obtient 1.680
— plus élevé, mais toujours non distinguable du bruit (p=0.98). Le mécanisme "phase géométrique" est
un artefact de lissage sur une tendance haussière.

**Enseignement :** C'est ici que MÉTIS Q2 est le plus précieux. Sans permutation, on certifierait une
stratégie inutile avec D-SIG excellent.

---

### 3.4 PhaseCoherence — Composante Phase Seule

**Verdict Studio :** ARCHIVE_FAIL_Q1_Q2 | CNSR=3.801 | D-SIG 92/100 EXCELLENT

Cas extrême : **CNSR=3.801 et DSR=1.0000** mais MIF FAIL_PHASE_0 (T5 — asymétrie directionnelle
quasi-nulle) et Q2 p=0.99. Le signal ne fait presque pas varier l'allocation entre PAXG et BTC,
ce qui en période bull market donne l'illusion d'une stratégie parfaite (hold BTC ≈ hold PAXG+BTC).

**Avec N=1**, le DSR=1.0 est mathématiquement correct mais sans valeur — aucune correction pour les tests
multiples. Le vrai test est Q2 (permutation) qui révèle : n'importe quel signal aléatoire ferait aussi bien.

**Enseignement :** Le MIF T5 (asymétrie directionnelle) détecte ce cas : si half_delta ≈ 0,
la stratégie ne choisit pas entre actifs — elle détient essentiellement un portefeuille fixe qui profite
passivement du marché. Artefact géométrique le plus évident du dataset.

---

### 3.5 H9+EMA60j sur ETH/BTC — Généralisation

**Verdict Studio :** ARCHIVE_FAIL_Q1_Q2_Q4 | CNSR=0.821 | D-SIG 20/100 CRITICAL

Le mécanisme H9+EMA60j ne se généralise pas directement sur ETH/BTC. Plusieurs facteurs :

1. **Max DD = 42.5%** (vs 13.5% sur PAXG/BTC) — ETH/BTC est un instrument beaucoup plus volatile
2. **MIF G1/G2/G3 FAIL** — même problème que sur PAXG/BTC : bear market et crash défavorables  
3. **Q3 PASS** — la stabilité EMA est préservée (IS CNSR=0.61, plateau visible)
4. **CNSR=0.821 < B_5050=0.861** — légèrement sous le benchmark passif ETH/BTC

**Benchmarks ETH/BTC :**  B_5050=0.861, B_BTC=1.244, B_ETH=0.456

**Enseignement :** La logique H9 (mean-reversion lissée) fonctionne mieux sur PAXG/BTC que sur ETH/BTC.
PAXG a une propriété stabilisatrice (adossé à l'or) qui n'existe pas pour ETH. Un filtre MA200 sur ETH/BTC
serait la prochaine étape naturelle — mais résultats préliminaires dans PAXG suggèrent que ce ne sera
pas suffisant (H9+MA200 également archivé).

---

## 4. Avant / Après — Correction de Numéraire

| Hypothèse    | CNSR pré-Studio (BTC/pair-dén.) | CNSR Studio OOS 2023-2024 (USD) | Différence |
|--------------|----------------------------------|----------------------------------|------------|
| MR_pur       | -0.80                            | -0.544                           | +0.256     |
| H9_brut      | +0.32                            | -0.541                           | -0.861     |
| QAAF-R       | +0.99 *(avant numéraire)*        | +1.680                           | +0.690     |
| PhaseCoherence| null (non calculé)               | +3.801                           | —          |

> Les différences reflètent : (1) correction de numéraire USD, (2) période OOS 2023-2024 vs périodes
> historiques variées, (3) implémentation Studio vs implémentations ad-hoc pré-Studio.

---

## 5. Synthèse et Recommandations

### Ce que MÉTIS a sauvé

Sans MÉTIS Q2 (permutation), on aurait certifié :
- **QAAF-R** avec CNSR=1.680, D-SIG=86/100 EXCELLENT → artefact
- **PhaseCoherence** avec CNSR=3.801, D-SIG=92/100 EXCELLENT → artefact pur

**MÉTIS Q2 est le test critique pour la famille geometric_regime.**

### Position famille EMA_span_variants

| Hypothèse              | N   | CNSR  | Q2 p-val | Statut   |
|------------------------|-----|-------|----------|----------|
| H9_brut                | 101 | -0.541| 0.328    | ARCHIVE  |
| H9+EMA60j              | 101 | 1.285 | 0.487    | ARCHIVE  |
| H9+EMA60j+MA200_hard   | 103 | 0.894 | —        | ARCHIVE  |
| H9+EMA60j+MA200_soft   | 103 | 1.082 | —        | ARCHIVE  |
| H9+EMA60j_ETH-BTC      | 103 | 0.821 | 0.962    | ARCHIVE  |

Toute la famille est en ARCHIVE. La prochaine piste : **régimes dynamiques** ou **allocation multi-actifs**
plutôt que de continuer les variantes PAXG/BTC en isolation.

### Prochaines pistes (non ordonnées)

1. H9+EMA60j+MA200 sur ETH/BTC (si le filtre MA200 corrige les régimes bear)
2. Signal multi-actifs : combiner PAXG/BTC + ETH/BTC en portefeuille
3. Régime adaptatif : passer entre MR et momentum selon le régime détecté
4. Famille séparée pour ETH/BTC (différents N_trials, différent split)

---

## 6. Rapports HTML Générés

- `sessions/retrospective_mr_pur/results/MR_pur_report.html`
- `sessions/retrospective_h9_brut/results/H9_brut_report.html`
- `sessions/retrospective_qaaf_r/results/QAAF-R_report.html`
- `sessions/retrospective_phase_coherence/results/PhaseCoherence_report.html`
- `sessions/eth_btc_h9_ema60j/results/H9_EMA60j_ETH-BTC_report.html`
