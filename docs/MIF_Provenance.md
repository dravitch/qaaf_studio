# MIF — Provenance, justifications et risques résiduels

## 1. Déclaration d'origine

MIF (Market Integrity Framework) est un **framework interne QAAF Studio**, en cours
d'élaboration. Il n'est pas une implémentation d'un standard externe publié.

Le versionnement présent dans le code (`MIF v4` dans `metrics/vol_ratio/v1.0/certification.yaml`,
`MIF v5` dans `layer2_qualification/mif/mif_runner.py`) désigne des **itérations du
framework interne**, pas des références à un document de normalisation externe.

**DQF (Data Quality Framework)** est la seule composante MIF formellement spécifiée,
testée et packageable à ce jour (`mif_dqf` sur PyPI, utilisé en mode DIAGNOSTIC par
`layer1_engine/data_loader.py`). Les autres composantes (PAF, tests MIF Phase 0/1/2,
MÉTIS) sont des spécifications internes QAAF documentées dans `docs/architecture.md`
et `docs/archive/`.

`docs/mif_v4_framework.md` **n'existe pas dans ce dépôt**. Toute référence à ce
fichier dans la documentation ou les commits est une référence anticipatoire non
encore matérialisée.

---

## 2. Justification de l'ordre PAF → MIF → MÉTIS

### Sources dans le dépôt

**Source 1** — `docs/architecture.md`, Règle d'or #5 :
> *"PAF → MIF → MÉTIS → D-SIG : ordre immuable, jamais de court-circuit."*

**Source 2** — `layer2_qualification/mif/mif_runner.py`, docstring ligne 5 :
> *"Règle d'arrêt à chaque phase : FAIL → stopper immédiat."*

**Source 3** — `docs/architecture.md`, flux de certification :
> *"→ FAIL → suspendu (sauf --force-metis)"*

### Justification économique

L'ordre PAF → MIF → MÉTIS est un **choix d'architecture QAAF**, non prescrit par
un document MIF externe.

Sa justification est causale :

- **PAF avant MIF** : PAF D3 détecte si le signal est un artefact de lissage. Un
  signal artefactuel qui passerait MIF produirait des métriques MÉTIS sans valeur
  informative. Le coût de PAF (quelques secondes) est négligeable devant le coût
  de MÉTIS Q2 (10 000 permutations).

- **MIF avant MÉTIS** : MIF Phase 0 détecte les biais algorithmiques (lookahead,
  instabilité numérique, dépendance au régime unique). Un signal présentant un biais
  de causalité (T1-T6) produirait des métriques OOS invalides en MÉTIS — les
  résultats seraient statistiquement non interprétables. Continuer serait du calcul
  gaspillé et un risque de faux positif.

- **MÉTIS après MIF** : MÉTIS valide la valeur économique réelle OOS d'un signal
  déjà qualifié algorithmiquement. Cette séquence découple la robustesse algorithmique
  (MIF) de la significativité statistique (MÉTIS), ce qui permet de diagnostiquer
  précisément la cause d'un échec.

---

## 3. Déviations délibérées et leur statut

### 3.1 Gate séquentiel tout-ou-rien (Phase 0 et Phase 1)

**Comportement** : si un seul test échoue en Phase 0 ou Phase 1, le pipeline s'arrête
immédiatement (`FAIL_PHASE_0` ou `FAIL_PHASE_1` dans `MIFSummary.verdict`).

**Justification Phase 0** : un biais de causalité (T1 : lookahead) ou une instabilité
algorithmique (T6 : crash) invalide l'intégrité du signal quelle que soit sa
performance. Le gate tout-ou-rien est causalement justifié.

**Justification Phase 1** : un régime G1-G5 en FAIL indique que le signal ne
généralise pas hors de son régime d'entraînement — overfitting caractérisé. L'arrêt
est conservateur mais défendable pour une qualification stricte.

**Statut** : le gate Phase 0 est définitif. Le gate Phase 1 est **révisable** : une
politique alternative (ex. gate 4/5 comme MÉTIS Q1) pourrait être appropriée selon
le contexte de certification. À adresser dans une session dédiée si nécessaire.

### 3.2 Gate proportionnel Phase 2 (GATE_RATIO = 0.75)

**Comportement** : Phase 2 exige que 3 paires sur 4 passent (75 %), pas un succès
universel. Défini dans `layer2_qualification/mif/phase2_multiasset.py`, ligne 18 :
```python
GATE_RATIO = 0.75
```

**Justification** : la nature multi-actifs de Phase 2 (M1-M4 : PAXG/BTC, ETH/BTC,
SOL/BTC, BNB/BTC) reconnaît qu'un signal peut être structurellement moins adapté
à une paire particulière sans être invalidé globalement. Un gate tout-ou-rien serait
trop restrictif pour une qualification de transferabilité. La valeur 0.75 est un
choix paramétrique — ajustable via `FilterConfig.params["gate_ratio"]`.

**Statut** : déviation délibérée et documentée, cohérente avec la nature du test.

### 3.3 Contournement --force-metis

**Comportement** : `docs/architecture.md` mentionne un flag `--force-metis` permettant
de contourner l'arrêt MIF FAIL et d'exécuter MÉTIS malgré l'échec.

**Statut** : documenté dans `architecture.md` mais **non implémenté** dans le périmètre
Phase 2 (QAAF Studio 3.1). Hors périmètre Phase 2 — à implémenter dans une session
post-Phase 2 si des cas d'usage le justifient (ex. recherche exploratoire sur signal
connu défaillant).

---

## 4. Table de correspondance filtres → origine empirique

| Couche | ID | Filtre | Objet | Source de spécification |
|--------|----|--------|-------|------------------------|
| PAF | D1 | Hiérarchie signal | Signal > benchmark (B_5050, B_BTC, B_PAXG) en CNSR-USD | `docs/archive/QAAF_ClaudeCode_Part3_PAF.md` |
| PAF | D2 | Isolation couches | Attribution des gains par couche du signal | `docs/archive/QAAF_ClaudeCode_Part3_PAF.md` |
| PAF | D3 | Source artefact | Artefact de lissage vs signal informatif (test trivial EMA) | `docs/archive/QAAF_ClaudeCode_Part3_PAF_v2.md` |
| MIF | T1-T6 | Phase 0 — Isolation | Biais algorithmiques sur 6 régimes synthétiques contrôlés | `layer2_qualification/mif/phase0_isolation.py` |
| MIF | G1-G5 | Phase 1 — Généralisation | Robustesse sur 5 régimes OOS (bear, latéral, crash, standard) | `layer2_qualification/mif/phase1_oos.py` |
| MIF | M1-M4 | Phase 2 — Multi-actifs | Transferabilité sur 4 paires synthétiques (gate 75 %) | `layer2_qualification/mif/phase2_multiasset.py` |
| MÉTIS | Q1 | Walk-forward | CNSR > 0.5 sur ≥ 4/5 fenêtres temporelles glissantes | `studio/filters/metis_q1.py`, Bailey & López de Prado (2014) |
| MÉTIS | Q2 | Permutation | p-value < 0.05 sur 10 000 permutations de l'allocation | `studio/filters/metis_q2.py` |
| MÉTIS | Q3 | Stabilité EMA | Absence de pic de performance sur grille de paramètres IS | `studio/filters/metis_q3.py` |
| MÉTIS | Q4 | DSR | Deflated Sharpe Ratio ≥ 0.95 avec N_effectif (Bailey & LdP 2014) | `studio/filters/metis_q4.py` |
| D-SIG | — | Score composite | Verdict lisible 0-100, label GOOD/DEGRADED/CRITICAL | `layer4_decision/` |

**Note** : aucun de ces filtres n'est la transposition directe d'une décomposition MIF
externe. L'ensemble constitue un design QAAF original, ancré dans les principes de
Bailey & López de Prado (2014) pour les tests MÉTIS, et dans les pratiques de
qualification de stratégies quantitatives pour PAF et MIF.

---

## 5. Risque résiduel — contamination alloc_btc inter-couche

### Description du risque

Le signal `alloc_btc` est produit par l'oracle en IS (période d'entraînement), puis
passé directement à MÉTIS Q1, Q2 et Q4 pour validation OOS. Ce passage expose MÉTIS
aux choix d'optimisation effectués en IS.

Concrètement :

- **MÉTIS Q4 (DSR)** : évalue `alloc_btc` sur la période OOS. Si `alloc_btc` a été
  optimisé pour maximiser le CNSR IS, le DSR OOS peut être gonflé par des effets de
  leakage indirect (ex. paramètres IS ajustés implicitement aux caractéristiques de la
  paire complète, pas uniquement IS).

- **MÉTIS Q1 (walk-forward)** : utilise `alloc_btc.reindex(df.index).fillna(0.5)` sur
  l'historique complet. L'allocation est pré-calculée — les fenêtres walk-forward ne
  testent pas une re-calibration indépendante, mais la stabilité temporelle d'une
  allocation fixe.

- **MÉTIS Q2 (permutation)** : permute l'allocation OOS pour calculer une p-value.
  Si l'allocation IS et OOS partagent des caractéristiques structurelles (ex. EMA
  span optimal stable), les permutations peuvent sous-estimer la significativité réelle.

### Statut

**Documenté, non résolu.** `docs/architecture.md` prescrit la séparation IS/OOS
(Règle #2 : *"L'OOS est vu une seule fois"*) et réserve Q3 à IS uniquement. Mais
le passage de `alloc_btc` pré-calculé entre couches n'est pas formellement traité
dans l'architecture actuelle.

Ce risque est inhérent à l'architecture de validation en deux étapes (qualification
→ validation). Sa résolution complète nécessiterait une re-calibration du signal
dans chaque fenêtre MÉTIS — ce qui changerait fondamentalement la nature de Q1
(de stabilité temporelle à walk-forward vrai) et est hors périmètre Phase 2.

**À adresser dans une session dédiée post-Phase 2**, idéalement comme premier sujet
d'audit interne dans le cadre FRAF (Framework Risk Audit Framework).
