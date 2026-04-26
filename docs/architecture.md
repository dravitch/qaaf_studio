# QAAF Studio 3.0 — Architecture

## Les 5 couches et leur responsabilité unique

| Couche | Module | Responsabilité unique |
|---|---|---|
| **Layer 1** — Moteur unifié | `layer1_engine/` | Métriques correctes et reproductibles (CNSR-USD, DSR, Sortino, Calmar). Conditions expérimentales fixes (splits, frais, taux sans risque). |
| **Layer 2** — Qualification | `layer2_qualification/` | Ce signal mérite-t-il d'être testé ? PAF (hiérarchie, attribution, sources) + MIF (robustesse algorithmique sur régimes variés). |
| **Layer 3** — Validation | `layer3_validation/` | Ce signal génère-t-il de la valeur réelle OOS ? MÉTIS Q1 (walk-forward), Q2 (permutation), Q3 (stabilité EMA), Q4 (DSR multiple testing). |
| **Layer 4** — Décision | `layer4_decision/` | Que retient-on ? Mémoire active (KB), verdict lisible (D-SIG score 0-100), compteur N_trials pour le DSR. |
| **sessions/** — Pilotes | `sessions/` | Orchestration complète d'une certification hypothèse. Wrapper autour de `SessionRunner` générique. |

## Flux de certification

```
Nouvelle hypothèse
       │
       ▼
  KB check (pré-session)
  → doublon ? → STOP
       │
       ▼
  Chargement données (réelles ou synthétiques fallback)
       │
       ▼
  Benchmarks OOS (B_5050, B_BTC, B_PAXG)
       │
       ▼
  PAF (D1 → D2 → D3)   ← certifié une fois, chargé depuis KB ensuite
  D1 : hiérarchie signal vs benchmarks
  D2 : isolation des couches du signal
  D3 : test source (artefact lissage ?)
       │
       ▼
  Métriques OOS brutes (CNSR-USD-Fed, Sortino, Calmar, Max DD, DSR)
       │
       ▼
  MIF Phase 0 + 1 (+ Phase 2 optionnelle)
  Phase 0 : tests algorithmiques T1-T6 (données synthétiques)
  Phase 1 : généralisation G1-G5 (bull, bear, lateral, crash, choppy)
  → FAIL → suspendu (sauf --force-metis)
       │
       ▼
  MÉTIS Q1-Q4
  Q1 : walk-forward 5 fenêtres (CNSR > 0.5 sur ≥ 4/5)
  Q2 : permutation 10 000 itérations (p-value < 0.05)
  Q3 : grille EMA IS, détection pic de paramètre
  Q4 : DSR ≥ 0.95 avec N_trials de la famille
       │
       ▼
  D-SIG (score 0-100, label GOOD/DEGRADED/CRITICAL, trend)
       │
       ▼
  Verdict final + KB update
  → CERTIFIE / ARCHIVE_FAIL_Q* / SUSPECT_DSR
```

## Règles d'or

1. **Iso-contexte** : toutes les hypothèses d'une famille partagent exactement les mêmes splits, frais et taux sans risque (définis dans `config.yaml`).
2. **Pas d'OOS pendant le développement** : la grille de paramètres (Q3) tourne sur IS uniquement. L'OOS est vu une seule fois.
3. **KB avant tout test** : 4 questions systématiques en pré-session (doublon ? famille ? N_trials à jour ? split cohérent ?).
4. **CNSR-USD comme seule référence de comparaison** inter-stratégies — Sharpe paire brute interdit.
5. **PAF → MIF → MÉTIS → D-SIG** : ordre immuable, jamais de court-circuit.
6. **N_trials incrémenté à chaque variante** : toute variante testée compte, même si rejetée en amont.

## Nomenclature : deux systèmes de "phases"

**Plan de mise en œuvre (couches 1-4 + sessions)** — phases de construction du studio :
- Phase 1 : Layer 1 (moteur) + Layer 4 (KB initiale)
- Phase 2 : Layer 2 (PAF + MIF)
- Phase 3 : Layer 3 (MÉTIS)
- Phase 4 : sessions/ (première certification pilote)

**MIF interne (layer2_qualification/mif/)** — étapes de tests algorithmiques :
- Phase 0 : tests d'isolation T1-T6 (données synthétiques contrôlées)
- Phase 1 : tests de généralisation G1-G5 (régimes OOS variés)
- Phase 2 : tests multi-actifs M1-M4 (4 paires, gate 75%)
- Phase 3 : intégration post-déploiement — **hors périmètre v1.0**

> Ces deux nomenclatures sont indépendantes. "Phase 3" du plan de mise en œuvre ≠ "Phase 3" MIF.

## Hors périmètre v1.0

- **MIF Phase 3** (I1-I5) : monitoring post-déploiement — prévu après déploiement live.
  Archive : `docs/archive/phase3_integration_draft.py`
- **Score composite** : agrégation expérimentale multi-métriques — non intégrée au pipeline.
  Archive : `docs/archive/score_composite/`
