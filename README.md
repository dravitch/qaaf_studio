# QAAF Studio 3.0

**Andrei | Avril 2026 | Claude (Sonnet 4.6)**

Environnement de recherche contraint pour la certification de stratégies de
rebalancement PAXG/BTC. Métrique de référence : CNSR-USD. Protocole : PAF → MIF → MÉTIS.

---

## Structure des fichiers

```
qaaf_studio/
│
├── config.yaml                    ← paramètres globaux (splits, frais, Rf)
├── requirements.txt
│
├── layer1_engine/                 ── COUCHE 1 : MOTEUR UNIFIÉ ──────────────
│   │                                 "La lumière de la pièce" — conditions fixes
│   │                                 Phase 1 du plan de mise en œuvre
│   │
│   ├── metrics_engine.py          compute_cnsr() · deflated_sharpe_ratio()
│   │                              MetricsEngine.compute_all()
│   │
│   ├── data_loader.py             DataLoader.load_prices()
│   │                              _dqf_stub() (Layer 0 intégré)
│   │
│   ├── split_manager.py           SplitManager — splits figés + N_trials
│   ├── backtester.py              Backtester.run(allocation_fn, prices_df, r_btc_usd)
│   └── benchmark_factory.py       BenchmarkFactory(backtester) → B_5050/BTC/PAXG
│
├── layer2_qualification/          ── COUCHE 2 : QUALIFICATION ──────────────
│   │                                 Phase 2 du plan de mise en œuvre
│   │                                 "Filtre d'entrée" — avant MÉTIS
│   │
│   ├── paf/
│   │   ├── paf_runner.py          PAFRunner.run() — orchestre D1→D2→D3
│   │   │                          Précondition : DQF PASS vérifié en entrée
│   │   ├── paf_d1_hierarchy.py    D1 — hiérarchie signal vs benchmarks passifs
│   │   ├── paf_d2_attribution.py  D2 — isolation des couches du signal
│   │   └── paf_d3_source.py       D3 — test iso-variance (artefact lissage ?)
│   │
│   └── mif/
│       ├── synthetic_data.py      generate_synthetic_paxgbtc() — F1-F5
│       ├── phase0_isolation.py    T1-T6 — détecte 80% des bugs algorithmiques
│       ├── phase1_oos.py          G1-G5 — généralisation OOS sur régimes variés
│       ├── phase2_multiasset.py   M1-M4 — transfert 4 paires (gate 75%)
│       └── mif_runner.py          MIFRunner.run() — Phase 0→1→2 avec arrêt
│
├── layer3_validation/             ── COUCHE 3 : VALIDATION ─────────────────
│   │                                 Phase 3 du plan de mise en œuvre
│   │                                 MÉTIS v2.1 — "le vrai test OOS"
│   │
│   ├── metis_q1_walkforward.py    Q1 — 5 fenêtres, CNSR > 0.5 sur ≥ 4/5
│   ├── metis_q2_permutation.py    Q2 — 10k permutations, p-value < 0.05
│   ├── metis_q3_ema_stability.py  Q3 — grille 20→120j IS, détection spike
│   ├── metis_q4_dsr.py            Q4 — DSR ≥ 0.95 (N_trials auto)
│   └── metis_runner.py            METISRunner.run() — orchestre Q1→Q4
│
├── layer4_decision/               ── COUCHE 4 : DÉCISION ───────────────────
│   │                                 Phase 1 (KB initiale) + usage continu
│   │                                 "La fiche patient" — mémoire active
│   │
│   ├── kb_manager.py              KBManager — CRUD KB + règles d'arrêt
│   │                              Vérification pré-session (4 questions)
│   ├── n_trials_tracker.py        NTrialsTracker — compteur DSR par famille
│   ├── kb_h9_ema60j.yaml          KB hypothèse active
│   ├── lentilles_inventory.yaml   Inventaire toutes lentilles
│   │
│   └── dsig/
│       ├── mapper.py              strategy_to_dsig() → score 0-100 + label
│       └── profiles.yaml          profils de pondération par famille
│
├── sessions/                      ── PHASE 4 : SESSION PILOTE ─────────────
│   └── certify_h9_ema60j.py       Certification complète H9+EMA60j
│                                  python sessions/certify_h9_ema60j.py --fast
│
└── tests/
    ├── conftest.py                 fixtures synthetic_prices / synthetic_returns
    ├── test_layer1_metrics.py      CNSR, DSR, cas limites
    └── test_layer1_backtester.py   Backtester, BenchmarkFactory, DQF stub
```

---

## Correspondance couches ↔ phases du plan

| Couche | Responsabilité | Phase plan |
|--------|---------------|-----------|
| **Layer 1** — Moteur unifié | Métriques correctes, conditions fixes | Phase 1 |
| **Layer 2** — Qualification | Ce signal mérite-t-il d'être testé ? | Phase 2 |
| **Layer 3** — Validation | Ce signal génère-t-il de la valeur OOS ? | Phase 3 |
| **Layer 4** — Décision | Que retient-on ? Quelle est la décision lisible ? | Phase 1 + continu |
| **sessions/** — Pilote | Première certification complète | Phase 4 |

> ⚠️ **Nomenclature MIF** : les "Phase 0, 1, 2" à l'intérieur de `mif/`
> sont des étapes de tests algorithmiques internes à la Couche 2.
> Elles ne correspondent **pas** aux phases du plan de mise en œuvre (1-4).

---

## Usage rapide

```bash
# Installer les dépendances
pip install -r requirements.txt

# Lancer les tests Layer 1
pytest tests/ -k "layer1" -v

# Session de certification complète
python sessions/certify_h9_ema60j.py

# Mode rapide (test, n_perm=500, skip Q2)
python sessions/certify_h9_ema60j.py --fast --skip-q2
```

---

## Scripts disponibles

| Script | Rôle |
|---|---|
| `deploy.sh` | Commit + sync + push (avec fallback rebase Option B). Usage : `./deploy.sh -m "message"` |
| `tools/local/test-qaaf-studio.sh` | Suite de tests CI locale. Usage : `bash tools/local/test-qaaf-studio.sh --layer all` |
| `tools/local/test-qaaf-studio.ps1` | Équivalent Windows PowerShell |
| `tools/cleanup_v2.py` | Nettoyage des fichiers temporaires. Usage : `python tools/cleanup_v2.py` (dry-run) |

---

## Invariants du studio

1. **Lump sum uniquement** pour la certification — DCA désactivé avec avertissement explicite.
2. **CNSR-USD** comme métrique de référence — Sharpe paire brute interdit en comparaison inter-stratégies.
3. **Splits figés** — définis dans `config.yaml`, jamais modifiés mid-session.
4. **OOS vu une seule fois** — toute grille de paramètres tourne sur IS uniquement.
5. **PAF avant MIF avant MÉTIS** — toujours dans cet ordre.
6. **KB check avant tout test** — 4 questions systématiques, N_trials mis à jour.

---

*QAAF Studio 3.0 | Avril 2026 | Andrei + Claude (Sonnet 4.6)*
*Prochaine révision : après certification de H9+EMA60j (MÉTIS Q1/Q2/Q3/Q4)*
