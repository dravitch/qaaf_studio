# KB Famille : EMA_span_variants — Sprint A

**Date** : Avril 2026  
**Famille** : `EMA_span_variants`  
**N_trials total** : 103 (H9+EMA60j = 101 + H9+EMA60j+MA200 = 2 variantes)

---

## Hypothèses testées

### H9+EMA60j (lentille pilote)

| Dimension | Valeur |
|---|---|
| Hypothèse précise | H9 + EMA 60j surpasse B_5050 sur CNSR-USD-Fed, OOS 2023-06 → 2024-12, paire PAXG/BTC |
| N_trials au test | 101 |
| CNSR-USD-Fed OOS | 1.285 |
| Sortino OOS | 1.928 |
| Max DD OOS | 13.6% |
| DSR(N=101) | 0.3246 |
| PAF D1 | HIERARCHIE_CONFIRMEE |
| PAF D2 | REGIMES_NEUTRES |
| PAF D3 | H9_LISSE_SUPERIEUR |
| MIF Phase 1 | FAIL (G1 bear, G2 lateral, G3 crash) |
| MÉTIS Q1 | PASS |
| MÉTIS Q2 | FAIL (p=0.487) |
| MÉTIS Q4 | FAIL (DSR=0.325 < 0.95) |
| D-SIG | 59/100 — DEGRADED |
| **Verdict** | **ARCHIVE_FAIL_Q2_Q4** |

**Diagnostic** : Le signal H9+EMA60j présente une hiérarchie confirmée (D1-D3) mais échoue sur la robustesse de régime (MIF Phase 1) et la significativité statistique (Q2 p=0.487, DSR=0.325). Le Sharpe brut de 1.77 documenté dans QAAF-R (avant correction numéraire) était trompeur.

---

### H9+EMA60j+MA200 (Sprint A — variantes hard & soft)

**Hypothèse précise** : H9+EMA60j avec filtre MA200 sur BTC corrige les déficits MIF G1/G2/G3 documentés sur H9+EMA60j.

| Dimension | hard | soft |
|---|---|---|
| N_trials (cumul famille) | 102 | 103 |
| Logique filtre | allocation = 0 si BTC < MA200 | allocation × 0.5 si BTC < MA200 |
| Référence certifiée | H9+EMA60j (PAF D1-D3 chargés depuis KB) | idem |

*Résultats détaillés à compléter après run complet Sprint A.*

---

## Chronologie Sprint A

| Date | Événement |
|---|---|
| 2026-04-01 | PAF D1/D2/D3 certifiés pour H9+EMA60j |
| 2026-04-22 | Run certify_h9_ema60j.py — verdict ARCHIVE_FAIL_Q2_Q4 |
| 2026-04-26 | Session H9+MA200 (hard + soft) lancée — Sprint A |

---

## Leçons apprises

1. **MIF Phase 1 comme filtre précoce** : les déficits G1/G2/G3 sur H9+EMA60j auraient pu stopper le test MÉTIS en mode complet. Le flag `--force-metis` a été nécessaire pour diagnostiquer.
2. **N_trials depuis KB** : avec N=101 dès la première session, le seuil DSR de 0.95 est très exigeant. Prévoir N_trials réaliste avant de définir le critère de succès.
3. **Filtre MA200** : hypothèse naturelle pour corriger les déficits de régime bear/crash — à vérifier via MIF Phase 1 sur hard vs soft.

---

*Famille EMA_span_variants — Sprint A | QAAF Studio 3.0 | Avril 2026*
