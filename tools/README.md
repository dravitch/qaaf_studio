# tools/ — Outils QAAF Studio 3.0

## Outils actifs

| Fichier | Rôle | Usage |
|---|---|---|
| `certify_metric.py` | Certifie une métrique individuelle via MIF v4.0 | `python tools/certify_metric.py --metric vol_ratio` |
| `test_metrics_orthogonality.py` | Vérifie l'orthogonalité des 4 métriques clés | `pytest tools/test_metrics_orthogonality.py` |
| `cleanup_v2.py` | Nettoie les fichiers temporaires et backups | `python tools/cleanup_v2.py` (dry-run) / `--execute` |

## Répertoires

| Répertoire | Rôle |
|---|---|
| `local/` | Scripts CI locaux — `test-qaaf-studio.sh` (.sh Linux, .ps1 Windows) |
| `migration_tools/` | Promotion d'une métrique de cooking → production |

## Scripts CI

```bash
# Lancer tous les tests (toutes les couches)
bash tools/local/test-qaaf-studio.sh --layer all

# Lancer uniquement une couche
bash tools/local/test-qaaf-studio.sh --layer layer1
```

## Archive

Les outils expérimentaux ou obsolètes sont dans `docs/archive/` :
- `score_composite/` — composite score expérimental (non intégré au pipeline v1.0)
