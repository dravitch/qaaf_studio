# Gate 0 — Problème Yahoo Finance / yfinance

**Statut** : En investigation — stable (11 skipped sur NixOS/CI, 11 PASS sur Windows)
**Date** : Avril 2026
**Tests concernés** : `tests/test_benchmark_calibration.py` (11 tests)

---

## Symptôme

Gate 0 retourne systématiquement `11 skipped` dans certains environnements, même avec
yfinance installé et le réseau disponible. Le skip vient du `_skip_if_no_data` decorator
qui attrape toute exception levée dans `_load_data()` au niveau module.

```
[!!] Ces tests necessitent un acces reseau (yfinance). Skip si hors ligne.
>> Calibrage Benchmarks (attendu : 11 tests)...
[WARN] Calibrage Benchmarks : 11 skipped
```

---

## Comportement observé par environnement

### Windows — Python 3.12.10, uv

| Version yfinance | Résultat Gate 0 | Notes |
|------------------|-----------------|-------|
| Latest (via requirements.txt) | **11 PASS** | Réseau OK, mif-dqf 1.2.0.post1 actif |

**Commandes :**
```powershell
# Setup
git pull origin main
.\tools\local\test-qaaf-studio.ps1

# Isoler Gate 0 uniquement
.\tools\local\test-qaaf-studio.ps1 -Layer 1

# Direct pytest
.\.venv\Scripts\python.exe -m pytest tests/test_benchmark_calibration.py -v

# Diagnostiquer le chargement des données
.\.venv\Scripts\python.exe -c "
from layer1_engine.data_loader import DataLoader
dl = DataLoader()
p, b, rp, rb = dl.load_prices('2019-01-01', '2024-12-31')
print('PAXG:', len(p), 'points —', p.index[0].date(), '->', p.index[-1].date())
print('BTC: ', len(b), 'points')
"

# Vérifier la version et le backend actif
.\.venv\Scripts\python.exe -c "
import yfinance as yf
print('version:', yf.__version__)
import requests; print('requests OK:', requests.__version__)
try:
    import curl_cffi; print('curl_cffi:', curl_cffi.__version__)
except ImportError:
    print('curl_cffi: absent (backend requests)')
"
```

---

### NixOS — Python 3.12 (nix-shell), venv --system-site-packages

| Version yfinance | Résultat Gate 0 | Cause |
|------------------|-----------------|-------|
| Latest (>=0.2.0) | 11 skipped | curl_cffi wheel ABI python3.13 ≠ venv python3.12 |
| Latest + `YFINANCE_NO_CURL=1` | 11 skipped | Variable ignorée dans certaines versions |
| Latest + `pip uninstall curl_cffi` | 11 skipped | Encore en investigation |
| **0.2.51 (pin actuel)** | 11 skipped | yfinance import OK, mais _load_data() échoue pour une autre raison |

**Commandes :**
```bash
# Setup depuis zéro
rm -rf .venv
nix-shell   # crée le venv, installe yfinance==0.2.51

# Isoler Gate 0
bash tools/local/test-qaaf-studio.sh --layer 1

# Direct pytest
pytest tests/test_benchmark_calibration.py -v

# Diagnostiquer l'erreur exacte dans _load_data()
python -c "
import sys, traceback
sys.path.insert(0, '.')
from tests.test_benchmark_calibration import _LOAD_ERROR, _DATA
print('_DATA:', 'OK' if _DATA else 'None')
print('_LOAD_ERROR:', _LOAD_ERROR)
"

# Tester yfinance seul (hors DataLoader)
python -c "
import yfinance as yf
print('version:', yf.__version__)
df = yf.download('PAXG-USD', start='2023-01-01', end='2023-06-01', progress=False)
print('rows:', len(df))
print(df.tail(3))
"

# Vérifier la présence de curl_cffi
python -c "
try:
    import curl_cffi; print('curl_cffi present:', curl_cffi.__version__)
except ImportError:
    print('curl_cffi: absent (ok pour 0.2.51)')
import requests; print('requests:', requests.__version__)
"
```

---

## Historique des approches testées

### Approche 1 — Compiler curl_cffi depuis les sources (abandonnée)
`pip install curl_cffi --no-binary curl_cffi` avec `pkgs.libffi` dans buildInputs.
Résultat : build échoue ou produit un wheel inutilisable sur NixOS.

### Approche 2 — Désinstaller curl_cffi + YFINANCE_NO_CURL=1 (abandonnée)
```bash
pip install yfinance
pip uninstall -y curl_cffi
export YFINANCE_NO_CURL=1
```
Résultat : skip persiste. `_load_data()` lève une exception non-réseau.

### Approche 3 — Pin yfinance==0.2.51 (approche actuelle)
Dernière version antérieure à l'introduction de curl_cffi.
Utilise `requests` nativement, aucune dépendance ABI.
Résultat : yfinance importable, mais Gate 0 encore skippée sur NixOS.
**Conclusion** : l'ABI curl_cffi n'est PAS la seule cause du skip sur NixOS.

---

## Hypothèses à investiguer

1. **API yfinance changée** — `data["Close"]` retourne un `DataFrame` au lieu d'une `Series`
   depuis yfinance 0.2.x selon la configuration multi-ticker. `.squeeze()` peut échouer.
   → Tester : `df = yf.download("PAXG-USD", ...); print(type(df["Close"]), df["Close"].shape)`

2. **Réponse Yahoo Finance différente par géolocalisation/IP**
   Yahoo Finance bloque ou limite certaines IP (serveurs cloud, VPN, NixOS).
   Les données retournées peuvent être vides ou malformées.
   → Tester : `df = yf.download("PAXG-USD", start="2019-01-01", end="2024-12-31"); print(len(df))`

3. **Index non-UTC sur NixOS** — timezone naive vs aware selon le système.
   L'intersection d'index `paxg_usd.index.intersection(btc_usd.index)` pourrait être vide.
   → Tester : `print(paxg_usd.index.dtype, paxg_usd.index[:3])`

4. **DQF FAIL sur données retournées** — le stub DQF lève `ValueError` si status="FAIL".
   Sur NixOS, yfinance pourrait retourner des données avec trop de NaN (IP bloquée).
   → Vérifier : le message dans `_LOAD_ERROR` sur NixOS.

---

## Données de référence KB (session certify_h9_ema60j, Avril 2026)

Ces valeurs sont valides sur Windows avec yfinance latest :

| Benchmark | CNSR OOS | Tolérance |
|-----------|----------|-----------|
| B_5050    | 1.343    | ±0.15     |
| B_BTC     | 1.244    | ±0.15     |

Période OOS : 2023-06-01 → 2024-12-31 (≥ 380 jours crypto)

---

## Prochaine étape

Reproduire l'erreur exacte sur NixOS avec `_LOAD_ERROR` puis trier entre
hypothèses 1, 2, 3, 4 ci-dessus. Ne pas lancer de session comparative tant
que Gate 0 est rouge ou skippée.

**Voir aussi** : `tests/test_benchmark_calibration.py` — décorateur `_skip_if_no_data`

---

## Résolution — Mai 2026

**Cause racine confirmée : H2 — rate-limiting Yahoo Finance (429) sur NixOS**

```bash
curl -s -o /dev/null -w "%{http_code}" \
  "https://query1.finance.yahoo.com/v8/finance/chart/PAXG-USD?interval=1d&range=1mo"
# → 429 sans User-Agent
# → 200 avec User-Agent browser
```

H1, H3, H4 étaient des symptômes en cascade : df vide → shape (0,1) →
index vide → DQF FAIL. La cause unique est le blocage réseau Yahoo.

**Solution appliquée : fallback requests natif dans `_download_ticker`**

- yfinance tenté en premier (fonctionne sur Windows)
- Si données vides (< 30 points) → fallback `requests` avec User-Agent browser
- Index normalisé tz-naive pour cohérence avec yfinance 0.2.51
- Guard anti-corruption cache : fichier < 30 points → suppression automatique

**Commit** : `fix(data_loader): fallback requests natif pour 429 Yahoo sur NixOS`
**Branche** : `fix/yfinance-nixos-diagnosis`
**Résultat** : Gate 0 — 11/11 PASS (était 11 skipped sur NixOS)

**Note yfinance long terme** : la baseline 0.2.51 est fonctionnelle mais pas
pérenne. Migration vers yfinance 1.3.0 à planifier en session dédiée après
certification H9+EMA60j — le MultiIndex et la structure JSON ont changé.

**Statut** : RÉSOLU — archivé.
```

Applique cet addendum dans le fichier, puis commit sur la même branche avant la PR :

```bash
git add docs/gate0_yfinance_troubleshooting.md  # ou le chemin exact
git commit -m "docs: archiver gate0_yfinance_troubleshooting avec résolution H2"
git push origin fix/yfinance-nixos-diagnosis
```

Ensuite on ouvre la PR et on passe aux Points 2 et 3.