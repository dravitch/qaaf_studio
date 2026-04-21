{ pkgs ? import <nixpkgs> {} }:

let
  pythonEnv = pkgs.python312.withPackages (ps: with ps; [
    # QAAF Studio 3.0 — packages nixpkgs
    numpy
    pandas
    scipy
    matplotlib
    requests
    pyyaml
    python-dateutil
    pytz
  ]);
in
pkgs.mkShell {
  buildInputs = [
    pythonEnv
    pkgs.gcc
    pkgs.pkg-config
    pkgs.stdenv.cc.cc.lib
    pkgs.zlib
    pkgs.openssl
    pkgs.curl.dev      # libcurl headers pour curl_cffi
    pkgs.cacert        # certificats SSL pour pip + yfinance downloads
    pkgs.libffi        # libffi headers pour compiler curl_cffi depuis source
  ];

  shellHook = ''
    echo "🔬 Initialisation de l'environnement QAAF Studio 3.0..."

    # 1. Chemins bibliothèques C
    export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:${pkgs.openssl.out}/lib:${pkgs.curl.out}/lib:${pkgs.libffi}/lib:$LD_LIBRARY_PATH"
    export PKG_CONFIG_PATH="${pkgs.openssl.dev}/lib/pkgconfig:${pkgs.zlib}/lib/pkgconfig:${pkgs.curl.dev}/lib/pkgconfig:$PKG_CONFIG_PATH"
    export NIX_SSL_CERT_FILE="${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
    export LDFLAGS="-L${pkgs.libffi}/lib"
    export CFLAGS="-I${pkgs.libffi}/include"

    # 2. Venv local
    # --system-site-packages : hérite de numpy/pandas/scipy du pythonEnv nix
    VENV_DIR=".venv"

    if [ ! -d "$VENV_DIR" ]; then
      echo "📦 Création de l'environnement virtuel Python..."
      python -m venv "$VENV_DIR" --system-site-packages
    fi

    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip setuptools wheel --quiet

    # 3. Packages pip uniquement (absent de nixpkgs ou inaccessibles via --system-site-packages)
    pip install --quiet scipy pyyaml matplotlib
    # curl_cffi : compiler depuis source — la wheel binaire cible python3.13 (Nix store)
    #             mais le venv tourne en python3.12 → _cffi_backend absent sans recompilation
    pip install curl_cffi --no-binary curl_cffi --quiet \
      || echo "   ⚠  curl_cffi: compilation échouée (libffi manquant ?)"
    # yfinance : utilise curl_cffi installé ci-dessus
    pip install yfinance --quiet || echo "   ⚠  yfinance: pip install échoué — vérifier réseau"
    # pytest : tests
    pip install --quiet pytest
    # mif-dqf : DQF complet (stub actif sans)
    # pip install --quiet mif-dqf

    # 4. PYTHONPATH : imports from project root
    export PYTHONPATH="$(pwd):$PYTHONPATH"

    # 5. Vérifications — utiliser le Python du venv explicitement
    PY="$VENV_DIR/bin/python"
    echo ""
    echo "📊 Vérification des dépendances QAAF Studio 3.0 :"
    echo ""
    $PY -c "import numpy;      print('   ✓ numpy:',      numpy.__version__)"      2>/dev/null || echo "   ✗ numpy"
    $PY -c "import pandas;     print('   ✓ pandas:',     pandas.__version__)"     2>/dev/null || echo "   ✗ pandas"
    $PY -c "import scipy;      print('   ✓ scipy:',      scipy.__version__)"      2>/dev/null || echo "   ✗ scipy"
    $PY -c "import yaml;       print('   ✓ pyyaml:',     yaml.__version__)"       2>/dev/null || echo "   ✗ pyyaml"
    $PY -c "import requests;   print('   ✓ requests:',   requests.__version__)"   2>/dev/null || echo "   ✗ requests"
    $PY -c "import matplotlib; print('   ✓ matplotlib:', matplotlib.__version__)" 2>/dev/null || echo "   ✗ matplotlib"
    $PY -c "import yfinance;   print('   ✓ yfinance:',   yfinance.__version__)"   2>/dev/null || echo "   ✗ yfinance"
    $PY -c "import pytest;     print('   ✓ pytest:',     pytest.__version__)"     2>/dev/null || echo "   ✗ pytest"
    echo ""
    $PY -c "from mif_dqf import DQFValidator; print('   ✓ mif-dqf: mode DIAGNOSTIC complet')" \
      2>/dev/null || echo "   ℹ  mif-dqf: non installé — stub DQF actif"
    echo ""

    # 6. Imports QAAF Studio
    $PY -c "
from layer1_engine import compute_cnsr, deflated_sharpe_ratio, Backtester, BenchmarkFactory
from layer2_qualification.mif.mif_runner import MIFRunner
from layer3_validation.metis_runner import METISRunner
from layer4_decision.dsig.mapper import strategy_to_dsig
print('   ✓ QAAF Studio 3.0 — tous les imports OK')
" 2>/dev/null || echo "   ⚠  Imports QAAF Studio KO — vérifier PYTHONPATH ($(pwd))"

    echo ""
    echo "✅ Environnement QAAF Studio 3.0 prêt"
    echo "🐍 Python: $($PY --version)"
    echo ""
    echo "🚀 Commandes disponibles :"
    echo "   python -m unittest tests/test_layer1_metrics.py tests/test_layer1_backtester.py -v"
    echo "   pytest tests/ -k 'layer1' -v"
    echo "   python sessions/h9_ema60j/certify_h9_ema60j.py --fast"
    echo "   python sessions/h9_ema60j/certify_h9_ema60j.py"
    echo ""
    echo "💡 Venv : $VENV_DIR — pour désactiver : deactivate"
    echo ""
  '';

  PYTHON_KEYRING_BACKEND = "keyring.backends.null.Keyring";
}
