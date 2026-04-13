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
  ];

  shellHook = ''
    echo "🔬 Initialisation de l'environnement QAAF Studio 3.0..."

    # 1. Chemins bibliothèques C
    export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:${pkgs.openssl.out}/lib:$LD_LIBRARY_PATH"
    export PKG_CONFIG_PATH="${pkgs.openssl.dev}/lib/pkgconfig:${pkgs.zlib}/lib/pkgconfig:$PKG_CONFIG_PATH"

    # 2. Venv local
    # --system-site-packages : hérite de numpy/pandas/scipy du pythonEnv nix
    VENV_DIR=".venv_qaaf"

    if [ ! -d "$VENV_DIR" ]; then
      echo "📦 Création de l'environnement virtuel Python..."
      python -m venv "$VENV_DIR" --system-site-packages
    fi

    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip setuptools wheel --quiet

    # 3. Packages pip uniquement
    # yfinance : absent de nixpkgs
    pip install --quiet yfinance
    # pytest : tests
    pip install --quiet pytest
    # mif-dqf : DQF complet (stub actif sans)
    # pip install --quiet mif-dqf

    # 4. PYTHONPATH : imports from project root
    export PYTHONPATH="$(pwd):$PYTHONPATH"

    # 5. Vérifications
    echo ""
    echo "📊 Vérification des dépendances QAAF Studio 3.0 :"
    echo ""
    python -c "import numpy;      print('   ✓ numpy:',      numpy.__version__)"      2>/dev/null || echo "   ✗ numpy"
    python -c "import pandas;     print('   ✓ pandas:',     pandas.__version__)"     2>/dev/null || echo "   ✗ pandas"
    python -c "import scipy;      print('   ✓ scipy:',      scipy.__version__)"      2>/dev/null || echo "   ✗ scipy"
    python -c "import yaml;       print('   ✓ pyyaml:',     yaml.__version__)"       2>/dev/null || echo "   ✗ pyyaml"
    python -c "import requests;   print('   ✓ requests:',   requests.__version__)"   2>/dev/null || echo "   ✗ requests"
    python -c "import matplotlib; print('   ✓ matplotlib:', matplotlib.__version__)" 2>/dev/null || echo "   ✗ matplotlib"
    python -c "import yfinance;   print('   ✓ yfinance:',   yfinance.__version__)"   2>/dev/null || echo "   ✗ yfinance"
    python -c "import pytest;     print('   ✓ pytest:',     pytest.__version__)"     2>/dev/null || echo "   ✗ pytest"
    echo ""
    python -c "from mif_dqf import DQFValidator; print('   ✓ mif-dqf: mode DIAGNOSTIC complet')" \
      2>/dev/null || echo "   ℹ  mif-dqf: non installé — stub DQF actif"
    echo ""

    # 6. Imports QAAF Studio
    python -c "
from layer1_engine import compute_cnsr, deflated_sharpe_ratio, Backtester, BenchmarkFactory
from layer2_qualification.mif.mif_runner import MIFRunner
from layer3_validation.metis_runner import METISRunner
from layer4_decision.dsig.mapper import strategy_to_dsig
print('   ✓ QAAF Studio 3.0 — tous les imports OK')
" 2>/dev/null || echo "   ⚠  Imports QAAF Studio KO — vérifier PYTHONPATH ($(pwd))"

    echo ""
    echo "✅ Environnement QAAF Studio 3.0 prêt"
    echo "🐍 Python: $(python --version)"
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
