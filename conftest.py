"""
conftest.py — racine du projet QAAF Studio 3.0

Ce fichier est chargé automatiquement par pytest avant toute collecte.
Il ajoute la racine du projet à sys.path, ce qui permet les imports
relatifs depuis n'importe quel sous-répertoire de tests :

    from sessions.comparative_001.signals import signal_h9_ema
    from layer1_engine import compute_cnsr
    from tests.conftest import make_synthetic_prices

Sans ce fichier, pytest sur Windows ne trouve pas les modules du projet.
"""

import sys
from pathlib import Path

# Ajouter la racine du projet en premier dans sys.path
_ROOT = Path(__file__).parent.resolve()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
