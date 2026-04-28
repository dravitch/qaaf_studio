"""
layer3_validation/metis.py
Alias de certification pour MetisQ2Permutation (Étape E).

MetisQ2Permutation = MetisQ2 avec support config=None.
"""

from studio.filters.metis_q2 import MetisQ2
from studio.interfaces import FilterConfig as _FC


class MetisQ2Permutation(MetisQ2):
    """Alias de certification : MetisQ2 avec config=None supporté."""

    def evaluate(self, signal, config=None):
        if config is None:
            config = _FC(name="certification", params={})
        return super().evaluate(signal, config)
