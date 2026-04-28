"""
layer3_validation/dsr.py
Alias de certification pour DSRFilter (Étape E).

DSRFilter = MetisQ4 avec support config=None.
"""

from studio.filters.metis_q4 import MetisQ4
from studio.interfaces import FilterConfig as _FC


class DSRFilter(MetisQ4):
    """Alias de certification : MetisQ4 avec config=None supporté."""

    def evaluate(self, signal, config=None):
        if config is None:
            config = _FC(name="certification", params={})
        return super().evaluate(signal, config)
