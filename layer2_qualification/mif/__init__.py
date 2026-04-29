from .mif_runner import MIFRunner, MIFSummary  # noqa: F401

# Adapter pour certification oracle (Étape E)
# MetisQ1WalkForward = MIFPhase1 avec config=None → FilterConfig par défaut
from studio.filters.mif_phase1 import MIFPhase1
from studio.interfaces import FilterConfig as _FC


# MetisQ1WalkForward : alias historique de MIFPhase1 (MIF G1-G5).
# Le walk-forward Layer 3 réel est dans studio/filters/metis_q1.py (MetisQ1).
# Ce nom est conservé pour ne pas casser la structure PIPELINE_AVAILABLE.
class MetisQ1WalkForward(MIFPhase1):
    """Alias de certification : MIFPhase1 avec config=None supporté."""

    def evaluate(self, signal, config=None):
        if config is None:
            config = _FC(name="certification", params={})
        return super().evaluate(signal, config)
