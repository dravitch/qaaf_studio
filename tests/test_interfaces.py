"""
tests/test_interfaces.py
MWE pour l'interface Filter.

Ces tests vérifient le contrat d'interface, pas un filtre métier.
Ils sont volontairement courts — chaque assertion teste un fait.

Signal bon      → FilterVerdict(passed=True)  avec diagnosis valide
Signal mauvais  → FilterVerdict(passed=False) avec diagnosis actionnable
Erreur technique → FilterError (jamais en cas de signal invalide)
"""

import pytest
import pandas as pd
import numpy as np
from studio.interfaces import (
    Filter, FilterConfig, FilterVerdict, FilterError,
    SignalData, FilterProtocol,
)


# ─── Fixtures ────────────────────────────────

def make_signal(alloc_value: float = 0.5, n_days: int = 500) -> SignalData:
    """Signal synthétique minimal pour les tests d'interface."""
    idx    = pd.date_range("2020-01-01", periods=n_days, freq="D")
    prices = pd.Series(
        np.random.default_rng(42).lognormal(0, 0.02, n_days).cumprod() * 100,
        index=idx,
    )
    return SignalData(
        alloc_btc        = pd.Series(alloc_value, index=idx),
        prices_pair      = prices,
        prices_base_usd  = prices * 50_000,
        prices_quote_usd = prices * 1_800,
        is_start         = "2020-01-01",
        is_end           = "2022-12-31",
        oos_start        = "2023-01-01",
        oos_end          = "2024-12-31",
    )


# ─── Filtres minimaux de test ─────────────────

class AlwaysPassFilter(Filter):
    """Filtre trivial qui passe toujours — teste le contrat positif."""
    NAME = "always_pass"

    def evaluate(self, signal: SignalData, config: FilterConfig) -> FilterVerdict:
        return FilterVerdict(
            passed      = True,
            filter_name = self.NAME,
            metrics     = {"value": 1.0},
            diagnosis   = "Signal accepté : toutes les allocations sont dans [0,1].",
        )


class AlwaysFailFilter(Filter):
    """Filtre trivial qui échoue toujours — teste le contrat négatif."""
    NAME = "always_fail"

    def evaluate(self, signal: SignalData, config: FilterConfig) -> FilterVerdict:
        return FilterVerdict(
            passed      = False,
            filter_name = self.NAME,
            metrics     = {"value": 0.0},
            diagnosis   = (
                "Signal rejeté : la valeur est toujours nulle. "
                "Pour passer ce filtre, le signal doit produire "
                "une allocation variable, pas constante."
            ),
        )


class TechnicalErrorFilter(Filter):
    """Filtre qui lève FilterError — distingue erreur technique de verdict négatif."""
    NAME = "technical_error"

    def evaluate(self, signal: SignalData, config: FilterConfig) -> FilterVerdict:
        raise FilterError(
            filter_name = self.NAME,
            cause       = "Module 'scipy' non disponible",
            action      = "Installer scipy : pip install scipy",
        )


# ─── Tests du contrat Filter ──────────────────

class TestFilterContract:

    def test_always_pass_returns_verdict(self):
        """Un filtre valide retourne toujours un FilterVerdict, jamais None."""
        f       = AlwaysPassFilter()
        verdict = f.evaluate(make_signal(), FilterConfig(name="test"))
        assert isinstance(verdict, FilterVerdict)

    def test_pass_verdict_is_true(self):
        """Un filtre qui accepte retourne passed=True."""
        verdict = AlwaysPassFilter().evaluate(make_signal(), FilterConfig(name="test"))
        assert verdict.passed is True

    def test_fail_verdict_is_false(self):
        """Un filtre qui rejette retourne passed=False, pas une exception."""
        verdict = AlwaysFailFilter().evaluate(make_signal(), FilterConfig(name="test"))
        assert verdict.passed is False

    def test_verdict_filter_name_matches(self):
        """filter_name dans le verdict correspond au NAME du filtre."""
        f       = AlwaysPassFilter()
        verdict = f.evaluate(make_signal(), FilterConfig(name="test"))
        assert verdict.filter_name == f.NAME

    def test_verdict_has_metrics_dict(self):
        """metrics est toujours un dict, même vide."""
        verdict = AlwaysPassFilter().evaluate(make_signal(), FilterConfig(name="test"))
        assert isinstance(verdict.metrics, dict)

    def test_verdict_diagnosis_is_sentence(self):
        """diagnosis est une phrase d'au moins 20 caractères."""
        verdict = AlwaysPassFilter().evaluate(make_signal(), FilterConfig(name="test"))
        assert len(verdict.diagnosis) >= 20

    def test_fail_diagnosis_explains_what_to_change(self):
        """En cas d'échec, diagnosis contient un mot-clé actionnable."""
        verdict = AlwaysFailFilter().evaluate(make_signal(), FilterConfig(name="test"))
        assert not verdict.passed
        assert any(
            word in verdict.diagnosis.lower()
            for word in ["pour", "afin", "envisager", "augmenter",
                         "réduire", "modifier", "changer"]
        )

    def test_technical_error_raises_filter_error(self):
        """Un problème technique lève FilterError, pas une exception générique."""
        with pytest.raises(FilterError) as exc_info:
            TechnicalErrorFilter().evaluate(make_signal(), FilterConfig(name="test"))
        assert exc_info.value.filter_name == "technical_error"
        assert exc_info.value.cause
        assert exc_info.value.action

    def test_filter_error_not_raised_for_bad_signal(self):
        """Un signal mauvais retourne passed=False, jamais FilterError."""
        bad_signal = make_signal(alloc_value=0.0)
        verdict    = AlwaysFailFilter().evaluate(bad_signal, FilterConfig(name="test"))
        assert isinstance(verdict, FilterVerdict)

    def test_filter_protocol_conformance(self):
        """AlwaysPassFilter est conforme au FilterProtocol."""
        assert isinstance(AlwaysPassFilter(), FilterProtocol)

    def test_verdict_summary_contains_status(self):
        """summary() retourne une ligne lisible avec le statut et le nom."""
        verdict = AlwaysPassFilter().evaluate(make_signal(), FilterConfig(name="test"))
        summary = verdict.summary()
        assert "PASS" in summary
        assert verdict.filter_name in summary


# ─── Tests de validation de FilterVerdict ────

class TestFilterVerdictValidation:

    def test_short_diagnosis_raises(self):
        """FilterVerdict refuse un diagnosis trop court."""
        with pytest.raises(ValueError, match="trop court"):
            FilterVerdict(
                passed      = True,
                filter_name = "test",
                metrics     = {},
                diagnosis   = "OK",
            )

    def test_empty_diagnosis_raises(self):
        """FilterVerdict refuse un diagnosis vide."""
        with pytest.raises(ValueError):
            FilterVerdict(
                passed      = False,
                filter_name = "test",
                metrics     = {},
                diagnosis   = "",
            )

    def test_metrics_must_be_dict(self):
        """FilterVerdict refuse un metrics qui n'est pas un dict."""
        with pytest.raises(TypeError):
            FilterVerdict(
                passed      = True,
                filter_name = "test",
                metrics     = "pas un dict",
                diagnosis   = "Signal accepté : toutes les conditions sont remplies.",
            )


# ─── Tests de FilterConfig ───────────────────

class TestFilterConfig:

    def test_config_get_with_default(self):
        """FilterConfig.get() retourne la valeur ou le défaut si absente."""
        config = FilterConfig(name="test", params={"seuil": 0.7})
        assert config.get("seuil") == 0.7
        assert config.get("absent", 42) == 42

    def test_config_max_params_convention(self):
        """Un FilterConfig avec > 5 paramètres documente la règle Complexity Budget."""
        config = FilterConfig(
            name   = "trop_complexe",
            params = {f"p{i}": i for i in range(6)},
        )
        # Convention : > 5 params → ce filtre doit être scindé.
        assert len(config.params) > 5
