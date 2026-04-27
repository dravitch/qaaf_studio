"""
studio/interfaces.py
Interface unique pour tous les filtres du pipeline QAAF Studio.

Contrat (API Contract First) :
  - Tout filtre du pipeline implémente Filter.
  - Tout filtre retourne un FilterVerdict.
  - Tout filtre lève FilterError en cas d'échec technique,
    jamais en cas d'échec de validation (un signal mauvais
    retourne passed=False, pas une exception).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable
import pandas as pd


# ─────────────────────────────────────────────
# Types de base
# ─────────────────────────────────────────────

@dataclass
class SignalData:
    """
    Données d'entrée standardisées pour tout filtre.

    Invariants :
      - alloc_btc.index == prices_pair.index
      - alloc_btc est dans [0, 1]
      - prices_base_usd est toujours en USD
    """
    alloc_btc:        pd.Series   # Allocation BTC ∈ [0,1], indexée par date
    prices_pair:      pd.Series   # Prix de la paire (ex. PAXG/BTC)
    prices_base_usd:  pd.Series   # Prix de l'actif de base en USD (ex. BTC-USD)
    prices_quote_usd: pd.Series   # Prix de l'actif de cotation en USD (ex. PAXG-USD)
    is_start:         str         # Date début période IS (format YYYY-MM-DD)
    is_end:           str         # Date fin période IS
    oos_start:        str         # Date début période OOS
    oos_end:          str         # Date fin période OOS


@dataclass
class FilterConfig:
    """
    Configuration d'un filtre.
    Maximum 5 paramètres (Complexity Budget).

    Les filtres complexes scindent leur config en
    deux FilterConfig plutôt que d'en dépasser 5.
    """
    name:   str
    params: dict = field(default_factory=dict)

    def get(self, key: str, default=None):
        return self.params.get(key, default)


@dataclass
class FilterVerdict:
    """
    Résultat structuré de tout filtre.

    Contrat (Explainability First) :
      - diagnosis est toujours une phrase complète en français.
      - En cas d'échec (passed=False), diagnosis indique
        ce qu'il faudrait changer pour passer — pas seulement
        ce qui a échoué.
      - metrics contient les valeurs numériques brutes
        permettant de reproduire le verdict.

    Exemples de diagnosis corrects :
      PASS : "Signal robuste sur 4/5 fenêtres (CNSR médiane=1.00) —
              walk-forward confirmé."
      FAIL : "Seulement 3/5 fenêtres passent le seuil CNSR > 0.5.
              Pour passer Q1, le signal doit être profitable en régime
              bear 2021-2022 (fenêtre 2) — envisager un filtre de régime."

    Exemples de diagnosis incorrects (interdits) :
      "FAIL" (trop court)
      "Le test a échoué" (ne dit pas quoi changer)
      "p-value = 0.487" (chiffre sans interprétation)
    """
    passed:      bool
    filter_name: str
    metrics:     dict
    diagnosis:   str

    def __post_init__(self):
        if not self.diagnosis or len(self.diagnosis) < 20:
            raise ValueError(
                f"FilterVerdict.diagnosis trop court pour '{self.filter_name}'. "
                f"Le diagnostic doit être une phrase complète expliquant le résultat "
                f"et, en cas d'échec, ce qu'il faudrait changer."
            )
        if not isinstance(self.metrics, dict):
            raise TypeError("FilterVerdict.metrics doit être un dict.")

    def summary(self) -> str:
        """Résumé une ligne pour les logs et le rapport HTML."""
        status = "✅ PASS" if self.passed else "🔴 FAIL"
        return f"{status} [{self.filter_name}] — {self.diagnosis}"


# ─────────────────────────────────────────────
# Exception technique
# ─────────────────────────────────────────────

class FilterError(Exception):
    """
    Levée uniquement pour les échecs techniques (données manquantes,
    import manquant, division par zéro non gérée).

    Ne jamais lever FilterError pour un signal qui ne passe pas
    un critère statistique — utiliser FilterVerdict(passed=False).

    Contient obligatoirement :
      - filter_name : quel filtre a échoué
      - cause       : description technique de la cause
      - action      : ce que l'opérateur doit faire pour corriger
    """
    def __init__(self, filter_name: str, cause: str, action: str):
        self.filter_name = filter_name
        self.cause       = cause
        self.action      = action
        super().__init__(
            f"[{filter_name}] Erreur technique : {cause}. "
            f"Action requise : {action}"
        )


# ─────────────────────────────────────────────
# Interface abstraite
# ─────────────────────────────────────────────

class Filter(ABC):
    """
    Interface unique pour tous les filtres du pipeline QAAF Studio.

    Contrat (API Contract First) :
      - evaluate() est la seule méthode publique.
      - evaluate() ne lève jamais d'exception pour un signal invalide.
      - evaluate() lève FilterError uniquement pour des problèmes techniques.
      - Chaque implémentation respecte le Complexity Budget :
          * ≤ 50 lignes dans evaluate()
          * ≤ 3 niveaux d'imbrication
          * ≤ 5 paramètres dans FilterConfig.params
      - Le MWE de chaque filtre vérifie deux cas (MWE First / Test Before Trust) :
          * Un signal délibérément bon  → passed=True
          * Un signal délibérément mauvais → passed=False

    Héritage :
      Chaque filtre du pipeline est une classe qui hérite de Filter
      et implémente evaluate(). Les filtres composites (ex. MIF qui
      regroupe T1-T6 + G1-G5 + M1-M4) sont des agrégats de filtres
      atomiques, pas des filtres monolithiques.
    """

    @property
    @abstractmethod
    def NAME(self) -> str:
        """Identifiant unique du filtre, utilisé dans FilterVerdict.filter_name."""
        ...

    @abstractmethod
    def evaluate(
        self,
        signal: SignalData,
        config: FilterConfig,
    ) -> FilterVerdict:
        """
        Évalue le signal et retourne un verdict structuré.

        Args:
            signal : données d'entrée standardisées (prix, allocations, splits).
            config : paramètres du filtre (≤ 5 clés dans config.params).

        Returns:
            FilterVerdict avec passed, filter_name, metrics, diagnosis.

        Raises:
            FilterError : uniquement pour des problèmes techniques
                          (données manquantes, import raté, etc.).
                          Jamais pour un signal qui ne passe pas le critère.

        Guarantees:
            - Ne modifie jamais signal ni config (side-effect free).
            - Retourne toujours un FilterVerdict, même si le signal est
              dégénéré (toutes allocations à 0.5, prix constants, etc.).
            - diagnosis est une phrase complète (≥ 20 caractères).
        """
        ...


# ─────────────────────────────────────────────
# Protocol pour la vérification de conformité
# ─────────────────────────────────────────────

@runtime_checkable
class FilterProtocol(Protocol):
    """
    Version Protocol de Filter pour la vérification statique.
    Utilisé dans les tests pour vérifier qu'un objet
    est bien un filtre valide sans héritage explicite.

    Usage dans les tests :
        assert isinstance(mon_filtre, FilterProtocol)
    """
    NAME: str

    def evaluate(
        self,
        signal: SignalData,
        config: FilterConfig,
    ) -> FilterVerdict: ...
