"""
BOUND COHERENCE METRIC - VERSION PRODUCTION
============================================

Implémentation production-ready de bound_coherence avec:
- Gestion d'erreurs robuste
- Monitoring en temps réel
- Logging structuré
- Circuit breakers
- Health checks

Version: 1.0.0
Date: 2025-10-13
Auteur: Système QAAF
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import json
from collections import deque
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

class MetricStatus(Enum):
    """États possibles de la métrique"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"
    CIRCUIT_OPEN = "circuit_open"


@dataclass
class ProductionConfig:
    """Configuration production"""
    # Paramètres de la métrique
    window: int = 30
    lambda_param: float = 2.0
    
    # Limites et validations
    min_window: int = 10
    max_window: int = 100
    min_lambda: float = 0.5
    max_lambda: float = 5.0
    
    # Gestion des erreurs
    max_consecutive_errors: int = 5
    circuit_breaker_threshold: int = 10
    circuit_breaker_timeout: int = 300  # secondes
    
    # Monitoring
    enable_monitoring: bool = True
    alert_threshold_low: float = 0.30
    alert_threshold_critical: float = 0.15
    history_size: int = 1000
    
    # Performance
    nan_tolerance: float = 0.10  # 10% max de NaN
    min_valid_points: int = 100
    
    def validate(self) -> bool:
        """Valide la configuration"""
        if not self.min_window <= self.window <= self.max_window:
            raise ValueError(f"Window {self.window} hors limites [{self.min_window}, {self.max_window}]")
        if not self.min_lambda <= self.lambda_param <= self.max_lambda:
            raise ValueError(f"Lambda {self.lambda_param} hors limites [{self.min_lambda}, {self.max_lambda}]")
        return True


@dataclass
class MetricHealth:
    """État de santé de la métrique"""
    status: MetricStatus
    last_value: Optional[float]
    nan_rate: float
    error_count: int
    consecutive_errors: int
    last_error: Optional[str]
    last_update: datetime
    uptime_seconds: float
    total_calculations: int
    successful_calculations: int


@dataclass
class MonitoringMetrics:
    """Métriques de monitoring"""
    timestamp: datetime
    value: float
    computation_time_ms: float
    is_alert: bool
    alert_level: Optional[str]
    metadata: Dict = field(default_factory=dict)


# ============================================================================
# LOGGER STRUCTURÉ
# ============================================================================

class StructuredLogger:
    """Logger avec sortie JSON structurée"""
    
    def __init__(self, name: str, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Handler avec format JSON
        handler = logging.StreamHandler()
        handler.setLevel(level)
        
        # Format simple pour démonstration
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        if not self.logger.handlers:
            self.logger.addHandler(handler)
    
    def log_metric_calculation(self, value: float, computation_time: float, 
                              metadata: Dict):
        """Log le calcul d'une métrique"""
        log_data = {
            'event': 'metric_calculation',
            'value': value,
            'computation_time_ms': computation_time,
            **metadata
        }
        self.logger.info(json.dumps(log_data))
    
    def log_error(self, error_type: str, error_msg: str, context: Dict):
        """Log une erreur"""
        log_data = {
            'event': 'error',
            'error_type': error_type,
            'error_message': error_msg,
            **context
        }
        self.logger.error(json.dumps(log_data))
    
    def log_alert(self, alert_level: str, message: str, value: float):
        """Log une alerte"""
        log_data = {
            'event': 'alert',
            'level': alert_level,
            'message': message,
            'value': value
        }
        self.logger.warning(json.dumps(log_data))


# ============================================================================
# CIRCUIT BREAKER
# ============================================================================

class CircuitBreaker:
    """Circuit breaker pour protéger contre les erreurs en cascade"""
    
    def __init__(self, threshold: int, timeout: int):
        self.threshold = threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func, *args, **kwargs):
        """Execute une fonction avec protection circuit breaker"""
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """Reset après succès"""
        self.failure_count = 0
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
    
    def _on_failure(self):
        """Incrémente les échecs"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.threshold:
            self.state = "OPEN"
    
    def _should_attempt_reset(self) -> bool:
        """Vérifie si on peut tenter de fermer le circuit"""
        if self.last_failure_time is None:
            return True
        
        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        return elapsed >= self.timeout


# ============================================================================
# CALCULATEUR PRODUCTION
# ============================================================================

class BoundCoherenceProduction:
    """
    Calculateur production de bound_coherence avec monitoring complet
    """
    
    def __init__(self, config: ProductionConfig):
        config.validate()
        self.config = config
        
        # Logging
        self.logger = StructuredLogger("BoundCoherence", logging.INFO)
        
        # Circuit breaker
        self.circuit_breaker = CircuitBreaker(
            threshold=config.circuit_breaker_threshold,
            timeout=config.circuit_breaker_timeout
        )
        
        # État de santé
        self.health = MetricHealth(
            status=MetricStatus.HEALTHY,
            last_value=None,
            nan_rate=0.0,
            error_count=0,
            consecutive_errors=0,
            last_error=None,
            last_update=datetime.now(),
            uptime_seconds=0.0,
            total_calculations=0,
            successful_calculations=0
        )
        
        # Historique pour monitoring
        self.history = deque(maxlen=config.history_size)
        
        # Timestamp de démarrage
        self.start_time = datetime.now()
    
    # ------------------------------------------------------------------------
    # CALCUL PRINCIPAL
    # ------------------------------------------------------------------------
    
    def calculate(self, prices_btc: np.ndarray, 
                 prices_paxg: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Calcule bound_coherence avec monitoring complet
        
        Args:
            prices_btc: Prix BTC
            prices_paxg: Prix PAXG
            
        Returns:
            (coherence_array, metadata_dict)
        """
        start_time = datetime.now()
        
        try:
            # Validation des entrées
            self._validate_inputs(prices_btc, prices_paxg)
            
            # Calcul via circuit breaker
            coherence = self.circuit_breaker.call(
                self._compute_coherence,
                prices_btc,
                prices_paxg
            )
            
            # Post-traitement
            coherence, metadata = self._post_process(coherence, prices_btc, prices_paxg)
            
            # Calcul du temps
            computation_time = (datetime.now() - start_time).total_seconds() * 1000
            metadata['computation_time_ms'] = computation_time
            
            # Update health
            self._update_health_success(coherence, computation_time)
            
            # Monitoring
            if self.config.enable_monitoring:
                self._monitor(coherence, computation_time, metadata)
            
            # Logging
            self.logger.log_metric_calculation(
                value=float(np.nanmean(coherence)),
                computation_time=computation_time,
                metadata=metadata
            )
            
            return coherence, metadata
            
        except Exception as e:
            self._handle_error(e, prices_btc, prices_paxg)
            raise
    
    def _compute_coherence(self, prices_btc: np.ndarray, 
                          prices_paxg: np.ndarray) -> np.ndarray:
        """Calcul core de bound_coherence"""
        
        # Calcul des rendements
        ret_btc = np.diff(prices_btc) / prices_btc[:-1]
        ret_paxg = np.diff(prices_paxg) / prices_paxg[:-1]
        
        # Initialisation
        coherence = np.full(len(ret_btc), np.nan)
        
        # Calcul glissant
        for t in range(self.config.window, len(ret_btc)):
            # Protection division par zéro
            if abs(ret_paxg[t]) < 1e-6:
                continue
            
            # Ratio de rendements
            ret_ratio = ret_btc[t] / ret_paxg[t]
            
            # Bornes historiques
            ret_btc_window = ret_btc[t-self.config.window:t]
            ret_paxg_window = ret_paxg[t-self.config.window:t]
            all_rets = np.concatenate([ret_btc_window, ret_paxg_window])
            
            min_ret = np.min(all_rets)
            max_ret = np.max(all_rets)
            
            # Score de cohérence
            if min_ret <= ret_ratio <= max_ret:
                coherence[t] = 1.0
            else:
                distance = min(
                    abs(ret_ratio - min_ret),
                    abs(ret_ratio - max_ret)
                )
                coherence[t] = np.exp(-self.config.lambda_param * distance**2)
        
        return coherence
    
    # ------------------------------------------------------------------------
    # VALIDATION ET POST-TRAITEMENT
    # ------------------------------------------------------------------------
    
    def _validate_inputs(self, prices_btc: np.ndarray, prices_paxg: np.ndarray):
        """Valide les entrées"""
        
        # Vérification des types
        if not isinstance(prices_btc, np.ndarray) or not isinstance(prices_paxg, np.ndarray):
            raise TypeError("Les prix doivent être des numpy arrays")
        
        # Vérification de la taille
        if len(prices_btc) < self.config.min_valid_points:
            raise ValueError(f"Pas assez de points BTC: {len(prices_btc)} < {self.config.min_valid_points}")
        
        if len(prices_paxg) < self.config.min_valid_points:
            raise ValueError(f"Pas assez de points PAXG: {len(prices_paxg)} < {self.config.min_valid_points}")
        
        # Vérification des NaN
        btc_nan_rate = np.sum(np.isnan(prices_btc)) / len(prices_btc)
        paxg_nan_rate = np.sum(np.isnan(prices_paxg)) / len(prices_paxg)
        
        if btc_nan_rate > self.config.nan_tolerance:
            raise ValueError(f"Trop de NaN dans BTC: {btc_nan_rate:.2%}")
        
        if paxg_nan_rate > self.config.nan_tolerance:
            raise ValueError(f"Trop de NaN dans PAXG: {paxg_nan_rate:.2%}")
        
        # Vérification des valeurs
        if np.any(prices_btc <= 0):
            raise ValueError("Prix BTC négatifs ou nuls détectés")
        
        if np.any(prices_paxg <= 0):
            raise ValueError("Prix PAXG négatifs ou nuls détectés")
    
    def _post_process(self, coherence: np.ndarray, 
                     prices_btc: np.ndarray,
                     prices_paxg: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Post-traitement et génération de métadonnées"""
        
        valid_values = coherence[~np.isnan(coherence)]
        
        metadata = {
            'n_total': len(coherence),
            'n_valid': len(valid_values),
            'n_nan': np.sum(np.isnan(coherence)),
            'nan_rate': float(np.sum(np.isnan(coherence)) / len(coherence)),
            'mean': float(np.mean(valid_values)) if len(valid_values) > 0 else None,
            'median': float(np.median(valid_values)) if len(valid_values) > 0 else None,
            'std': float(np.std(valid_values)) if len(valid_values) > 0 else None,
            'min': float(np.min(valid_values)) if len(valid_values) > 0 else None,
            'max': float(np.max(valid_values)) if len(valid_values) > 0 else None,
            'pct_high': float(np.sum(valid_values >= 0.9) / len(valid_values)) if len(valid_values) > 0 else 0,
            'pct_low': float(np.sum(valid_values < 0.6) / len(valid_values)) if len(valid_values) > 0 else 0,
        }
        
        # Vérification du taux de NaN
        if metadata['nan_rate'] > self.config.nan_tolerance:
            self.logger.log_alert(
                alert_level="WARNING",
                message=f"Taux de NaN élevé: {metadata['nan_rate']:.2%}",
                value=metadata['nan_rate']
            )
        
        return coherence, metadata
    
    # ------------------------------------------------------------------------
    # MONITORING ET ALERTING
    # ------------------------------------------------------------------------
    
    def _monitor(self, coherence: np.ndarray, computation_time: float, 
                metadata: Dict):
        """Surveillance en temps réel"""
        
        valid_values = coherence[~np.isnan(coherence)]
        if len(valid_values) == 0:
            return
        
        current_value = float(np.mean(valid_values))
        
        # Détection d'alertes
        is_alert = False
        alert_level = None
        
        if current_value < self.config.alert_threshold_critical:
            is_alert = True
            alert_level = "CRITICAL"
            self.logger.log_alert(
                alert_level="CRITICAL",
                message=f"Cohérence critique: {current_value:.3f}",
                value=current_value
            )
        elif current_value < self.config.alert_threshold_low:
            is_alert = True
            alert_level = "WARNING"
            self.logger.log_alert(
                alert_level="WARNING",
                message=f"Cohérence basse: {current_value:.3f}",
                value=current_value
            )
        
        # Enregistrement dans l'historique
        monitoring_record = MonitoringMetrics(
            timestamp=datetime.now(),
            value=current_value,
            computation_time_ms=computation_time,
            is_alert=is_alert,
            alert_level=alert_level,
            metadata=metadata
        )
        self.history.append(monitoring_record)
    
    def _update_health_success(self, coherence: np.ndarray, computation_time: float):
        """Met à jour l'état de santé après un succès"""
        
        valid_values = coherence[~np.isnan(coherence)]
        nan_rate = np.sum(np.isnan(coherence)) / len(coherence)
        
        self.health.status = MetricStatus.HEALTHY
        self.health.last_value = float(np.mean(valid_values)) if len(valid_values) > 0 else None
        self.health.nan_rate = float(nan_rate)
        self.health.consecutive_errors = 0
        self.health.last_update = datetime.now()
        self.health.uptime_seconds = (datetime.now() - self.start_time).total_seconds()
        self.health.total_calculations += 1
        self.health.successful_calculations += 1
        
        # Dégradé si trop de NaN
        if nan_rate > self.config.nan_tolerance * 0.8:
            self.health.status = MetricStatus.DEGRADED
    
    def _handle_error(self, error: Exception, prices_btc: np.ndarray,
                     prices_paxg: np.ndarray):
        """Gère les erreurs"""
        
        self.health.error_count += 1
        self.health.consecutive_errors += 1
        self.health.last_error = str(error)
        self.health.total_calculations += 1
        
        # Mise à jour du statut
        if self.health.consecutive_errors >= self.config.max_consecutive_errors:
            self.health.status = MetricStatus.FAILING
        else:
            self.health.status = MetricStatus.DEGRADED
        
        # Logging
        self.logger.log_error(
            error_type=type(error).__name__,
            error_msg=str(error),
            context={
                'prices_btc_len': len(prices_btc),
                'prices_paxg_len': len(prices_paxg),
                'consecutive_errors': self.health.consecutive_errors
            }
        )
    
    # ------------------------------------------------------------------------
    # HEALTH CHECK
    # ------------------------------------------------------------------------
    
    def get_health(self) -> Dict:
        """Retourne l'état de santé complet"""
        return {
            'status': self.health.status.value,
            'last_value': self.health.last_value,
            'nan_rate': self.health.nan_rate,
            'error_count': self.health.error_count,
            'consecutive_errors': self.health.consecutive_errors,
            'last_error': self.health.last_error,
            'last_update': self.health.last_update.isoformat(),
            'uptime_seconds': self.health.uptime_seconds,
            'total_calculations': self.health.total_calculations,
            'successful_calculations': self.health.successful_calculations,
            'success_rate': (self.health.successful_calculations / self.health.total_calculations 
                           if self.health.total_calculations > 0 else 0),
            'circuit_breaker_state': self.circuit_breaker.state,
        }
    
    def get_monitoring_stats(self) -> Dict:
        """Retourne les statistiques de monitoring"""
        if len(self.history) == 0:
            return {'message': 'No data available'}
        
        values = [m.value for m in self.history]
        computation_times = [m.computation_time_ms for m in self.history]
        alerts = [m for m in self.history if m.is_alert]
        
        return {
            'n_records': len(self.history),
            'mean_value': float(np.mean(values)),
            'std_value': float(np.std(values)),
            'mean_computation_time_ms': float(np.mean(computation_times)),
            'max_computation_time_ms': float(np.max(computation_times)),
            'n_alerts': len(alerts),
            'alert_rate': len(alerts) / len(self.history),
            'recent_alerts': [
                {
                    'timestamp': a.timestamp.isoformat(),
                    'level': a.alert_level,
                    'value': a.value
                }
                for a in list(alerts)[-5:]  # 5 dernières alertes
            ]
        }
    
    # ------------------------------------------------------------------------
    # UTILITAIRES
    # ------------------------------------------------------------------------
    
    def reset_circuit_breaker(self):
        """Reset manuel du circuit breaker"""
        self.circuit_breaker.failure_count = 0
        self.circuit_breaker.state = "CLOSED"
        self.logger.logger.info("Circuit breaker manually reset")
    
    def export_history(self, filepath: str):
        """Exporte l'historique de monitoring"""
        data = []
        for record in self.history:
            data.append({
                'timestamp': record.timestamp.isoformat(),
                'value': record.value,
                'computation_time_ms': record.computation_time_ms,
                'is_alert': record.is_alert,
                'alert_level': record.alert_level,
                **record.metadata
            })
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.logger.info(f"History exported to {filepath}")


# ============================================================================
# WRAPPER SIMPLIFIÉ
# ============================================================================

class BoundCoherenceSimple:
    """Interface simplifiée pour usage rapide"""
    
    def __init__(self, window: int = 30, lambda_param: float = 2.0):
        config = ProductionConfig(window=window, lambda_param=lambda_param)
        self.calculator = BoundCoherenceProduction(config)
    
    def calculate(self, prices_btc: np.ndarray, 
                 prices_paxg: np.ndarray) -> np.ndarray:
        """Calcul simplifié retournant seulement le array"""
        coherence, _ = self.calculator.calculate(prices_btc, prices_paxg)
        return coherence
    
    def health_check(self) -> bool:
        """Health check simple"""
        health = self.calculator.get_health()
        return health['status'] in ['healthy', 'degraded']


# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def main():
    """Démonstration de l'usage en production"""
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║           BOUND COHERENCE - VERSION PRODUCTION                               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Configuration
    config = ProductionConfig(
        window=30,
        lambda_param=2.0,
        enable_monitoring=True,
        alert_threshold_low=0.50,
        alert_threshold_critical=0.30
    )
    
    # Création du calculateur
    calculator = BoundCoherenceProduction(config)
    
    # Simulation de données
    np.random.seed(42)
    n_points = 1000
    
    # Données normales
    print("\n1. Test avec données normales...")
    btc_returns = np.random.normal(0.001, 0.02, n_points)
    prices_btc = 40000 * np.exp(np.cumsum(btc_returns))
    paxg_returns = np.random.normal(0.0002, 0.005, n_points)
    prices_paxg = 2000 * np.exp(np.cumsum(paxg_returns))
    
    coherence, metadata = calculator.calculate(prices_btc, prices_paxg)
    print(f"   ✓ Calcul réussi: mean={metadata['mean']:.3f}, nan_rate={metadata['nan_rate']:.2%}")
    
    # Test avec anomalie
    print("\n2. Test avec anomalie (crash)...")
    btc_returns[500:510] = -0.05  # Crash brutal
    prices_btc = 40000 * np.exp(np.cumsum(btc_returns))
    
    coherence, metadata = calculator.calculate(prices_btc, prices_paxg)
    print(f"   ✓ Calcul réussi: mean={metadata['mean']:.3f}")
    
    # Health check
    print("\n3. Health check...")
    health = calculator.get_health()
    print(f"   Status: {health['status']}")
    print(f"   Success rate: {health['success_rate']:.2%}")
    print(f"   Total calculations: {health['total_calculations']}")
    
    # Monitoring stats
    print("\n4. Monitoring stats...")
    stats = calculator.get_monitoring_stats()
    print(f"   Mean value: {stats['mean_value']:.3f}")
    print(f"   Mean computation time: {stats['mean_computation_time_ms']:.2f}ms")
    print(f"   Alerts: {stats['n_alerts']}")
    
    # Export
    calculator.export_history("bound_coherence_production_history.json")
    print("\n✓ Historique exporté")
    
    print(f"\n{'='*80}")
    print("✅ DÉMONSTRATION TERMINÉE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()