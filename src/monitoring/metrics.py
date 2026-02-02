from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import wraps
import asyncio
import structlog
import time

from prometheus_client import (
    Counter, Histogram, Gauge, Summary, 
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
)
from prometheus_client.multiprocess import MultiProcessCollector

from ..core.drift_detector import DriftResult, DriftSeverity
from ..core.embedding_store import EmbeddingMetadata
from ..exceptions import MetricsError, AlertingError


logger = structlog.get_logger(__name__)


@dataclass
class AlertConfig:
    """Configuration for drift alerting."""
    drift_threshold: float = 0.15
    degradation_threshold: float = 0.20
    evaluation_window_minutes: int = 15
    min_samples_for_alert: int = 100
    cooldown_minutes: int = 30


class DriftMetrics:
    """Prometheus metrics collector for embedding drift monitoring."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self._registry = registry or CollectorRegistry()
        self._alert_config = AlertConfig()
        self._alert_cooldowns: Dict[str, datetime] = {}
        
        # Initialize metrics
        self._init_metrics()
        
    def _init_metrics(self) -> None:
        """Initialize all Prometheus metrics."""
        
        # Drift detection metrics
        self.drift_detected_total = Counter(
            'embedding_drift_detected_total',
            'Total number of drift detections',
            ['model_id', 'severity', 'drift_type'],
            registry=self._registry
        )
        
        self.drift_score = Histogram(
            'embedding_drift_score',
            'Distribution of drift scores',
            ['model_id', 'drift_type'],
            buckets=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 1.0],
            registry=self._registry
        )
        
        self.model_performance_score = Gauge(
            'embedding_model_performance_score',
            'Current model performance score',
            ['model_id', 'metric_type'],
            registry=self._registry
        )
        
        # Embedding storage metrics
        self.embeddings_stored_total = Counter(
            'embeddings_stored_total',
            'Total number of embeddings stored',
            ['model_id', 'source'],
            registry=self._registry
        )
        
        self.embedding_storage_latency = Histogram(
            'embedding_storage_latency_seconds',
            'Time spent storing embeddings',
            ['model_id', 'operation'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
            registry=self._registry
        )
        
        # System health metrics
        self.active_monitoring_sessions = Gauge(
            'active_monitoring_sessions_total',
            'Number of active monitoring sessions',
            registry=self._registry
        )
        
        self.monitoring_errors_total = Counter(
            'monitoring_errors_total',
            'Total monitoring errors',
            ['error_type', 'component'],
            registry=self._registry
        )
        
        # Alert metrics
        self.alerts_fired_total = Counter(
            'drift_alerts_fired_total',
            'Total alerts fired',
            ['alert_type', 'model_id', 'severity'],
            registry=self._registry
        )
        
        self.alert_processing_duration = Summary(
            'alert_processing_duration_seconds',
            'Time spent processing alerts',
            ['alert_type'],
            registry=self._registry
        )

    def record_drift_detection(
        self, 
        model_id: str, 
        drift_result: DriftResult
    ) -> None:
        """Record drift detection event."""
        try:
            # Record drift detection
            self.drift_detected_total.labels(
                model_id=model_id,
                severity=drift_result.severity.value,
                drift_type=drift_result.drift_type.value
            ).inc()
            
            # Record drift score
            self.drift_score.labels(
                model_id=model_id,
                drift_type=drift_result.drift_type.value
            ).observe(drift_result.drift_score)
            
            # Update performance score if available
            if drift_result.performance_impact:
                self.model_performance_score.labels(
                    model_id=model_id,
                    metric_type='drift_impact'
                ).set(drift_result.performance_impact)
            
            logger.info(
                "drift_detection_recorded",
                model_id=model_id,
                drift_score=drift_result.drift_score,
                severity=drift_result.severity.value
            )
            
        except Exception as e:
            self.monitoring_errors_total.labels(
                error_type="metrics_recording",
                component="drift_detection"
            ).inc()
            logger.error("failed_to_record_drift_metrics", error=str(e))
            raise MetricsError(f"Failed to record drift metrics: {e}") from e

    def record_embedding_storage(
        self,
        model_id: str,
        operation: str,
        duration: float,
        count: int = 1,
        source: str = "unknown"
    ) -> None:
        """Record embedding storage operation."""
        try:
            self.embeddings_stored_total.labels(
                model_id=model_id,
                source=source
            ).inc(count)
            
            self.embedding_storage_latency.labels(
                model_id=model_id,
                operation=operation
            ).observe(duration)
            
        except Exception as e:
            self.monitoring_errors_total.labels(
                error_type="metrics_recording",
                component="embedding_storage"
            ).inc()
            logger.error("failed_to_record_storage_metrics", error=str(e))

    def record_model_performance(
        self,
        model_id: str,
        metric_type: str,
        score: float
    ) -> None:
        """Record model performance metrics."""
        try:
            self.model_performance_score.labels(
                model_id=model_id,
                metric_type=metric_type
            ).set(score)
            
        except Exception as e:
            self.monitoring_errors_total.labels(
                error_type="metrics_recording",
                component="performance"
            ).inc()
            logger.error("failed_to_record_performance_metrics", error=str(e))

    def increment_active_sessions(self) -> None:
        """Increment active monitoring sessions."""
        self.active_monitoring_sessions.inc()

    def decrement_active_sessions(self) -> None:
        """Decrement active monitoring sessions."""
        self.active_monitoring_sessions.dec()

    def get_metrics_text(self) -> str:
        """Get metrics in Prometheus text format."""
        try:
            return generate_latest(self._registry)
        except Exception as e:
            logger.error("failed_to_generate_metrics", error=str(e))
            raise MetricsError(f"Failed to generate metrics: {e}") from e


def timing_metric(metric_histogram):
    """Decorator to automatically time function execution."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                metric_histogram.observe(duration)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                metric_histogram.observe(duration)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


class AlertManager:
    """Manages drift alerting logic."""
    
    def __init__(self, metrics: DriftMetrics, config: Optional[AlertConfig] = None):
        self._metrics = metrics
        self._config = config or AlertConfig()
        self._alert_history: Dict[str, List[datetime]] = {}

    async def evaluate_drift_alert(
        self,
        model_id: str,
        drift_result: DriftResult
    ) -> bool:
        """Evaluate if a drift alert should be fired."""
        try:
            alert_key = f"{model_id}:{drift_result.drift_type.value}"
            
            # Check cooldown
            if self._is_in_cooldown(alert_key):
                logger.debug(
                    "alert_in_cooldown",
                    model_id=model_id,
                    drift_type=drift_result.drift_type.value
                )
                return False
            
            # Check thresholds
            should_alert = (
                drift_result.drift_score >= self._config.drift_threshold or
                drift_result.severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]
            )
            
            if should_alert:
                await self._fire_alert(model_id, drift_result, alert_key)
                return True
                
            return False
            
        except Exception as e:
            self._metrics.monitoring_errors_total.labels(
                error_type="alert_evaluation",
                component="alert_manager"
            ).inc()
            logger.error("failed_to_evaluate_alert", error=str(e))
            raise AlertingError(f"Failed to evaluate drift alert: {e}") from e

    async def evaluate_performance_alert(
        self,
        model_id: str,
        performance_score: float,
        metric_type: str
    ) -> bool:
        """Evaluate if a performance degradation alert should be fired."""
        try:
            alert_key = f"{model_id}:performance:{metric_type}"
            
            if self._is_in_cooldown(alert_key):
                return False
            
            should_alert = performance_score <= self._config.degradation_threshold
            
            if should_alert:
                await self._fire_performance_alert(
                    model_id, performance_score, metric_type, alert_key
                )
                return True
                
            return False
            
        except Exception as e:
            self._metrics.monitoring_errors_total.labels(
                error_type="alert_evaluation",
                component="alert_manager"
            ).inc()
            logger.error("failed_to_evaluate_performance_alert", error=str(e))
            raise AlertingError(f"Failed to evaluate performance alert: {e}") from e

    def _is_in_cooldown(self, alert_key: str) -> bool:
        """Check if alert is in cooldown period."""
        if alert_key not in self._metrics._alert_cooldowns:
            return False
            
        last_alert = self._metrics._alert_cooldowns[alert_key]
        cooldown_end = last_alert + timedelta(minutes=self._config.cooldown_minutes)
        
        return datetime.utcnow() < cooldown_end

    @timing_metric
    async def _fire_alert(
        self,
        model_id: str,
        drift_result: DriftResult,
        alert_key: str
    ) -> None:
        """Fire a drift alert."""
        alert_type = f"drift_{drift_result.drift_type.value}"
        
        self._metrics.alerts_fired_total.labels(
            alert_type=alert_type,
            model_id=model_id,
            severity=drift_result.severity.value
        ).inc()
        
        # Record cooldown
        self._metrics._alert_cooldowns[alert_key] = datetime.utcnow()
        
        # Track alert history
        if alert_key not in self._alert_history:
            self._alert_history[alert_key] = []
        self._alert_history[alert_key].append(datetime.utcnow())
        
        logger.warning(
            "drift_alert_fired",
            model_id=model_id,
            drift_type=drift_result.drift_type.value,
            drift_score=drift_result.drift_score,
            severity=drift_result.severity.value,
            alert_key=alert_key
        )

    @timing_metric
    async def _fire_performance_alert(
        self,
        model_id: str,
        performance_score: float,
        metric_type: str,
        alert_key: str
    ) -> None:
        """Fire a performance degradation alert."""
        alert_type = f"performance_{metric_type}"
        
        self._metrics.alerts_fired_total.labels(
            alert_type=alert_type,
            model_id=model_id,
            severity="high"
        ).inc()
        
        # Record cooldown
        self._metrics._alert_cooldowns[alert_key] = datetime.utcnow()
        
        logger.warning(
            "performance_alert_fired",
            model_id=model_id,
            metric_type=metric_type,
            performance_score=performance_score,
            threshold=self._config.degradation_threshold,
            alert_key=alert_key
        )

    def get_alert_history(self, model_id: str) -> Dict[str, List[datetime]]:
        """Get alert history for a model."""
        return {
            key: alerts 
            for key, alerts in self._alert_history.items() 
            if key.startswith(f"{model_id}:")
        }


# Global metrics instance
_metrics_registry = CollectorRegistry()
drift_metrics = DriftMetrics(_metrics_registry)
alert_manager = AlertManager(drift_metrics)


def get_metrics_registry() -> CollectorRegistry:
    """Get the global metrics registry."""
    return _metrics_registry


def get_drift_metrics() -> DriftMetrics:
    """Get the global drift metrics instance."""
    return drift_metrics


def get_alert_manager() -> AlertManager:
    """Get the global alert manager instance."""
    return alert_manager