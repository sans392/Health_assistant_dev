"""Пакет модулей обработки данных (Data Processing).

Содержит вычислительные модули без LLM:
- activity_summary: суммарная статистика тренировок
- training_load: расчёт тренировочной нагрузки
- trend_analyzer: анализ трендов метрик
- recovery_score: passthrough для нативного recovery score
"""

from app.services.data_processing.activity_summary import ActivitySummary, compute_activity_summary
from app.services.data_processing.training_load import TrainingLoad, compute_training_load
from app.services.data_processing.trend_analyzer import TrendResult, analyze_trend
from app.services.data_processing.recovery_score import RecoveryScoreResult, get_recovery_score

__all__ = [
    "ActivitySummary",
    "compute_activity_summary",
    "TrainingLoad",
    "compute_training_load",
    "TrendResult",
    "analyze_trend",
    "RecoveryScoreResult",
    "get_recovery_score",
]
