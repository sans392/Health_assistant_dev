"""Пакет модулей обработки данных (Data Processing).

Phase 2 расширения:
- heart_rate_zones: расчёт зон пульса по Карвонену
- strain_score: TRIMP-подобный Strain Score
- recovery_score (v2): расчётный + Whoop passthrough
- overtraining_detection: маркеры перетренированности
- training_load (расширен): monotony, strain_weekly, load_warning
- periodization: TODO v3
"""

from app.services.data_processing.activity_summary import ActivitySummary, compute_activity_summary
from app.services.data_processing.training_load import TrainingLoad, compute_training_load
from app.services.data_processing.trend_analyzer import TrendResult, analyze_trend
from app.services.data_processing.recovery_score import (
    RecoveryScoreResult,
    get_recovery_score,
    compute_recovery_score,
)
from app.services.data_processing.heart_rate_zones import HRZones, compute_hr_zones
from app.services.data_processing.strain_score import StrainScoreResult, compute_strain_score
from app.services.data_processing.overtraining_detection import (
    OvertrainingResult,
    detect_overtraining,
)
from app.services.data_processing.summary_builder import (
    ActivityPromptSummary,
    MetricSummary,
    annotate_anomalies,
    build_activity_summary,
    build_metric_summary,
    format_activity_summary,
    format_metric_summary,
    format_structured_block,
)

__all__ = [
    "ActivitySummary",
    "compute_activity_summary",
    "TrainingLoad",
    "compute_training_load",
    "TrendResult",
    "analyze_trend",
    "RecoveryScoreResult",
    "get_recovery_score",
    "compute_recovery_score",
    "HRZones",
    "compute_hr_zones",
    "StrainScoreResult",
    "compute_strain_score",
    "OvertrainingResult",
    "detect_overtraining",
    "ActivityPromptSummary",
    "MetricSummary",
    "annotate_anomalies",
    "build_activity_summary",
    "build_metric_summary",
    "format_activity_summary",
    "format_metric_summary",
    "format_structured_block",
]
