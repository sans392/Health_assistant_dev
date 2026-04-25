"""Тесты summary_builder (Issue #56).

Проверяем:
- annotate_anomalies: граничные случаи (строго 2σ, нулевое std, normal)
- build_metric_summary: пустой список, одна точка, все одинаковые, с baseline
- build_activity_summary: edge-cases + выбор самой интенсивной
- format_structured_block: распознавание daily_facts/activities и fallback
"""

from __future__ import annotations

import pytest

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


# ---------------------------------------------------------------------------
# annotate_anomalies
# ---------------------------------------------------------------------------


class TestAnnotateAnomalies:
    def test_normal_value_no_flag(self) -> None:
        # В пределах ±2σ — не аномалия
        assert annotate_anomalies(value=50.0, baseline_mean=50.0, baseline_std=5.0) == []

    def test_below_baseline_anomaly(self) -> None:
        flags = annotate_anomalies(value=30.0, baseline_mean=50.0, baseline_std=5.0)
        assert "ANOMALY" in flags
        assert "below_baseline" in flags

    def test_above_baseline_anomaly(self) -> None:
        flags = annotate_anomalies(value=80.0, baseline_mean=50.0, baseline_std=5.0)
        assert "ANOMALY" in flags
        assert "above_baseline" in flags

    def test_exactly_two_sigma_is_not_flagged(self) -> None:
        # Граничный случай: строгое неравенство, не включая 2σ.
        assert annotate_anomalies(value=60.0, baseline_mean=50.0, baseline_std=5.0) == []

    def test_zero_std_value_equal_mean_no_flag(self) -> None:
        assert annotate_anomalies(value=50.0, baseline_mean=50.0, baseline_std=0.0) == []

    def test_zero_std_value_differs_flags(self) -> None:
        flags = annotate_anomalies(value=51.0, baseline_mean=50.0, baseline_std=0.0)
        assert "ANOMALY" in flags
        assert "above_baseline" in flags


# ---------------------------------------------------------------------------
# build_metric_summary
# ---------------------------------------------------------------------------


class TestBuildMetricSummary:
    def test_empty_list(self) -> None:
        summary = build_metric_summary([], "hrv_rmssd_milli")
        assert summary.count == 0
        assert summary.total == 0
        assert summary.mean == 0
        assert summary.latest_value is None
        assert summary.baseline_mean is None
        assert summary.anomaly_flags == []

    def test_single_value_no_baseline(self) -> None:
        facts = [{"iso_date": "2026-04-23", "hrv_rmssd_milli": 50.0}]
        summary = build_metric_summary(facts, "hrv_rmssd_milli")
        assert summary.count == 1
        assert summary.mean == 50.0
        assert summary.latest_value == 50.0
        # С одной точкой baseline не считается
        assert summary.baseline_mean is None
        assert summary.anomaly_flags == []

    def test_all_values_identical(self) -> None:
        facts = [
            {"iso_date": f"2026-04-{20+i:02d}", "hrv_rmssd_milli": 50.0}
            for i in range(5)
        ]
        summary = build_metric_summary(facts, "hrv_rmssd_milli")
        assert summary.count == 5
        assert summary.mean == 50.0
        assert summary.min_value == summary.max_value == 50.0
        # baseline_std == 0, latest равен baseline → нет аномалии
        assert summary.baseline_std == 0
        assert summary.anomaly_flags == []
        assert summary.delta_pct == 0.0

    def test_latest_below_baseline_gets_anomaly(self) -> None:
        # Baseline: 50, 52, 48, 51, 49, 50 — stable ~50. Последний: 30 (ниже 2σ).
        facts = [
            {"iso_date": "2026-04-17", "hrv_rmssd_milli": 50},
            {"iso_date": "2026-04-18", "hrv_rmssd_milli": 52},
            {"iso_date": "2026-04-19", "hrv_rmssd_milli": 48},
            {"iso_date": "2026-04-20", "hrv_rmssd_milli": 51},
            {"iso_date": "2026-04-21", "hrv_rmssd_milli": 49},
            {"iso_date": "2026-04-22", "hrv_rmssd_milli": 50},
            {"iso_date": "2026-04-23", "hrv_rmssd_milli": 30},
        ]
        summary = build_metric_summary(facts, "hrv_rmssd_milli")
        assert summary.latest_value == 30
        assert summary.baseline_mean is not None
        assert 49 <= summary.baseline_mean <= 51
        assert summary.delta_pct is not None
        # Средний период ниже baseline → delta отрицательный
        assert summary.delta_pct < 0
        assert "ANOMALY" in summary.anomaly_flags
        assert "below_baseline" in summary.anomaly_flags

    def test_uses_explicit_baseline_facts(self) -> None:
        """Если переданы baseline_facts — они используются для baseline."""
        facts = [{"iso_date": "2026-04-23", "hrv_rmssd_milli": 40}]
        baseline = [
            {"iso_date": f"2026-04-{i:02d}", "hrv_rmssd_milli": 60}
            for i in range(1, 22)
        ]
        summary = build_metric_summary(facts, "hrv_rmssd_milli", baseline_facts=baseline)
        assert summary.latest_value == 40
        assert summary.baseline_mean == 60
        assert summary.delta_pct is not None
        # Δ = (40 - 60)/60 *100 ≈ -33%
        assert summary.delta_pct < -30
        # 40 vs baseline 60, std=0 → аномалия
        assert "ANOMALY" in summary.anomaly_flags

    def test_ignores_none_values(self) -> None:
        facts = [
            {"iso_date": "2026-04-21", "hrv_rmssd_milli": None},
            {"iso_date": "2026-04-22", "hrv_rmssd_milli": 50},
            {"iso_date": "2026-04-23", "hrv_rmssd_milli": 52},
        ]
        summary = build_metric_summary(facts, "hrv_rmssd_milli")
        assert summary.count == 2

    def test_has_baseline_property(self) -> None:
        facts = [
            {"iso_date": f"2026-04-{20+i:02d}", "hrv_rmssd_milli": 50.0 + i}
            for i in range(5)
        ]
        summary = build_metric_summary(facts, "hrv_rmssd_milli")
        assert summary.has_baseline is True

    def test_format_metric_summary_contains_delta_pct(self) -> None:
        facts = [
            {"iso_date": f"2026-04-{10+i:02d}", "steps": 7000 + i * 100}
            for i in range(7)
        ]
        summary = build_metric_summary(facts, "steps")
        text = format_metric_summary(summary)
        assert "Шаги" in text
        assert "% от базы" in text


# ---------------------------------------------------------------------------
# build_activity_summary
# ---------------------------------------------------------------------------


class TestBuildActivitySummary:
    def test_empty_list(self) -> None:
        summary = build_activity_summary([])
        assert isinstance(summary, ActivityPromptSummary)
        assert summary.total_activities == 0
        assert summary.total_duration_seconds == 0
        assert summary.most_intense is None

    def test_single_activity(self) -> None:
        act = {
            "title": "Бег",
            "sport_type": "running",
            "duration_seconds": 1800,
            "calories": 300,
            "distance_meters": 5000,
            "start_time": "2026-04-23T09:00:00",
        }
        summary = build_activity_summary([act])
        assert summary.total_activities == 1
        assert summary.total_duration_seconds == 1800
        assert summary.total_calories == 300
        assert summary.most_intense is not None
        assert summary.most_intense["title"] == "Бег"

    def test_most_intense_by_calories(self) -> None:
        activities = [
            {"sport_type": "running", "duration_seconds": 1800, "calories": 200,
             "distance_meters": 4000, "start_time": "2026-04-22T09:00:00", "title": "Короткая"},
            {"sport_type": "cycling", "duration_seconds": 3600, "calories": 600,
             "distance_meters": 20000, "start_time": "2026-04-23T10:00:00", "title": "Длинная"},
            {"sport_type": "running", "duration_seconds": 1200, "calories": 150,
             "distance_meters": 3000, "start_time": "2026-04-21T09:00:00", "title": "Разминка"},
        ]
        summary = build_activity_summary(activities)
        assert summary.total_activities == 3
        assert summary.most_intense["title"] == "Длинная"
        assert summary.by_sport_counts == {"running": 2, "cycling": 1}

    def test_missing_fields_handled(self) -> None:
        activities = [
            {"sport_type": "running"},
            {"sport_type": "cycling", "duration_seconds": None, "calories": None},
        ]
        # Не падает, not crashes
        summary = build_activity_summary(activities)
        assert summary.total_activities == 2

    def test_format_activity_summary_shows_most_intense(self) -> None:
        activities = [
            {"sport_type": "running", "duration_seconds": 1800, "calories": 200,
             "distance_meters": 4000, "start_time": "2026-04-22T09:00:00",
             "avg_heart_rate": 140, "title": "Утренняя пробежка"},
        ]
        summary = build_activity_summary(activities)
        text = format_activity_summary(summary)
        assert "Тренировок: 1" in text
        assert "Утренняя пробежка" in text


# ---------------------------------------------------------------------------
# format_structured_block
# ---------------------------------------------------------------------------


class TestFormatStructuredBlock:
    def test_empty_returns_empty(self) -> None:
        assert format_structured_block(None) == ""
        assert format_structured_block({}) == ""

    def test_daily_facts_list_rendered(self) -> None:
        """daily_facts-шейп рендерится как метрики, а не как JSON."""
        data = {
            "get_daily_facts": [
                {"iso_date": f"2026-04-{20+i:02d}", "hrv_rmssd_milli": 50 + i,
                 "steps": 8000 + i * 100}
                for i in range(5)
            ]
        }
        text = format_structured_block(data)
        # Метрики распознаны и отрендерены
        assert "HRV" in text
        assert "Шаги" in text
        # Не должно быть голого JSON-дампа
        assert '"iso_date"' not in text

    def test_activities_list_rendered(self) -> None:
        data = {
            "get_activities": [
                {"sport_type": "running", "duration_seconds": 1800, "calories": 300,
                 "distance_meters": 5000, "start_time": "2026-04-23T09:00:00",
                 "title": "Пробежка"},
            ]
        }
        text = format_structured_block(data)
        assert "Тренировок: 1" in text
        assert "Пробежка" in text

    def test_unknown_keys_fallback_to_truncated_json(self) -> None:
        """Неизвестные структуры — truncated JSON (первые 500 символов)."""
        big_value = {"x": "y" * 1000}
        data = {"unknown_key": big_value}
        text = format_structured_block(data)
        assert "unknown_key" in text
        assert "…" in text
        # Длина обрезанной секции ≤ 500 + ellipsis
        payload_start = text.find("- ")
        if payload_start >= 0:
            payload = text[payload_start:]
            assert len(payload) <= 520

    def test_nested_tool_data_rendered(self) -> None:
        """tool_data структура (route tool_simple) тоже разбирается."""
        data = {
            "tool_data": {
                "get_daily_facts": [
                    {"iso_date": "2026-04-23", "hrv_rmssd_milli": 50},
                    {"iso_date": "2026-04-22", "hrv_rmssd_milli": 55},
                    {"iso_date": "2026-04-21", "hrv_rmssd_milli": 52},
                ],
            }
        }
        text = format_structured_block(data)
        assert "HRV" in text

    def test_rag_keys_are_skipped(self) -> None:
        """RAG-чанки не рендерятся в этом блоке (их делает отдельный блок)."""
        data = {
            "rag_retrieve_training_principles": [{"text": "sleep matters"}],
            "get_daily_facts": [
                {"iso_date": "2026-04-23", "hrv_rmssd_milli": 50},
                {"iso_date": "2026-04-22", "hrv_rmssd_milli": 52},
            ],
        }
        text = format_structured_block(data)
        assert "sleep matters" not in text
        assert "HRV" in text

    def test_baseline_facts_propagated_into_output(self) -> None:
        """Явный baseline даёт «% от базы» в выводе."""
        data = {
            "get_daily_facts": [
                {"iso_date": "2026-04-23", "hrv_rmssd_milli": 40},
            ]
        }
        baseline = [
            {"iso_date": f"2026-04-{i:02d}", "hrv_rmssd_milli": 55}
            for i in range(1, 22)
        ]
        text = format_structured_block(data, baseline_facts=baseline)
        assert "% от базы" in text


# ---------------------------------------------------------------------------
# Adaptive detail level — daily facts
# ---------------------------------------------------------------------------


class TestDailyFactsDetailLevel:
    def test_short_window_includes_per_day_breakdown(self) -> None:
        """Окно ≤ 7 дней — после агрегатов идёт построчная разбивка по датам."""
        data = {
            "get_daily_facts": [
                {"iso_date": f"2026-04-{20+i:02d}", "steps": 8000 + i * 100}
                for i in range(4)
            ]
        }
        text = format_structured_block(data)
        assert "По дням:" in text
        # Все даты присутствуют
        for i in range(4):
            assert f"2026-04-{20+i:02d}" in text
        # И значения тоже
        assert "8 000" in text
        assert "8 300" in text

    def test_seven_days_still_includes_breakdown(self) -> None:
        """Граница включительно: 7 дней попадают в детальный режим."""
        data = {
            "get_daily_facts": [
                {"iso_date": f"2026-04-{15+i:02d}", "steps": 7000 + i * 100}
                for i in range(7)
            ]
        }
        text = format_structured_block(data)
        assert "По дням:" in text

    def test_long_window_skips_per_day_breakdown(self) -> None:
        """Окно > 7 дней — только агрегаты, без построчной разбивки."""
        data = {
            "get_daily_facts": [
                {"iso_date": f"2026-04-{1+i:02d}", "steps": 7000 + i * 50}
                for i in range(10)
            ]
        }
        text = format_structured_block(data)
        assert "По дням:" not in text
        # Но агрегаты остаются
        assert "Шаги" in text

    def test_breakdown_lists_only_present_metrics_per_row(self) -> None:
        """В разбивке метрика выводится только если у дня она не None."""
        data = {
            "get_daily_facts": [
                {"iso_date": "2026-04-22", "steps": 9000, "hrv_rmssd_milli": None},
                {"iso_date": "2026-04-23", "steps": 9500, "hrv_rmssd_milli": 55},
            ]
        }
        text = format_structured_block(data)
        # Берём только строки из блока «По дням» — у них префикс "  · YYYY-".
        breakdown_lines = [
            ln for ln in text.splitlines() if ln.startswith("  · 2026-")
        ]
        first_line = next((ln for ln in breakdown_lines if "2026-04-22" in ln), "")
        assert "Шаги" in first_line
        assert "HRV" not in first_line
        second_line = next((ln for ln in breakdown_lines if "2026-04-23" in ln), "")
        assert "Шаги" in second_line
        assert "HRV" in second_line


# ---------------------------------------------------------------------------
# Adaptive detail level — activities
# ---------------------------------------------------------------------------


class TestActivitiesDetailLevel:
    def test_short_window_lists_each_activity(self) -> None:
        """≤ 5 тренировок — выводится «По тренировкам» с каждой сессией."""
        data = {
            "get_activities": [
                {
                    "sport_type": "running", "title": "Утренний бег",
                    "duration_seconds": 1800, "calories": 250,
                    "distance_meters": 4000,
                    "start_time": "2026-04-22T07:00:00",
                    "avg_heart_rate": 145,
                },
                {
                    "sport_type": "cycling", "title": "Вело",
                    "duration_seconds": 3600, "calories": 500,
                    "distance_meters": 20000,
                    "start_time": "2026-04-23T18:00:00",
                    "avg_heart_rate": 130,
                },
            ]
        }
        text = format_structured_block(data)
        assert "По тренировкам:" in text
        assert "Утренний бег" in text
        assert "Вело" in text
        assert "2026-04-22" in text
        assert "2026-04-23" in text
        # Атрибуты по конкретной сессии
        assert "ЧСС 145" in text
        assert "20.0 км" in text or "20 км" in text

    def test_long_window_skips_per_session_listing(self) -> None:
        """> 5 тренировок — только сводка, без построчного перечисления."""
        activities = [
            {
                "sport_type": "running", "title": f"Бег {i}",
                "duration_seconds": 1800, "calories": 250,
                "distance_meters": 4000,
                "start_time": f"2026-04-{10+i:02d}T07:00:00",
            }
            for i in range(6)
        ]
        text = format_structured_block({"get_activities": activities})
        assert "По тренировкам:" not in text
        # Но агрегаты на месте
        assert "Тренировок: 6" in text

    def test_five_activities_still_lists_each(self) -> None:
        """Граница включительно: 5 активностей попадают в детальный режим."""
        activities = [
            {
                "sport_type": "running", "title": f"Бег {i}",
                "duration_seconds": 1800, "calories": 250,
                "distance_meters": 4000,
                "start_time": f"2026-04-{10+i:02d}T07:00:00",
            }
            for i in range(5)
        ]
        text = format_structured_block({"get_activities": activities})
        assert "По тренировкам:" in text
