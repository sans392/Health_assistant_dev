"""Тесты для модуля проверки безопасности (SafetyChecker)."""

import pytest

from app.pipeline.safety_check import SafetyChecker, SafetyResult


@pytest.fixture
def checker() -> SafetyChecker:
    return SafetyChecker()


class TestHighPriorityPatterns:
    """Тесты блокирующих паттернов высокого приоритета."""

    def test_chest_pain_blocks(self, checker: SafetyChecker) -> None:
        result = checker.check("У меня боль в груди")
        assert result.is_safe is False
        assert result.safety_level == "high_priority"
        assert result.redirect_message is not None
        assert result.warning_suffix is None

    def test_heart_pain_blocks(self, checker: SafetyChecker) -> None:
        result = checker.check("Боль в сердце уже час")
        assert result.is_safe is False
        assert result.safety_level == "high_priority"

    def test_cannot_breathe_blocks(self, checker: SafetyChecker) -> None:
        result = checker.check("Не могу дышать после тренировки")
        assert result.is_safe is False
        assert result.safety_level == "high_priority"

    def test_loss_of_consciousness_blocks(self, checker: SafetyChecker) -> None:
        result = checker.check("Потеря сознания после бега")
        assert result.is_safe is False
        assert result.safety_level == "high_priority"

    def test_numbness_extremities_blocks(self, checker: SafetyChecker) -> None:
        result = checker.check("Онемение конечностей во время упражнений")
        assert result.is_safe is False
        assert result.safety_level == "high_priority"

    def test_severe_dizziness_blocks(self, checker: SafetyChecker) -> None:
        result = checker.check("Сильное головокружение после тренировки")
        assert result.is_safe is False
        assert result.safety_level == "high_priority"

    def test_blood_in_urine_blocks(self, checker: SafetyChecker) -> None:
        result = checker.check("Заметил кровь в моче")
        assert result.is_safe is False
        assert result.safety_level == "high_priority"

    def test_redirect_message_not_empty(self, checker: SafetyChecker) -> None:
        result = checker.check("Боль в груди")
        assert result.redirect_message
        assert len(result.redirect_message) > 10


class TestMediumPriorityPatterns:
    """Тесты паттернов среднего приоритета."""

    def test_pain_for_weeks_warns(self, checker: SafetyChecker) -> None:
        result = checker.check("Болит уже неделю колено")
        assert result.is_safe is True
        assert result.safety_level == "medium_priority"
        assert result.warning_suffix is not None
        assert result.redirect_message is None

    def test_chronic_fatigue_warns(self, checker: SafetyChecker) -> None:
        result = checker.check("Постоянная усталость мешает тренироваться")
        assert result.is_safe is True
        assert result.safety_level == "medium_priority"

    def test_headache_after_workout_warns(self, checker: SafetyChecker) -> None:
        result = checker.check("Головная боль после тренировки")
        assert result.is_safe is True
        assert result.safety_level == "medium_priority"

    def test_cannot_gain_weight_warns(self, checker: SafetyChecker) -> None:
        result = checker.check("Не могу набрать вес уже несколько месяцев")
        assert result.is_safe is True
        assert result.safety_level == "medium_priority"

    def test_warning_suffix_not_empty(self, checker: SafetyChecker) -> None:
        result = checker.check("Постоянная усталость")
        assert result.warning_suffix
        assert len(result.warning_suffix) > 10


class TestSafeQueries:
    """Тесты безопасных запросов."""

    def test_show_workouts_is_safe(self, checker: SafetyChecker) -> None:
        result = checker.check("Покажи тренировки за неделю")
        assert result.is_safe is True
        assert result.safety_level == "ok"
        assert result.redirect_message is None
        assert result.warning_suffix is None

    def test_create_plan_is_safe(self, checker: SafetyChecker) -> None:
        result = checker.check("Составь план тренировок")
        assert result.is_safe is True
        assert result.safety_level == "ok"

    def test_general_chat_is_safe(self, checker: SafetyChecker) -> None:
        result = checker.check("Привет!")
        assert result.is_safe is True
        assert result.safety_level == "ok"

    def test_analysis_request_is_safe(self, checker: SafetyChecker) -> None:
        result = checker.check("Проанализируй прогресс в беге")
        assert result.is_safe is True
        assert result.safety_level == "ok"

    def test_simple_knee_pain_without_duration(self, checker: SafetyChecker) -> None:
        """Простая боль в колене без хронических признаков — не medium, не high."""
        result = checker.check("Болит колено после приседаний")
        # Не high_priority (нет критических паттернов)
        assert result.safety_level != "high_priority"
