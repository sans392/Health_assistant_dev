"""Тесты для модуля определения намерений (IntentDetector)."""

import pytest

from app.pipeline.intent_detection import IntentDetector, IntentResult


@pytest.fixture
def detector() -> IntentDetector:
    return IntentDetector()


class TestIntentClassification:
    """Тесты классификации намерений."""

    def test_data_retrieval_show_workouts(self, detector: IntentDetector) -> None:
        result = detector.detect("Покажи тренировки за неделю")
        assert result.intent == "data_retrieval"
        assert result.confidence >= 0.85

    def test_data_retrieval_give_activities(self, detector: IntentDetector) -> None:
        result = detector.detect("Дай историю занятий")
        assert result.intent == "data_retrieval"
        assert result.confidence >= 0.85

    def test_plan_request_create(self, detector: IntentDetector) -> None:
        result = detector.detect("Составь план тренировок на месяц")
        assert result.intent == "plan_request"
        assert result.confidence >= 0.85

    def test_plan_request_make_program(self, detector: IntentDetector) -> None:
        result = detector.detect("Сделай программу тренировок")
        assert result.intent == "plan_request"
        assert result.confidence >= 0.85

    def test_health_concern_knee_pain(self, detector: IntentDetector) -> None:
        result = detector.detect("Болит колено после приседаний")
        assert result.intent == "health_concern"
        assert result.confidence >= 0.9

    def test_health_concern_discomfort(self, detector: IntentDetector) -> None:
        result = detector.detect("Чувствую дискомфорт в пояснице")
        assert result.intent == "health_concern"
        assert result.confidence >= 0.9

    def test_direct_question_pulse(self, detector: IntentDetector) -> None:
        result = detector.detect("Какой у меня пульс?")
        assert result.intent == "direct_question"
        assert result.confidence >= 0.8

    def test_direct_question_weight(self, detector: IntentDetector) -> None:
        result = detector.detect("Сколько я вешу?")
        assert result.intent == "direct_question"
        assert result.confidence >= 0.8

    def test_data_analysis_progress(self, detector: IntentDetector) -> None:
        result = detector.detect("Проанализируй прогресс в беге")
        assert result.intent == "data_analysis"
        assert result.confidence >= 0.85

    def test_data_analysis_trend(self, detector: IntentDetector) -> None:
        result = detector.detect("Покажи динамику моих тренировок")
        assert result.intent == "data_analysis"
        assert result.confidence >= 0.85

    def test_general_chat_hello(self, detector: IntentDetector) -> None:
        result = detector.detect("Привет!")
        assert result.intent == "general_chat"
        assert result.confidence >= 0.9

    def test_general_chat_thanks(self, detector: IntentDetector) -> None:
        result = detector.detect("Спасибо")
        assert result.intent == "general_chat"
        assert result.confidence >= 0.9

    def test_fallback_unclear_query(self, detector: IntentDetector) -> None:
        result = detector.detect("хм")
        assert result.intent == "general_chat"
        assert result.confidence < 0.5

    def test_result_has_raw_query(self, detector: IntentDetector) -> None:
        query = "Покажи тренировки за неделю"
        result = detector.detect(query)
        assert result.raw_query == query


class TestEntityExtraction:
    """Тесты извлечения сущностей."""

    def test_time_range_today(self, detector: IntentDetector) -> None:
        result = detector.detect("Покажи активность за сегодня")
        assert result.entities.get("time_range") == "сегодня"

    def test_time_range_yesterday(self, detector: IntentDetector) -> None:
        result = detector.detect("Какие тренировки были вчера?")
        assert result.entities.get("time_range") == "вчера"

    def test_time_range_week(self, detector: IntentDetector) -> None:
        result = detector.detect("Покажи тренировки за неделю")
        assert result.entities.get("time_range") == "за неделю"

    def test_time_range_month(self, detector: IntentDetector) -> None:
        result = detector.detect("Статистика за месяц")
        assert result.entities.get("time_range") == "за месяц"

    def test_sport_running(self, detector: IntentDetector) -> None:
        result = detector.detect("Проанализируй мой бег")
        assert result.entities.get("sport_type") == "running"

    def test_sport_cycling(self, detector: IntentDetector) -> None:
        result = detector.detect("Покажи тренировки на велосипеде")
        assert result.entities.get("sport_type") == "cycling"

    def test_sport_swimming(self, detector: IntentDetector) -> None:
        result = detector.detect("Сколько раз я плавал?")
        assert result.entities.get("sport_type") == "swimming"

    def test_sport_gym(self, detector: IntentDetector) -> None:
        result = detector.detect("Сколько раз я ходил в зал?")
        assert result.entities.get("sport_type") == "gym"

    def test_metric_pulse(self, detector: IntentDetector) -> None:
        result = detector.detect("Какой у меня пульс?")
        assert result.entities.get("metric") == "пульс"

    def test_metric_weight(self, detector: IntentDetector) -> None:
        result = detector.detect("Сколько я вешу?")
        assert result.entities.get("metric") == "вес"

    def test_metric_calories(self, detector: IntentDetector) -> None:
        result = detector.detect("Сколько калорий я сжёг?")
        assert result.entities.get("metric") == "калории"

    def test_no_entities_general_chat(self, detector: IntentDetector) -> None:
        result = detector.detect("Привет!")
        assert result.entities == {}
