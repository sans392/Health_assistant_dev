"""Тесты для модуля определения намерений (IntentDetector) — Phase 2."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.pipeline.intent_detection import IntentDetector, IntentResult, _parse_llm_json


@pytest.fixture
def detector() -> IntentDetector:
    return IntentDetector()


# ---------------------------------------------------------------------------
# Классификация намерений (rule-based, теперь async)
# ---------------------------------------------------------------------------

class TestIntentClassification:
    """Тесты классификации намерений."""

    @pytest.mark.asyncio
    async def test_data_query_show_workouts(self, detector: IntentDetector) -> None:
        result = await detector.detect("Покажи тренировки за неделю")
        assert result.intent == "data_query"
        assert result.confidence >= 0.85

    @pytest.mark.asyncio
    async def test_data_query_give_activities(self, detector: IntentDetector) -> None:
        result = await detector.detect("Дай историю занятий")
        assert result.intent == "data_query"
        assert result.confidence >= 0.85

    @pytest.mark.asyncio
    async def test_plan_request_create(self, detector: IntentDetector) -> None:
        result = await detector.detect("Составь план тренировок на месяц")
        assert result.intent == "plan_request"
        assert result.confidence >= 0.85

    @pytest.mark.asyncio
    async def test_plan_request_make_program(self, detector: IntentDetector) -> None:
        result = await detector.detect("Сделай программу тренировок")
        assert result.intent == "plan_request"
        assert result.confidence >= 0.85

    @pytest.mark.asyncio
    async def test_health_concern_knee_pain(self, detector: IntentDetector) -> None:
        result = await detector.detect("Болит колено после приседаний")
        assert result.intent == "health_concern"
        assert result.confidence >= 0.9

    @pytest.mark.asyncio
    async def test_health_concern_discomfort(self, detector: IntentDetector) -> None:
        result = await detector.detect("Чувствую дискомфорт в пояснице")
        assert result.intent == "health_concern"
        assert result.confidence >= 0.9

    @pytest.mark.asyncio
    async def test_direct_question_pulse(self, detector: IntentDetector) -> None:
        result = await detector.detect("Какой у меня пульс?")
        assert result.intent == "direct_question"
        assert result.confidence >= 0.8

    @pytest.mark.asyncio
    async def test_direct_question_weight(self, detector: IntentDetector) -> None:
        result = await detector.detect("Сколько я вешу?")
        assert result.intent == "direct_question"
        assert result.confidence >= 0.8

    @pytest.mark.asyncio
    async def test_data_query_progress(self, detector: IntentDetector) -> None:
        result = await detector.detect("Проанализируй прогресс в беге")
        assert result.intent == "data_query"
        assert result.confidence >= 0.85

    @pytest.mark.asyncio
    async def test_data_query_trend(self, detector: IntentDetector) -> None:
        result = await detector.detect("Покажи динамику моих тренировок")
        assert result.intent == "data_query"
        assert result.confidence >= 0.85

    @pytest.mark.asyncio
    async def test_general_chat_hello(self, detector: IntentDetector) -> None:
        result = await detector.detect("Привет!")
        assert result.intent == "general_chat"
        assert result.confidence >= 0.9

    @pytest.mark.asyncio
    async def test_general_chat_thanks(self, detector: IntentDetector) -> None:
        result = await detector.detect("Спасибо")
        assert result.intent == "general_chat"
        assert result.confidence >= 0.9

    @pytest.mark.asyncio
    async def test_fallback_unclear_query(self, detector: IntentDetector) -> None:
        result = await detector.detect("хм")
        assert result.intent == "general_chat"
        assert result.confidence < 0.5

    @pytest.mark.asyncio
    async def test_result_has_raw_query(self, detector: IntentDetector) -> None:
        query = "Покажи тренировки за неделю"
        result = await detector.detect(query)
        assert result.raw_query == query


# ---------------------------------------------------------------------------
# Извлечение сущностей (расширенный Phase 2)
# ---------------------------------------------------------------------------

class TestEntityExtraction:
    """Тесты извлечения сущностей."""

    @pytest.mark.asyncio
    async def test_time_range_today(self, detector: IntentDetector) -> None:
        result = await detector.detect("Покажи активность за сегодня")
        assert result.entities.get("time_range") == "сегодня"

    @pytest.mark.asyncio
    async def test_time_range_yesterday(self, detector: IntentDetector) -> None:
        result = await detector.detect("Какие тренировки были вчера?")
        assert result.entities.get("time_range") == "вчера"

    @pytest.mark.asyncio
    async def test_time_range_week(self, detector: IntentDetector) -> None:
        result = await detector.detect("Покажи тренировки за неделю")
        assert result.entities.get("time_range") == "за неделю"

    @pytest.mark.asyncio
    async def test_time_range_month(self, detector: IntentDetector) -> None:
        result = await detector.detect("Статистика за месяц")
        assert result.entities.get("time_range") == "за месяц"

    @pytest.mark.asyncio
    async def test_sport_running(self, detector: IntentDetector) -> None:
        result = await detector.detect("Проанализируй мой бег")
        assert result.entities.get("sport_type") == "running"

    @pytest.mark.asyncio
    async def test_sport_cycling(self, detector: IntentDetector) -> None:
        result = await detector.detect("Покажи тренировки на велосипеде")
        assert result.entities.get("sport_type") == "cycling"

    @pytest.mark.asyncio
    async def test_sport_swimming(self, detector: IntentDetector) -> None:
        result = await detector.detect("Сколько раз я плавал?")
        assert result.entities.get("sport_type") == "swimming"

    @pytest.mark.asyncio
    async def test_sport_gym(self, detector: IntentDetector) -> None:
        result = await detector.detect("Сколько раз я ходил в зал?")
        assert result.entities.get("sport_type") == "gym"

    @pytest.mark.asyncio
    async def test_metric_pulse(self, detector: IntentDetector) -> None:
        result = await detector.detect("Какой у меня пульс?")
        assert result.entities.get("metric") == "heart_rate"

    @pytest.mark.asyncio
    async def test_metric_weight(self, detector: IntentDetector) -> None:
        result = await detector.detect("Сколько я вешу?")
        assert result.entities.get("metric") == "вес"

    @pytest.mark.asyncio
    async def test_metric_calories(self, detector: IntentDetector) -> None:
        result = await detector.detect("Сколько калорий я сжёг?")
        assert result.entities.get("metric") == "калории"

    @pytest.mark.asyncio
    async def test_metric_hrv(self, detector: IntentDetector) -> None:
        result = await detector.detect("Какой у меня HRV?")
        assert result.entities.get("metric") == "hrv"

    @pytest.mark.asyncio
    async def test_metric_sleep(self, detector: IntentDetector) -> None:
        result = await detector.detect("Как я сплю последнюю неделю?")
        assert result.entities.get("metric") == "сон"

    @pytest.mark.asyncio
    async def test_body_part_knee(self, detector: IntentDetector) -> None:
        result = await detector.detect("Болит колено после пробежки")
        assert result.entities.get("body_part") == "колено"

    @pytest.mark.asyncio
    async def test_body_part_back(self, detector: IntentDetector) -> None:
        result = await detector.detect("Болит спина после тренировки")
        assert result.entities.get("body_part") == "спина"

    @pytest.mark.asyncio
    async def test_intensity_heavy(self, detector: IntentDetector) -> None:
        result = await detector.detect("После тяжёлой тренировки болит всё тело")
        assert result.entities.get("intensity") == "тяжело"

    @pytest.mark.asyncio
    async def test_no_entities_general_chat(self, detector: IntentDetector) -> None:
        result = await detector.detect("Привет!")
        assert result.entities == {}


# ---------------------------------------------------------------------------
# LLM stage 2 — fallback поведение
# ---------------------------------------------------------------------------

class TestLLMStage:
    """Тесты LLM stage 2 (LLM fallback при низкой уверенности)."""

    def _make_registry(self, json_response: str) -> MagicMock:
        """Создать mock LLMRegistry, возвращающий заданный JSON через /api/chat."""
        llm_response = MagicMock()
        llm_response.content = json_response
        llm_response.model = "test-model"

        client = AsyncMock()
        client.chat = AsyncMock(return_value=llm_response)

        registry = MagicMock()
        registry.get_client.return_value = client
        return registry

    @pytest.mark.asyncio
    async def test_high_confidence_skips_llm(self, detector: IntentDetector) -> None:
        """Высокая уверенность → LLM не вызывается."""
        registry = MagicMock()
        # "Болит колено" → health_concern с confidence=0.95
        result = await detector.detect(
            "Болит колено", llm_registry=registry
        )
        assert result.intent == "health_concern"
        assert result.llm_used is False
        registry.get_client.assert_not_called()

    @pytest.mark.asyncio
    async def test_low_confidence_calls_llm(self, detector: IntentDetector) -> None:
        """Низкая уверенность → LLM вызывается и возвращает корректный intent."""
        registry = self._make_registry(
            '{"intent": "data_query", "confidence": 0.9, "entities": {}}'
        )
        # Неоднозначный запрос → rule-based даст низкую уверенность
        result = await detector.detect(
            "хм, интересно что там с моими показателями", llm_registry=registry
        )
        assert result.intent == "data_query"
        assert result.llm_used is True
        assert result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_llm_without_registry_no_crash(self, detector: IntentDetector) -> None:
        """Без registry → возвращается rule-based результат без ошибки."""
        result = await detector.detect("хм", llm_registry=None)
        assert result.intent == "general_chat"
        assert result.llm_used is False

    @pytest.mark.asyncio
    async def test_json_parse_fail_falls_back_to_rule_based(
        self, detector: IntentDetector
    ) -> None:
        """Невалидный JSON от LLM → возврат rule-based результата + llm_used=False."""
        registry = self._make_registry("Извините, я не могу ответить в JSON формате.")
        result = await detector.detect(
            "непонятный запрос без ключевых слов", llm_registry=registry
        )
        assert result.llm_used is False  # fallback сработал
        assert result.intent in {"general_chat", "data_query",
                                  "plan_request", "health_concern", "direct_question",
                                  "emergency", "off_topic",
                                  "reference_question", "capability_question"}

    @pytest.mark.asyncio
    async def test_llm_unknown_intent_falls_back(self, detector: IntentDetector) -> None:
        """Неизвестный intent от LLM → fallback на rule-based."""
        registry = self._make_registry(
            '{"intent": "неизвестный_тип", "confidence": 0.9, "entities": {}}'
        )
        result = await detector.detect("непонятный вопрос", llm_registry=registry)
        assert result.llm_used is False

    @pytest.mark.asyncio
    async def test_llm_error_falls_back(self, detector: IntentDetector) -> None:
        """Ошибка LLM-вызова → fallback на rule-based без исключения."""
        client = AsyncMock()
        client.chat = AsyncMock(side_effect=Exception("Ollama недоступен"))

        registry = MagicMock()
        registry.get_client.return_value = client

        result = await detector.detect("непонятный вопрос", llm_registry=registry)
        assert result.llm_used is False

    @pytest.mark.asyncio
    async def test_llm_entities_merged_with_rule_based(
        self, detector: IntentDetector
    ) -> None:
        """Entities из LLM объединяются с rule-based; rule-based имеет приоритет."""
        registry = self._make_registry(
            '{"intent": "data_query", "confidence": 0.85, '
            '"entities": {"metric": "hrv", "extra": "value"}}'
        )
        # sport_type "бег" → rule-based найдёт running
        result = await detector.detect(
            "как мой бег в целом?", llm_registry=registry
        )
        # Rule-based sport_type должен быть сохранён
        assert result.entities.get("sport_type") == "running"
        # LLM-only entity должен быть включён
        assert result.entities.get("extra") == "value"

    @pytest.mark.asyncio
    async def test_history_passed_to_llm(self, detector: IntentDetector) -> None:
        """История диалога передаётся при вызове LLM как role=user/assistant."""
        registry = self._make_registry(
            '{"intent": "data_query", "confidence": 0.88, "entities": {}}'
        )
        history = [
            {"role": "user", "content": "Привет"},
            {"role": "assistant", "content": "Здравствуйте!"},
        ]
        await detector.detect(
            "покажи что у меня", llm_registry=registry, history=history
        )
        client = registry.get_client.return_value
        call_kwargs = client.chat.call_args.kwargs
        messages = call_kwargs.get("messages", [])
        # Ожидаем role=user/assistant в messages с текущим запросом в конце
        assert {"role": "user", "content": "Привет"} in messages
        assert {"role": "assistant", "content": "Здравствуйте!"} in messages
        assert messages[-1] == {"role": "user", "content": "покажи что у меня"}

    @pytest.mark.asyncio
    async def test_llm_stage_uses_json_format(self, detector: IntentDetector) -> None:
        """Intent LLM stage 2 должен вызываться с format='json'."""
        registry = self._make_registry(
            '{"intent": "data_query", "confidence": 0.9, "entities": {}}'
        )
        await detector.detect(
            "хм, интересно что там с показателями", llm_registry=registry
        )
        client = registry.get_client.return_value
        call_kwargs = client.chat.call_args.kwargs
        assert call_kwargs.get("format") == "json"
        assert call_kwargs.get("system_prompt")


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

class TestParseJson:
    """Тесты парсера JSON ответов LLM."""

    def test_valid_json(self) -> None:
        result = _parse_llm_json('{"intent": "plan_request", "confidence": 0.9}')
        assert result is not None
        assert result["intent"] == "plan_request"

    def test_json_in_markdown_block(self) -> None:
        text = '```json\n{"intent": "health_concern", "confidence": 0.8}\n```'
        result = _parse_llm_json(text)
        assert result is not None
        assert result["intent"] == "health_concern"

    def test_json_embedded_in_text(self) -> None:
        text = 'Вот мой ответ: {"intent": "data_query", "confidence": 0.75} и всё.'
        result = _parse_llm_json(text)
        assert result is not None
        assert result["intent"] == "data_query"

    def test_invalid_text_returns_none(self) -> None:
        result = _parse_llm_json("Я не могу определить намерение.")
        assert result is None
