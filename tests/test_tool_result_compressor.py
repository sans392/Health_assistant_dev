"""Тесты schema-aware сжатия tool-результатов (Issue #58).

Покрываем:
- list-компрессия get_activities / get_daily_facts (summary + sample + flag).
- dict pass-through для compute_recovery / check_overtraining / get_user_profile.
- RAG trim: top-1 целиком, остальные — snippet.
- Edge-cases: None, пустой список, короткий список (≤ threshold).
"""

from __future__ import annotations

from app.pipeline.tool_result_compressor import (
    CompressedResult,
    compress_for_planner,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mk_activity(i: int, calories: int = 500, sport: str = "running") -> dict:
    return {
        "id": i,
        "user_id": "u1",
        "title": f"run-{i}",
        "sport_type": sport,
        "distance_meters": 5000 + i * 100,
        "duration_seconds": 1800 + i * 60,
        "start_time": f"2026-04-{10 + (i % 20):02d}T10:00:00",
        "end_time": f"2026-04-{10 + (i % 20):02d}T10:30:00",
        "avg_speed": 3.0,
        "calories": calories,
        "avg_heart_rate": 140,
        "source": "seed",
    }


def _mk_fact(iso_date: str, hrv: float | None = 50.0, rhr: int | None = 55) -> dict:
    return {
        "id": iso_date,
        "user_id": "u1",
        "iso_date": iso_date,
        "hrv_rmssd_milli": hrv,
        "resting_heart_rate": rhr,
        "steps": 8000,
        "recovery_score": None,
        "strain_score": None,
    }


# ---------------------------------------------------------------------------
# get_activities
# ---------------------------------------------------------------------------


class TestCompressActivities:
    def test_short_list_pass_through(self) -> None:
        raw = [_mk_activity(i) for i in range(3)]
        out = compress_for_planner("get_activities", raw)
        assert out.compressed is False
        assert out.payload == raw
        assert out.to_message_payload() == raw

    def test_long_list_is_compressed(self) -> None:
        raw = [_mk_activity(i, calories=100 + i * 10) for i in range(30)]
        out = compress_for_planner("get_activities", raw)

        assert out.compressed is True
        assert out.full_count == 30
        assert out.shown == 5

        payload = out.payload
        assert isinstance(payload, dict)
        assert payload["total_count"] == 30
        assert len(payload["sample"]) == 5
        assert "summary" in payload

        # Sample должен содержать самые «интенсивные» (максимум калорий).
        sample_calories = [a["calories"] for a in payload["sample"]]
        assert max(sample_calories) == 390  # i=29: 100 + 29*10
        # И сортировка по убыванию.
        assert sample_calories == sorted(sample_calories, reverse=True)

        # Summary — это ActivityPromptSummary.
        assert payload["summary"]["total_activities"] == 30

    def test_message_payload_wraps_with_flag(self) -> None:
        raw = [_mk_activity(i) for i in range(20)]
        out = compress_for_planner("get_activities", raw)
        msg = out.to_message_payload()
        assert isinstance(msg, dict)
        assert msg["compressed"] is True
        assert msg["full_count"] == 20
        assert msg["shown"] == 5
        assert "tool_result" in msg
        assert msg["tool_result"]["total_count"] == 20

    def test_empty_list_returns_empty_payload(self) -> None:
        out = compress_for_planner("get_activities", [])
        assert out.compressed is False
        assert out.payload == []


# ---------------------------------------------------------------------------
# get_daily_facts
# ---------------------------------------------------------------------------


class TestCompressDailyFacts:
    def test_short_list_pass_through(self) -> None:
        raw = [_mk_fact(f"2026-04-{i:02d}") for i in range(1, 4)]
        out = compress_for_planner("get_daily_facts", raw)
        assert out.compressed is False
        assert out.payload == raw

    def test_long_list_is_compressed(self) -> None:
        # 14 дней, среди них один явный outlier по HRV.
        raw = [_mk_fact(f"2026-04-{i:02d}", hrv=50.0) for i in range(1, 15)]
        raw[5]["hrv_rmssd_milli"] = 20.0  # большое отклонение

        out = compress_for_planner("get_daily_facts", raw)
        assert out.compressed is True
        assert out.full_count == 14
        assert out.shown == 5

        payload = out.payload
        assert payload["total_count"] == 14
        # Sample содержит 5 записей, и outlier должен быть в них.
        sample_dates = {s["iso_date"] for s in payload["sample"]}
        assert "2026-04-06" in sample_dates

        metrics = payload["summary"]["metrics"]
        assert "hrv_rmssd_milli" in metrics
        assert metrics["hrv_rmssd_milli"]["min"] == 20.0

    def test_daily_facts_without_metrics_fallback_to_latest(self) -> None:
        # Записи без единого числового значения среди приоритетных метрик.
        raw = [
            {"iso_date": f"2026-04-{i:02d}", "id": i} for i in range(1, 10)
        ]
        out = compress_for_planner("get_daily_facts", raw)
        assert out.compressed is True
        # Выбраны последние 5 по дате.
        dates = [s["iso_date"] for s in out.payload["sample"]]
        assert dates[0] == "2026-04-09"
        assert len(dates) == 5


# ---------------------------------------------------------------------------
# Dict tools
# ---------------------------------------------------------------------------


class TestCompressDictTools:
    def test_compute_recovery_pass_through(self) -> None:
        raw = {"score": 72, "components": {"hrv": 30, "sleep": 25}, "trend": "stable"}
        out = compress_for_planner("compute_recovery", raw)
        assert out.compressed is False
        assert out.payload is raw
        assert out.to_message_payload() is raw

    def test_check_overtraining_pass_through(self) -> None:
        raw = {"is_overtrained": False, "score": 0.3, "details": {"hrv_drop": False}}
        out = compress_for_planner("check_overtraining", raw)
        assert out.compressed is False
        assert out.payload == raw

    def test_get_user_profile_pass_through(self) -> None:
        raw = {"name": "Иван", "age": 30, "weight_kg": 75}
        out = compress_for_planner("get_user_profile", raw)
        assert out.compressed is False
        assert out.payload == raw


# ---------------------------------------------------------------------------
# rag_retrieve
# ---------------------------------------------------------------------------


class TestCompressRag:
    def test_single_chunk_pass_through(self) -> None:
        raw = [{"text": "x" * 500, "category": "physiology_norms", "source": "doc1",
                "confidence": "high", "score": 0.9}]
        out = compress_for_planner("rag_retrieve", raw)
        assert out.compressed is False
        assert out.payload == raw

    def test_multiple_chunks_trim_non_top(self) -> None:
        raw = [
            {"text": "A" * 500, "category": "physiology_norms", "source": "doc1",
             "confidence": "high", "score": 0.95},
            {"text": "B" * 500, "category": "training_principles", "source": "doc2",
             "confidence": "medium", "score": 0.7},
            {"text": "C" * 80, "category": "recovery_science", "source": "doc3",
             "confidence": "medium", "score": 0.5},
        ]
        out = compress_for_planner("rag_retrieve", raw)
        assert out.compressed is True
        assert out.full_count == 3
        assert out.shown == 3

        chunks = out.payload
        # top-1 — сохранён полностью
        assert chunks[0]["text"] == "A" * 500

        # второй — обрезан до snippet
        assert "snippet" in chunks[1]
        assert chunks[1]["snippet"].startswith("B")
        assert len(chunks[1]["snippet"]) <= 151  # 150 + «…»
        assert chunks[1]["snippet"].endswith("…")
        assert "text" not in chunks[1]

        # третий — короткий, snippet не заканчивается на «…»
        assert chunks[2]["snippet"] == "C" * 80


# ---------------------------------------------------------------------------
# None / unknown
# ---------------------------------------------------------------------------


class TestCompressMiscellaneous:
    def test_none_returns_none_payload(self) -> None:
        out = compress_for_planner("get_activities", None)
        assert isinstance(out, CompressedResult)
        assert out.payload is None
        assert out.compressed is False

    def test_unknown_tool_list_compressed_when_long(self) -> None:
        raw = [{"i": i} for i in range(10)]
        out = compress_for_planner("some_new_tool", raw)
        assert out.compressed is True
        assert out.full_count == 10
        assert out.payload["sample"] == raw[:5]

    def test_unknown_tool_dict_passthrough(self) -> None:
        raw = {"anything": "ok"}
        out = compress_for_planner("some_new_tool", raw)
        assert out.compressed is False
        assert out.payload == raw
