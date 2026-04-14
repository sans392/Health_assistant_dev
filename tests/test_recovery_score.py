"""Тесты модуля recovery_score (app/services/data_processing/recovery_score.py)."""

import pytest

from app.services.data_processing.recovery_score import (
    RecoveryScoreResult,
    get_recovery_score,
)


class TestGetRecoveryScore:
    """Тесты get_recovery_score."""

    def test_empty_list_returns_unavailable(self) -> None:
        result = get_recovery_score([])
        assert result.available is False
        assert result.score is None
        assert result.source is None
        assert result.iso_date is None

    def test_single_fact_with_score(self) -> None:
        facts = [{"iso_date": "2026-04-10", "recovery_score": 75}]
        result = get_recovery_score(facts)
        assert result.available is True
        assert result.score == 75
        assert result.source == "whoop"
        assert result.iso_date == "2026-04-10"

    def test_returns_latest_score(self) -> None:
        """Возвращает самую свежую запись."""
        facts = [
            {"iso_date": "2026-04-08", "recovery_score": 60},
            {"iso_date": "2026-04-09", "recovery_score": 70},
            {"iso_date": "2026-04-10", "recovery_score": 85},
        ]
        result = get_recovery_score(facts)
        assert result.score == 85
        assert result.iso_date == "2026-04-10"

    def test_skips_none_scores(self) -> None:
        """Возвращает последний непустой score."""
        facts = [
            {"iso_date": "2026-04-09", "recovery_score": 70},
            {"iso_date": "2026-04-10", "recovery_score": None},  # нет данных
        ]
        result = get_recovery_score(facts)
        assert result.score == 70
        assert result.iso_date == "2026-04-09"

    def test_all_none_returns_unavailable(self) -> None:
        facts = [
            {"iso_date": "2026-04-09", "recovery_score": None},
            {"iso_date": "2026-04-10", "recovery_score": None},
        ]
        result = get_recovery_score(facts)
        assert result.available is False
        assert result.score is None

    def test_score_is_int(self) -> None:
        facts = [{"iso_date": "2026-04-10", "recovery_score": 82.5}]
        result = get_recovery_score(facts)
        assert isinstance(result.score, int)
        assert result.score == 82
