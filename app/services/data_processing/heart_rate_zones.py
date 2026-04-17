"""Модуль расчёта зон пульса по формуле Карвонена (Phase 2, Issue #27)."""

from dataclasses import dataclass


@dataclass
class HRZones:
    """Пять зон пульса рассчитанных по формуле Карвонена."""

    z1: tuple[int, int]   # 50–60% HRR — активное восстановление
    z2: tuple[int, int]   # 60–70% HRR — аэробная база
    z3: tuple[int, int]   # 70–80% HRR — аэробный порог
    z4: tuple[int, int]   # 80–90% HRR — анаэробный порог
    z5: tuple[int, int]   # 90–100% HRR — максимальная нагрузка
    max_hr: int
    resting_hr: int

    def zone_for_hr(self, hr: int) -> int | None:
        """Определить зону для заданного значения пульса.

        Returns:
            Номер зоны (1–5) или None если пульс вне диапазона всех зон.
        """
        for zone_num, zone_range in enumerate(
            [self.z1, self.z2, self.z3, self.z4, self.z5], start=1
        ):
            if zone_range[0] <= hr <= zone_range[1]:
                return zone_num
        return None


def compute_hr_zones(
    age: int,
    resting_hr: int,
    max_hr: int | None = None,
) -> HRZones:
    """Рассчитать зоны пульса по формуле Карвонена.

    Формула:
        HRR = max_hr - resting_hr
        HR_zone_boundary = resting_hr + HRR × zone_percentage

    Args:
        age: Возраст пользователя (лет).
        resting_hr: Пульс покоя (уд/мин).
        max_hr: Максимальный пульс. Если None — используется 220 - age.

    Returns:
        HRZones с нижней и верхней границей для каждой из 5 зон.
    """
    effective_max_hr = max_hr if max_hr and max_hr > 0 else (220 - age)
    hrr = effective_max_hr - resting_hr

    def _boundary(pct: float) -> int:
        return int(resting_hr + hrr * pct)

    return HRZones(
        z1=(_boundary(0.50), _boundary(0.60)),
        z2=(_boundary(0.60), _boundary(0.70)),
        z3=(_boundary(0.70), _boundary(0.80)),
        z4=(_boundary(0.80), _boundary(0.90)),
        z5=(_boundary(0.90), _boundary(1.00)),
        max_hr=effective_max_hr,
        resting_hr=resting_hr,
    )
