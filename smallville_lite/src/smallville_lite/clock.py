"""Game clock: ticks <-> game datetimes (DESIGN §3). Game time is naive local time."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta


@dataclass(frozen=True)
class GameClock:
    start: datetime
    tick_minutes: int = 10

    def time_at(self, tick: int) -> datetime:
        return self.start + timedelta(minutes=tick * self.tick_minutes)

    def tick_at(self, when: datetime) -> int:
        """Index of the tick whose interval contains ``when`` (floor)."""
        return int((when - self.start).total_seconds() // (self.tick_minutes * 60))

    def ticks_for(self, minutes: float) -> int:
        """Number of whole ticks needed to cover ``minutes`` (at least 1)."""
        return max(1, -(-int(minutes) // self.tick_minutes))

    @property
    def step(self) -> timedelta:
        return timedelta(minutes=self.tick_minutes)


def fmt_time(t: datetime | time) -> str:
    return t.strftime("%H:%M")


def fmt_datetime(t: datetime) -> str:
    """Human-readable, as used in prompts: 'Monday February 13, 2023, 08:30'."""
    return t.strftime("%A %B %d, %Y, %H:%M").replace(" 0", " ")


def fmt_date(d: date | datetime) -> str:
    return d.strftime("%A %B %d, %Y").replace(" 0", " ")


def parse_hhmm(text: str, day: date) -> datetime:
    """Parse 'HH:MM' (24h) on ``day``. '24:00' means midnight at the end of ``day``."""
    text = text.strip()
    try:
        hh, mm = text.split(":")
        h, m = int(hh), int(mm)
    except ValueError:
        raise ValueError(f"time {text!r} is not in HH:MM 24-hour format") from None
    if h == 24 and m == 0:
        return datetime.combine(day, time()) + timedelta(days=1)
    if not (0 <= h < 24 and 0 <= m < 60):
        raise ValueError(f"time {text!r} is out of range")
    return datetime.combine(day, time(h, m))
