"""Source interval defaults and safe display names."""

from __future__ import annotations

import math
import unicodedata
from pathlib import PurePosixPath, PureWindowsPath

from tone_listener_mcp.workbench.models import Interval


def automatic_intervals(duration_seconds: float) -> list[Interval]:
    """Choose up to three disjoint, bounded intervals from known decoded duration."""
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("Source duration must be positive and finite.")
    if duration_seconds > 86_400:
        raise ValueError("Sources longer than 24 hours are not supported.")
    count = min(3, max(1, int(duration_seconds // 10)))
    length = min(30.0, duration_seconds / count)
    if count == 1:
        starts = [max(0.0, (duration_seconds - length) / 2)]
    else:
        spacing = (duration_seconds - length) / (count - 1)
        starts = [spacing * index for index in range(count)]
    intervals = []
    for index, start in enumerate(starts):
        end = min(duration_seconds, start + length)
        if index + 1 < len(starts):
            end = min(end, starts[index + 1])
        intervals.append(Interval(start_seconds=start, end_seconds=end))
    return intervals


def safe_display_name(filename: str | None) -> str:
    if not filename:
        return "uploaded audio"
    name = PureWindowsPath(filename).name
    name = PurePosixPath(name).name
    name = unicodedata.normalize("NFC", name)
    printable = "".join(character for character in name if character.isprintable())
    return printable[:120] or "uploaded audio"
