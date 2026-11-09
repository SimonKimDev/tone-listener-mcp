from itertools import pairwise

import pytest

from tone_listener_mcp.workbench.sources import automatic_intervals, safe_display_name


@pytest.mark.parametrize(
    ("duration", "expected_count"),
    [(4.5, 1), (19.9, 1), (20, 2), (60, 3), (240, 3)],
)
def test_automatic_intervals_are_bounded_and_disjoint(duration, expected_count):
    intervals = automatic_intervals(duration)

    assert len(intervals) == expected_count
    assert all(0 <= interval.start_seconds < interval.end_seconds <= duration for interval in intervals)
    assert all(interval.end_seconds - interval.start_seconds <= 30 for interval in intervals)
    assert all(
        left.end_seconds <= right.start_seconds
        for left, right in pairwise(intervals)
    )


def test_automatic_intervals_reject_unknown_or_excessive_duration():
    for duration in (0, float("nan"), float("inf"), 86_401):
        with pytest.raises(ValueError):
            automatic_intervals(duration)


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("../../unsafe/song.wav", "song.wav"),
        (r"C:\Users\someone\Desktop\riff.wav", "riff.wav"),
        ("bad\x00name.wav", "badname.wav"),
        ("", "uploaded audio"),
    ],
)
def test_safe_display_name_never_displays_a_path_or_control_character(filename, expected):
    assert safe_display_name(filename) == expected
