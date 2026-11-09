#!/usr/bin/env python3
"""Create an explicitly synthetic multi-interval UI fixture, with no Gemini calls."""

import argparse

from tone_listener_mcp.listener import ToneProfile
from tone_listener_mcp.workbench.adapters.soundshed import SoundshedAdapter, new_project
from tone_listener_mcp.workbench.models import Interval, SourceRef
from tone_listener_mcp.workbench.renderer import NativeRenderer
from tone_listener_mcp.workbench.storage import LocalStorage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", action="store_true", help="Create an invented quota-diagnostic example instead.")
    args = parser.parse_args()
    adapter = SoundshedAdapter()
    renderer = NativeRenderer(adapter)
    renderer.initialize()
    if args.diagnostics:
        interval = Interval(start_seconds=0, end_seconds=10)
        analysis = {
            "interval": interval.model_dump(), "model": "offline-synthetic-diagnostics",
            "unavailable": "SYNTHETIC quota example for UI testing only; no provider request was made.",
            "provider_attempted": False,
            "failure_info": {
                "http_status": 429, "category": "daily_quota",
                "retry_after_seconds": 60, "daily_request_limit": None,
            },
        }
        project = new_project(
            name="OFFLINE SYNTHETIC diagnostic example",
            source=SourceRef(
                kind="audio_file", display_name="Synthetic diagnostic metadata; no audio upload",
                duration_seconds=10, selected_intervals=[interval],
            ),
            analysis=[analysis],
            intent={"summary": "Invented HTTP 429 / Retry-After example, not an actual provider response or quota limit."},
            chain=adapter.default_chain([analysis]), engine_version=renderer.engine_version,
        )
        project.synthetic_fixture = True
        LocalStorage().save_project(project)
        print(f"OFFLINE SYNTHETIC diagnostic fixture {project.id}; invented Retry-After=60s; no Gemini calls.")
        return
    absent = {"presence": "absent", "prominence": "none"}
    analyses = []
    for index, (gain, brightness, mids) in enumerate((
        ("clean", "balanced", "present"),
        ("crunch", "bright", "forward"),
        ("high-gain", "dark", "scooped"),
    )):
        profile = ToneProfile(
            guitar_dominance="clear", gain=gain, brightness=brightness,
            bass_level="typical", bass_tightness="tight", mids=mids,
            attack="balanced", compression="moderate", sustain="typical",
            reverb=absent, delay=absent, modulation=absent, modulation_type="none",
            evidence=[], uncertainties=["Invented test data; not a listening result."],
        )
        analyses.append({
            "interval": Interval(start_seconds=index * 10, end_seconds=(index + 1) * 10).model_dump(),
            "model": "offline-synthetic-fixture", "profile": profile.model_dump(),
            "provider_attempted": False,
        })
    project = new_project(
        name="OFFLINE SYNTHETIC multi-interval fixture",
        source=SourceRef(
            kind="audio_file", display_name="Synthetic input metadata; no reference audio",
            duration_seconds=30, selected_intervals=[item["interval"] for item in analyses],
        ),
        analysis=analyses,
        intent={"summary": "Invented categorical profiles for UI testing only; no reference was analyzed."},
        chain=adapter.default_chain([analyses[0]]), engine_version=renderer.engine_version,
    )
    project.synthetic_fixture = True
    storage = LocalStorage()
    storage.save_project(project)
    print(f"OFFLINE SYNTHETIC fixture {project.id}; 3 invented intervals; no Gemini calls.")


if __name__ == "__main__":
    main()
