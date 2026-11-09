"""Soundshed Guitar 1.7.3 resource resolution, validation, render state and export."""

from __future__ import annotations

import hashlib
import io
import itertools
import json
import sqlite3
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from tone_listener_mcp.listener import ToneProfile
from tone_listener_mcp.workbench.models import (
    AdapterCatalog,
    AssetRef,
    BlockDefinition,
    ChainDraft,
    ChainNode,
    ParameterSpec,
    PreviewSnapshot,
    ResourceDefinition,
    SourceRef,
    ToneProject,
)

ADAPTER_ID = "soundshed-guitar"
SUPPORTED_ENGINE_VERSION = "1.7.3"

_FX_NAM = "c3263344-65e4-4b7e-b102-ea625700e12f"
_AMP_NAM = "49ea214c-91e6-41f9-bd27-ad6eec0ae90a"
_CAB_IR = "94fa2577-e904-43b8-968b-9c569c511160"
_EQ_PARAMETRIC = "4b4025ca-64cd-4180-be79-81873b618dba"
_REVERB_ROOM = "7467cbf1-6c7f-4f07-b5dd-a303d25b475c"
_DELAY_DIGITAL = "673d3e7a-e9ef-4c5d-a4c4-619dff3355ed"
_CHORUS = "decdd132-029a-46a5-a362-edcde007a450"
_GAIN = "0bcd895e-5d36-4247-a351-6bed1fcb37a8"

_PARAMETERS: dict[str, list[ParameterSpec]] = {
    "input": [
        ParameterSpec(
            id="gainDb", label="Input trim", minimum=-24, maximum=24,
            default=0, unit="dB", description="Level entering the saved chain.",
        )
    ],
    "output": [
        ParameterSpec(
            id="gainDb", label="Output trim", minimum=-24, maximum=24,
            default=0, unit="dB", description="Saved preset level; monitor volume is separate.",
        )
    ],
    "nam_fx": [
        ParameterSpec(
            id="inputGain", label="Input trim", minimum=-24, maximum=24,
            default=0, unit="dB", description="Level into the fixed NAM capture, not its physical Drive knob.",
        ),
        *[
            ParameterSpec(
                id=key, label=label, minimum=-10, maximum=10,
                default=0, unit="dB", description="Soundshed post-capture tone EQ, not a physical pedal control.",
            )
            for key, label in (
                ("bass", "Bass EQ"),
                ("mid", "Mid EQ"),
                ("treble", "Treble EQ"),
                ("presence", "Presence EQ"),
            )
        ],
        ParameterSpec(
            id="outputGain", label="Output trim", minimum=-24, maximum=24,
            default=0, unit="dB", description="Level after the fixed NAM capture.",
        ),
        ParameterSpec(
            id="mix", label="Dry / wet", minimum=0, maximum=1,
            default=1, unit="ratio", description="Blend the capture with the input signal.",
        ),
    ],
    "nam_amp": [
        ParameterSpec(
            id="inputGain", label="Input trim", minimum=-24, maximum=24,
            default=0, unit="dB", description="Level into the fixed NAM capture, not an amp knob.",
        ),
        *[
            ParameterSpec(
                id=key, label=label, minimum=-10, maximum=10,
                default=0, unit="dB", description="Soundshed post-capture tone EQ, not a physical amp control.",
            )
            for key, label in (
                ("bass", "Bass EQ"),
                ("mid", "Mid EQ"),
                ("treble", "Treble EQ"),
                ("presence", "Presence EQ"),
            )
        ],
        ParameterSpec(
            id="outputGain", label="Output trim", minimum=-24, maximum=24,
            default=0, unit="dB", description="Level after the fixed NAM capture.",
        ),
        ParameterSpec(
            id="mix", label="Dry / wet", minimum=0, maximum=1,
            default=1, unit="ratio", description="Blend the capture with the input signal.",
        ),
    ],
    "cab_ir": [
        ParameterSpec(
            id="mix", label="Mix", minimum=0, maximum=1,
            default=1, unit="ratio", description="Blend the cabinet IR with the input signal.",
        ),
        ParameterSpec(
            id="outputGain", label="Output trim", minimum=-24, maximum=24,
            default=0, unit="dB", description="Level after the cabinet stage.",
        ),
        ParameterSpec(
            id="lowCutHz", label="Low cut", minimum=20, maximum=1000,
            default=85, unit="Hz", description="High-pass filter before output.",
        ),
        ParameterSpec(
            id="highCutHz", label="High cut", minimum=1000, maximum=20000,
            default=6500, unit="Hz", description="Low-pass filter before output.",
        ),
    ],
    "parametric_eq": [
        ParameterSpec(
            id=key, label=label, minimum=-12, maximum=12,
            default=0, unit="dB", description=f"Gain for the {label.lower()} band.",
        )
        for key, label in (
            ("lowGain", "Low"),
            ("lowMidGain", "Low-mid"),
            ("highMidGain", "High-mid"),
            ("highGain", "High"),
        )
    ],
    "room_reverb": [
        ParameterSpec(
            id="decay", label="Decay", minimum=0, maximum=1,
            default=0.46, unit="ratio", description="Room reverb decay amount.",
        ),
        ParameterSpec(
            id="size", label="Size", minimum=0, maximum=1,
            default=0.42, unit="ratio", description="Perceived room size.",
        ),
        ParameterSpec(
            id="damping", label="Damping", minimum=0, maximum=1,
            default=0.56, unit="ratio", description="High-frequency decay damping.",
        ),
        ParameterSpec(
            id="preDelay", label="Pre-delay", minimum=0, maximum=220,
            default=8, unit="ms", description="Delay before the room response.",
        ),
        ParameterSpec(
            id="mix", label="Mix", minimum=0, maximum=1,
            default=0.22, unit="ratio", description="Wet reverb level.",
        ),
    ],
    "digital_delay": [
        ParameterSpec(
            id="time", label="Time", minimum=1, maximum=2000,
            default=300, unit="ms", description="Delay time; unsynced mode is fixed in the exported preset.",
        ),
        ParameterSpec(
            id="feedback", label="Feedback", minimum=0, maximum=0.95,
            default=0.4, unit="ratio", description="Amount of delayed signal repeated.",
        ),
        ParameterSpec(
            id="mix", label="Mix", minimum=0, maximum=1,
            default=0.3, unit="ratio", description="Wet delay level.",
        ),
    ],
    "chorus": [
        ParameterSpec(
            id="rate", label="Rate", minimum=0.1, maximum=10,
            default=0.5, unit="Hz", description="Chorus modulation rate.",
        ),
        ParameterSpec(
            id="depth", label="Depth", minimum=0, maximum=20,
            default=2, unit="ms", description="Chorus modulation depth.",
        ),
        ParameterSpec(
            id="delay", label="Delay", minimum=1, maximum=30,
            default=15, unit="ms", description="Base delay time.",
        ),
        ParameterSpec(
            id="feedback", label="Feedback", minimum=0, maximum=0.95,
            default=0.1, unit="ratio", description="Amount of modulated signal repeated.",
        ),
        ParameterSpec(
            id="mix", label="Mix", minimum=0, maximum=1,
            default=0.3, unit="ratio", description="Wet chorus level.",
        ),
    ],
    "gain": [
        ParameterSpec(
            id="gainDb", label="Gain", minimum=-24, maximum=24,
            default=0, unit="dB", description="Linear utility gain.",
        )
    ],
    "gear_note": [],
}

_BLOCK_LABELS = {
    "input": ("Input", "Fixed serial-chain input."),
    "nam_fx": ("Pedal NAM", "A real pedal capture. Capture Drive/Tone settings are fixed; controls are surrounding trims and post-capture EQ."),
    "nam_amp": ("Amp NAM", "A real amp capture. Amp controls are fixed in the capture; controls are surrounding trims and post-capture EQ."),
    "cab_ir": ("Cabinet IR", "A local impulse response; the referenced audio is not bundled."),
    "parametric_eq": ("Parametric EQ", "Soundshed 4-band parametric EQ."),
    "room_reverb": ("Room reverb", "Soundshed room reverb; not a claim about identified physical gear."),
    "digital_delay": ("Digital delay", "Soundshed stereo digital delay."),
    "chorus": ("Chorus", "Soundshed chorus modulation."),
    "gain": ("Gain", "Soundshed linear utility gain."),
    "gear_note": ("Gear note", "A positional note for missing or unsupported equipment; it is not an audio processor."),
    "output": ("Output", "Fixed serial-chain output."),
}

_EFFECT_TYPES = {
    "nam_fx": _FX_NAM,
    "nam_amp": _AMP_NAM,
    "cab_ir": _CAB_IR,
    "parametric_eq": _EQ_PARAMETRIC,
    "room_reverb": _REVERB_ROOM,
    "digital_delay": _DELAY_DIGITAL,
    "chorus": _CHORUS,
    "gain": _GAIN,
}

_FIXED_PARAMS: dict[str, dict[str, float]] = {
    "nam_fx": {"useCalibration": 0.0},
    "nam_amp": {"useCalibration": 0.0},
    "cab_ir": {"irBlend": 0.0, "air": 0.0, "quality": 1.0},
    "parametric_eq": {
        "lowFreq": 100.0,
        "lowQ": 0.707,
        "lowMidFreq": 400.0,
        "lowMidQ": 1.0,
        "highMidFreq": 2000.0,
        "highMidQ": 1.0,
        "highFreq": 8000.0,
        "highQ": 0.707,
        "lowShape": 0.0,
        "highShape": 0.0,
    },
    "digital_delay": {
        "syncMode": 0.0,
        "syncDivision": 4.0,
        "highCut": 8000.0,
        "lowCut": 20.0,
        "drive": 0.0,
        "stereoMode": 0.0,
        "spread": 0.0,
        "modRate": 0.0,
        "modDepth": 0.0,
        "ducking": 0.0,
        "direction": 0.0,
    },
    "chorus": {"syncMode": 0.0, "syncDivision": 4.0},
}

_DEFAULT_ASSET_IDS = {
    "pedal": ("nam", "tone3000:383084"),
    "amp": (
        "nam",
        "Preset-Pack-Vol.-1__fca3d4b65aab6cd846a939c0ea185d97f009c315c8cdf8ebe7e14e1124f76e97",
    ),
    "cabinet": (
        "ir",
        "Preset-Pack-Vol.-1__146e69d17d67dda236f720bac6cba5f530f7c671bb5450979026e5e14050a26d",
    ),
}


class AdapterError(ValueError):
    """A chain, installed resource, or target state is not safe to use."""


@dataclass(frozen=True)
class ResolvedResource:
    asset: AssetRef
    path: Path
    provider: str
    file_size_bytes: int


class SoundshedAdapter:
    adapter_id = ADAPTER_ID

    def __init__(
        self,
        resource_root: Path | None = None,
        database_path: Path | None = None,
        engine_version: str | None = None,
    ) -> None:
        data_root = Path.home() / "Library" / "Soundshed Guitar" / "data" / "v1"
        self.resource_root = (resource_root or data_root / "resources").resolve()
        self.database_path = database_path or data_root / "soundshed.db"
        self._engine_version = engine_version
        self._hash_cache: dict[tuple[str, int, int], str] = {}
        self._records: dict[str, dict[str, object]] | None = None

    @property
    def engine_version(self) -> str:
        return self._engine_version or "unavailable"

    def set_engine_version(self, value: str) -> None:
        self._engine_version = value

    def catalog(self) -> AdapterCatalog:
        resources = []
        for record_id, record in self._resource_records().items():
            resolved = self._resolve_record(record_id, record)
            if resolved:
                resources.append(
                    ResourceDefinition(
                        asset=resolved.asset,
                        provider=resolved.provider,
                        file_size_bytes=resolved.file_size_bytes,
                    )
                )
        blocks = [
            BlockDefinition(
                kind=kind,
                label=label,
                description=description,
                parameters=_PARAMETERS[kind],
            )
            for kind, (label, description) in _BLOCK_LABELS.items()
        ]
        return AdapterCatalog(
            adapter_id=self.adapter_id,
            adapter_label="Soundshed Guitar",
            engine_version=self.engine_version,
            blocks=blocks,
            resources=sorted(resources, key=lambda item: (item.asset.resource_type, item.asset.display_name.casefold())),
        )

    def self_test_chain(self) -> ChainDraft:
        """Fixed pinned resources for DSP canaries, never reference generation."""
        analysis: Sequence[object] = ()
        records = self._resource_records()
        nodes = [
            ChainNode(id="input", kind="input", title="Input", params={"gainDb": 0.0}),
        ]
        stages = (
            ("pedal", "nam_fx", "BOSS BD-2w pedal NAM", {"inputGain": 0, "bass": 0, "mid": 0, "treble": 0, "presence": 0, "outputGain": 0, "mix": 1}),
            ("amp", "nam_amp", "Two-Rock SSS amp NAM", {"inputGain": 0, "bass": 0, "mid": 0, "treble": 0, "presence": 0, "outputGain": 0, "mix": 1}),
            ("cabinet", "cab_ir", "Two Rocks 2x12 cabinet IR", {"mix": 1, "outputGain": 0, "lowCutHz": 85, "highCutHz": 6500}),
        )
        for role, kind, title, params in stages:
            asset_type, resource_id = _DEFAULT_ASSET_IDS[role]
            record = records.get(f"{asset_type}:{resource_id}")
            resolved = self._resolve_record(f"{asset_type}:{resource_id}", record) if record else None
            if resolved:
                nodes.append(
                    ChainNode(
                        id=role,
                        kind=kind,
                        title=f"{title} · capture controls fixed",
                        params=params,
                        asset=resolved.asset,
                    )
                )
            else:
                nodes.append(
                    ChainNode(
                        id=f"{role}-note",
                        kind="gear_note",
                        title=f"Missing local {role} resource",
                        note=f"Expected the exact Soundshed library resource {resource_id}; no substitute was selected.",
                    )
                )
        for kind, effect, title, params in (
            ("chorus", "modulation", "Subtle chorus", {"rate": 0.5, "depth": 2, "delay": 15, "feedback": 0, "mix": 0.08}),
            ("digital_delay", "delay", "Subtle digital delay", {"time": 300, "feedback": 0.25, "mix": 0.1}),
            ("room_reverb", "reverb", "Subtle room ambience", {"decay": 0.25, "size": 0.3, "damping": 0.6, "preDelay": 12, "mix": 0.08}),
        ):
            presence, note = self._effect_observation(analysis, effect)
            if presence == "present" and (kind != "chorus" or "chorus" in note.casefold()):
                nodes.append(
                    ChainNode(
                        id=f"effect-{effect}",
                        kind=kind,
                        title=f"{title} · generic approximation",
                        params=params,
                    )
                )
            elif presence in ("present", "possible"):
                explanation = (
                    "Audible effect was reported, but its specific type is not established; no substitute was selected."
                    if effect == "modulation"
                    else "An effect may be audible, but no processor was added without a clear observation."
                )
                nodes.append(
                    ChainNode(
                        id=f"note-{effect}",
                        kind="gear_note",
                        title=f"{effect.title()} needs review",
                        note=f"{explanation} Analysis note: {note or 'no supporting details'}",
                    )
                )
        nodes.append(ChainNode(id="output", kind="output", title="Output", params={"gainDb": 0.0}))
        return ChainDraft(revision=1, nodes=nodes)

    def default_chain(self, analysis: Sequence[object] = ()) -> ChainDraft:
        """Propose one selected audible profile; never blend different intervals."""
        segment = analysis[0] if analysis else None
        profile = (
            segment.get("profile") if isinstance(segment, dict)
            else getattr(segment, "profile", None)
        )
        if isinstance(profile, dict):
            profile = ToneProfile.model_validate(profile)
        nodes = [ChainNode(id="input", kind="input", title="Input", params={"gainDb": 0})]
        if not isinstance(profile, ToneProfile) or profile.gain == "uncertain":
            nodes.append(ChainNode(
                id="target-note", kind="gear_note", title="Reference tone needs analysis",
                note="No reliable selected gain profile is available. Analyze a guitar-dominant interval or choose equipment manually; no reference recipe was substituted.",
            ))
        else:
            # Curated LOCAL capture candidates, not identification of the original rig.
            # Capture Drive/Tone remain fixed. All adjustable values are Soundshed trims/EQ.
            recipes = {
                "clean": (None, "tone3000:494014", "fender"),
                "edge-of-breakup": (None, "tone3000:758458", "greenback"),
                "crunch": (None, "tone3000:758459", "greenback"),
                "overdrive": ("tone3000:383084", "tone3000:494017", "greenback"),
                "distortion": ("Preset-Pack-Vol.-1__0a1b65893aa6ea2bb729b670f6ecf363bd8ca18b1aa1f710a98c2b4fd80a1be3",
                               "Preset-Pack-Vol.-1__a6fd4ff796c110dd8b7f7dc0918f8b9ddef3b256d519fe7df4c0d676e91f86a5", "mesa"),
                "high-gain": ("Preset-Pack-Vol.-1__56d562b14a717370ea108d272d4734b999d861e3d70e3d803f58f4206ede2853",
                              "Preset-Pack-Vol.-1__a6fd4ff796c110dd8b7f7dc0918f8b9ddef3b256d519fe7df4c0d676e91f86a5", "mesa"),
            }
            pedal, amp, cabinet = recipes[profile.gain]
            records = self._resource_records()
            def add_capture(role: str, kind: str, resource_id: str | None):
                if resource_id is None:
                    return
                record_id = f"nam:{resource_id}"
                record = records.get(record_id)
                resolved = self._resolve_record(record_id, record) if record else None
                if resolved is None:
                    nodes.append(ChainNode(
                        id=f"{role}-note", kind="gear_note", title=f"Missing {profile.gain} {role} candidate",
                        note=f"This {profile.gain} starting recipe requires an unavailable local {role} capture. Select a suitable local capture manually; no alternate rig was substituted.",
                    ))
                    return
                nodes.append(ChainNode(
                    id=role, kind=kind, title=f"{resolved.asset.display_name[:75]} (candidate)",
                    asset=resolved.asset, params={
                        "inputGain": {"clean": -6, "edge-of-breakup": -3, "crunch": 0,
                                      "overdrive": 0, "distortion": 0, "high-gain": 3}[profile.gain],
                        "outputGain": 0, "mix": 1, "bass": 0, "mid": 0,
                        "treble": 0, "presence": 0,
                    },
                ))
            add_capture("pedal", "nam_fx", pedal)
            add_capture("amp", "nam_amp", amp)
            # Only curated cabinet IRs, never a room IR or arbitrary catalog first item.
            wanted = {"fender": "Fender Deluxe Reverb", "greenback": "Celestion Greenback",
                      "mesa": "Mesa OS 4x12"}[cabinet]
            candidates = [
                self._resolve_record(key, record) for key, record in records.items()
                if record.get("type") == "ir" and record.get("category") == "cab"
                and str(record.get("name", "")).startswith(wanted)
            ]
            resolved_cab = next((item for item in candidates if item is not None), None)
            if resolved_cab is None:
                nodes.append(ChainNode(
                    id="cabinet-note", kind="gear_note", title=f"Missing {cabinet} cabinet candidate",
                    note="The matching curated cabinet IR is unavailable locally. Choose a cabinet IR explicitly; no substitute was selected.",
                ))
            else:
                nodes.append(ChainNode(
                    id="cabinet", kind="cab_ir", title=f"{cabinet.title()} cabinet IR (candidate)",
                    asset=resolved_cab.asset, params={
                        "mix": 1, "outputGain": 0,
                        "lowCutHz": {"loose": 65, "neutral": 85, "tight": 120, "uncertain": 85}[profile.bass_tightness],
                        "highCutHz": {"dark": 4000, "slightly-dark": 5000, "balanced": 6500,
                                      "slightly-bright": 8000, "bright": 10000, "uncertain": 6500}[profile.brightness],
                    },
                ))
            nodes.append(ChainNode(
                id="profile-eq", kind="parametric_eq", title="Selected profile EQ direction",
                params={
                    "lowGain": {"low": -3, "typical": 0, "high": 3, "uncertain": 0}[profile.bass_level],
                    "lowMidGain": {"scooped": -3, "neutral": 0, "present": 2, "forward": 3, "uncertain": 0}[profile.mids],
                    "highMidGain": {"scooped": -2, "neutral": 0, "present": 1, "forward": 2, "uncertain": 0}[profile.mids],
                    "highGain": {"dark": -4, "slightly-dark": -2, "balanced": 0,
                                 "slightly-bright": 2, "bright": 4, "uncertain": 0}[profile.brightness],
                },
            ))
            for field, kind, params in (
                ("reverb", "room_reverb", {"decay": 0.3, "size": 0.35, "damping": 0.6, "preDelay": 12}),
                ("delay", "digital_delay", {"time": 300, "feedback": 0.25}),
            ):
                observation = getattr(profile, field)
                if observation.presence == "present" and observation.prominence != "uncertain":
                    params["mix"] = {"none": 0, "subtle": 0.08, "moderate": 0.18, "prominent": 0.3}[observation.prominence]
                    nodes.append(ChainNode(
                        id=field, kind=kind, title=f"Generic {field} candidate", params=params,
                    ))
                elif observation.presence in ("possible", "uncertain", "present"):
                    nodes.append(ChainNode(
                        id=f"{field}-note", kind="gear_note", note_role="omitted_optional",
                        title=f"{field.title()} omitted (uncertain)",
                        note=(
                            f"Analysis reported {field} as {observation.presence}/{observation.prominence}. "
                            "This candidate deliberately OMITS it: no effect is rendered or exported here, "
                            "and none was invented. Add one explicitly if you hear it."
                        ),
                    ))
            if (
                profile.modulation.presence == "present"
                and profile.modulation_type == "chorus"
                and profile.modulation.prominence not in ("none", "uncertain")
            ):
                nodes.append(ChainNode(
                    id="chorus", kind="chorus", title="Generic chorus candidate",
                    params={"rate": 0.5, "depth": 2, "delay": 15, "feedback": 0,
                            "mix": {"subtle": 0.08, "moderate": 0.18, "prominent": 0.3}[profile.modulation.prominence]},
                ))
            elif profile.modulation.presence != "absent":
                nodes.append(ChainNode(
                    id="modulation-note", kind="gear_note", note_role="omitted_optional",
                    title="Modulation omitted (unsupported/uncertain)",
                    note=f"Reported modulation family: {profile.modulation_type}. Only clearly reported chorus is mapped in this adapter; this candidate omits modulation and nothing was substituted. Choose supported equipment manually.",
                ))
            if profile.guitar_dominance != "clear":
                nodes.append(ChainNode(
                    id="mix-note", kind="gear_note", title="Full-mix attribution needs review",
                    note="Guitar is not clearly separable in the selected interval. This candidate is inconclusive; select a clearer interval before treating it as a match.",
                ))
        nodes.append(ChainNode(id="output", kind="output", title="Output", params={"gainDb": 0}))
        return ChainDraft(revision=1, nodes=nodes)

    def resolve_asset(self, asset: AssetRef) -> ResolvedResource:
        record_id = f"{asset.resource_type}:{asset.resource_id}"
        record = self._resource_records().get(record_id)
        if record is None:
            raise AdapterError(f"Local Soundshed resource is unavailable: {asset.display_name}")
        resolved = self._resolve_record(record_id, record)
        if resolved is None:
            raise AdapterError(f"Local Soundshed resource file is missing: {asset.display_name}")
        if resolved.asset.sha256 != asset.sha256:
            raise AdapterError(f"Local Soundshed resource changed since selection: {asset.display_name}")
        return resolved

    def validate_chain(self, chain: ChainDraft, *, for_render: bool = True) -> list[str]:
        errors: list[str] = []
        for node in chain.nodes:
            if node.kind not in ("input", "output", "gear_note") and node.id in (
                "__input__", "__output__"
            ):
                errors.append(f"{node.title}: node id is reserved for a Soundshed boundary")
            if node.kind == "gear_note":
                if for_render and node.note_role == "blocking":
                    errors.append(f"{node.title}: unresolved gear note prevents rendering")
                continue
            specs = {spec.id: spec for spec in _PARAMETERS[node.kind]}
            for key, value in node.params.items():
                spec = specs.get(key)
                if spec is None:
                    errors.append(f"{node.title}: unsupported parameter")
                elif not isinstance(value, (int, float)) or not float("-inf") < value < float("inf"):
                    errors.append(f"{node.title}: {key} must be finite")
                elif value < spec.minimum or value > spec.maximum:
                    errors.append(
                        f"{node.title}: {key} must be between {spec.minimum:g} and {spec.maximum:g} {spec.unit}"
                    )
            if node.kind in ("nam_fx", "nam_amp", "cab_ir") and node.asset:
                expected_type = "ir" if node.kind == "cab_ir" else "nam"
                if node.asset.resource_type != expected_type:
                    errors.append(f"{node.title}: resource type must be {expected_type}")
                elif node.kind == "nam_fx" and node.asset.capture_scope not in ("pedal", "unknown"):
                    errors.append(f"{node.title}: a pedal NAM capture is required")
                elif node.kind == "nam_amp" and node.asset.capture_scope not in ("amp", "unknown"):
                    errors.append(f"{node.title}: an amp NAM capture is required")
                else:
                    try:
                        self.resolve_asset(node.asset)
                    except AdapterError as exc:
                        errors.append(str(exc))
        if self._engine_version and self._engine_version != SUPPORTED_ENGINE_VERSION:
            errors.append(
                f"Soundshed {self._engine_version} is not validated; this adapter is pinned to "
                f"{SUPPORTED_ENGINE_VERSION}"
            )
        return errors

    def host_state(self, project: ToneProject, trailer: bytes) -> bytes:
        errors = self.validate_chain(project.chain)
        if errors:
            raise AdapterError("; ".join(errors))
        if not trailer.startswith(b"\0" * 8 + b"JUCEPrivateData") or not 32 <= len(trailer) <= 512:
            raise AdapterError("The installed Soundshed version returned an unsupported JUCE state trailer")
        preset = self._preset(project)
        # Runtime paths are resolved by this adapter, never supplied by an API caller.
        # The worker can therefore load resources without reading the profile database.
        for node in preset["scenes"][0]["graph"]["nodes"]:
            source = next((item for item in project.chain.nodes if item.id == node["id"]), None)
            if source is not None and source.asset is not None:
                node["resources"][0]["filePath"] = str(self.resolve_asset(source.asset).path)
        return self.state_for_preset(preset, trailer)

    def state_for_preset(self, preset: dict[str, object], trailer: bytes) -> bytes:
        """Wrap an adapter-built preset with explicit single-slot host conditions."""
        state = {
            "version": 1,
            "preset": preset,
            "presetId": preset["id"],
            "activeSceneId": "main",
            "appSettings": {},
            "uiSettings": {},
            "uiViewState": {},
            "globalSignalChain": self._isolated_global_chain(),
            "namQuality": {
                "slimmableSize": 1.0,
                "oversampling": 0,
                "antiAliasPhase": 0,
            },
            "mixer": {
                "masterGain": 1.0,
                "mixGainDb": 0.0,
                "limiterEnabled": False,
                "monoOutputUnity": False,
                "activePresetIds": [preset["id"]],
                "presets": {
                    preset["id"]: {
                        "name": preset["name"],
                        "mix": 1.0,
                        "pan": 0.0,
                        "mute": False,
                        "solo": False,
                    }
                },
                "presetData": {},
            },
            "automation": {},
            "automationValues": {},
        }
        return json.dumps(state, separators=(",", ":"), ensure_ascii=False).encode("utf-8") + trailer

    def export_archive(self, project: ToneProject, preview: PreviewSnapshot) -> bytes:
        chain_sha = self.chain_sha256(project.chain)
        if (
            preview.project_id != project.id
            or preview.project_revision != project.revision
            or preview.chain_revision != project.chain.revision
            or preview.chain_sha256 != chain_sha
            or preview.engine_version != self.engine_version
        ):
            raise AdapterError("The requested chain revision has not been previewed by this engine")
        errors = self.validate_chain(project.chain)
        if errors:
            raise AdapterError("; ".join(errors))

        payload = {
            "formatVersion": 1,
            "preset": self._preset(project, preview),
            "resources": [],
        }
        memory = io.BytesIO()
        with zipfile.ZipFile(memory, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                "preset.json",
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            )
        return memory.getvalue()

    @staticmethod
    def chain_sha256(chain: ChainDraft) -> str:
        dumped = chain.model_dump(mode="json")
        for node in dumped["nodes"]:
            # Pre-existing chains keep their hashes; only a non-default role is hashed.
            if node.get("note_role") == "blocking":
                del node["note_role"]
        content = json.dumps(dumped, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _resource_records(self) -> dict[str, dict[str, object]]:
        if self._records is not None:
            return self._records
        records: dict[str, dict[str, object]] = {}
        if not self.database_path.is_file():
            self._records = records
            return records
        try:
            connection = sqlite3.connect(
                self.database_path.as_uri() + "?mode=ro",
                uri=True,
                timeout=2,
            )
            connection.execute("PRAGMA query_only=ON")
            rows = connection.execute(
                "SELECT id, json FROM items WHERE type = ? "
                "AND (id LIKE 'nam:%' OR id LIKE 'ir:%') ORDER BY id LIMIT 2000",
                ("resource",),
            )
            for resource_id, record_json in rows:
                try:
                    parsed = json.loads(record_json)
                except (TypeError, json.JSONDecodeError):
                    continue
                if isinstance(parsed, dict) and parsed.get("type") in ("nam", "ir"):
                    records[resource_id] = parsed
            connection.close()
        except sqlite3.Error as exc:
            raise AdapterError("Could not read the local Soundshed resource catalog") from exc
        self._records = records
        return records

    def _resolve_record(
        self, record_id: str, record: dict[str, object]
    ) -> ResolvedResource | None:
        resource_type = record.get("type")
        if resource_type not in ("nam", "ir"):
            return None
        relative = record.get("filePath")
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            return None
        try:
            path = (self.resource_root / relative).resolve(strict=True)
        except (OSError, RuntimeError):
            return None
        if not path.is_relative_to(self.resource_root) or not path.is_file():
            return None
        stat = path.stat()
        if stat.st_size <= 0 or stat.st_size > 512 * 1024 * 1024:
            return None
        digest = self._sha256(path, stat.st_size, stat.st_mtime_ns)
        metadata = record.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        if resource_type == "nam":
            scope = metadata.get("captureScope", record.get("category", "unknown"))
            capture_scope = scope if scope in ("pedal", "amp") else "unknown"
        else:
            capture_scope = "unknown"
        name = record.get("name")
        display_name = name if isinstance(name, str) and name else record_id
        asset = AssetRef(
            resource_type=resource_type,
            resource_id=record_id.split(":", 1)[1],
            sha256=digest,
            display_name=display_name[:200],
            capture_scope=capture_scope,
        )
        provider = metadata.get("provider", "local library")
        return ResolvedResource(
            asset=asset,
            path=path,
            provider=provider if isinstance(provider, str) else "local library",
            file_size_bytes=stat.st_size,
        )

    def _sha256(self, path: Path, size: int, modified_ns: int) -> str:
        key = (str(path), size, modified_ns)
        cached = self._hash_cache.get(key)
        if cached:
            return cached
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        value = digest.hexdigest()
        self._hash_cache[key] = value
        return value

    def _preset(
        self, project: ToneProject, preview: PreviewSnapshot | None = None
    ) -> dict[str, object]:
        rendered_nodes = []
        graph_order = []
        for node in project.chain.nodes:
            if node.kind == "gear_note":
                continue
            graph_order.append(node)
        for node in graph_order:
            if node.kind == "input":
                rendered_nodes.append(
                    {
                        "id": "__input__",
                        "type": "input",
                        "category": "utility",
                        "label": "Input",
                        "enabled": node.enabled,
                        "params": {"gainDb": node.params.get("gainDb", 0.0)},
                    }
                )
                continue
            if node.kind == "output":
                rendered_nodes.append(
                    {
                        "id": "__output__",
                        "type": "output",
                        "category": "utility",
                        "label": "Output",
                        "enabled": node.enabled,
                        "params": {"gainDb": node.params.get("gainDb", 0.0)},
                    }
                )
                continue
            params = dict(_FIXED_PARAMS.get(node.kind, {}))
            for spec in _PARAMETERS[node.kind]:
                params[spec.id] = node.params.get(spec.id, spec.default)
            resources = []
            if node.asset:
                self.resolve_asset(node.asset)
                resources = [
                    {
                        "resourceType": node.asset.resource_type,
                        "resourceId": node.asset.resource_id,
                    }
                ]
            rendered_nodes.append(
                {
                    "id": node.id,
                    "type": _EFFECT_TYPES[node.kind],
                    "category": self._category(node.kind),
                    "label": node.title,
                    "title": node.title,
                    "enabled": node.enabled,
                    "params": params,
                    "resources": resources,
                }
            )
        edges = [
            {"from": left["id"], "to": right["id"]}
            for left, right in itertools.pairwise(rendered_nodes)
        ]
        description = self._preset_description(project, preview)
        from ..privacy import redact_paths
        return redact_paths({
            "id": project.id,
            "name": project.name,
            "author": "Local Tone Workbench",
            "category": "Tone Workbench",
            "description": description,
            "version": 2,
            "createdAt": project.created_at,
            "modifiedAt": project.updated_at,
            "tags": ["tone-workbench", "synthetic-fixture" if project.synthetic_fixture else "analysis-derived" if (
                project.analysis[project.selected_analysis_index].listening is not None
                or project.analysis[project.selected_analysis_index].profile is not None
            ) else "manual-starting-point"],
            "scenes": [
                {
                    "id": "main",
                    "title": "Main",
                    "graph": {"nodes": rendered_nodes, "edges": edges},
                }
            ],
        })

    @staticmethod
    def _category(kind: str) -> str:
        if kind in ("nam_fx", "nam_amp"):
            return "amp"
        if kind == "cab_ir":
            return "cab"
        if kind in ("room_reverb", "digital_delay", "chorus"):
            return "effects"
        if kind == "parametric_eq":
            return "eq"
        return "utility"

    def _preset_description(
        self, project: ToneProject, preview: PreviewSnapshot | None
    ) -> str:
        source = project.source
        intervals = ", ".join(
            f"{item.start_seconds:g}-{item.end_seconds:g}s"
            for item in source.selected_intervals
        )
        source_line = source.url if source.kind == "youtube" else source.display_name
        analyses = [
            segment.listening.summary
            for segment in project.analysis
            if segment.listening is not None
        ]
        gear_notes = [
            (
                f"OMITTED optional effect at position {index + 1} / {node.title} (not in this preset or preview): {node.note}"
                if node.note_role == "omitted_optional"
                else f"Position {index + 1} / {node.title}: {node.note}"
            )
            for index, node in enumerate(project.chain.nodes)
            if node.note
        ]
        successful_models = sorted({
            segment.model for segment in [project.analysis[project.selected_analysis_index]]
            if segment.listening is not None or segment.profile is not None
        })
        if project.synthetic_fixture:
            successful_models = []
        lines = [
            (
                "OFFLINE SYNTHETIC FIXTURE: invented categorical profiles; no reference was listened to or validated."
                if project.synthetic_fixture else "Generated as an analysis-informed starting point."
                if successful_models else
                "Manual starting chain: reference analysis was unavailable; no reference tone was inferred."
            ) + " This does not identify or reproduce the original player's rig.",
            f"Source: {source_line}; selected intervals: {intervals}.",
            f"Recipe target: interval {project.selected_analysis_index + 1} only; other profiles were not blended.",
            f"Successful analysis model(s): {', '.join(successful_models) or 'none'}.",
            "NAM captures and cabinet IRs are referenced from the existing local Soundshed library; no audio resources are included. This preset is not portable without those resources.",
            "Preview conditions: a single allowlisted preset, no user mixer slots, no global FX, no calibration, full NAM model size, 44.1 kHz, and unity host mixer gain. The shared Soundshed profile was not modified. The adapter reads local resource records and resource files read-only.",
        ]
        lines.extend(f"Audible analysis: {item}" for item in analyses)
        profile = project.analysis[project.selected_analysis_index].profile
        if profile is not None:
            lines.append(
                f"{'Synthetic test profile' if project.synthetic_fixture else 'Selected audible profile'}: gain={profile.gain}, brightness={profile.brightness}, "
                f"bass={profile.bass_level}/{profile.bass_tightness}, mids={profile.mids}, "
                f"attack={profile.attack}, compression={profile.compression}, sustain={profile.sustain}."
            )
        lines.extend(gear_notes)
        if preview is not None:
            if preview.dsp_conditions.get("state_canary_version") != 1:
                raise AdapterError("Renderer state-sensitivity validation is missing; render again")
            lines.append(
                f"Preview snapshot {preview.id}; engine {preview.engine_version}; "
                f"chain SHA-256 {preview.chain_sha256}; source DI SHA-256 {preview.source_sample_sha256}."
            )
        from ..privacy import redact_paths
        return redact_paths("\n".join(lines))

    @staticmethod
    def _effect_observation(analysis: Sequence[object], effect: str) -> tuple[str, str]:
        observations = []
        for segment in analysis:
            listening = (
                segment.get("listening")
                if isinstance(segment, dict)
                else getattr(segment, "listening", None)
            )
            if hasattr(listening, "model_dump"):
                listening = listening.model_dump(mode="json")
            if not isinstance(listening, dict):
                continue
            effect_value = listening.get("effects", {}).get(effect, {})
            if not isinstance(effect_value, dict):
                continue
            presence = effect_value.get("presence")
            if presence in ("possible", "present"):
                observations.append((presence, str(effect_value.get("notes", ""))))
        if any(value == "present" for value, _ in observations):
            presence = "present"
        elif observations:
            presence = "possible"
        else:
            presence = "absent"
        note = "; ".join(value for _, value in observations if value)
        return presence, note

    @staticmethod
    def _isolated_global_chain() -> dict[str, object]:
        def make_graph(
            stages: list[tuple[str, str]],
        ) -> dict[str, object]:
            nodes = [
                {"id": "__input__", "type": "input", "category": "utility", "enabled": True, "params": {}}
            ]
            nodes.extend(
                {
                    "id": node_id,
                    "type": effect_type,
                    "category": "utility",
                    "enabled": False,
                    "params": {},
                }
                for node_id, effect_type in stages
            )
            nodes.append(
                {"id": "__output__", "type": "output", "category": "utility", "enabled": True, "params": {}}
            )
            edges = [
                {"from": left["id"], "to": right["id"]}
                for left, right in itertools.pairwise(nodes)
            ]
            return {"nodes": nodes, "edges": edges}

        return {
            "preChainGraph": make_graph(
                [
                    ("global_gate", "e8388de1-d262-4123-a123-8dbc56f657bc"),
                    ("global_pickup_match", "df1f4ed4-5d84-4a21-9474-ad6ab014fca5"),
                    ("global_transpose", "9b89cc46-e05b-4f06-981e-1d74d1f628cf"),
                ]
            ),
            "postChainGraph": make_graph(
                [
                    ("global_eq", _EQ_PARAMETRIC),
                    ("global_doubler", "778aaef4-40e3-4efa-8782-6a8bfa1d1661"),
                ]
            ),
            "inputGain": 0.0,
            "monoMode": False,
            "inputChannel": 0,
            "dualMono": False,
            "outputGain": 0.0,
            "limiterEnabled": False,
        }


def new_project(
    source: SourceRef,
    analysis: list[dict[str, object]],
    intent: dict[str, object],
    chain: ChainDraft,
    engine_version: str,
    name: str,
) -> ToneProject:
    now = datetime.now(UTC).isoformat()
    from tone_listener_mcp.workbench.models import (
        AnalysisSegment,
        TargetBinding,
        ToneIntent,
    )

    segments = [AnalysisSegment.model_validate(item) for item in analysis]
    binding_assets = {
        node.id: node.asset
        for node in chain.nodes
        if node.asset is not None
    }
    return ToneProject(
        name=name,
        source=source,
        analysis=segments,
        intent=ToneIntent.model_validate(intent),
        chain=chain,
        target=TargetBinding(
            adapter_id=ADAPTER_ID,
            engine_version=engine_version,
            resources=binding_assets,
        ),
        created_at=now,
        updated_at=now,
    )
