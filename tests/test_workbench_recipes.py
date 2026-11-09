import json

import pytest

from tone_listener_mcp.listener import ToneProfile
from tone_listener_mcp.workbench.adapters.soundshed import SoundshedAdapter


def profile(**values):
    effect = {"presence": "absent", "prominence": "none"}
    return ToneProfile.model_validate({
        "guitar_dominance": "clear", "gain": "clean", "brightness": "balanced",
        "bass_level": "typical", "bass_tightness": "neutral", "mids": "neutral",
        "attack": "balanced", "compression": "dynamic", "sustain": "typical",
        "reverb": effect, "delay": effect, "modulation": effect,
        "evidence": [], "uncertainties": [], **values,
    })


@pytest.fixture
def adapter(tmp_path):
    root = tmp_path / "resources"
    root.mkdir()
    ids = [
        ("nam", "tone3000:494014", "VOX clean candidate", "amp"),
        ("nam", "tone3000:758459", "Plexi crunch candidate", "amp"),
        ("nam", "Preset-Pack-Vol.-1__56d562b14a717370ea108d272d4734b999d861e3d70e3d803f58f4206ede2853",
         "Ubermetal candidate", "pedal"),
        ("nam", "Preset-Pack-Vol.-1__a6fd4ff796c110dd8b7f7dc0918f8b9ddef3b256d519fe7df4c0d676e91f86a5",
         "JCM gain10 candidate", "amp"),
        ("ir", "fender-ir", "Fender Deluxe Reverb test IR", "cab"),
        ("ir", "greenback-ir", "Celestion Greenback test IR", "cab"),
        ("ir", "mesa-ir", "Mesa OS 4x12 test IR", "cab"),
    ]
    records = {}
    for index, (kind, resource_id, name, category) in enumerate(ids):
        path = root / f"asset-{index}"
        path.write_bytes(f"Offline contract fixture {index}".encode())
        records[f"{kind}:{resource_id}"] = {
            "type": kind, "filePath": path.name, "name": name, "category": category,
            "metadata": {"captureScope": category if category in ("amp", "pedal") else "unknown"},
        }
    result = SoundshedAdapter(root, tmp_path / "absent.db", engine_version="1.7.3")
    result._records = records
    return result


def test_contrasting_profiles_change_core_resources_and_real_controls(adapter):
    clean = adapter.default_chain([{"profile": profile()}])
    dark_gain = adapter.default_chain([{"profile": profile(
        gain="high-gain", brightness="dark", bass_level="high", bass_tightness="tight",
        mids="scooped", compression="compressed",
    )}])
    bright_crunch = adapter.default_chain([{"profile": profile(
        gain="crunch", brightness="bright", mids="forward",
    )}])
    chains = [clean, dark_gain, bright_crunch]
    assert all(not adapter.validate_chain(chain) for chain in chains)
    amp_ids = [next(node.asset.resource_id for node in chain.nodes if node.kind == "nam_amp") for chain in chains]
    assert len(set(amp_ids)) == 3
    assert not any(node.kind == "nam_fx" for node in clean.nodes)
    assert any(node.kind == "nam_fx" for node in dark_gain.nodes)
    high_cuts = [next(node.params["highCutHz"] for node in chain.nodes if node.kind == "cab_ir") for chain in chains]
    assert high_cuts == [6500, 4000, 10000]
    eqs = [next(node.params for node in chain.nodes if node.kind == "parametric_eq") for chain in chains]
    assert eqs[0]["lowGain"] == 0 and eqs[1]["lowGain"] == 3
    assert eqs[1]["highGain"] == -4 and eqs[2]["highGain"] == 4


def test_unavailable_capture_is_positional_note_not_amane_fallback(adapter):
    del adapter._records["nam:tone3000:758459"]
    chain = adapter.default_chain([{"profile": profile(gain="crunch")}])
    assert any(node.kind == "gear_note" and node.id == "amp-note" for node in chain.nodes)
    assert not any(node.kind == "nam_amp" for node in chain.nodes)
    assert "383084" not in chain.model_dump_json()
    assert adapter.validate_chain(chain)


def test_second_interval_is_not_ored_into_selected_recipe(adapter):
    clean = {"profile": profile()}
    driven = {"profile": profile(
        gain="high-gain", reverb={"presence": "present", "prominence": "prominent"},
    )}
    selected = adapter.default_chain([clean, driven])
    assert not any(node.kind in ("nam_fx", "room_reverb") for node in selected.nodes)
    alternate = adapter.default_chain([driven])
    assert any(node.kind == "room_reverb" for node in alternate.nodes)


def test_failed_analysis_does_not_seed_universal_chain(adapter):
    chain = adapter.default_chain([{"unavailable": "Provider unavailable."}])
    assert [node.kind for node in chain.nodes] == ["input", "gear_note", "output"]
    assert all(node.asset is None for node in chain.nodes)


def test_modulation_is_not_silently_mapped_to_chorus(adapter):
    chain = adapter.default_chain([{"profile": profile(
        modulation={"presence": "present", "prominence": "moderate"},
    )}])
    assert any(node.id == "modulation-note" for node in chain.nodes)
    assert not any(node.kind == "chorus" for node in chain.nodes)


def test_clearly_reported_chorus_maps_to_supported_actual_effect(adapter):
    chain = adapter.default_chain([{"profile": profile(
        modulation={"presence": "present", "prominence": "moderate"},
        modulation_type="chorus",
    )}])
    assert any(node.kind == "chorus" for node in chain.nodes)
    assert not any(node.kind == "gear_note" for node in chain.nodes)


def test_typed_selected_profile_never_uses_prose_similarity(adapter):
    chain = adapter.default_chain([{"profile": json.loads(profile().model_dump_json())}])
    assert chain.nodes[1].asset is not None
    assert not any("Drive" in key or "Tone" in key for node in chain.nodes for key in node.params)
