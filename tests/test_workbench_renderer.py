import json
import os

import numpy as np
import pytest

from tone_listener_mcp.workbench.adapters.soundshed import SoundshedAdapter
from tone_listener_mcp.workbench.renderer import NativeRenderer, RenderError


def test_constant_default_audio_is_rejected_by_canary(monkeypatch):
    from scripts.verify_workbench_renderer import verification_project
    from tone_listener_mcp.workbench.models import AssetRef, ChainDraft, ChainNode

    asset = AssetRef(
        resource_type="nam", resource_id="test", sha256="a" * 64, display_name="Test",
    )
    chain = ChainDraft(nodes=[
        ChainNode(id="input", kind="input", title="Input"),
        ChainNode(id="pedal", kind="nam_fx", title="Pedal", asset=asset),
        ChainNode(id="amp", kind="nam_amp", title="Amp", asset=asset),
        ChainNode(id="cab", kind="cab_ir", title="Cab",
                  asset=asset.model_copy(update={"resource_type": "ir"})),
        ChainNode(id="output", kind="output", title="Output"),
    ])
    adapter = SoundshedAdapter()
    monkeypatch.setattr(adapter, "self_test_chain", lambda: chain)
    monkeypatch.setattr(adapter, "validate_chain", lambda value: [])
    monkeypatch.setattr(adapter, "host_state", lambda project, trailer: project.model_dump_json().encode())
    renderer = NativeRenderer(adapter)
    monkeypatch.setattr(
        renderer, "_render_raw",
        lambda state, mono, event=None: np.repeat(mono[:, None], 2, axis=1) / np.sqrt(2),
    )
    with pytest.raises(RenderError, match=r"\+6 dB state-sensitivity"):
        renderer._verify_state_application(b"", "1.7.3", None)
    assert not renderer.verified
    assert renderer.canary_results is None
    # No captured or unrelated user state is used in the fake contract.
    assert verification_project(adapter).analysis[0].listening is None


def test_old_snapshot_without_canary_is_not_verified():
    renderer = NativeRenderer(SoundshedAdapter())
    from types import SimpleNamespace
    snapshot = SimpleNamespace(dsp_conditions={})
    assert not renderer.verifies_snapshot(snapshot)
    renderer._trailer = b"test"
    renderer._canary = {}
    assert not renderer.verifies_snapshot(snapshot)


def test_parameter_save_validation_rejects_ranges_and_unknown_keys():
    from tone_listener_mcp.workbench.models import ChainDraft, ChainNode
    adapter = SoundshedAdapter()
    chain = ChainDraft(nodes=[
        ChainNode(id="input", kind="input", title="Input", params={"gainDb": 1e9}),
        ChainNode(id="output", kind="output", title="Output", params={"__proto__": 1}),
    ])
    errors = adapter.validate_chain(chain, for_render=False)
    assert any("between -24 and 24" in item for item in errors)
    assert any("unsupported parameter" in item for item in errors)
    assert "__proto__" not in json.dumps(errors)


@pytest.mark.native_integration
@pytest.mark.skipif(
    os.environ.get("TONE_WORKBENCH_NATIVE_TESTS") != "1",
    reason="Requires opt-in installed Soundshed and validated local resources",
)
def test_installed_worker_state_sensitivity_export_and_tails(tmp_path):
    from scripts.verify_workbench_renderer import run_verification
    report = run_verification(tmp_path)
    assert abs(report["checks"]["utility_gain_delta_db"] - 6) <= 0.1
    assert report["checks"]["export_fresh_worker_max_error"] == 0
