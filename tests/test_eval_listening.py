import json
import subprocess
import sys

import numpy as np
import pytest

from scripts.eval_listening import evaluate, load_credentials, prediction, run_split, score
from scripts.make_validation_set import generate
from tone_listener_mcp.audio_io import load_audio
from tone_listener_mcp.metrics import gated_power


def test_prediction_swap_invariance():
    assert prediction({"brighter": "A"}, "brighter", False) == "left"
    assert prediction({"brighter": "B"}, "brighter", True) == "left"
    assert prediction({"brighter": "uncertain"}, "brighter", True) == "uncertain"
    assert prediction(None, "brighter", False) is None
    payload = {axis: "similar" for axis in (
        "brighter", "more_saturated", "more_compressed", "more_ambient"
    )}
    assert prediction(payload, "tone_similarity", False) == "similar"
    payload["more_ambient"] = "uncertain"
    assert prediction(payload, "tone_similarity", False) == "uncertain"


def test_scoring_does_not_reward_unavailable_or_uncertain():
    rows = [
        {"pair": "eq", "repeat": 0, "expected": "right", "prediction": "right"},
        {"pair": "eq", "repeat": 0, "expected": "right", "prediction": "left"},
        {"pair": "drive", "repeat": 0, "expected": "right", "prediction": None},
        {"pair": "drive", "repeat": 0, "expected": "right", "prediction": "uncertain"},
    ]
    analyses = [{
        "results": [{"listening": {"gain_level": gain}}
                    for gain in ("clean", "crunch", "distortion")]
    }, {"results": [{"listening": None}] * 3}]
    result = score(rows, analyses)
    assert result["direction_accuracy"] == 0.25
    assert result["order_swap_consistency"] == 0
    assert result["gain_order_accuracy"] == 0.5
    assert result["unavailable_comparisons"] == 1


def test_no_key_skips_without_calls():
    result = evaluate(["tuning", "held_out"])
    assert result["status"] == "skipped"
    assert result["max_requests"] == 90
    assert result["splits"] == {}


@pytest.mark.parametrize("repeats", [0, 2, 6])
def test_repeat_bounds(repeats):
    with pytest.raises(ValueError, match="repeats"):
        evaluate(["tuning"], repeats)


def test_mocked_blind_split(monkeypatch):
    calls = []
    payload = {axis: "similar" for axis in (
        "brighter", "more_saturated", "more_compressed", "more_ambient"
    )}

    def compare(a, b):
        assert a.source == b.source == "blind"
        assert a.duration == b.duration
        assert gated_power(a)[0] == pytest.approx(gated_power(b)[0], rel=1e-4)
        calls.append((a, b))
        return {"listening": payload, "listening_unavailable": None}

    monkeypatch.setattr("scripts.eval_listening.listen_comparison", compare)
    monkeypatch.setattr("scripts.eval_listening.listen", lambda clip: {
        "listening": {"gain_level": "uncertain"}, "listening_unavailable": None,
    })
    result = run_split("tuning", 3, 0)
    assert result["status"] == "complete"
    assert len(calls) == 36
    assert len(result["analyses"]) == 3
    assert result["scores"]["direction_total"] == 36
    assert result["scores"]["order_swap_total"] == 18
    assert result["scores"]["direction_accuracy"] == pytest.approx(1 / 3)
    assert result["scores"]["gain_order_accuracy"] == 0
    assert all(sum(row["swapped"] for row in result["rows"] if row["pair"] == pair) == 3
               for pair in result["scores"]["per_pair"])


def test_blocked_api_stops_budget(monkeypatch):
    monkeypatch.setattr("scripts.eval_listening.listen_comparison", lambda a, b: {
        "listening": None, "listening_unavailable": "Gemini listening failed (ClientError)",
    })
    result = run_split("tuning", 3, 0)
    assert result["status"] == "blocked"
    assert len(result["rows"]) == 1
    assert result["scores"]["direction_accuracy"] == 0


def test_held_out_source_is_distinct_and_reproducible(tmp_path):
    tuning = generate(tmp_path / "tuning", seconds=1, seed=42)
    held_out = generate(tmp_path / "held", seconds=1, seed=90210, held_out=True)
    repeated = generate(tmp_path / "repeat", seconds=1, seed=90210, held_out=True)
    a = load_audio(str(tuning["reference"])).samples
    b = load_audio(str(held_out["reference"])).samples
    assert not np.array_equal(a, b)
    for name in held_out:
        np.testing.assert_array_equal(
            load_audio(str(held_out[name])).samples, load_audio(str(repeated[name])).samples
        )


def test_cli_skip_writes_external_report(tmp_path):
    report = tmp_path / "eval.json"
    result = subprocess.run(
        [sys.executable, "-m", "scripts.eval_listening", "--report", str(report)],
        check=True, capture_output=True, text=True, timeout=30,
    )
    assert json.loads(result.stdout)["status"] == "skipped"
    assert json.loads(report.read_text())["status"] == "skipped"


def test_literal_credentials(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("TONE_LISTENER_MODEL", "before")
    path = tmp_path / "env"
    path.write_text('# comments\nGEMINI_API_KEY="test-key"\nTONE_LISTENER_MODEL=test-model\n')
    path.chmod(0o600)
    load_credentials(path)
    import os

    assert os.environ["GEMINI_API_KEY"] == "test-key"
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.delenv("TONE_LISTENER_MODEL")


@pytest.mark.parametrize("text", [
    "GEMINI_API_KEY=$(touch /tmp/should-never-run)",
    "GEMINI_API_KEY=test\nGEMINI_API_KEY=duplicate",
    "GEMINI_API_KEY=test\nUNKNOWN=value",
    "GEMINI_API_KEY=",
    "GEMINI_API_KEY=test-key'",
])
def test_credentials_reject_unsafe_or_empty_entries(tmp_path, text):
    path = tmp_path / "env"
    path.write_text(text)
    path.chmod(0o600)
    with pytest.raises(ValueError):
        load_credentials(path)


def test_credentials_reject_public_permissions(tmp_path):
    path = tmp_path / "env"
    path.write_text("GEMINI_API_KEY=test")
    path.chmod(0o644)
    with pytest.raises(ValueError, match="0600"):
        load_credentials(path)
