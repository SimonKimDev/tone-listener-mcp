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


def build_manifest(tmp_path):
    from scripts.make_validation_set import build_dataset

    build_dataset(tmp_path, {"take1": None, "take2": None}, {"take2"}, seconds=1)
    return tmp_path / "manifest.json"


def edit_manifest(path, change):
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))
    return path


def test_dataset_manifest_holds_out_whole_performances(tmp_path):
    from scripts.eval_listening import load_manifest, manifest_judgments

    manifest = load_manifest(build_manifest(tmp_path))
    assert manifest["splits"] == {"tuning": ["take1"], "held_out": ["take2"]}
    assert len(manifest["clips"]) == 18
    assert {comparison["task"] for comparison in manifest["comparisons"]} == {
        "eq", "drive", "ambience", "level_control", "silence_control"
    }
    held = manifest_judgments(manifest, "held_out", 2)
    assert len(held) == 7 * 2 * 2
    assert all(comparison["a"].startswith("take2/") for comparison, _, _ in held)
    padded = load_audio(manifest["clips"]["take1/padded"]["path"])
    reference = load_audio(manifest["clips"]["take1/reference"]["path"])
    assert padded.duration == pytest.approx(reference.duration + 3, abs=1e-3)
    assert gated_power(padded)[0] == pytest.approx(gated_power(reference)[0], rel=0.05)


@pytest.mark.parametrize(("change", "message"), [
    (lambda data: data["splits"].update(held_out=["take1", "take2"]), "both"),
    (lambda data: data["splits"].pop("held_out"), "not assigned"),
    (lambda data: data["comparisons"][0].update(b="take2/eq_bright"), "different splits"),
    (lambda data: data["comparisons"][0].update(axis="louder"), "axis"),
    (lambda data: data["comparisons"][0].update(expected="different"), "expected"),
    (lambda data: data["comparisons"][1].update(id=data["comparisons"][0]["id"]), "unique"),
    (lambda data: data["comparisons"][0].update(b="missing"), "two different clips"),
    (lambda data: data["clips"]["take1/reference"].update(path="nope.wav"), "does not exist"),
    (lambda data: data.update(version=2), "version 1"),
])
def test_manifest_validation(tmp_path, change, message):
    from scripts.eval_listening import load_manifest

    with pytest.raises(ValueError, match=message):
        load_manifest(edit_manifest(build_manifest(tmp_path), change))


def test_manifest_youtube_reference_only_as_a(tmp_path):
    from scripts.eval_listening import load_manifest, manifest_judgments

    def add_youtube(data):
        data["clips"]["take1/yt"] = {
            "youtube": {"url": "https://youtu.be/9hE5-98ZeCg", "start": 1, "end": 5},
            "performance": "take1",
        }
        data["comparisons"].append({
            "id": "yt", "a": "take1/yt", "b": "take1/reference", "task": "visual_bias",
            "axis": "brighter", "expected": "similar", "bias_group": "same-audio",
        })

    path = edit_manifest(build_manifest(tmp_path), add_youtube)
    manifest = load_manifest(path)
    assert manifest["clips"]["take1/yt"]["youtube"]["url"].endswith("v=9hE5-98ZeCg")
    youtube = [item for item in manifest_judgments(manifest, "tuning", 1) if item[0]["id"] == "yt"]
    assert [swapped for _, _, swapped in youtube] == [False]
    edit_manifest(path, lambda data: data["comparisons"][-1].update(a="take1/reference", b="take1/yt"))
    with pytest.raises(ValueError, match="only reference A"):
        load_manifest(path)


def test_manifest_scoring_by_task_order_abstention_and_bias():
    from scripts.eval_listening import manifest_prediction, score_manifest

    assert manifest_prediction({"brighter": "A"}, "brighter", True) == "B"
    assert manifest_prediction({"brighter": "B"}, "brighter", False) == "B"

    def row(comparison, task, expected, prediction, swapped=False, repeat=0, bias=None,
            axis="brighter", b="take1/reference"):
        return {"comparison": comparison, "task": task, "expected": expected,
                "prediction": prediction, "swapped": swapped, "repeat": repeat,
                "performance": "take1", "bias_group": bias, "axis": axis, "b": b}

    result = score_manifest([
        row("eq", "eq", "B", "B"), row("eq", "eq", "B", "B", swapped=True),
        row("amb", "ambience", "B", "uncertain"), row("amb", "ambience", "B", "B", swapped=True),
        row("none", "no_guitar", "uncertain", "uncertain"),
        row("yt1", "visual", "similar", "similar", bias="g"),
        row("yt2", "visual", "similar", "A", bias="g"),
        row("down", "eq", "A", None),
    ])
    assert result["judgments"] == 8
    assert result["correct"] == 5
    assert result["unavailable"] == 1
    assert result["abstentions"] == 2
    assert result["abstention_rate"] == pytest.approx(2 / 7)
    assert (result["order_consistent"], result["order_pairs"]) == (1, 2)
    assert (result["bias_groups_consistent"], result["bias_groups"]) == (0, 1)
    assert result["by_task"]["eq"] == {
        "correct": 2, "total": 3, "accuracy": pytest.approx(2 / 3), "abstentions": 0,
        "abstention_rate": 0, "comparisons": 2, "performances": 1,
    }
    assert result["by_task"]["no_guitar"]["accuracy"] == 1
    assert result["sample_counts"] == {"judgments": 8, "comparisons": 6, "performances": 1}


def test_mocked_manifest_run_maps_swaps_and_disables_cache(tmp_path, monkeypatch):
    path = build_manifest(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("TONE_LISTENER_CACHE_ENTRIES", "64")
    calls = []

    def compare(a, b):
        import os

        assert os.environ["TONE_LISTENER_CACHE_ENTRIES"] == "0"
        assert a.source == b.source == "blind"
        assert gated_power(a)[0] == pytest.approx(gated_power(b)[0], rel=1e-4)
        calls.append((a, b))
        # Always call the second-presented clip brighter: correct only in one order.
        payload = {axis: "similar" for axis in (
            "brighter", "more_saturated", "more_compressed", "more_ambient"
        )}
        return {"listening": {**payload, "brighter": "B"}, "listening_unavailable": None}

    monkeypatch.setattr("scripts.eval_listening.listen_comparison", compare)
    report = evaluate(["tuning"], repeats=1, pause=0, manifest_path=path)
    import os

    assert os.environ["TONE_LISTENER_CACHE_ENTRIES"] == "64"
    assert report["status"] == "complete"
    assert report["mode"] == "manifest"
    assert report["max_requests"] == 14 == len(calls)
    scores = report["splits"]["tuning"]["scores"]
    assert scores["by_task"]["eq"]["total"] == 4
    assert scores["by_task"]["eq"]["correct"] == 2
    assert scores["order_consistency"] == pytest.approx(5 / 7)
    assert scores["by_task"]["silence_control"]["correct"] == 0


def test_manifest_skip_reports_budget_without_calls(tmp_path):
    report = evaluate(["tuning", "held_out"], repeats=1, manifest_path=build_manifest(tmp_path))
    assert report["status"] == "skipped"
    assert report["max_requests"] == 28
    assert report["manifest"]["sha256"]
    with pytest.raises(ValueError, match="repeats"):
        evaluate(["tuning"], repeats=0, manifest_path=tmp_path / "manifest.json")


def test_bias_groups_only_compare_answers_to_the_same_question():
    from scripts.eval_listening import score_manifest

    def row(comparison, prediction, axis="brighter", b="take1/reference", repeat=0):
        return {"comparison": comparison, "task": "visual", "expected": "similar",
                "prediction": prediction, "swapped": False, "repeat": repeat,
                "performance": "take1", "bias_group": "cover", "axis": axis, "b": b}

    result = score_manifest([
        row("yt1-bright", "similar"), row("yt2-bright", "similar"),
        # Different questions in the same group: never compared with each other.
        row("yt1-drive", "B", axis="more_saturated"),
        row("yt2-other", "A", b="take1/eq_dark"),
        row("yt1-bright-r1", "similar", repeat=1), row("yt2-bright-r1", "A", repeat=1),
    ])
    assert (result["bias_groups_consistent"], result["bias_groups"]) == (1, 2)


def test_dataset_pairs_have_equal_length_except_silence_control(tmp_path):
    import soundfile as sf

    from scripts.eval_listening import load_manifest

    manifest = load_manifest(build_manifest(tmp_path))
    frames = {name: sf.info(clip["path"]).frames for name, clip in manifest["clips"].items()}
    for comparison in manifest["comparisons"]:
        a, b = frames[comparison["a"]], frames[comparison["b"]]
        if comparison["task"] == "silence_control":
            assert b - a == 3 * 48000
        else:
            assert a == b, comparison["id"]
    reverb = load_audio(manifest["clips"]["take1/reverb"]["path"])
    reference = load_audio(manifest["clips"]["take1/reference"]["path"])
    assert reference.duration == pytest.approx(1.25, abs=1e-3)
    # Padding is silent: the reference's appended tail adds no energy.
    assert not reference.samples[:, 48000:].any()
    assert reverb.samples[:, 48000:].any()


def corrupt(path):
    path.write_bytes(b"not audio")


def three_channels(path):
    import soundfile as sf

    sf.write(path, np.zeros((4800, 3)), 48000)


@pytest.mark.parametrize(("change", "message"), [
    (lambda data, root: data["clips"]["take1/eq_bright"].update(start="1"), "number of seconds"),
    (lambda data, root: data["clips"]["take1/eq_bright"].update(end=True), "number of seconds"),
    (lambda data, root: data["clips"]["take1/eq_bright"].update(start=0.5, end=0.2), "greater"),
    (lambda data, root: data["clips"]["take1/eq_bright"].update(start=5), "no audio"),
    (lambda data, root: corrupt(root / "take1" / "drive_heavy.wav"), "Cannot decode"),
    (lambda data, root: three_channels(root / "take1" / "reverb.wav"), "mono and stereo"),
])
def test_manifest_clips_are_checked_before_any_request(tmp_path, monkeypatch, change, message):
    from scripts.eval_listening import ManifestError

    path = build_manifest(tmp_path)
    edit_manifest(path, lambda data: change(data, tmp_path))
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    def forbidden(*args, **kwargs):
        raise AssertionError("no request may be made for an invalid manifest")

    monkeypatch.setattr("scripts.eval_listening.listen_comparison", forbidden)
    with pytest.raises(ManifestError, match=message):
        evaluate(["tuning"], repeats=1, pause=0, manifest_path=path)


def test_manifest_preflight_rejects_over_length_clips_and_skips_other_splits(
    tmp_path, monkeypatch
):
    from scripts.eval_listening import ManifestError

    path = build_manifest(tmp_path)
    corrupt(tmp_path / "take2" / "reference.wav")
    # Only the selected split is decoded; a broken held-out clip does not block tuning.
    assert evaluate(["tuning"], repeats=1, manifest_path=path)["status"] == "skipped"
    with pytest.raises(ManifestError, match="take2/reference"):
        evaluate(["held_out"], repeats=1, manifest_path=path)
    monkeypatch.setenv("TONE_LISTENER_MAX_SECONDS", "0.5")
    with pytest.raises(ManifestError, match="exceeds 0.5 seconds"):
        evaluate(["tuning"], repeats=1, manifest_path=path)


def test_mid_run_failure_writes_partial_error_report(tmp_path, monkeypatch):
    from scripts import eval_listening

    path = build_manifest(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    calls = []

    def compare(a, b):
        calls.append(1)
        if len(calls) == 4:
            raise ValueError("local failure after paid calls")
        payload = {axis: "similar" for axis in (
            "brighter", "more_saturated", "more_compressed", "more_ambient"
        )}
        return {"listening": payload, "listening_unavailable": None}

    monkeypatch.setattr("scripts.eval_listening.listen_comparison", compare)
    report_path = tmp_path / "out" / "report.json"
    monkeypatch.setattr(sys, "argv", [
        "eval_listening", "--manifest", str(path), "--repeats", "1", "--pause", "0",
        "--report", str(report_path),
    ])
    with pytest.raises(SystemExit) as exit_info:
        eval_listening.main()
    assert exit_info.value.code == 1
    report = json.loads(report_path.read_text())
    assert report["status"] == "error"
    split = report["splits"]["tuning"]
    assert split["status"] == "error"
    assert len(split["rows"]) == 3
    assert split["scores"]["judgments"] == 3
    assert split["error"] == {"type": "ValueError", "message": "local failure after paid calls"}
