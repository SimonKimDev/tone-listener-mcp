"""Offline regressions for the v3 tone-copier path. No provider calls are made."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import numpy as np
import pytest
from google.genai import errors, types

from tests.test_workbench_recipes import adapter, profile  # noqa: F401  (fixture reuse)
from tone_listener_mcp.audio_io import AudioClip
from tone_listener_mcp.workbench import evaluation, evaluation_v3, relative_judge
from tone_listener_mcp.workbench.adapters.soundshed import new_project
from tone_listener_mcp.workbench.models import ChainDraft, ChainNode, Interval, SourceRef

EFFECT_NONE = {"presence": "absent", "prominence": "none"}


def amane(**values):
    base = {"gain": "overdrive", "brightness": "bright", "mids": "present", "attack": "balanced",
            "compression": "moderate", "reverb": {"presence": "present", "prominence": "moderate"}}
    return profile(**{**base, **values})


# --- v3 grader --------------------------------------------------------------------------------

def test_known_large_mismatch_fails_even_when_another_axis_is_unknown():
    result = evaluation_v3.compare_profiles_v3(amane(), amane(gain="clean", attack="uncertain"))
    assert result["outcome"] == "fail"
    assert result["core"]["known_failures"] and result["core"]["unknown_axes"] == ["attack"]
    # The frozen v2 grader masks the same failure behind the unknown axis; it stays unchanged.
    assert evaluation.compare_profiles(amane(), amane(gain="clean", attack="uncertain"))["outcome"] == "inconclusive"


def test_known_partial_sum_fails_while_remaining_axis_unknown():
    result = evaluation_v3.compare_profiles_v3(
        amane(), amane(gain="distortion", brightness="slightly-bright", mids="forward", attack="uncertain"))
    assert result["outcome"] == "fail"
    assert any("exceeds 2" in item for item in result["core"]["known_failures"])


def test_unresolved_effect_never_becomes_pass_and_aggregate_takes_worst():
    reference = amane(delay={"presence": "possible", "prominence": "subtle"})
    result = evaluation_v3.compare_profiles_v3(reference, amane(), omitted_effects=["delay"])
    assert result["outcome"] == "core_match_effects_unresolved"
    assert result["effects"]["axes"]["delay"]["omitted_in_candidate_chain"] is True
    assert result["perceptual_match_claim"] is False
    exact = evaluation_v3.compare_profiles_v3(amane(), amane())
    assert exact["outcome"] == "pass"
    assert evaluation_v3.aggregate_v3([exact, result]) == "core_match_effects_unresolved"
    assert evaluation_v3.aggregate_v3([exact, {"outcome": "fail"}, result]) == "fail"
    assert evaluation_v3.aggregate_v3([]) == "inconclusive"


def test_prominent_effect_contradiction_fails_and_unclear_guitar_is_inconclusive():
    contradiction = evaluation_v3.compare_profiles_v3(
        amane(), amane(reverb=EFFECT_NONE))
    assert contradiction["outcome"] == "fail"
    unclear = evaluation_v3.compare_profiles_v3(amane(), amane(guitar_dominance="mixed", gain="clean"))
    assert unclear["outcome"] == "inconclusive"
    assert unclear["core"]["known_failures"]


# --- required gear vs omitted optional effects ------------------------------------------------

def test_possible_delay_is_labeled_omission_that_does_not_block_render(adapter):  # noqa: F811
    for resource_id, category in (("383084", "pedal"), ("494017", "amp")):
        adapter._records[f"nam:tone3000:{resource_id}"] = {
            "type": "nam", "filePath": "asset-0", "name": f"Test {category} capture", "category": category,
            "metadata": {"captureScope": category}}
    chain = adapter.default_chain([{"profile": amane(
        delay={"presence": "possible", "prominence": "subtle"},
        modulation={"presence": "uncertain", "prominence": "uncertain"})}])
    omitted = [node for node in chain.nodes if node.kind == "gear_note" and node.note_role == "omitted_optional"]
    assert {node.id for node in omitted} >= {"delay-note", "modulation-note"}
    assert not any(node.kind in ("delay", "chorus") for node in chain.nodes)
    assert adapter.validate_chain(chain) == []
    interval = Interval(start_seconds=0, end_seconds=10)
    project = new_project(
        source=SourceRef(kind="audio_file", display_name="clip.wav", duration_seconds=20, selected_intervals=[interval]),
        analysis=[{"interval": interval.model_dump(), "model": "m", "profile": amane().model_dump()}],
        intent={"summary": "s"}, chain=chain, engine_version="1.7.3", name="t")
    description = adapter._preset_description(project, None)
    assert "OMITTED optional effect" in description and "Delay omitted (uncertain)" in description


def test_missing_required_gear_still_blocks(adapter):  # noqa: F811
    del adapter._records["nam:tone3000:494014"]
    del adapter._records["nam:tone3000:758459"]
    chain = adapter.default_chain([{"profile": profile()}])
    blocking = [node for node in chain.nodes if node.kind == "gear_note" and node.note_role == "blocking"]
    assert blocking and adapter.validate_chain(chain)


def test_note_role_only_on_gear_notes_and_existing_chain_hash_stable(adapter):  # noqa: F811
    with pytest.raises(ValueError):
        ChainNode(id="x", kind="input", title="Input", note_role="omitted_optional")
    chain = adapter.default_chain([{"profile": profile()}])
    legacy = json.loads(chain.model_dump_json())
    for node in legacy["nodes"]:
        node.pop("note_role", None)
    assert adapter.chain_sha256(ChainDraft.model_validate(legacy)) == adapter.chain_sha256(chain)


# --- relative judge ---------------------------------------------------------------------------

def tone(freq, amplitude, seconds=1.0, rate=44100):
    t = np.arange(int(seconds * rate)) / rate
    return AudioClip(np.stack([amplitude * np.sin(2 * np.pi * freq * t)] * 2), rate, "clip", 0)


def test_level_match_and_blind_labels():
    matched, gains = relative_judge.level_match([tone(220, 0.5), tone(440, 0.05)])
    levels = [20 * np.log10(np.sqrt(np.mean(item.samples ** 2))) for item in matched]
    assert all(abs(level - relative_judge.TARGET_RMS_DBFS) < 0.1 for level in levels)
    assert gains[1] - gains[0] == pytest.approx(20, abs=0.1)
    labels = relative_judge.blind_labels(["base", "dark", "bright"], seed=7)
    assert sorted(labels.values()) == ["base", "bright", "dark"]
    assert all(len(label) == 2 and label[0] != label[1] for label in labels)
    assert labels == relative_judge.blind_labels(["base", "dark", "bright"], seed=7)
    assert labels != relative_judge.blind_labels(["base", "dark", "bright"], seed=8)
    with pytest.raises(ValueError):
        relative_judge.level_match([tone(220, 0.0)])


def fake_client(generate):
    client = MagicMock()
    client.__enter__.return_value = client
    client.files.upload.side_effect = lambda **_: SimpleNamespace(
        name=f"files/{client.files.upload.call_count}", state=types.FileState.ACTIVE, uri="gs://x")
    client.models.generate_content.side_effect = generate
    return client


def judgment_for(labels, closest, gain="similar"):
    return json.dumps({
        "reference_guitar_audible": "clear",
        "comparisons": [{"clip": label, **dict.fromkeys(relative_judge.RELATIVE_AXES, "similar"), "gain": gain}
                        for label in labels],
        "closest_to_reference_order": closest, "closest_is_close_match": "uncertain", "notes": [],
    })


def test_judge_relative_one_attempt_blinded_and_scored(tmp_path):
    candidates = {"dup": tone(220, 0.2), "gain_down": tone(220, 0.02)}
    seen = {}

    def generate(**kwargs):
        seen.update(kwargs)
        text = [part for part in kwargs["contents"] if isinstance(part, str)]
        labels = [item.split()[1] for item in text if item.startswith("Candidate ")]
        return SimpleNamespace(text=judgment_for(labels, labels), candidates=[SimpleNamespace(finish_reason="STOP")],
                               usage_metadata=None)

    client = fake_client(generate)
    result = relative_judge.judge_relative(model="m", candidates=candidates, seed=3, anchor_clip=tone(220, 0.1),
                                           client_factory=lambda: client)
    assert result["request_counts"] == {"generate": 1, "upload": 3, "poll_upload": 0, "delete_upload": 3}
    assert seen["config"].max_output_tokens == relative_judge.MAX_OUTPUT_TOKENS
    prompt = json.dumps([part for part in seen["contents"] if isinstance(part, str)])
    assert "dup" not in prompt and "gain_down" not in prompt
    score = relative_judge.score_calibration(result, {"gain_down": {"gain": {"less", "much_less"}}}, "dup")
    assert score["valid"] and score["scored"] == 2
    assert relative_judge.agreement(result, result)["axis_exact_agreement"] == 1


def test_judge_relative_429_is_recorded_once_without_retry():
    error = errors.ClientError(429, {"error": {"message": "secret-key", "details": [
        {"violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel"}]}]}},
        httpx.Response(429, headers={"Retry-After": "30"}))
    client = fake_client(MagicMock(side_effect=error))
    result = relative_judge.judge_relative(model="m", candidates={"a": tone(220, 0.2)}, seed=1,
                                           anchor_clip=tone(220, 0.1), client_factory=lambda: client)
    assert client.models.generate_content.call_count == 1
    assert result["failure_info"]["http_status"] == 429
    assert result["failure_info"]["retry_after_seconds"] == 30
    assert result["request_counts"]["delete_upload"] == 2
    assert "secret-key" not in json.dumps(result)
    assert relative_judge.unblind(result) == {}


# --- batch ledger -----------------------------------------------------------------------------

def test_ledger_counts_failures_enforces_aggregate_ceiling_and_pauses_on_429(tmp_path):
    from scripts import tone_copier_batch as batch_module

    batch = batch_module.Batch(tmp_path / "batch", ceiling=3)
    first = batch.reserve("relative", "flash", {})
    batch.settle(first, {"failure_info": {"http_status": 503, "category": "capacity"}, "request_counts": {}})
    second = batch.reserve("relative", "argon", {})
    batch.settle(second, {"failure_info": None, "request_counts": {"generate": 1}})
    reloaded = batch_module.Batch(tmp_path / "batch", ceiling=99)
    assert reloaded.ledger["ceiling"] == 3
    assert [item["status"] for item in reloaded.ledger["attempts"]] == ["failed", "ok"]
    third = reloaded.reserve("absolute_profile", "flash", {})
    reloaded.settle(third, {"failure_info": {"http_status": 429, "category": "rate_window",
                                             "retry_after_seconds": 10}})
    with pytest.raises(SystemExit, match="paused"):
        reloaded.reserve("relative", "flash", {})
    ledger = reloaded.ledger
    ledger["provider_pause"] = None
    batch_module._save(reloaded.ledger_path, ledger)
    with pytest.raises(SystemExit, match="ceiling"):
        reloaded.reserve("relative", "flash", {})
    assert len(reloaded.ledger["attempts"]) == 3


def test_freeze_change_is_refused(tmp_path, monkeypatch):
    from scripts import tone_copier_batch as batch_module

    renderer = SimpleNamespace(engine_version="1.7.3", _worker_sha256="a" * 64, _plugin_sha256="b" * 64)
    batch = batch_module.Batch(tmp_path / "batch")
    digest = batch.freeze(renderer)
    assert batch.freeze(renderer) == digest
    monkeypatch.setattr(relative_judge, "MAX_OUTPUT_TOKENS", 1024)
    with pytest.raises(SystemExit, match="changed"):
        batch.freeze(renderer)


# --- independent-check regressions (grader/calibration/admission) ------------------------------

def test_unknown_presence_effect_difference_is_unresolved_not_contradiction():
    reference = amane(delay={"presence": "possible", "prominence": "subtle"})
    result = evaluation_v3.compare_profiles_v3(reference, amane(delay={"presence": "present", "prominence": "moderate"}))
    assert result["outcome"] == "core_match_effects_unresolved"
    assert result["effects"]["axes"]["delay"]["status"] == "unresolved" and not result["effects"]["mismatches"]
    # A reference reverb that is present/moderate against a candidate "possible" is also unresolved.
    candidate = amane(reverb={"presence": "possible", "prominence": "subtle"})
    assert evaluation_v3.compare_profiles_v3(amane(), candidate)["effects"]["axes"]["reverb"]["status"] == "unresolved"
    # Known present-vs-absent with a moderate effect is still a contradiction.
    absent = amane(reverb={"presence": "absent", "prominence": "none"})
    assert evaluation_v3.compare_profiles_v3(amane(), absent)["outcome"] == "fail"
    # Known core failures still fail even with an unresolved effect.
    assert evaluation_v3.compare_profiles_v3(reference, amane(gain="clean"))["outcome"] == "fail"


def test_modulation_family_requires_known_presence_for_a_contradiction():
    chorus = {"modulation": {"presence": "present", "prominence": "moderate"}, "modulation_type": "chorus"}
    possible = {"modulation": {"presence": "possible", "prominence": "subtle"}, "modulation_type": "phaser"}
    result = evaluation_v3.compare_profiles_v3(amane(**chorus), amane(**possible))
    assert "modulation family contradiction" not in result["effects"]["mismatches"]
    assert result["effects"]["axes"]["modulation"]["status"] == "unresolved"
    assert result["outcome"] == "core_match_effects_unresolved"
    phaser = {"modulation": {"presence": "present", "prominence": "moderate"}, "modulation_type": "phaser"}
    assert evaluation_v3.compare_profiles_v3(amane(**chorus), amane(**phaser))["outcome"] == "fail"


@pytest.mark.parametrize("bad", [[{"outcome": "blocked"}], [{"outcome": "pass"}, {}], [{"outcome": None}]])
def test_aggregate_rejects_unknown_or_unperformed_outcomes(bad):
    with pytest.raises(ValueError):
        evaluation_v3.aggregate_v3(bad)
    assert evaluation_v3.aggregate_v3([{"outcome": "pass"}, {"outcome": "core_match_effects_unresolved"}]) == (
        "core_match_effects_unresolved")


@pytest.mark.parametrize("audibility", ["partial", "unclear"])
def test_calibration_and_agreement_require_a_clearly_audible_anchor(audibility):
    def judged(value):
        return {"label_mapping": {"AB": "identical", "CD": "gain_up"}, "invalid": None, "failure_info": None,
                "judgment": {"reference_guitar_audible": value, "closest_to_reference_order": ["AB", "CD"],
                             "closest_is_close_match": "yes", "notes": [],
                             "comparisons": [{"clip": "AB", **dict.fromkeys(relative_judge.RELATIVE_AXES, "similar")},
                                             {"clip": "CD", **dict.fromkeys(relative_judge.RELATIVE_AXES, "more")}]}}
    truth = {"gain_up": {"gain": {"more", "much_more"}}}
    score = relative_judge.score_calibration(judged(audibility), truth, "identical")
    assert score["valid"] is False and score["accuracy"] is None and score["observed_correct"] == 2
    assert relative_judge.score_calibration(judged("clear"), truth, "identical")["valid"] is True
    assert relative_judge.agreement(judged("clear"), judged(audibility))["valid"] is False
    assert relative_judge.agreement(judged("clear"), judged("clear"))["valid"] is True


def _reserve_in_process(root, tag, barrier):
    from scripts import tone_copier_batch as batch_module

    batch = batch_module.Batch(root, ceiling=3)
    barrier.wait()
    try:
        batch.reserve("relative", "offline", {"tag": tag})
        return "admitted"
    except SystemExit as exc:
        return str(exc)


def test_separate_processes_never_over_admit_or_lose_reservations(tmp_path):
    import multiprocessing

    from scripts import tone_copier_batch as batch_module

    root = tmp_path / "batch"
    batch_module.Batch(root, ceiling=3)
    context = multiprocessing.get_context("spawn")
    with context.Manager() as manager:
        barrier = manager.Barrier(8)
        with context.Pool(8) as pool:
            outcomes = pool.starmap(_reserve_in_process, [(root, f"tag-{i % 6}", barrier) for i in range(8)])
    ledger = batch_module.Batch(root).ledger
    assert outcomes.count("admitted") == 3
    assert len(ledger["attempts"]) == 3 and [a["index"] for a in ledger["attempts"]] == [1, 2, 3]
    assert len({a["tag"] for a in ledger["attempts"]}) == 3


def test_duplicate_tag_reservation_is_atomic_and_relative_refuses_before_call(tmp_path, monkeypatch):
    from scripts import tone_copier_batch as batch_module

    batch = batch_module.Batch(tmp_path / "batch")
    batch.reserve("relative", "offline", {"tag": "once"})
    with pytest.raises(SystemExit, match="already reserved"):
        batch_module.Batch(tmp_path / "batch").reserve("absolute_profile", "offline", {"tag": "once"})
    calls = []
    monkeypatch.setattr(batch_module, "_require_live", lambda args: None)
    monkeypatch.setattr(batch_module, "SoundshedAdapter", lambda: None)
    monkeypatch.setattr(batch_module, "NativeRenderer", lambda _adapter: SimpleNamespace(initialize=lambda: None))
    monkeypatch.setattr(batch_module.Batch, "freeze", lambda *args: "a" * 64)
    monkeypatch.setattr(relative_judge, "judge_relative", lambda **kwargs: calls.append(1))
    args = SimpleNamespace(tag="once", live=True, candidates=["a"], sample="riff-1", anchor=None, model="m", seed=1)
    with pytest.raises(SystemExit, match="already used"):
        batch_module.cmd_relative(args, batch)
    assert calls == [] and len(batch.ledger["attempts"]) == 1


def test_freeze_identity_covers_preparation_and_rubric_axes(monkeypatch):
    from scripts import tone_copier_batch as batch_module

    identity = batch_module.frozen_identity()
    assert {"audio_io_preparation", "grader_axes_v2_module", "rubric_axes", "relative_axes"} <= set(identity)
    before = json.dumps(identity, sort_keys=True)
    monkeypatch.setitem(evaluation.AXES, "gain", ["clean", "overdrive"])
    assert json.dumps(batch_module.frozen_identity(), sort_keys=True) != before
