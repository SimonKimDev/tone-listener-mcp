"""Only allowlisted numeric recovery information may cross the provider boundary."""

import json

import httpx
import pytest
from google.genai import errors
from test_schemas import undeclared

from tone_listener_mcp.listener import failure_info
from tone_listener_mcp.schemas import Failure

QUOTA_TYPE = "type.googleapis.com/google.rpc.QuotaFailure"
RETRY_TYPE = "type.googleapis.com/google.rpc.RetryInfo"


def quota_error(violations, *, status=429, header=None, retry="2.5s"):
    response = httpx.Response(
        status, headers={"Retry-After": header, "X-Secret": "secret"} if header is not None else {}
    )
    return errors.ClientError(status, {"error": {
        "message": "secret", "details": [
            {"@type": QUOTA_TYPE, "violations": violations, "secret": "secret"},
            {"@type": RETRY_TYPE, "retryDelay": retry, "secret": "secret"},
        ],
    }}, response)


def test_daily_limits_and_retry_hint_are_allowlisted():
    result = failure_info(quota_error([
        {"quotaId": "GenerateRequestsPerDay", "quotaValue": "20", "subject": "secret"},
        {"quotaId": "GenerateRequestsPerDay", "quotaValue": 10},
        {"quotaId": "GenerateTokensPerDay", "quotaValue": "1000"},
        {"quotaId": "GenerateRequestsPerMinute", "quotaValue": "5"},
    ], header="4"))
    assert result["category"] == "daily_quota"
    assert result["http_status"] == 429
    assert result["retryable"] is False
    assert result["retry_after_seconds"] == 4
    assert result["retry_after_invalid"] is False
    assert result["daily_request_limit"] == 10
    assert result["daily_token_limit"] == 1000
    assert "secret" not in json.dumps(result)
    assert undeclared(Failure.model_validate(result)) == []


@pytest.mark.parametrize(("identifier", "category"), [
    ("GenerateRequestsPerMinute", "rate_window"),
    ("generate_requests_per_minute", "rate_window"),
    ("generate_requests_per_day", "daily_quota"),
    ("unknown", "quota"),
])
def test_quota_category_from_structured_identifier_only(identifier, category):
    result = failure_info(quota_error([{"quotaId": identifier}]))
    assert result["category"] == category
    assert result["retry_after_seconds"] == 2.5


@pytest.mark.parametrize("header", ["secret", "nan", "inf", "-1", "", "Mon, 99 Foo 2026"])
def test_invalid_retry_headers_are_reported_without_echo(header):
    result = failure_info(quota_error([], header=header))
    assert result["retry_after_invalid"] is True
    assert result["retry_after_seconds"] == 2.5
    assert "secret" not in json.dumps(result)


def test_http_date_retry_header(monkeypatch):
    monkeypatch.setattr("tone_listener_mcp.listener.time.time", lambda: 0.0)
    result = failure_info(quota_error([], header="Thu, 01 Jan 1970 00:01:00 GMT"))
    assert result["retry_after_seconds"] == 60
    assert not result["retry_after_invalid"]


@pytest.mark.parametrize("payload", [
    None, "secret", [], {"error": "secret"}, {"error": {"details": "secret"}},
    {"error": {"details": [None, "secret", {"@type": QUOTA_TYPE, "violations": "secret"}]}},
    {"details": [{"@type": QUOTA_TYPE, "violations": [None, "secret", {"quotaId": []}]}]},
])
def test_malformed_provider_details_are_ignored_safely(payload):
    result = failure_info(errors.ClientError(429, payload))
    assert result["category"] == "quota"
    assert result["retry_after_seconds"] is None
    assert result["daily_request_limit"] is None
    assert "secret" not in json.dumps(result)


@pytest.mark.parametrize("value", ["secret", "-1", "nan", "1.5", "1e9", "9" * 13, True, {}, []])
def test_malformed_limits_are_never_returned(value):
    result = failure_info(quota_error([{
        "quotaId": "GenerateRequestsPerDay", "quotaValue": value,
    }], retry="secret"))
    assert result["daily_request_limit"] is None
    assert result["retry_after_seconds"] is None
    assert "secret" not in json.dumps(result)


def test_unrecognized_detail_types_cannot_set_metadata():
    result = failure_info(errors.ClientError(429, {"error": {"details": [{
        "@type": "secret.RetryInfo", "retryDelay": "123s",
        "violations": [{"quotaId": "GenerateRequestsPerDay", "quotaValue": 10}],
    }]}}))
    assert result["category"] == "quota"
    assert result["retry_after_seconds"] is None and result["daily_request_limit"] is None


def test_quota_metadata_does_not_override_authentication():
    result = failure_info(quota_error([{"quotaId": "GenerateRequestsPerDay"}], status=403))
    assert result["category"] == "permission"
    assert result["http_status"] == 403
