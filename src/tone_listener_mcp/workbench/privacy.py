"""Redact filesystem locations from workbench presentation and persistence."""

from __future__ import annotations

import logging
import re

from pydantic import BaseModel

_PATHS = re.compile(
    r"""file://[^\n"'<>]*|(?<![\w:/])/(?!api(?:/|$)|assets(?:/|$))[A-Za-z_][^\n"'<>]*|\b[A-Za-z]:[\\/][^\n"'<>]*"""
)


def redact_paths(value):
    if isinstance(value, str):
        return _PATHS.sub("[local path removed]", value)
    if isinstance(value, BaseModel):
        return redact_paths(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {key: redact_paths(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_paths(item) for item in value]
    return value


class SafeLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return redact_paths(super().format(record))
