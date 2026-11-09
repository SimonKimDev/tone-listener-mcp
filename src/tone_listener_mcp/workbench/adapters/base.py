"""Small target adapter contract; product-specific state never leaks into projects."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from tone_listener_mcp.workbench.models import (
    AdapterCatalog,
    ChainDraft,
    PreviewSnapshot,
    ToneProject,
)


class TargetAdapter(Protocol):
    adapter_id: str

    def catalog(self) -> AdapterCatalog: ...

    def default_chain(self, analysis: Sequence[object] = ()) -> ChainDraft: ...

    def validate_chain(self, chain: ChainDraft, *, for_render: bool = True) -> list[str]: ...

    def host_state(self, project: ToneProject, trailer: bytes) -> bytes: ...

    def export_archive(self, project: ToneProject, preview: PreviewSnapshot) -> bytes: ...
