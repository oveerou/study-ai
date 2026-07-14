

from __future__ import annotations

from dataclasses import dataclass


VALID_OPERATIONS = frozenset({"chat", "list_sources", "read_source", "search"})
VALID_READ_MODES = frozenset({"overview", "full_text"})
VALID_SOURCE_SCOPES = frozenset({"none", "selected", "active", "all", "auto"})


@dataclass(frozen=True)
class OperationPlan:
    

    operation: str
    source_ids: tuple[str, ...]
    query: str
    confidence: float
    reason: str = ""
    read_mode: str | None = None
    scope: str = "auto"


@dataclass(frozen=True)
class EvidenceCitation:
    

    source_id: str
    source_name: str
    page_start: int | str | None
    page_end: int | str | None
    chunk_id: str
