

from __future__ import annotations

import hashlib
import ntpath
import re
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, Sequence

from langchain_core.documents import Document


@dataclass(frozen=True)
class SourceRecord:
    

    source_id: str
    source_name: str
    normalized_name: str
    source_type: str
    session_id: str
    file_hash: str
    source_path: str


def normalize_source_name(source_name: str) -> str:
    

    name = unicodedata.normalize("NFKC", str(source_name or "")).strip().casefold()
    name = re.sub(r"\.(?:pdf|docx?|md|txt|html?|json|ya?ml|toml|py|sql)\s*$", "", name)
    return "".join(character for character in name if character.isalnum())


def _normalize_source_identity(source_path: str, source_name: str, source_type: str) -> str:
    identity = unicodedata.normalize("NFKC", source_path or source_name).strip()
    normalized_path = ntpath.normcase(ntpath.normpath(identity)).replace("\\", "/")
    return f"{source_type.casefold()}:{normalized_path}"


def build_source_records(documents: Iterable[Document], session_id: str) -> list[SourceRecord]:
    

    grouped: OrderedDict[tuple[str, str, str], list[Document]] = OrderedDict()
    for document in documents:
        metadata = document.metadata
        source_name = str(metadata.get("source_name") or metadata.get("source") or "未命名来源")
        source_path = str(metadata.get("source") or source_name)
        source_type = str(metadata.get("source_type") or "unknown").casefold()
        grouped.setdefault((source_path, source_name, source_type), []).append(document)

    records: list[SourceRecord] = []
    for (source_path, source_name, source_type), source_documents in grouped.items():
        digest = hashlib.sha256()
        for index, document in enumerate(source_documents):
            if index:
                digest.update(b"\x1e")
            digest.update((document.page_content or "").encode("utf-8"))
        file_hash = digest.hexdigest()
        source_identity = _normalize_source_identity(source_path, source_name, source_type)
        source_digest = hashlib.sha256(f"{file_hash}\x1f{source_identity}".encode("utf-8")).hexdigest()
        records.append(
            SourceRecord(
                source_id=f"src_{source_digest[:20]}",
                source_name=source_name,
                normalized_name=normalize_source_name(source_name),
                source_type=source_type,
                session_id=str(session_id),
                file_hash=file_hash,
                source_path=source_path,
            )
        )
    return records


def attach_source_records(
    documents: Iterable[Document],
    records: Sequence[SourceRecord],
) -> list[Document]:
    

    by_key = {
        (record.source_path, record.source_name, record.source_type): record
        for record in records
    }
    by_name: dict[tuple[str, str], list[SourceRecord]] = {}
    for record in records:
        by_name.setdefault((record.source_name, record.source_type), []).append(record)

    attached: list[Document] = []
    for document in documents:
        metadata = dict(document.metadata)
        source_name = str(metadata.get("source_name") or metadata.get("source") or "未命名来源")
        source_path = str(metadata.get("source") or source_name)
        source_type = str(metadata.get("source_type") or "unknown").casefold()
        record = by_key.get((source_path, source_name, source_type))
        if record is None:
            same_name = by_name.get((source_name, source_type), [])
            record = same_name[0] if len(same_name) == 1 else None
        if record is None:
            raise ValueError(f"未找到来源记录: {source_name}")

        metadata.update(
            {
                "source_id": record.source_id,
                "session_id": record.session_id,
                "file_hash": record.file_hash,
                "normalized_name": record.normalized_name,
            }
        )
        attached.append(Document(page_content=document.page_content, metadata=metadata))
    return attached
