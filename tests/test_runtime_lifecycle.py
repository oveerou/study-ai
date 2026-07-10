from __future__ import annotations

import json
from pathlib import Path

from qdrant_client import QdrantClient

from ragbase.runtime import archive_conversation, close_vector_store


class _VectorStore:
    def __init__(self, client: QdrantClient):
        self.client = client


def test_close_vector_store_releases_local_qdrant_lock(tmp_path: Path):
    database_dir = tmp_path / "docs-db"
    first_client = QdrantClient(path=str(database_dir))

    close_vector_store(_VectorStore(first_client))

    second_client = QdrantClient(path=str(database_dir))
    second_client.close()


def test_archive_conversation_writes_messages_and_sources(tmp_path: Path):
    archive_path = archive_conversation(
        history_dir=tmp_path / "history",
        session_id="session-123",
        messages=[
            {"role": "user", "content": "总结资料"},
            {"role": "assistant", "content": "资料主要讲图像处理。"},
        ],
        source_names=["复习大纲.pdf"],
    )

    assert archive_path is not None
    payload = json.loads(archive_path.read_text(encoding="utf-8"))
    assert payload["session_id"] == "session-123"
    assert payload["source_names"] == ["复习大纲.pdf"]
    assert payload["messages"][0]["content"] == "总结资料"


def test_archive_conversation_skips_unused_blank_session(tmp_path: Path):
    archive_path = archive_conversation(
        history_dir=tmp_path / "history",
        session_id="blank-session",
        messages=[{"role": "assistant", "content": "你好"}],
        source_names=[],
    )

    assert archive_path is None
    assert not (tmp_path / "history").exists()
