

from __future__ import annotations

import json
from pathlib import Path

from ragbase.conversation_store import (
    delete_conversation,
    list_conversations,
    load_conversation,
    save_conversation,
)


def test_save_conversation_upserts_one_local_file(tmp_path: Path):
    
    history_dir = tmp_path / "history"

    first_path = save_conversation(
        history_dir,
        session_id="session-1",
        messages=[{"role": "user", "content": "Explain image gradients"}],
        source_names=["course.pdf"],
    )
    second_path = save_conversation(
        history_dir,
        session_id="session-1",
        messages=[
            {"role": "user", "content": "Explain image gradients"},
            {"role": "assistant", "content": "A gradient measures spatial change."},
        ],
        source_names=["course.pdf"],
    )

    assert first_path == second_path
    assert len(list(history_dir.glob("*.json"))) == 1
    saved = load_conversation(second_path)
    assert saved["title"] == "Explain image gradients"
    assert saved["messages"][-1]["role"] == "assistant"


def test_list_conversations_reads_legacy_archives_and_deduplicates_session(tmp_path: Path):
    
    history_dir = tmp_path / "history"
    history_dir.mkdir()
    legacy = {
        "session_id": "legacy-session",
        "saved_at": "2026-07-10T10:00:00+08:00",
        "source_names": ["old.pdf"],
        "messages": [{"role": "user", "content": "Old question"}],
    }
    (history_dir / "20260710-legacy.json").write_text(
        json.dumps(legacy, ensure_ascii=False),
        encoding="utf-8",
    )
    save_conversation(
        history_dir,
        session_id="legacy-session",
        messages=[
            {"role": "user", "content": "Old question"},
            {"role": "assistant", "content": "Latest answer"},
        ],
        source_names=["old.pdf"],
    )

    conversations = list_conversations(history_dir)

    assert len(conversations) == 1
    assert conversations[0]["session_id"] == "legacy-session"
    assert conversations[0]["message_count"] == 2


def test_delete_conversation_removes_stable_and_legacy_files(tmp_path: Path):
    
    history_dir = tmp_path / "history"
    stable_path = save_conversation(
        history_dir,
        session_id="session-delete",
        messages=[{"role": "user", "content": "Delete me"}],
        source_names=[],
    )
    legacy_path = history_dir / "legacy-copy.json"
    legacy_path.write_text(stable_path.read_text(encoding="utf-8"), encoding="utf-8")

    deleted = delete_conversation(history_dir, "session-delete")

    assert deleted == 2
    assert list(history_dir.glob("*.json")) == []
