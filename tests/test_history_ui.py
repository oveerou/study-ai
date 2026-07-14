

from __future__ import annotations

from pathlib import Path


APP = Path(__file__).resolve().parents[1] / "app.py"


def test_app_auto_persists_user_and_assistant_messages():
    
    text = APP.read_text(encoding="utf-8")

    assert "from ragbase.conversation_store import" in text
    assert "def persist_current_conversation()" in text
    assert "def record_message(" in text
    assert 'record_message("user", question)' in text
    assert 'record_message("assistant", result.answer)' in text
    persist_body = text.split("def persist_current_conversation()", 1)[1].split("\ndef ", 1)[0]
    assert "st.session_state.viewing_history" in persist_body


def test_sidebar_lists_opens_and_deletes_local_history():
    
    text = APP.read_text(encoding="utf-8")

    assert "def render_history_controls()" in text
    assert "list_conversations(" in text
    assert "load_conversation(" in text
    assert "delete_conversation(" in text
    assert "历史对话" in text


def test_history_view_is_rendered_without_claiming_sources_are_indexed():
    
    text = APP.read_text(encoding="utf-8")

    assert 'if "viewing_history" not in st.session_state' in text
    assert "def render_history_view()" in text
    assert "历史记录为只读" in text


def test_empty_workspace_still_renders_current_conversation_messages():
    
    text = APP.read_text(encoding="utf-8")
    empty_body = text.split("def render_empty_state() -> None:", 1)[1].split("\ndef ", 1)[0]

    assert "render_message_history()" in empty_body
