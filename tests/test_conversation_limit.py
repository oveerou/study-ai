from __future__ import annotations

from pathlib import Path

from ragbase.config import Config


def test_conversation_has_no_turn_limit_blocker():
    app_text = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")

    assert Config.CONVERSATION_MESSAGES_LIMIT == 0
    assert "当前对话已达到轮数上限" not in app_text
