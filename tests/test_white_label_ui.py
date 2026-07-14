

from pathlib import Path


APP = Path(__file__).resolve().parents[1] / "app.py"


def test_app_uses_learning_agent_branding():
    
    text = APP.read_text(encoding="utf-8")

    assert "学习助手" in text
    assert "知识库问答" in text
    assert "执行过程" in text
    assert "引用来源" in text


def test_app_does_not_show_original_ragbase_branding():
    
    text = APP.read_text(encoding="utf-8")

    forbidden = [
        "RagBase",
        "个人学习助手",
        "个人资料",
        "Get answers from your documents",
        "Upload PDF files",
        "Ask your question here",
        "🐧",
        "assistant-avatar.png",
        "user-avatar.png",
        "Source #",
    ]
    for value in forbidden:
        assert value not in text


def test_published_code_does_not_depend_on_local_machine_paths():
    root = Path(__file__).resolve().parents[1]
    checked_files = [root / "app.py", root / "ragbase" / "model.py"]

    for path in checked_files:
        text = path.read_text(encoding="utf-8")
        assert "universal-knowledge-agent" not in text
        assert "D:\\A_shixi" not in text
        assert "D:/A_shixi" not in text
