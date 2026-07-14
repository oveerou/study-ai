

from __future__ import annotations

import importlib.util
from pathlib import Path

from ragbase.config import Config


APP = Path(__file__).resolve().parents[1] / "app.py"


def test_app_exposes_local_knowledge_graph_generation_and_view():
    
    text = APP.read_text(encoding="utf-8")

    assert hasattr(Config.Path, "KNOWLEDGE_GRAPH_DIR")
    assert "def render_knowledge_graph_controls()" in text
    assert "extract_knowledge_graph(" in text
    assert "save_knowledge_graph(" in text
    assert "graph_to_interactive_html(" in text
    assert "components.html(" in text
    assert "st.graphviz_chart(" not in text
    assert "知识图谱" in text


def test_knowledge_graph_generation_does_not_crash_sidebar_on_exception():
    
    text = APP.read_text(encoding="utf-8")
    body = text.split("def render_knowledge_graph_controls() -> None:", 1)[1].split("\n\ndef render_sidebar", 1)[0]

    assert "try:" in body
    assert "except Exception as exc:" in body
    assert "st.error(" in body


def test_knowledge_graph_view_does_not_nest_expanders():
    
    text = APP.read_text(encoding="utf-8")
    body = text.split("def render_knowledge_graph_view() -> None:", 1)[1].split("\n\nasync def ask_chain", 1)[0]

    assert body.count("st.expander(") == 1
    assert 'st.markdown("**关系明细**")' in body


def test_knowledge_graph_view_shows_one_complete_graph_without_segment_picker():
    
    text = APP.read_text(encoding="utf-8")
    body = text.split("def render_knowledge_graph_view() -> None:", 1)[1].split("\n\nasync def ask_chain", 1)[0]

    assert "st.selectbox(" not in body
    assert "查看片段" not in body
    assert "全部关系" not in body
    assert "片段" not in body
    assert "graph_to_interactive_html(graph)" in body


def test_knowledge_graph_view_mentions_interactive_browser_capabilities():
    
    text = APP.read_text(encoding="utf-8")
    body = text.split("def render_knowledge_graph_view() -> None:", 1)[1].split("\n\nasync def ask_chain", 1)[0]

    assert "搜索、聚焦、拖拽、缩放、全屏" in body


def test_graphviz_runtime_dependency_is_installed():
    
    assert importlib.util.find_spec("graphviz") is not None
