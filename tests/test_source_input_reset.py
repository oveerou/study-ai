

from __future__ import annotations

from pathlib import Path


APP = Path(__file__).resolve().parents[1] / "app.py"


def test_new_session_clears_previous_runtime_sources():
    
    text = APP.read_text(encoding="utf-8")

    assert "def clear_runtime_sources()" in text
    assert "clear_runtime_sources()" in text
    assert "Config.Path.DATABASE_DIR" in text
    assert "Config.Path.DOCUMENTS_DIR" in text


def test_source_inputs_reset_after_import_and_clear():
    
    text = APP.read_text(encoding="utf-8")

    assert "def reset_source_inputs()" in text
    assert 'key=source_widget_key("uploaded_files")' in text
    assert 'key=source_widget_key("url_input")' in text
    assert 'key=source_widget_key("code_dir")' in text
    assert 'key=source_widget_key("mixed_items")' in text
    assert text.count("reset_source_inputs()") >= 2


def test_new_chat_archives_history_releases_qdrant_and_clears_sources():
    
    text = APP.read_text(encoding="utf-8")
    reset_chat_body = text.split("def reset_chat() -> None:", 1)[1].split("\ndef ", 1)[0]

    assert "persist_current_conversation()" in reset_chat_body
    assert "release_runtime_index()" in reset_chat_body
    assert "clear_runtime_sources()" in reset_chat_body
    assert "reset_source_inputs()" in reset_chat_body
    assert "source_names = []" in reset_chat_body


def test_chat_submission_generates_answer_without_forced_second_rerun():
    
    text = APP.read_text(encoding="utf-8")
    render_chat_body = text.split("def render_chat(chain) -> None:", 1)[1].split("\ndef ", 1)[0]

    assert "asyncio.run(ask_chain(question, chain))" in render_chat_body
    assert "st.session_state.pending_question = question" not in render_chat_body
    assert "st.rerun()" not in render_chat_body


def test_chat_view_keeps_scroll_at_bottom_after_updates():
    
    text = APP.read_text(encoding="utf-8")
    render_chat_body = text.split("def render_chat(chain) -> None:", 1)[1].split("\ndef ", 1)[0]

    assert "def scroll_chat_to_bottom()" in text
    assert "components.html(" in text
    assert "scroll_chat_to_bottom()" in render_chat_body


def test_main_answer_path_uses_model_planner_instead_of_keyword_router():
    
    text = APP.read_text(encoding="utf-8")
    ask_body = text.split("async def ask_chain", 1)[1].split("\ndef ", 1)[0]

    assert "execute_question(" in ask_body
    assert "plan_question(" not in ask_body
    assert "source_tool_route(" not in ask_body
    assert "active_source_ids" in ask_body


def test_agent_source_state_is_initialized_and_reset():
    
    text = APP.read_text(encoding="utf-8")

    assert 'if "active_source_names" not in st.session_state' in text
    assert 'if "active_source_ids" not in st.session_state' in text
    assert 'if "source_records" not in st.session_state' in text
    assert 'if "chunk_documents" not in st.session_state' in text
    assert text.count("st.session_state.active_source_names = []") >= 2
    assert text.count("st.session_state.active_source_ids = []") >= 2
    assert text.count("st.session_state.source_records = []") >= 2
    assert text.count("st.session_state.chunk_documents = []") >= 2


def test_indexing_attaches_source_records_before_chunking_and_keeps_chunks():
    
    text = APP.read_text(encoding="utf-8")
    index_body = text.split("def index_documents", 1)[1].split("\ndef ", 1)[0]

    assert "build_source_records(" in index_body
    assert "attach_source_records(" in index_body
    assert "source_records" in index_body
    assert "chunk_documents" in index_body


def test_indexing_releases_then_resets_storage_before_building():
    
    text = APP.read_text(encoding="utf-8")
    index_body = text.split("def index_documents", 1)[1].split("\ndef ", 1)[0]

    release_at = index_body.index("release_runtime_index()")
    reset_at = index_body.index("reset_index_storage(")
    build_at = index_body.index("build_chain_from_documents(")

    assert release_at < reset_at < build_at


def test_ingestor_exposes_recent_chunks_without_changing_return_value(monkeypatch):
    from langchain_core.documents import Document

    from ragbase import ingestor as ingestor_module

    source = Document(page_content="source", metadata={"source_name": "notes.pdf"})
    chunk = Document(page_content="chunk", metadata={"chunk_id": "c1"})
    instance = object.__new__(ingestor_module.Ingestor)
    instance.embeddings = object()
    instance.chunking_router = type(
        "Router",
        (),
        {"split_documents": lambda self, documents: [chunk]},
    )()
    sentinel = object()
    monkeypatch.setattr(
        ingestor_module.Qdrant,
        "from_documents",
        lambda **kwargs: sentinel,
    )

    result = instance.ingest_documents([source])

    assert result is sentinel
    assert instance.chunk_documents == [chunk]
