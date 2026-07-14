

from __future__ import annotations

import asyncio

from langchain_core.documents import Document

from ragbase.planner_schema import OperationPlan
from ragbase.source_registry import SourceRecord


def _record(source_id: str, name: str) -> SourceRecord:
    return SourceRecord(
        source_id=source_id,
        source_name=name,
        normalized_name=name.rsplit(".", 1)[0],
        source_type="pdf",
        session_id="session-1",
        file_hash=source_id,
        source_path=name,
    )


def _runtime(orchestrator):
    records = [_record("src_first", "first.pdf"), _record("src_second", "second.pdf")]
    chunks = [
        Document(
            page_content="FIRST EVIDENCE",
            metadata={"source_id": "src_first", "source_name": "first.pdf", "chunk_id": "c1"},
        ),
        Document(
            page_content="SECOND EVIDENCE",
            metadata={"source_id": "src_second", "source_name": "second.pdf", "chunk_id": "c2"},
        ),
    ]
    profiles = [
        {
            "source_id": "src_first",
            "name": "first.pdf",
            "sections": [
                {"page": 2, "content": "SECOND PAGE"},
                {"page": 1, "content": "FIRST PAGE"},
            ],
        },
        {
            "source_id": "src_second",
            "name": "second.pdf",
            "sections": [{"page": 1, "content": "OTHER SOURCE"}],
        },
    ]
    return orchestrator.OrchestratorRuntime(
        model=object(),
        source_records=records,
        source_profiles=profiles,
        active_source_ids=("src_first",),
        chunk_documents=chunks,
        vector_store=object(),
        recent_messages=[{"role": "user", "content": "earlier"}],
        session_id="session-1",
    )


def test_list_sources_reads_registry_without_retrieval_or_answer_generation(monkeypatch):
    import ragbase.orchestrator as orchestrator

    runtime = _runtime(orchestrator)

    async def fake_plan(**_kwargs):
        return OperationPlan("list_sources", (), "catalog", 1.0)

    monkeypatch.setattr(orchestrator, "plan_operation", fake_plan)
    monkeypatch.setattr(
        orchestrator,
        "create_retriever",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("retrieval must not run")),
    )
    monkeypatch.setattr(
        orchestrator,
        "generate_chat_answer",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("answer model must not run")),
    )

    result = asyncio.run(orchestrator.execute_question("catalog", runtime))

    assert "first.pdf" in result.answer
    assert "second.pdf" in result.answer
    assert result.documents == ()


def test_full_text_reads_only_selected_source_in_page_order(monkeypatch):
    import ragbase.orchestrator as orchestrator

    runtime = _runtime(orchestrator)

    async def fake_plan(**_kwargs):
        return OperationPlan("read_source", ("src_first",), "read", 1.0, read_mode="full_text")

    monkeypatch.setattr(orchestrator, "plan_operation", fake_plan)

    result = asyncio.run(orchestrator.execute_question("read", runtime))

    assert "FIRST PAGE" in result.answer
    assert "SECOND PAGE" in result.answer
    assert result.answer.index("FIRST PAGE") < result.answer.index("SECOND PAGE")
    assert "OTHER SOURCE" not in result.answer
    assert result.active_source_ids == ("src_first",)


def test_search_plan_query_builds_hybrid_retriever_with_source_ids(monkeypatch):
    import ragbase.orchestrator as orchestrator

    runtime = _runtime(orchestrator)
    captured = {}
    retrieved = Document(
        page_content="retrieved",
        metadata={"source_id": "src_second", "source_name": "second.pdf", "chunk_id": "c2"},
    )

    async def fake_plan(**_kwargs):
        return OperationPlan("search", ("src_second",), "rewritten query", 1.0)

    async def unexpected_rewrite(**_kwargs):
        raise AssertionError("search must not make a second rewrite-model call")

    def fake_create_retriever(model, **kwargs):
        captured.update(kwargs)
        return "hybrid-retriever"

    def fake_create_chain(model, retriever, use_history):
        captured["chain"] = (retriever, use_history)
        return "grounded-chain"

    async def fake_ask_question(chain, question, session_id):
        captured["ask"] = (chain, question, session_id)
        yield [retrieved]
        yield "grounded "
        yield "answer"

    monkeypatch.setattr(orchestrator, "plan_operation", fake_plan)
    monkeypatch.setattr(orchestrator, "rewrite_query", unexpected_rewrite, raising=False)
    monkeypatch.setattr(orchestrator, "create_retriever", fake_create_retriever)
    monkeypatch.setattr(orchestrator, "create_chain", fake_create_chain)
    monkeypatch.setattr(orchestrator, "ask_question", fake_ask_question)

    result = asyncio.run(orchestrator.execute_question("original", runtime))

    assert captured["source_ids"] == ("src_second",)
    assert captured["chunk_documents"] == runtime.chunk_documents
    assert captured["chain"] == ("hybrid-retriever", False)
    assert captured["ask"] == ("grounded-chain", "rewritten query", "session-1")
    assert result.answer == "grounded answer"
    assert result.documents == (retrieved,)


def test_chat_uses_recent_dialogue_without_source_context(monkeypatch):
    import ragbase.orchestrator as orchestrator

    runtime = _runtime(orchestrator)
    captured = {}

    async def fake_plan(**_kwargs):
        return OperationPlan("chat", (), "hello", 1.0)

    async def fake_chat_answer(model, question, recent_messages):
        captured["chat"] = (question, recent_messages)
        return "hello back"

    monkeypatch.setattr(orchestrator, "plan_operation", fake_plan)
    monkeypatch.setattr(orchestrator, "generate_chat_answer", fake_chat_answer)
    monkeypatch.setattr(
        orchestrator,
        "create_retriever",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("chat must not retrieve")),
    )

    result = asyncio.run(orchestrator.execute_question("hello", runtime))

    assert result.answer == "hello back"
    assert result.documents == ()
    assert captured["chat"] == ("hello", runtime.recent_messages)
    assert result.active_source_ids == runtime.active_source_ids


def test_ambiguous_selected_scope_requests_source_without_retrieval(monkeypatch):
    import ragbase.orchestrator as orchestrator
    from ragbase.source_resolver import SourceCandidate, SourceResolution

    runtime = _runtime(orchestrator)
    resolution = SourceResolution(
        (),
        0.65,
        (
            SourceCandidate("src_first", "first.pdf", 0.65, "similar name"),
            SourceCandidate("src_second", "second.pdf", 0.64, "similar name"),
        ),
    )

    async def fake_plan(**_kwargs):
        return OperationPlan("search", (), "question", 0.65, scope="selected")

    monkeypatch.setattr(orchestrator, "resolve_sources", lambda *_args: resolution)
    monkeypatch.setattr(orchestrator, "plan_operation", fake_plan)
    monkeypatch.setattr(
        orchestrator,
        "create_retriever",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("ambiguous source selection must not retrieve")
        ),
    )

    result = asyncio.run(orchestrator.execute_question("which one", runtime))

    assert result.documents == ()
    assert "first.pdf" in result.answer
    assert "second.pdf" in result.answer
    assert "请选择" in result.answer


def test_search_result_exposes_deterministic_citations(monkeypatch):
    import ragbase.orchestrator as orchestrator

    runtime = _runtime(orchestrator)
    first = Document(
        page_content="first evidence",
        metadata={
            "source_id": "src_second",
            "source_name": "second.pdf",
            "chunk_id": "chunk-2",
            "page_start": 3,
            "page_end": 4,
        },
    )
    second = Document(
        page_content="second evidence",
        metadata={
            "source_id": "src_first",
            "source_name": "first.pdf",
            "chunk_id": "chunk-1",
            "page": 2,
        },
    )

    async def fake_plan(**_kwargs):
        return OperationPlan("search", ("src_second",), "question", 1.0)

    async def fake_ask_question(*_args, **_kwargs):
        yield [first, second, first]
        yield "supported answer"

    monkeypatch.setattr(orchestrator, "plan_operation", fake_plan)
    monkeypatch.setattr(orchestrator, "create_retriever", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(orchestrator, "create_chain", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(orchestrator, "ask_question", fake_ask_question)

    result = asyncio.run(orchestrator.execute_question("question", runtime))

    assert [citation.chunk_id for citation in result.citations] == ["chunk-2", "chunk-1"]
    assert result.citations[0].source_name == "second.pdf"
    assert result.citations[0].page_start == 3
    assert result.citations[0].page_end == 4
    assert result.evidence_level == "high"
    assert result.missing_information is None


def test_insufficient_answer_marker_sets_low_evidence_and_missing_information(monkeypatch):
    import ragbase.orchestrator as orchestrator

    runtime = _runtime(orchestrator)
    retrieved = Document(
        page_content="unrelated evidence",
        metadata={
            "source_id": "src_second",
            "source_name": "second.pdf",
            "chunk_id": "chunk-2",
            "page": 3,
        },
    )

    async def fake_plan(**_kwargs):
        return OperationPlan("search", ("src_second",), "question", 1.0)

    async def fake_ask_question(*_args, **_kwargs):
        yield [retrieved]
        yield "当前资料依据不足，无法回答这个问题。"

    monkeypatch.setattr(orchestrator, "plan_operation", fake_plan)
    monkeypatch.setattr(orchestrator, "create_retriever", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(orchestrator, "create_chain", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(orchestrator, "ask_question", fake_ask_question)

    result = asyncio.run(orchestrator.execute_question("question", runtime))

    assert result.evidence_level == "low"
    assert result.missing_information
