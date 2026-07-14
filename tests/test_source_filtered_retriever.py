

from __future__ import annotations

from langchain_core.documents import Document

from ragbase.config import Config
from ragbase.hybrid_retriever import HybridRetriever
from ragbase.retriever import build_source_filter, create_retriever


def test_source_filter_targets_exact_metadata_source_names():
    
    source_filter = build_source_filter(["复习大纲.pdf", "图像增强.pdf"])

    assert source_filter is not None
    conditions = source_filter.should
    assert [condition.key for condition in conditions] == [
        "metadata.source_name",
        "metadata.source_name",
    ]
    assert [condition.match.value for condition in conditions] == [
        "复习大纲.pdf",
        "图像增强.pdf",
    ]


def test_empty_source_selection_does_not_filter_global_retrieval():
    
    assert build_source_filter([]) is None


def test_source_filter_can_target_stable_source_ids():
    source_filter = build_source_filter(
        ["src-course-a", "src-course-b"],
        metadata_key="metadata.source_id",
    )

    assert source_filter is not None
    assert [condition.key for condition in source_filter.should] == [
        "metadata.source_id",
        "metadata.source_id",
    ]
    assert [condition.match.value for condition in source_filter.should] == [
        "src-course-a",
        "src-course-b",
    ]


def test_retrieval_configuration_keeps_enough_fusion_candidates():
    assert Config.Retriever.RETRIEVAL_TOP_K == 20
    assert Config.Retriever.RERANK_TOP_N == 5
    assert Config.Retriever.FINAL_TOP_N == 5
    assert Config.Retriever.CONTEXT_MAX_TOKENS > 0


def test_flashrank_initialization_failure_keeps_hybrid_rrf_retrieval(monkeypatch):
    
    first = Document(page_content="dense first", metadata={"chunk_id": "chunk-1"})
    second = Document(page_content="dense second", metadata={"chunk_id": "chunk-2"})

    class VectorStore:
        def similarity_search(self, query, **kwargs):
            return [first, second]

    def fail_to_create_reranker():
        raise RuntimeError("FlashRank model unavailable")

    monkeypatch.setattr(Config.Retriever, "USE_RERANKER", True)
    monkeypatch.setattr(Config.Retriever, "USE_CHAIN_FILTER", False)
    monkeypatch.setattr("ragbase.retriever.create_reranker", fail_to_create_reranker)

    retriever = create_retriever(
        object(),
        vector_store=VectorStore(),
        chunk_documents=[first, second],
    )

    assert isinstance(retriever, HybridRetriever)
    assert retriever.reranker is None
    assert [doc.page_content for doc in retriever.invoke("missing lexical term")] == [
        "dense first",
        "dense second",
    ]
