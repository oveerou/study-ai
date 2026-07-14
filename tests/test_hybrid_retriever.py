

from __future__ import annotations

from langchain_core.documents import Document

from ragbase.hybrid_retriever import (
    HybridRetriever,
    RankedDocument,
    bm25_search,
    expand_context,
    lexical_tokens,
    reciprocal_rank_fusion,
)


def _chunk(
    chunk_id: str,
    text: str,
    *,
    source_id: str = "source-course",
    chunk_index: int = 0,
    parent_id: str = "parent-course",
    parent_content: str = "",
    token_count: int | None = None,
) -> Document:
    metadata = {
        "chunk_id": chunk_id,
        "source_id": source_id,
        "chunk_index": chunk_index,
        "parent_id": parent_id,
        "parent_content": parent_content,
    }
    if token_count is not None:
        metadata["token_count"] = token_count
    return Document(page_content=text, metadata=metadata)


def test_lexical_tokens_keep_chinese_terms_and_latin_abbreviations():
    tokens = set(lexical_tokens("Dm\u8ddd\u79bb\u3001PSNR\u4e0e\u56fe\u50cf\u8bc4\u4ef7"))

    assert {
        "dm",
        "psnr",
        "\u8ddd",
        "\u79bb",
        "\u8ddd\u79bb",
        "\u56fe\u50cf",
        "\u8bc4\u4ef7",
    }.issubset(tokens)


def test_bm25_recovers_exact_course_term():
    distance = _chunk("distance", "Dm distance includes Euclidean, D4, and D8 metrics")
    unrelated = _chunk("filter", "Mean filtering smooths image noise")

    results = bm25_search("Dm distance", [unrelated, distance], top_k=20)

    assert results[0].document.metadata["chunk_id"] == "distance"
    assert results[0].score > 0


def test_rrf_fuses_by_chunk_id_without_duplicates_deterministically():
    shared = _chunk("shared", "shared evidence")
    dense_only = _chunk("dense", "dense evidence")
    lexical_only = _chunk("lexical", "lexical evidence")
    dense = [RankedDocument(dense_only, 0.9), RankedDocument(shared, 0.8)]
    lexical = [RankedDocument(shared, 3.0), RankedDocument(lexical_only, 2.0)]

    first = reciprocal_rank_fusion([dense, lexical])
    second = reciprocal_rank_fusion([dense, lexical])

    first_ids = [item.document.metadata["chunk_id"] for item in first]
    assert first_ids == [item.document.metadata["chunk_id"] for item in second]
    assert first_ids.count("shared") == 1
    assert first_ids[0] == "shared"


def test_expand_context_uses_parent_without_duplicating_its_neighbors():
    previous = _chunk("previous", "previous section", chunk_index=0, token_count=2)
    hit = _chunk(
        "hit",
        "matched section",
        chunk_index=1,
        parent_content="complete parent section",
        token_count=2,
    )
    following = _chunk("following", "following section", chunk_index=2, token_count=2)
    documents_by_id = {doc.metadata["chunk_id"]: doc for doc in [previous, hit, following]}

    expanded = expand_context(
        [RankedDocument(hit, 1.0)],
        documents_by_id,
        budget_tokens=10,
    )

    assert expanded[0].metadata["chunk_id"] == "hit"
    assert expanded[0].metadata["context_role"] == "parent"
    assert expanded[0].page_content == "complete parent section"
    assert len(expanded) == 1


def test_expand_context_parent_uses_parent_page_range_metadata():
    hit = _chunk(
        "hit",
        "page two evidence",
        chunk_index=1,
        parent_content="[Page 1]\nfirst\n\n[Page 2]\nsecond\n\n[Page 3]\nthird",
        token_count=2,
    )
    hit.metadata.update(
        {
            "page": 2,
            "page_start": 2,
            "page_end": 2,
            "parent_page_start": 1,
            "parent_page_end": 3,
        }
    )

    expanded = expand_context(
        [RankedDocument(hit, 1.0)],
        {"hit": hit},
        budget_tokens=50,
    )

    parent = expanded[0]
    assert parent.metadata["context_role"] == "parent"
    assert parent.metadata["page_start"] == 1
    assert parent.metadata["page_end"] == 3
    assert "page" not in parent.metadata


class _VectorStore:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.calls: list[dict[str, object]] = []

    def similarity_search(self, query: str, **kwargs):
        self.calls.append({"query": query, **kwargs})
        return list(self.documents)


class _FailingReranker:
    def compress_documents(self, documents, query: str, callbacks=None):
        raise RuntimeError("reranker unavailable")


def test_hybrid_retriever_filters_source_ids_and_falls_back_to_fused_order():
    dense = _chunk("dense", "generic image processing", source_id="source-a")
    exact = _chunk("exact", "Sobel gradient operator", source_id="source-a")
    excluded = _chunk("excluded", "Sobel gradient operator", source_id="source-b")
    store = _VectorStore([dense, exact, excluded])
    retriever = HybridRetriever(
        vector_store=store,
        documents=[dense, exact, excluded],
        source_ids=("source-a",),
        reranker=_FailingReranker(),
        context_max_tokens=50,
    )

    results = retriever.invoke("Sobel gradient")

    assert results
    assert results[0].metadata["chunk_id"] == "exact"
    assert all(doc.metadata.get("source_id") == "source-a" for doc in results)
    assert store.calls[0]["k"] == 20
    conditions = store.calls[0]["filter"].should
    assert [condition.key for condition in conditions] == ["metadata.source_id"]
