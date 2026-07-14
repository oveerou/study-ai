

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from ragbase.config import Config


_TERM_RE = re.compile(r"[a-z0-9_]+|[\u3400-\u4dbf\u4e00-\u9fff]+")


@dataclass(frozen=True)
class RankedDocument:
    

    document: Document
    score: float


def lexical_tokens(text: str) -> list[str]:
    
    normalized = unicodedata.normalize("NFKC", text or "").casefold()
    tokens: list[str] = []
    for term in _TERM_RE.findall(normalized):
        if not any("\u3400" <= char <= "\u9fff" for char in term):
            tokens.append(term)
            continue
        characters = list(term)
        tokens.extend(characters)
        tokens.extend(
            characters[index] + characters[index + 1]
            for index in range(len(characters) - 1)
        )
        if len(characters) > 2:
            tokens.append(term)
    return tokens


def bm25_search(
    query: str,
    documents: Sequence[Document],
    top_k: int = 20,
) -> list[RankedDocument]:
    
    if top_k <= 0 or not documents:
        return []
    query_terms = set(lexical_tokens(query))
    if not query_terms:
        return []

    tokenized = [lexical_tokens(document.page_content) for document in documents]
    document_count = len(tokenized)
    average_length = sum(len(tokens) for tokens in tokenized) / document_count or 1.0
    document_frequency = {
        term: sum(term in tokens for tokens in tokenized) for term in query_terms
    }
    k1 = 1.5
    b = 0.75
    ranked: list[tuple[float, int, str, Document]] = []

    for position, (document, tokens) in enumerate(zip(documents, tokenized)):
        frequencies = Counter(tokens)
        score = 0.0
        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            frequency_count = document_frequency[term]
            inverse_frequency = math.log(
                1 + (document_count - frequency_count + 0.5) / (frequency_count + 0.5)
            )
            denominator = frequency + k1 * (
                1 - b + b * len(tokens) / average_length
            )
            score += inverse_frequency * frequency * (k1 + 1) / denominator
        if score > 0:
            ranked.append((score, position, _document_id(document), document))

    ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [RankedDocument(document=item[3], score=item[0]) for item in ranked[:top_k]]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RankedDocument]],
    k: int = 60,
) -> list[RankedDocument]:
    
    scores: dict[str, float] = {}
    documents: dict[str, Document] = {}
    first_positions: dict[str, tuple[int, int]] = {}
    for ranking_index, ranking in enumerate(rankings):
        seen_in_ranking: set[str] = set()
        for rank, item in enumerate(ranking, start=1):
            document_id = _document_id(item.document)
            if document_id in seen_in_ranking:
                continue
            seen_in_ranking.add(document_id)
            scores[document_id] = scores.get(document_id, 0.0) + 1.0 / (k + rank)
            documents.setdefault(document_id, item.document)
            first_positions.setdefault(document_id, (ranking_index, rank))

    ordered_ids = sorted(
        scores,
        key=lambda document_id: (
            -scores[document_id],
            first_positions[document_id],
            document_id,
        ),
    )
    return [RankedDocument(documents[item], scores[item]) for item in ordered_ids]


def expand_context(
    results: Sequence[RankedDocument],
    documents_by_id: Mapping[str, Document],
    budget_tokens: int,
) -> list[Document]:
    
    if budget_tokens <= 0:
        return []
    documents = list(documents_by_id.values())
    neighbors = {
        (_source_key(document), _chunk_index(document)): document
        for document in documents
        if _chunk_index(document) is not None
    }
    expanded: list[Document] = []
    seen: set[str] = set()
    remaining = budget_tokens

    def add(document: Document, identity: str) -> bool:
        nonlocal remaining
        if identity in seen:
            return False
        cost = _context_token_count(document)
        if cost > remaining:
            return False
        expanded.append(document)
        seen.add(identity)
        remaining -= cost
        return True

    for result in results:
        hit = result.document
        hit_id = _document_id(hit)
        parent_content = str(hit.metadata.get("parent_content") or "").strip()
        parent = None
        if parent_content and parent_content != hit.page_content.strip():
            metadata = dict(hit.metadata)
            metadata["context_role"] = "parent"
            metadata.pop("token_count", None)
            parent_page_start = metadata.get("parent_page_start")
            parent_page_end = metadata.get("parent_page_end")
            if parent_page_start is not None:
                metadata["page_start"] = parent_page_start
            if parent_page_end is not None:
                metadata["page_end"] = parent_page_end
            if parent_page_start is not None or parent_page_end is not None:
                metadata.pop("page", None)
            parent = Document(page_content=parent_content, metadata=metadata)

        parent_added = parent is not None and add(parent, hit_id)
        if not parent_added:
            add(hit, hit_id)
        else:
            continue

        index = _chunk_index(hit)
        if index is None:
            continue
        source = _source_key(hit)
        for neighbor_index in (index - 1, index + 1):
            neighbor = neighbors.get((source, neighbor_index))
            if neighbor is not None:
                add(neighbor, _document_id(neighbor))

    return expanded


class HybridRetriever(BaseRetriever):
    

    vector_store: Any
    documents: list[Document]
    source_ids: tuple[str, ...] = ()
    source_names: tuple[str, ...] = ()
    reranker: Any = None
    retrieval_top_k: int = Config.Retriever.RETRIEVAL_TOP_K
    rerank_top_n: int = Config.Retriever.RERANK_TOP_N
    final_top_n: int = Config.Retriever.FINAL_TOP_N
    context_max_tokens: int = Config.Retriever.CONTEXT_MAX_TOKENS

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        scoped_documents = [
            document for document in self.documents if self._source_matches(document)
        ]
        search_kwargs: dict[str, Any] = {"k": self.retrieval_top_k}
        values: Sequence[str] = self.source_ids or self.source_names
        if values:
            from ragbase.retriever import build_source_filter

            metadata_key = (
                "metadata.source_id" if self.source_ids else "metadata.source_name"
            )
            search_kwargs["filter"] = build_source_filter(
                values,
                metadata_key=metadata_key,
            )

        dense_documents = self.vector_store.similarity_search(query, **search_kwargs)
        dense = [
            RankedDocument(document, 1.0 / rank)
            for rank, document in enumerate(dense_documents, start=1)
            if self._source_matches(document)
        ]
        lexical = bm25_search(query, scoped_documents, top_k=self.retrieval_top_k)
        fused = reciprocal_rank_fusion([dense, lexical])
        selected = self._rerank(query, fused, run_manager)
        documents_by_id = {
            _document_id(document): document for document in scoped_documents
        }
        return expand_context(
            selected[: self.final_top_n],
            documents_by_id,
            self.context_max_tokens,
        )

    def _rerank(
        self,
        query: str,
        fused: Sequence[RankedDocument],
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[RankedDocument]:
        if self.reranker is None or not fused:
            return list(fused)
        candidates = [item.document for item in fused[: self.retrieval_top_k]]
        try:
            if hasattr(self.reranker, "top_n"):
                self.reranker.top_n = self.rerank_top_n
            reranked = self.reranker.compress_documents(
                candidates,
                query,
                callbacks=run_manager.get_child(),
            )
        except Exception:
            return list(fused)
        if not reranked:
            return list(fused)
        return [
            RankedDocument(
                document,
                float(document.metadata.get("relevance_score", len(reranked) - rank)),
            )
            for rank, document in enumerate(reranked)
        ]

    def _source_matches(self, document: Document) -> bool:
        if self.source_ids:
            return str(document.metadata.get("source_id") or "") in self.source_ids
        if self.source_names:
            return str(document.metadata.get("source_name") or "") in self.source_names
        return True


def _document_id(document: Document) -> str:
    chunk_id = str(document.metadata.get("chunk_id") or "").strip()
    if chunk_id:
        return chunk_id
    digest = hashlib.sha256(document.page_content.encode("utf-8")).hexdigest()[:24]
    return f"content-{digest}"


def _chunk_index(document: Document) -> int | None:
    try:
        value = document.metadata.get("chunk_index")
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _source_key(document: Document) -> str:
    return str(
        document.metadata.get("source_id")
        or document.metadata.get("source_name")
        or document.metadata.get("source")
        or ""
    )


def _context_token_count(document: Document) -> int:
    try:
        value = int(document.metadata.get("token_count", 0))
        if value > 0:
            return value
    except (TypeError, ValueError):
        pass
    text = document.page_content or ""
    cjk_count = sum("\u3400" <= char <= "\u9fff" for char in text)
    latin_count = len(re.findall(r"[a-zA-Z0-9_]+|[^\x00-\x7f]", text))
    return max(1, cjk_count + latin_count)
