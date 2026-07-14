

from typing import Optional, Sequence

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.chain_filter import LLMChainFilter
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from langchain_qdrant import Qdrant
from qdrant_client.http.models import FieldCondition, Filter, MatchValue

from ragbase.config import Config
from ragbase.hybrid_retriever import HybridRetriever
from ragbase.model import create_embeddings, create_reranker


def _create_optional_reranker():
    
    if not Config.Retriever.USE_RERANKER:
        return None
    try:
        return create_reranker()
    except Exception:
        return None


def build_source_filter(
    source_names: Sequence[str],
    metadata_key: str = "metadata.source_name",
) -> Filter | None:
    
    names = list(dict.fromkeys(str(name) for name in source_names if str(name).strip()))
    if not names:
        return None
    return Filter(
        should=[
            FieldCondition(
                key=metadata_key,
                match=MatchValue(value=name),
            )
            for name in names
        ]
    )


def create_retriever(
    llm: BaseLanguageModel,
    vector_store: Optional[VectorStore] = None,
    source_names: Optional[Sequence[str]] = None,
    source_ids: Optional[Sequence[str]] = None,
    chunk_documents: Optional[Sequence[Document]] = None,
) -> BaseRetriever:
    
    if not vector_store:
        vector_store = Qdrant.from_existing_collection(
            embedding=create_embeddings(),
            collection_name=Config.Database.DOCUMENTS_COLLECTION,
            path=Config.Path.DATABASE_DIR,
        )

    if chunk_documents is not None:
        retriever: BaseRetriever = HybridRetriever(
            vector_store=vector_store,
            documents=list(chunk_documents),
            source_ids=tuple(source_ids or ()),
            source_names=tuple(source_names or ()),
            reranker=_create_optional_reranker(),
        )
    else:
        search_kwargs = {"k": Config.Retriever.RETRIEVAL_TOP_K}
        filter_values = source_ids if source_ids is not None else source_names or []
        metadata_key = (
            "metadata.source_id" if source_ids is not None else "metadata.source_name"
        )
        source_filter = build_source_filter(filter_values, metadata_key=metadata_key)
        if source_filter is not None:
            search_kwargs["filter"] = source_filter

        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs=search_kwargs
        )

        reranker = _create_optional_reranker()
        if reranker is not None:
            retriever = ContextualCompressionRetriever(
                base_compressor=reranker, base_retriever=retriever
            )

    if Config.Retriever.USE_CHAIN_FILTER:
        retriever = ContextualCompressionRetriever(
            base_compressor=LLMChainFilter.from_llm(llm), base_retriever=retriever
        )

    return retriever
