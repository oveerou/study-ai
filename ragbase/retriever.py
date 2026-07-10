from typing import Optional, Sequence

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.chain_filter import LLMChainFilter
from langchain_core.language_models import BaseLanguageModel
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain_qdrant import Qdrant
from qdrant_client.http.models import FieldCondition, Filter, MatchValue

from ragbase.config import Config
from ragbase.model import create_embeddings, create_reranker


def build_source_filter(source_names: Sequence[str]) -> Filter | None:
    names = list(dict.fromkeys(str(name) for name in source_names if str(name).strip()))
    if not names:
        return None
    return Filter(
        should=[
            FieldCondition(
                key="metadata.source_name",
                match=MatchValue(value=name),
            )
            for name in names
        ]
    )


def create_retriever(
    llm: BaseLanguageModel,
    vector_store: Optional[VectorStore] = None,
    source_names: Optional[Sequence[str]] = None,
) -> VectorStoreRetriever:
    if not vector_store:
        vector_store = Qdrant.from_existing_collection(
            embedding=create_embeddings(),
            collection_name=Config.Database.DOCUMENTS_COLLECTION,
            path=Config.Path.DATABASE_DIR,
        )

    search_kwargs = {"k": 5}
    source_filter = build_source_filter(source_names or [])
    if source_filter is not None:
        search_kwargs["filter"] = source_filter

    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs=search_kwargs
    )

    if Config.Retriever.USE_RERANKER:
        retriever = ContextualCompressionRetriever(
            base_compressor=create_reranker(), base_retriever=retriever
        )

    if Config.Retriever.USE_CHAIN_FILTER:
        retriever = ContextualCompressionRetriever(
            base_compressor=LLMChainFilter.from_llm(llm), base_retriever=retriever
        )

    return retriever
