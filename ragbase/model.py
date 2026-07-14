





import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from ragbase.config import Config


def _load_model_env() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)


def create_llm() -> BaseLanguageModel:
    
    _load_model_env()
    if os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(
            temperature=Config.Model.TEMPERATURE,
            model=Config.Model.OPENAI_COMPATIBLE,
            base_url=Config.Model.OPENAI_BASE_URL,
            api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=Config.Model.MAX_TOKENS,
        )
    if Config.Model.USE_LOCAL:
        return ChatOllama(
            model=Config.Model.LOCAL_LLM,
            temperature=Config.Model.TEMPERATURE,
            keep_alive="1h",
            max_tokens=Config.Model.MAX_TOKENS,
        )
    else:
        return ChatGroq(
            temperature=Config.Model.TEMPERATURE,
            model_name=Config.Model.REMOTE_LLM,
            max_tokens=Config.Model.MAX_TOKENS,
        )


def create_embeddings() -> FastEmbedEmbeddings:
    
    return FastEmbedEmbeddings(model_name=Config.Model.EMBEDDINGS)


def create_reranker() -> FlashrankRerank:
    
    return FlashrankRerank(model=Config.Model.RERANKER)
