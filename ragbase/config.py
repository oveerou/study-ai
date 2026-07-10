import os
from pathlib import Path


class Config:
    class Path:
        APP_HOME = Path(os.getenv("APP_HOME", Path(__file__).parent.parent))
        DATABASE_DIR = APP_HOME / "docs-db"
        DOCUMENTS_DIR = APP_HOME / "tmp"
        IMAGES_DIR = APP_HOME / "images"
        HISTORY_DIR = APP_HOME / "history"
        KNOWLEDGE_GRAPH_DIR = APP_HOME / "knowledge-graphs"

    class Database:
        DOCUMENTS_COLLECTION = "documents"

    class Model:
        EMBEDDINGS = "BAAI/bge-small-zh-v1.5"
        RERANKER = "ms-marco-MultiBERT-L-12"
        LOCAL_LLM = "gemma2:9b"
        REMOTE_LLM = "llama-3.1-70b-versatile"
        OPENAI_COMPATIBLE = os.getenv("MODEL_NAME", "deepseek-chat")
        OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
        TEMPERATURE = 0.0
        MAX_TOKENS = 8000
        USE_LOCAL = False

    class Retriever:
        USE_RERANKER = True
        USE_CHAIN_FILTER = False

    DEBUG = False
    CONVERSATION_MESSAGES_LIMIT = 0
