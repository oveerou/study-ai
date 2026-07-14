





from __future__ import annotations

import html
import re
import zipfile
from pathlib import Path
from typing import Iterable, List
from urllib.parse import urlparse

from langchain_community.document_loaders import PyPDFium2Loader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_qdrant import Qdrant

from ragbase.chunking import ChunkingRouter, EmbeddingTokenizer
from ragbase.config import Config


FILE_EXTENSIONS = {".pdf", ".docx", ".md", ".txt"}
CODE_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".go",
    ".rs",
    ".cpp",
    ".c",
    ".h",
    ".cs",
    ".php",
    ".rb",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".html",
    ".css",
    ".sql",
}
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", "dist", "build", "docs-db", "tmp"}
MAX_CODE_FILE_BYTES = 1_000_000


def load_path_documents(path: str | Path) -> list[Document]:
    
    source_path = Path(path)
    suffix = source_path.suffix.lower()
    if suffix == ".pdf":
        docs = PyPDFium2Loader(source_path).load()
        for index, doc in enumerate(docs):
            doc.metadata.update(
                {
                    "source": str(source_path),
                    "source_name": source_path.name,
                    "source_type": "pdf",
                    "page": int(doc.metadata.get("page", index)) + 1,
                }
            )
        return [doc for doc in docs if (doc.page_content or "").strip()]
    if suffix == ".docx":
        return [
            Document(
                page_content=_read_docx_text(source_path),
                metadata={"source": str(source_path), "source_name": source_path.name, "source_type": "docx"},
            )
        ]
    if suffix in {".md", ".txt"}:
        return [
            Document(
                page_content=source_path.read_text(encoding="utf-8", errors="ignore"),
                metadata={"source": str(source_path), "source_name": source_path.name, "source_type": suffix.lstrip(".")},
            )
        ]
    raise ValueError(f"不支持的文件类型: {suffix}")


def load_url_documents(url: str) -> list[Document]:
    
    import requests

    url = url.strip()
    if not url:
        return []

    candidate_urls = _github_readme_candidates(url) or [url]
    last_error: Exception | None = None
    for candidate in candidate_urls:
        try:
            response = requests.get(
                candidate,
                timeout=20,
                headers={"User-Agent": "study-ai/1.0"},
            )
            response.raise_for_status()
            text = _html_to_text(response.text)
            if text.strip():
                return [
                    Document(
                        page_content=text,
                        metadata={"source": url, "source_name": url, "source_type": "url", "fetched_url": candidate},
                    )
                ]
        except Exception as exc:
            last_error = exc
    if last_error:
        raise RuntimeError(f"URL 读取失败: {last_error}") from last_error
    return []


def load_code_dir_documents(dir_path: str | Path) -> list[Document]:
    
    root = Path(dir_path)
    if not root.exists() or not root.is_dir():
        raise ValueError(f"代码目录不存在: {root}")

    docs: list[Document] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in CODE_EXTENSIONS:
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        if path.stat().st_size > MAX_CODE_FILE_BYTES:
            continue
        rel_path = path.relative_to(root).as_posix()
        content = path.read_text(encoding="utf-8", errors="ignore")
        if not content.strip():
            continue
        docs.append(
            Document(
                page_content=f"文件: {rel_path}\n\n{content}",
                metadata={
                    "source": str(path),
                    "source_name": rel_path,
                    "source_type": "code",
                    "root": str(root),
                },
            )
        )
    return docs


def load_mixed_documents(items: Iterable[str]) -> list[Document]:
    
    docs: list[Document] = []
    for raw_item in items:
        item = (raw_item or "").strip()
        if not item:
            continue
        if item.startswith(("http://", "https://")):
            docs.extend(load_url_documents(item))
            continue
        path = Path(item)
        if path.is_file():
            docs.extend(load_path_documents(path))
        elif path.is_dir():
            docs.extend(load_code_dir_documents(path))
        else:
            raise ValueError(f"无法识别来源: {item}")
    return docs


def source_names_from_documents(documents: Iterable[Document]) -> list[str]:
    
    names: list[str] = []
    for doc in documents:
        name = str(doc.metadata.get("source_name") or doc.metadata.get("source") or "未命名来源")
        if name not in names:
            names.append(name)
    return names


class Ingestor:
    

    def __init__(self):
        
        self.embeddings = FastEmbedEmbeddings(model_name=Config.Model.EMBEDDINGS)
        tokenizer = EmbeddingTokenizer(
            tokenizer=self.embeddings._model.model.tokenizer
        )
        self.chunking_router = ChunkingRouter(tokenizer=tokenizer)
        self.chunk_documents: list[Document] = []

    def ingest(self, doc_paths: List[Path]) -> VectorStore:
        
        documents: list[Document] = []
        for doc_path in doc_paths:
            documents.extend(load_path_documents(doc_path))
        return self.ingest_documents(documents)

    def ingest_documents(self, source_documents: list[Document]) -> VectorStore:
        
        if not source_documents:
            raise ValueError("没有解析到可索引内容")

        documents = []
        split_documents = self.chunking_router.split_documents(source_documents)
        for doc in split_documents:
            if (doc.page_content or "").strip():
                documents.append(doc)
        self.chunk_documents = documents

        return Qdrant.from_documents(
            documents=documents,
            embedding=self.embeddings,
            path=Config.Path.DATABASE_DIR,
            collection_name=Config.Database.DOCUMENTS_COLLECTION,
        )


def _read_docx_text(path: Path) -> str:
    
    try:
        with zipfile.ZipFile(path) as archive:
            raw_xml = archive.read("word/document.xml").decode("utf-8", errors="ignore")
    except Exception as exc:
        raise RuntimeError(f"DOCX 解析失败: {exc}") from exc

    raw_xml = re.sub(r"</w:p>", "\n", raw_xml)
    text_parts = re.findall(r"<w:t[^>]*>(.*?)</w:t>", raw_xml, flags=re.DOTALL)
    return html.unescape("\n".join(_strip_tags(part) for part in text_parts))


def _html_to_text(raw: str) -> str:
    
    raw = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", raw)
    raw = re.sub(r"(?s)<br\s*/?>", "\n", raw)
    raw = re.sub(r"(?s)</(p|div|h[1-6]|li|tr)>", "\n", raw)
    return re.sub(r"\n{3,}", "\n\n", html.unescape(_strip_tags(raw))).strip()


def _strip_tags(raw: str) -> str:
    
    return re.sub(r"(?s)<[^>]+>", " ", raw)


def _github_readme_candidates(url: str) -> list[str]:
    
    parsed = urlparse(url)
    if parsed.netloc.lower() not in {"github.com", "www.github.com"}:
        return []
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        return []
    owner, repo = parts[0], parts[1]
    if len(parts) >= 4 and parts[2] == "blob":
        branch = parts[3]
        file_path = "/".join(parts[4:])
        return [f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{file_path}"]
    return [
        f"https://raw.githubusercontent.com/{owner}/{repo}/main/README.md",
        f"https://raw.githubusercontent.com/{owner}/{repo}/master/README.md",
        url,
    ]
