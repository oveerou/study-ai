

from __future__ import annotations

import ast
import hashlib
import json
import re
import tomllib
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Sequence

from fastembed import TextEmbedding
from langchain_core.documents import Document
from tokenizers import Tokenizer

from ragbase.config import Config


MAX_EMBEDDING_TOKENS = 480
TARGET_CHUNK_TOKENS = 420
DEFAULT_OVERLAP_TOKENS = 50
NUMBERED_ITEM_RE = re.compile(r"(?m)^\s*(?:第\s*)?(\d{1,3})\s*[.、．)]\s*")
MARKDOWN_HEADING_RE = re.compile(r"(?m)^#{1,6}\s+.+$")
STRUCTURED_SUFFIXES = {".json", ".yaml", ".yml", ".toml"}


class EmbeddingTokenizer:
    

    def __init__(
        self,
        model_name: str | None = None,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        if tokenizer is None:
            embedding = TextEmbedding(
                model_name=model_name or Config.Model.EMBEDDINGS,
                lazy_load=True,
            )
            tokenizer = embedding.model.tokenizer

        
        
        self._tokenizer = Tokenizer.from_str(tokenizer.to_str())
        self._tokenizer.no_truncation()

    def count(self, text: str) -> int:
        
        return len(self._tokenizer.encode(text or "").ids)

    def split(
        self,
        text: str,
        max_tokens: int = MAX_EMBEDDING_TOKENS,
        overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    ) -> list[str]:
        
        text = (text or "").strip()
        if not text:
            return []
        if max_tokens <= 0:
            raise ValueError("max_tokens 必须大于 0")
        if self.count(text) <= max_tokens:
            return [text]

        encoding = self._tokenizer.encode(text)
        offsets = [(start, end) for start, end in encoding.offsets if end > start]
        if not offsets:
            return self._split_by_characters(text, max_tokens, overlap_tokens)

        special_tokens = max(0, len(encoding.ids) - len(offsets))
        payload_size = max(1, max_tokens - special_tokens)
        overlap = min(max(0, overlap_tokens), max(0, payload_size - 1))
        parts: list[str] = []
        start_index = 0

        while start_index < len(offsets):
            end_index = min(start_index + payload_size, len(offsets))
            part = text[offsets[start_index][0] : offsets[end_index - 1][1]].strip()

            
            while part and self.count(part) > max_tokens and end_index > start_index + 1:
                end_index -= 1
                part = text[offsets[start_index][0] : offsets[end_index - 1][1]].strip()

            if not part:
                break
            parts.append(part)
            if end_index >= len(offsets):
                break
            start_index = max(start_index + 1, end_index - overlap)

        return parts

    def _split_by_characters(
        self,
        text: str,
        max_tokens: int,
        overlap_tokens: int,
    ) -> list[str]:
        
        parts: list[str] = []
        start = 0
        while start < len(text):
            low, high = start + 1, len(text)
            best = low
            while low <= high:
                middle = (low + high) // 2
                if self.count(text[start:middle]) <= max_tokens:
                    best = middle
                    low = middle + 1
                else:
                    high = middle - 1
            part = text[start:best].strip()
            if part:
                parts.append(part)
            if best >= len(text):
                break
            start = max(start + 1, best - max(0, overlap_tokens))
        return parts


@dataclass
class _RawChunk:
    text: str
    parent_content: str
    element_type: str
    page_start: int
    page_end: int
    extra_metadata: dict[str, Any] = field(default_factory=dict)


class ChunkingRouter:
    

    def __init__(
        self,
        tokenizer: EmbeddingTokenizer | None = None,
        max_tokens: int = MAX_EMBEDDING_TOKENS,
    ) -> None:
        self.tokenizer = tokenizer or EmbeddingTokenizer()
        self.max_tokens = min(max_tokens, MAX_EMBEDDING_TOKENS)

    def split_documents(self, documents: Sequence[Document]) -> list[Document]:
        
        grouped: OrderedDict[str, list[Document]] = OrderedDict()
        for position, document in enumerate(documents):
            if not (document.page_content or "").strip():
                continue
            source_key = self._source_key(document, position)
            grouped.setdefault(source_key, []).append(document)

        chunks: list[Document] = []
        for source_key, source_documents in grouped.items():
            raw_chunks = self._split_source(source_documents)
            chunks.extend(self._finalize(source_key, source_documents[0], raw_chunks))
        return chunks

    def _split_source(self, documents: list[Document]) -> list[_RawChunk]:
        suffix = self._source_suffix(documents[0])
        source_type = str(documents[0].metadata.get("source_type") or "").lower()

        if source_type == "pdf":
            if len(documents) <= 4 and self._is_numbered_outline(documents):
                return self._split_numbered_outline(documents)
            if self._is_slide_deck(documents):
                return self._split_slides(documents)
            if self._is_numbered_outline(documents):
                return self._split_numbered_outline(documents)
            return self._split_prose(documents)
        if suffix == ".md" or source_type in {"md", "markdown"}:
            return self._split_markdown(documents)
        if suffix == ".py" or source_type == "python":
            return self._split_python(documents)
        if suffix in STRUCTURED_SUFFIXES or source_type in {"json", "yaml", "toml"}:
            return self._split_structured(documents, suffix)
        return self._split_prose(documents)

    def _is_slide_deck(self, documents: Sequence[Document]) -> bool:
        
        if len(documents) < 3:
            return False
        page_token_counts = [
            self.tokenizer.count(document.page_content)
            for document in documents
            if (document.page_content or "").strip()
        ]
        return bool(page_token_counts) and median(page_token_counts) <= 300

    def _is_numbered_outline(self, documents: Sequence[Document]) -> bool:
        matches = NUMBERED_ITEM_RE.findall("\n".join(doc.page_content for doc in documents))
        return len(matches) >= 3

    def _split_numbered_outline(self, documents: Sequence[Document]) -> list[_RawChunk]:
        chunks: list[_RawChunk] = []
        for document in documents:
            text = document.page_content.strip()
            items = self._numbered_items(text) or [text]
            page = self._page_number(document)
            current: list[str] = []

            def flush() -> None:
                if not current:
                    return
                chunks.extend(
                    self._split_text(
                        "\n".join(current),
                        parent_content=text,
                        element_type="numbered_item",
                        page_start=page,
                        page_end=page,
                        max_tokens=self.max_tokens,
                        overlap_tokens=0,
                    )
                )
                current.clear()

            for item in items:
                candidate = "\n".join([*current, item])
                if current and self.tokenizer.count(candidate) > 360:
                    flush()
                if self.tokenizer.count(item) > self.max_tokens:
                    flush()
                    chunks.extend(
                        self._split_text(
                            item,
                            parent_content=text,
                            element_type="numbered_item",
                            page_start=page,
                            page_end=page,
                            max_tokens=self.max_tokens,
                            overlap_tokens=0,
                        )
                    )
                else:
                    current.append(item)
            flush()
        return chunks

    def _split_slides(self, documents: Sequence[Document]) -> list[_RawChunk]:
        chunks: list[_RawChunk] = []
        for index, document in enumerate(documents):
            window = documents[max(0, index - 1) : min(len(documents), index + 2)]
            parent_pages = [
                (self._page_number(item), item.page_content.strip()) for item in window
            ]
            parent_content = "\n\n".join(
                f"[Page {page}]\n{content}" for page, content in parent_pages
            )
            page = self._page_number(document, index + 1)
            chunks.extend(
                self._split_text(
                    document.page_content,
                    parent_content=parent_content,
                    element_type="slide",
                    page_start=page,
                    page_end=page,
                    max_tokens=min(TARGET_CHUNK_TOKENS, self.max_tokens),
                    overlap_tokens=DEFAULT_OVERLAP_TOKENS,
                    extra_metadata={
                        "parent_page_start": parent_pages[0][0],
                        "parent_page_end": parent_pages[-1][0],
                    },
                )
            )
        return chunks

    def _split_markdown(self, documents: Sequence[Document]) -> list[_RawChunk]:
        chunks: list[_RawChunk] = []
        for document in documents:
            text = document.page_content.strip()
            matches = list(MARKDOWN_HEADING_RE.finditer(text))
            sections: list[str]
            if not matches:
                sections = [text]
            else:
                sections = []
                prefix = text[: matches[0].start()].strip()
                for index, match in enumerate(matches):
                    end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
                    section = text[match.start() : end].strip()
                    if index == 0 and prefix:
                        section = f"{prefix}\n\n{section}"
                    sections.append(section)
            page = self._page_number(document)
            for section in sections:
                chunks.extend(
                    self._split_text(
                        section,
                        parent_content=section,
                        element_type="markdown_section",
                        page_start=page,
                        page_end=page,
                    )
                )
        return chunks

    def _split_python(self, documents: Sequence[Document]) -> list[_RawChunk]:
        chunks: list[_RawChunk] = []
        for document in documents:
            code = self._strip_code_file_header(document.page_content)
            page = self._page_number(document)
            try:
                tree = ast.parse(code)
            except SyntaxError:
                chunks.extend(
                    self._split_text(code, code, "python_code", page, page, overlap_tokens=10)
                )
                continue

            lines = code.splitlines()
            blocks: list[tuple[str, str]] = []
            cursor = 1
            nodes = [
                node
                for node in tree.body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            for node in nodes:
                decorators = getattr(node, "decorator_list", [])
                start_line = min([node.lineno, *[item.lineno for item in decorators]])
                if cursor < start_line:
                    module_text = "\n".join(lines[cursor - 1 : start_line - 1]).strip()
                    if module_text:
                        blocks.append(("python_module", module_text))
                block_text = "\n".join(lines[start_line - 1 : node.end_lineno]).strip()
                element_type = "python_class" if isinstance(node, ast.ClassDef) else "python_function"
                blocks.append((element_type, block_text))
                cursor = node.end_lineno + 1
            tail = "\n".join(lines[cursor - 1 :]).strip()
            if tail:
                blocks.append(("python_module", tail))
            if not blocks and code.strip():
                blocks.append(("python_module", code.strip()))

            for element_type, block in blocks:
                chunks.extend(
                    self._split_text(
                        block,
                        parent_content=block,
                        element_type=element_type,
                        page_start=page,
                        page_end=page,
                        overlap_tokens=10,
                    )
                )
        return chunks

    def _split_structured(
        self,
        documents: Sequence[Document],
        suffix: str,
    ) -> list[_RawChunk]:
        chunks: list[_RawChunk] = []
        for document in documents:
            text = self._strip_code_file_header(document.page_content).strip()
            page = self._page_number(document)
            entries = self._structured_entries(text, suffix)
            for key_path, entry in entries:
                chunks.extend(
                    self._split_text(
                        entry,
                        parent_content=entry,
                        element_type="structured_object",
                        page_start=page,
                        page_end=page,
                        overlap_tokens=0,
                        extra_metadata={"key_path": key_path},
                    )
                )
        return chunks

    def _split_prose(self, documents: Sequence[Document]) -> list[_RawChunk]:
        chunks: list[_RawChunk] = []
        for document in documents:
            page_start = self._page_number(document)
            page_end = self._page_number(document, page_start, key="page_end")
            chunks.extend(
                self._split_text(
                    document.page_content,
                    parent_content=document.page_content.strip(),
                    element_type="prose",
                    page_start=page_start,
                    page_end=page_end,
                )
            )
        return chunks

    def _split_text(
        self,
        text: str,
        parent_content: str,
        element_type: str,
        page_start: int,
        page_end: int,
        max_tokens: int | None = None,
        overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[_RawChunk]:
        limit = min(max_tokens or TARGET_CHUNK_TOKENS, self.max_tokens)
        return [
            _RawChunk(
                text=part,
                parent_content=parent_content.strip(),
                element_type=element_type,
                page_start=page_start,
                page_end=page_end,
                extra_metadata=dict(extra_metadata or {}),
            )
            for part in self.tokenizer.split(text, limit, overlap_tokens)
            if part.strip()
        ]

    def _finalize(
        self,
        source_key: str,
        source_document: Document,
        raw_chunks: Sequence[_RawChunk],
    ) -> list[Document]:
        documents: list[Document] = []
        for chunk_index, raw in enumerate(raw_chunks):
            token_count = self.tokenizer.count(raw.text)
            if token_count > self.max_tokens:
                raise ValueError(f"切分结果超过 {self.max_tokens} tokens")
            parent_id = self._stable_id(
                "parent",
                source_key,
                raw.element_type,
                str(raw.page_start),
                str(raw.page_end),
                raw.parent_content,
            )
            chunk_id = self._stable_id(
                "chunk",
                parent_id,
                str(chunk_index),
                raw.text,
            )
            metadata = dict(source_document.metadata)
            metadata.update(raw.extra_metadata)
            metadata.update(
                {
                    "chunk_id": chunk_id,
                    "parent_id": parent_id,
                    "chunk_index": chunk_index,
                    "element_type": raw.element_type,
                    "page": raw.page_start,
                    "page_start": raw.page_start,
                    "page_end": raw.page_end,
                    "token_count": token_count,
                    "parent_content": raw.parent_content,
                }
            )
            documents.append(Document(page_content=raw.text, metadata=metadata))
        return documents

    @staticmethod
    def _source_key(document: Document, position: int) -> str:
        metadata = document.metadata
        return str(
            metadata.get("source_id")
            or metadata.get("source")
            or metadata.get("source_name")
            or f"source-{position}"
        )

    @staticmethod
    def _source_suffix(document: Document) -> str:
        source_name = str(document.metadata.get("source_name") or document.metadata.get("source") or "")
        return Path(source_name).suffix.lower()

    @staticmethod
    def _page_number(document: Document, default: int = 1, key: str = "page") -> int:
        value = document.metadata.get(key)
        if value is None and key == "page":
            value = document.metadata.get("page_start")
        try:
            return int(value) if value is not None else int(default)
        except (TypeError, ValueError):
            return int(default)

    @staticmethod
    def _numbered_items(text: str) -> list[str]:
        matches = list(NUMBERED_ITEM_RE.finditer(text))
        if not matches:
            return []
        items: list[str] = []
        prefix = text[: matches[0].start()].strip()
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            item = text[match.start() : end].strip()
            if index == 0 and prefix:
                item = f"{prefix}\n{item}"
            items.append(item)
        return items

    @staticmethod
    def _strip_code_file_header(text: str) -> str:
        return re.sub(r"^文件:\s*[^\n]+\n+", "", text.strip(), count=1)

    @staticmethod
    def _structured_entries(text: str, suffix: str) -> list[tuple[str, str]]:
        try:
            parsed = tomllib.loads(text) if suffix == ".toml" else json.loads(text)
        except (json.JSONDecodeError, tomllib.TOMLDecodeError):
            return ChunkingRouter._yaml_like_entries(text)

        if isinstance(parsed, dict):
            return [
                (
                    str(key),
                    json.dumps({key: value}, ensure_ascii=False, indent=2, default=str),
                )
                for key, value in parsed.items()
            ]
        if isinstance(parsed, list):
            return [
                (str(index), json.dumps(value, ensure_ascii=False, indent=2, default=str))
                for index, value in enumerate(parsed)
            ]
        return [("root", str(parsed))]

    @staticmethod
    def _yaml_like_entries(text: str) -> list[tuple[str, str]]:
        matches = list(re.finditer(r"(?m)^([A-Za-z_][\w.-]*):(?:\s|$)", text))
        if not matches:
            return [("root", text)]
        return [
            (
                match.group(1),
                text[
                    match.start() : matches[index + 1].start() if index + 1 < len(matches) else len(text)
                ].strip(),
            )
            for index, match in enumerate(matches)
        ]

    @staticmethod
    def _stable_id(prefix: str, *parts: str) -> str:
        digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:24]
        return f"{prefix}-{digest}"
