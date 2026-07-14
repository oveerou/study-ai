

from __future__ import annotations

import json

import pytest
from fastembed import TextEmbedding
from langchain_core.documents import Document

from ragbase.chunking import ChunkingRouter, EmbeddingTokenizer


REQUIRED_METADATA = {
    "chunk_id",
    "parent_id",
    "chunk_index",
    "element_type",
    "page_start",
    "page_end",
    "token_count",
    "parent_content",
}


@pytest.fixture(scope="module")
def tokenizer() -> EmbeddingTokenizer:
    return EmbeddingTokenizer()


@pytest.fixture(scope="module")
def router(tokenizer: EmbeddingTokenizer) -> ChunkingRouter:
    return ChunkingRouter(tokenizer=tokenizer)


def assert_valid_chunks(chunks: list[Document], tokenizer: EmbeddingTokenizer) -> None:
    assert chunks
    for chunk in chunks:
        assert chunk.page_content.strip()
        assert REQUIRED_METADATA <= chunk.metadata.keys()
        assert chunk.metadata["token_count"] == tokenizer.count(chunk.page_content)
        assert chunk.metadata["token_count"] <= 480


def test_tokenizer_counts_beyond_fastembed_truncation_and_caps_split_tokens(
    tokenizer: EmbeddingTokenizer,
):
    text = "图像处理与机器视觉是课程的重要内容。" * 200

    assert tokenizer.count(text) > 512

    parts = tokenizer.split(text, max_tokens=480, overlap_tokens=50)

    assert len(parts) > 1
    assert all(tokenizer.count(part) <= 480 for part in parts)


def test_tokenizer_does_not_mutate_fastembed_shared_truncation():
    embedding = TextEmbedding(model_name="BAAI/bge-small-zh-v1.5", lazy_load=True)
    shared_tokenizer = embedding.model.tokenizer
    original_truncation = dict(shared_tokenizer.truncation or {})

    independent = EmbeddingTokenizer(tokenizer=shared_tokenizer)

    assert independent.count("机器视觉。" * 400) > 512
    assert shared_tokenizer.truncation == original_truncation


def test_numbered_outline_keeps_all_items_and_caps_tokens(
    router: ChunkingRouter,
    tokenizer: EmbeddingTokenizer,
):
    outline = "\n".join(
        f"{index}. 第{index}项复习内容：图像处理概念、计算方法与应用场景。"
        for index in range(1, 43)
    )
    page = Document(
        page_content=outline,
        metadata={"source": "outline.pdf", "source_name": "复习大纲.pdf", "source_type": "pdf", "page": 1},
    )

    chunks = router.split_documents([page])
    joined = "\n".join(chunk.page_content for chunk in chunks)

    assert "1. 第1项复习内容" in joined
    assert "42. 第42项复习内容" in joined
    assert all(chunk.metadata["element_type"] == "numbered_item" for chunk in chunks)
    assert_valid_chunks(chunks, tokenizer)


def test_slide_page_keeps_page_and_parent_window_metadata(
    router: ChunkingRouter,
    tokenizer: EmbeddingTokenizer,
):
    slide_pages = [
        Document(
            page_content=f"第{page}页标题\n第{page}页的核心知识点",
            metadata={
                "source": "course.pdf",
                "source_name": "课程课件.pdf",
                "source_type": "pdf",
                "page": page,
            },
        )
        for page in range(1, 4)
    ]

    chunks = router.split_documents(slide_pages)
    page_two = next(chunk for chunk in chunks if chunk.metadata["page_start"] == 2)

    assert page_two.metadata["page_end"] == 2
    assert page_two.metadata["page"] == 2
    assert page_two.metadata["element_type"] == "slide"
    assert page_two.metadata["parent_page_start"] == 1
    assert page_two.metadata["parent_page_end"] == 3
    assert page_two.metadata["parent_content"].startswith("[Page 1]\n")
    assert "[Page 2]\n" in page_two.metadata["parent_content"]
    assert "[Page 3]\n" in page_two.metadata["parent_content"]
    assert slide_pages[2].page_content in page_two.metadata["parent_content"]
    assert_valid_chunks(chunks, tokenizer)


def test_long_pdf_with_numbered_exercises_is_still_a_slide_deck(
    router: ChunkingRouter,
):
    pages = [
        Document(
            page_content=f"第 {page} 页课件\n1. 课堂练习\n2. 关键概念\n3. 应用示例",
            metadata={
                "source": "course.pdf",
                "source_name": "课程课件.pdf",
                "source_type": "pdf",
                "page": page,
            },
        )
        for page in range(1, 13)
    ]

    chunks = router.split_documents(pages)

    assert {chunk.metadata["element_type"] for chunk in chunks} == {"slide"}


def test_dense_multi_page_pdf_uses_prose_strategy(router: ChunkingRouter):
    pages = [
        Document(
            page_content="。".join(f"第 {index} 段连续正文内容" for index in range(450)),
            metadata={
                "source": "paper.pdf",
                "source_name": "论文.pdf",
                "source_type": "pdf",
                "page": page,
            },
        )
        for page in range(1, 10)
    ]

    chunks = router.split_documents(pages)

    assert {chunk.metadata["element_type"] for chunk in chunks} == {"prose"}


def test_markdown_splits_by_heading(
    router: ChunkingRouter,
    tokenizer: EmbeddingTokenizer,
):
    markdown = Document(
        page_content="# 安装\n安装说明。\n\n## 使用\n使用说明。\n\n## 测试\n测试说明。",
        metadata={"source": "README.md", "source_name": "README.md", "source_type": "md"},
    )

    chunks = router.split_documents([markdown])

    assert [chunk.page_content.splitlines()[0] for chunk in chunks] == ["# 安装", "## 使用", "## 测试"]
    assert all(chunk.metadata["element_type"] == "markdown_section" for chunk in chunks)
    assert_valid_chunks(chunks, tokenizer)


def test_python_uses_ast_blocks(
    router: ChunkingRouter,
    tokenizer: EmbeddingTokenizer,
):
    python_source = Document(
        page_content=(
            "文件: sample.py\n\n"
            "import math\n\n"
            "class Calculator:\n"
            "    def square(self, value):\n"
            "        return value * value\n\n"
            "def normalize(value):\n"
            "    return value / math.pi\n"
        ),
        metadata={"source": "sample.py", "source_name": "sample.py", "source_type": "code"},
    )

    chunks = router.split_documents([python_source])
    element_types = {chunk.metadata["element_type"] for chunk in chunks}

    assert "python_class" in element_types
    assert "python_function" in element_types
    assert "class Calculator" in "\n".join(chunk.page_content for chunk in chunks)
    assert_valid_chunks(chunks, tokenizer)


def test_structured_json_splits_top_level_objects(
    router: ChunkingRouter,
    tokenizer: EmbeddingTokenizer,
):
    content = json.dumps(
        {
            "model": {"name": "bge-small-zh", "limit": 512},
            "retrieval": {"dense": True, "top_k": 20},
        },
        ensure_ascii=False,
    )
    source = Document(
        page_content=content,
        metadata={"source": "config.json", "source_name": "config.json", "source_type": "code"},
    )

    chunks = router.split_documents([source])

    assert {chunk.metadata.get("key_path") for chunk in chunks} == {"model", "retrieval"}
    assert all(chunk.metadata["element_type"] == "structured_object" for chunk in chunks)
    assert_valid_chunks(chunks, tokenizer)


def test_prose_is_token_bounded_and_ids_are_deterministic(
    router: ChunkingRouter,
    tokenizer: EmbeddingTokenizer,
):
    source = Document(
        page_content="。".join(f"这是第{index}段关于图像增强的说明" for index in range(700)),
        metadata={"source": "notes.txt", "source_name": "notes.txt", "source_type": "txt"},
    )

    first = router.split_documents([source])
    second = router.split_documents([source])

    assert [chunk.metadata["chunk_id"] for chunk in first] == [
        chunk.metadata["chunk_id"] for chunk in second
    ]
    assert all(chunk.metadata["element_type"] == "prose" for chunk in first)
    assert_valid_chunks(first, tokenizer)
