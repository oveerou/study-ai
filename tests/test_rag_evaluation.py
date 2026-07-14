

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from langchain_core.documents import Document

from ragbase.chunking import ChunkingRouter
from ragbase.hybrid_retriever import (
    RankedDocument,
    bm25_search,
    reciprocal_rank_fusion,
)
from ragbase.ingestor import load_path_documents
from ragbase.source_registry import (
    SourceRecord,
    attach_source_records,
    build_source_records,
)
from ragbase.source_resolver import resolve_sources


CASES_PATH = Path(__file__).parent / "fixtures" / "rag_cases.json"
REAL_OUTLINE_PATH = Path(
    r"D:\我的文档\大三下 机器视觉\期末复习\2025-2026-2复习大纲.pdf"
)


@dataclass(frozen=True)
class EvaluationCorpus:
    records: tuple[SourceRecord, ...]
    chunks: tuple[Document, ...]
    source_ids: dict[str, str]
    used_real_outline: bool


@pytest.fixture(scope="session")
def rag_cases() -> list[dict[str, Any]]:
    payload = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    return payload["cases"]


@pytest.fixture(scope="session")
def evaluation_corpus() -> EvaluationCorpus:
    documents, used_real_outline = _evaluation_documents()
    records = build_source_records(documents, session_id="rag-evaluation")
    attached = attach_source_records(documents, records)
    chunks = ChunkingRouter().split_documents(attached)
    source_ids = {
        "intro": _source_id(records, "0 入门篇.pdf"),
        "system": _source_id(records, "1 基础篇-2机器视觉系统.pdf"),
        "outline": _source_id(records, "2025-2026-2复习大纲.pdf"),
        "enhancement": _source_id(records, "1 基础篇-3图像增强.pdf"),
    }
    return EvaluationCorpus(tuple(records), tuple(chunks), source_ids, used_real_outline)


def test_fixture_covers_required_normal_chinese_cases(rag_cases):
    categories = {case["category"] for case in rag_cases}
    tags = {tag for case in rag_cases for tag in case["tags"]}

    assert len(rag_cases) >= 10
    assert categories == {"source_resolution", "all_source_read", "retrieval", "irrelevant"}
    assert {
        "partial_filename",
        "filename_typo",
        "ordinal_reference",
        "active_pronoun",
        "all_sources",
        "exact_term",
        "irrelevant",
    }.issubset(tags)


def test_registry_and_resolver_handle_natural_source_references(
    rag_cases,
    evaluation_corpus,
):
    assert len(evaluation_corpus.records) == 4
    assert len({record.source_id for record in evaluation_corpus.records}) == 4
    assert {record.session_id for record in evaluation_corpus.records} == {"rag-evaluation"}

    cases = [case for case in rag_cases if case["category"] == "source_resolution"]
    for case in cases:
        active_ids = tuple(
            evaluation_corpus.source_ids[key] for key in case.get("active_sources", [])
        )
        result = resolve_sources(case["query"], evaluation_corpus.records, active_ids)
        expected = tuple(
            evaluation_corpus.source_ids[key] for key in case["expected_sources"]
        )

        assert result.source_ids == expected, case["id"]
        assert result.confidence >= case["minimum_confidence"], case["id"]


def test_all_source_read_uses_the_complete_registry(rag_cases, evaluation_corpus):
    case = next(case for case in rag_cases if case["category"] == "all_source_read")

    resolution = resolve_sources(case["query"], evaluation_corpus.records)
    selected_ids = tuple(record.source_id for record in evaluation_corpus.records)
    expected_ids = tuple(
        evaluation_corpus.source_ids[key] for key in case["expected_sources"]
    )

    assert resolution.source_ids == ()
    assert selected_ids == expected_ids


def test_outline_chunking_retains_the_last_numbered_item(evaluation_corpus):
    outline_id = evaluation_corpus.source_ids["outline"]
    outline_chunks = [
        chunk
        for chunk in evaluation_corpus.chunks
        if chunk.metadata.get("source_id") == outline_id
    ]
    complete_text = "\n".join(chunk.page_content for chunk in outline_chunks)

    assert outline_chunks
    assert {chunk.metadata["element_type"] for chunk in outline_chunks} == {"numbered_item"}
    assert all(chunk.metadata["token_count"] <= 480 for chunk in outline_chunks)
    assert "1. 图像质量评价指标" in complete_text
    assert "42. 计算机视觉技术的前沿和发展趋势" in complete_text
    if REAL_OUTLINE_PATH.is_file():
        assert evaluation_corpus.used_real_outline


def test_exact_terms_are_recalled_after_bm25_and_rrf(rag_cases, evaluation_corpus):
    cases = [case for case in rag_cases if case["category"] == "retrieval"]
    for case in cases:
        lexical = bm25_search(case["query"], evaluation_corpus.chunks, top_k=20)
        baseline = [
            RankedDocument(document=chunk, score=1.0 / rank)
            for rank, chunk in enumerate(evaluation_corpus.chunks, start=1)
        ]
        fused = reciprocal_rank_fusion([baseline, lexical])[:5]
        evidence = "\n".join(item.document.page_content for item in fused)
        evidence_source_ids = {
            item.document.metadata.get("source_id") for item in fused
        }

        assert lexical, case["id"]
        assert evaluation_corpus.source_ids[case["expected_source"]] in evidence_source_ids, case["id"]
        assert all(term in evidence for term in case["expected_terms"]), case["id"]


def test_irrelevant_queries_select_no_source_or_lexical_evidence(
    rag_cases,
    evaluation_corpus,
):
    cases = [case for case in rag_cases if case["category"] == "irrelevant"]
    for case in cases:
        resolution = resolve_sources(case["query"], evaluation_corpus.records)
        lexical = bm25_search(case["query"], evaluation_corpus.chunks, top_k=20)

        assert resolution.source_ids == (), case["id"]
        assert lexical == [], case["id"]


def _evaluation_documents() -> tuple[list[Document], bool]:
    intro = _document(
        "0 入门篇.pdf",
        "数字图像处理入门介绍了课程目标、图像应用和机器视觉的基本任务。",
    )
    system = _document(
        "1 基础篇-2机器视觉系统.pdf",
        "机器视觉系统由光源、镜头、相机、采集卡和处理软件组成。",
    )
    enhancement = _document(
        "1 基础篇-3图像增强.pdf",
        "图像增强包括灰度变换、平滑、锐化以及空间域和频率域处理。",
    )

    if REAL_OUTLINE_PATH.is_file():
        outline = load_path_documents(REAL_OUTLINE_PATH)
        used_real_outline = True
    else:
        outline = _synthetic_outline_documents()
        used_real_outline = False

    return [intro, system, *outline, enhancement], used_real_outline


def _document(source_name: str, content: str, page: int = 1) -> Document:
    return Document(
        page_content=content,
        metadata={
            "source": f"synthetic/{source_name}",
            "source_name": source_name,
            "source_type": "pdf",
            "page": page,
        },
    )


def _synthetic_outline_documents() -> list[Document]:
    items = {
        1: "图像质量评价指标有哪些？峰值信噪比 PSNR 的含义；",
        5: "像素间的距离：欧式距离、D4、D8、Dm 距离；",
        25: "大津阈值法 Otsu 的基本原理；",
        27: "Canny 算法的基本思想；",
        31: "图像特征描述和提取方法：SIFT、LBP、HOG、Haar、GLCM；",
        41: "AI视觉模型部署的大概流程；",
        42: "计算机视觉技术的前沿和发展趋势。",
    }
    lines = [
        f"{index}. {items.get(index, f'第{index}项图像处理复习内容；')}"
        for index in range(1, 43)
    ]
    return [
        _document("2025-2026-2复习大纲.pdf", "\n".join(lines[:37]), page=1),
        _document("2025-2026-2复习大纲.pdf", "\n".join(lines[37:]), page=2),
    ]


def _source_id(records: list[SourceRecord], source_name: str) -> str:
    return next(record.source_id for record in records if record.source_name == source_name)
