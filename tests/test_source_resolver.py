from langchain_core.documents import Document

from ragbase.source_registry import build_source_records
from ragbase.source_resolver import resolve_sources


def _records():
    names = [
        "0 入门篇.pdf",
        "1 基础篇-机器视觉系统.pdf",
        "2025-2026-2复习大纲.pdf",
        "图像复原练习.pdf",
        "1 基础篇-图像增强.pdf",
    ]
    documents = [
        Document(
            page_content=f"{name}的唯一内容",
            metadata={
                "source": f"D:/资料/{name}",
                "source_name": name,
                "source_type": "pdf",
            },
        )
        for name in names
    ]
    return build_source_records(documents, "session-1")


def test_resolves_exact_file_name():
    records = _records()

    result = resolve_sources("请总结 1 基础篇-机器视觉系统.pdf", records)

    assert result.source_ids == (records[1].source_id,)
    assert result.confidence == 1.0
    assert result.candidates[0].source_name == records[1].source_name


def test_resolves_extensionless_file_name():
    records = _records()

    result = resolve_sources("1 基础篇-机器视觉系统讲了哪些内容", records)

    assert result.source_ids == (records[1].source_id,)
    assert result.confidence >= 0.95


def test_resolves_unique_partial_name():
    records = _records()

    result = resolve_sources("复习大纲主要考哪些内容", records)

    assert result.source_ids == (records[2].source_id,)
    assert result.candidates[0].reason in {"名称匹配", "部分名称匹配"}


def test_resolves_minor_typo_by_similarity():
    records = _records()

    result = resolve_sources("复习大钢主要考哪些内容", records)

    assert result.source_ids == (records[2].source_id,)
    assert 0.7 <= result.confidence < 1.0
    assert result.candidates[0].reason == "相似名称"


def test_resolves_ordinal_reference_in_catalog_order():
    records = _records()

    result = resolve_sources("第二个文件说了什么", records)

    assert result.source_ids == (records[1].source_id,)
    assert result.confidence == 1.0
    assert result.candidates[0].reason == "序号指代"


def test_resolves_arabic_ordinal_reference():
    records = _records()

    result = resolve_sources("看一下第3份资料", records)

    assert result.source_ids == (records[2].source_id,)


def test_resolves_ordinal_reference_without_repeating_source_noun():
    records = _records() + [
        build_source_records(
            [
                Document(
                    page_content=f"扩展来源 {index}",
                    metadata={
                        "source": f"D:/资料/扩展来源{index}.pdf",
                        "source_name": f"扩展来源{index}.pdf",
                        "source_type": "pdf",
                    },
                )
            ],
            "session-1",
        )[0]
        for index in range(6, 11)
    ]

    result = resolve_sources("第十份大概讲什么", records)

    assert result.source_ids == (records[9].source_id,)


def test_descriptive_name_beats_generic_course_series_prefix():
    names = [
        "1 基础篇-1基本概念1.pdf",
        "1 基础篇——传统图像处理方法.pdf",
        "1 基础篇-3图像增强.pdf",
    ]
    records = build_source_records(
        [
            Document(
                page_content=name,
                metadata={"source": name, "source_name": name, "source_type": "pdf"},
            )
            for name in names
        ],
        "session-1",
    )

    result = resolve_sources("基础概念那份怎么解释数字图像处理", records)

    assert result.source_ids == (records[0].source_id,)


def test_explicit_source_description_beats_active_pronoun_in_same_question():
    names = [
        "1 基础篇-1基本概念1.pdf",
        "1 基础篇-3图像增强.pdf",
    ]
    records = build_source_records(
        [
            Document(
                page_content=name,
                metadata={"source": name, "source_name": name, "source_type": "pdf"},
            )
            for name in names
        ],
        "session-1",
    )

    result = resolve_sources(
        "基础概念那份怎么解释数字图像处理",
        records,
        [records[1].source_id],
    )

    assert result.source_ids == (records[0].source_id,)


def test_pronoun_followup_inherits_active_source():
    records = _records()

    result = resolve_sources("它里面还有哪些题目", records, [records[2].source_id])

    assert result.source_ids == (records[2].source_id,)
    assert result.confidence >= 0.9
    assert result.candidates[0].reason == "当前来源指代"


def test_location_pronoun_followup_inherits_active_source():
    records = _records()

    result = resolve_sources("那里面关于距离写了什么", records, [records[2].source_id])

    assert result.source_ids == (records[2].source_id,)
    assert result.confidence >= 0.9


def test_unknown_active_ids_are_ignored():
    records = _records()

    result = resolve_sources("这个文件讲了什么", records, ["src_not_exists"])

    assert result.source_ids == ()
    assert result.confidence == 0.0


def test_ambiguous_partial_name_returns_ranked_candidates_without_guessing():
    records = _records()

    result = resolve_sources("基础篇的内容", records)

    assert result.source_ids == ()
    assert result.confidence < 0.7
    assert result.candidates
    assert result.candidates[0].source_name == "1 基础篇-机器视觉系统.pdf"


def test_unrelated_query_does_not_select_a_source():
    records = _records()

    result = resolve_sources("你好，今天天气怎么样", records)

    assert result.source_ids == ()
    assert result.confidence == 0.0
