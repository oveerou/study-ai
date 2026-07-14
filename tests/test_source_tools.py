

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from ragbase.source_registry import attach_source_records, build_source_records
from ragbase.source_tools import answer_source_tool, build_source_profiles, source_tool_route


def test_file_inventory_questions_return_imported_file_names():
    
    names = ["0 入门篇.pdf", "2025-2026-2复习大纲.pdf"]

    answer = answer_source_tool("我给了你什么文件", names, [])

    assert source_tool_route("我给了你什么文件") == "inventory"
    assert "当前已导入 2 个来源" in answer
    assert "0 入门篇.pdf" in answer
    assert "2025-2026-2复习大纲.pdf" in answer
    assert "资料中没有明确给出文件名称" not in answer


def test_file_name_question_is_inventory_not_document_qa():
    
    names = ["1 基础篇-3图像增强.pdf"]

    answer = answer_source_tool("文件的名称", names, [])

    assert source_tool_route("文件的名称") == "inventory"
    assert "1 基础篇-3图像增强.pdf" in answer


def test_source_overview_uses_all_source_profiles():
    
    documents = [
        Document(
            page_content="PSNR、SSIM、图像质量评价、数字图像处理基础。",
            metadata={"source_name": "2025-2026-2复习大纲.pdf", "source_type": "pdf"},
        ),
        Document(
            page_content="机器视觉系统由光源、镜头、相机、图像采集卡和处理软件构成。",
            metadata={"source_name": "1 基础篇-2机器视觉系统.pdf", "source_type": "pdf"},
        ),
    ]
    profiles = build_source_profiles(documents)

    answer = answer_source_tool("资料都有什么", [profile["name"] for profile in profiles], profiles)

    assert source_tool_route("资料都有什么") == "overview"
    assert "当前已导入 2 个来源" in answer
    assert "2025-2026-2复习大纲.pdf" in answer
    assert "1 基础篇-2机器视觉系统.pdf" in answer
    assert "PSNR" in answer
    assert "机器视觉系统" in answer


@pytest.mark.parametrize(
    "question",
    [
        "把资料原文全部展示出来",
        "不要概括，我要看完整正文",
        "内容全给我，别省略",
        "逐字输出这份文档",
        "所有内容",
        "给我完整版，不要简略",
    ],
)
def test_full_text_intent_uses_compositional_language_cues(question):
    
    assert source_tool_route(question) == "full_text"


def test_full_text_answer_preserves_every_page_without_excerpt_truncation():
    
    documents = [
        Document(
            page_content="第一页开头\n第一题：什么是图像质量评价？",
            metadata={"source_name": "复习大纲.pdf", "source_type": "pdf", "page": 1},
        ),
        Document(
            page_content="第二页内容\n最后一题：说明模型部署流程。",
            metadata={"source_name": "复习大纲.pdf", "source_type": "pdf", "page": 2},
        ),
    ]
    profiles = build_source_profiles(documents, max_chars_per_source=12)

    answer = answer_source_tool("完整原文全部输出", ["复习大纲.pdf"], profiles)

    assert "第 1 页" in answer
    assert "第 2 页" in answer
    assert "第一页开头" in answer
    assert "最后一题：说明模型部署流程。" in answer


def test_full_text_request_can_select_a_source_by_partial_file_name():
    
    documents = [
        Document(
            page_content="复习大纲正文",
            metadata={"source_name": "2025-2026-2复习大纲.pdf", "source_type": "pdf", "page": 1},
        ),
        Document(
            page_content="图像增强课件正文",
            metadata={"source_name": "1 基础篇-3图像增强.pdf", "source_type": "pdf", "page": 1},
        ),
    ]
    profiles = build_source_profiles(documents)

    answer = answer_source_tool(
        "把复习大纲的完整正文给我",
        ["2025-2026-2复习大纲.pdf", "1 基础篇-3图像增强.pdf"],
        profiles,
    )

    assert "复习大纲正文" in answer
    assert "图像增强课件正文" not in answer


def test_full_text_executor_accepts_validated_source_names_without_reinterpreting_question():
    
    documents = [
        Document(
            page_content="FIRST_SOURCE_CONTENT",
            metadata={"source_name": "first.pdf", "source_type": "pdf", "page": 1},
        ),
        Document(
            page_content="LAST_SOURCE_CONTENT",
            metadata={"source_name": "last.pdf", "source_type": "pdf", "page": 1},
        ),
    ]
    profiles = build_source_profiles(documents)

    answer = answer_source_tool(
        "ambiguous follow-up",
        ["first.pdf", "last.pdf"],
        profiles,
        route="full_text",
        selected_source_names=["last.pdf"],
    )

    assert "LAST_SOURCE_CONTENT" in answer
    assert "FIRST_SOURCE_CONTENT" not in answer


def test_profiles_keep_source_ids_all_pages_and_summary_units():
    
    documents = [
        Document(
            page_content="PAGE TWO",
            metadata={"source_id": "src_notes", "source_name": "notes.pdf", "page": 2},
        ),
        Document(
            page_content="PAGE ONE",
            metadata={"source_id": "src_notes", "source_name": "notes.pdf", "page": 1},
        ),
    ]

    profile = build_source_profiles(documents, max_chars_per_source=4)[0]

    assert profile["source_id"] == "src_notes"
    assert [section["page"] for section in profile["sections"]] == [1, 2]
    assert "PAGE ONE" in "\n".join(profile["summary_units"])
    assert "PAGE TWO" in "\n".join(profile["summary_units"])
    assert len(profile["excerpt"]) <= 4


def test_profiles_keep_identical_content_files_as_separate_named_sources():
    documents = [
        Document(
            page_content="SHARED BODY",
            metadata={
                "source": "D:/materials/first.pdf",
                "source_name": "first.pdf",
                "source_type": "pdf",
                "page": 1,
            },
        ),
        Document(
            page_content="SHARED BODY",
            metadata={
                "source": "D:/archive/second.pdf",
                "source_name": "second.pdf",
                "source_type": "pdf",
                "page": 1,
            },
        ),
    ]
    records = build_source_records(documents, "session-1")
    attached = attach_source_records(documents, records)

    profiles = build_source_profiles(attached)

    assert [profile["name"] for profile in profiles] == ["first.pdf", "second.pdf"]
    assert [profile["source_id"] for profile in profiles] == [
        records[0].source_id,
        records[1].source_id,
    ]
