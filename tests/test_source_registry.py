from langchain_core.documents import Document

from ragbase.source_registry import (
    attach_source_records,
    build_source_records,
)


def _document(
    content: str,
    *,
    name: str = "复习大纲.pdf",
    source: str = "D:/资料/复习大纲.pdf",
    page: int = 1,
) -> Document:
    return Document(
        page_content=content,
        metadata={
            "source": source,
            "source_name": name,
            "source_type": "pdf",
            "page": page,
        },
    )


def test_build_source_records_assigns_stable_id_hash_and_session():
    records = build_source_records([_document("第一题：图像质量评价指标")], "session-1")

    assert len(records) == 1
    assert records[0].source_id.startswith("src_")
    assert records[0].file_hash
    assert records[0].session_id == "session-1"
    assert records[0].source_name == "复习大纲.pdf"
    assert records[0].normalized_name == "复习大纲"
    assert records[0].source_path == "D:/资料/复习大纲.pdf"


def test_same_content_uses_same_source_id_across_sessions():
    documents = [_document("第一题"), _document("第二题", page=2)]

    first = build_source_records(documents, "session-a")[0]
    second = build_source_records(documents, "session-b")[0]

    assert first.source_id == second.source_id
    assert first.file_hash == second.file_hash
    assert first.session_id != second.session_id


def test_identical_content_from_different_sources_has_distinct_stable_ids():
    documents = [
        _document("SHARED BODY", name="first.pdf", source="D:/materials/first.pdf"),
        _document("SHARED BODY", name="second.pdf", source="D:/archive/second.pdf"),
    ]

    first_session = build_source_records(documents, "session-a")
    second_session = build_source_records(documents, "session-b")

    assert first_session[0].file_hash == first_session[1].file_hash
    assert first_session[0].source_id != first_session[1].source_id
    assert [record.source_id for record in first_session] == [
        record.source_id for record in second_session
    ]


def test_changed_content_changes_hash_and_source_id():
    first = build_source_records([_document("第一版内容")], "session-1")[0]
    second = build_source_records([_document("第二版内容")], "session-1")[0]

    assert first.file_hash != second.file_hash
    assert first.source_id != second.source_id


def test_build_source_records_groups_pages_and_preserves_source_order():
    documents = [
        _document("大纲第一页", page=1),
        _document(
            "入门课",
            name="0 入门篇.pdf",
            source="D:/资料/0 入门篇.pdf",
            page=1,
        ),
        _document("大纲第二页", page=2),
    ]

    records = build_source_records(documents, "session-1")

    assert [record.source_name for record in records] == ["复习大纲.pdf", "0 入门篇.pdf"]


def test_attach_source_records_adds_internal_identity_without_mutating_input():
    documents = [_document("第一页"), _document("第二页", page=2)]
    records = build_source_records(documents, "session-1")

    attached = attach_source_records(documents, records)

    assert [doc.metadata["source_id"] for doc in attached] == [records[0].source_id] * 2
    assert [doc.metadata["session_id"] for doc in attached] == ["session-1", "session-1"]
    assert [doc.metadata["file_hash"] for doc in attached] == [records[0].file_hash] * 2
    assert attached[0].metadata["page"] == 1
    assert "source_id" not in documents[0].metadata


def test_attach_source_records_keeps_identical_content_sources_separate():
    documents = [
        _document("SHARED BODY", name="first.pdf", source="D:/materials/first.pdf"),
        _document("SHARED BODY", name="second.pdf", source="D:/archive/second.pdf"),
    ]
    records = build_source_records(documents, "session-1")

    attached = attach_source_records(documents, records)

    assert attached[0].metadata["source_id"] == records[0].source_id
    assert attached[1].metadata["source_id"] == records[1].source_id
    assert attached[0].metadata["source_id"] != attached[1].metadata["source_id"]
    assert attached[0].metadata["file_hash"] == attached[1].metadata["file_hash"]


def test_attach_source_records_rejects_a_document_without_a_record():
    document = _document("未登记内容", name="未登记.pdf", source="D:/资料/未登记.pdf")

    try:
        attach_source_records([document], [])
    except ValueError as exc:
        assert "未找到来源记录" in str(exc)
    else:
        raise AssertionError("未登记的文档不应被静默接受")
