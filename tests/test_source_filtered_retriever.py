from __future__ import annotations

from ragbase.retriever import build_source_filter


def test_source_filter_targets_exact_metadata_source_names():
    source_filter = build_source_filter(["复习大纲.pdf", "图像增强.pdf"])

    assert source_filter is not None
    conditions = source_filter.should
    assert [condition.key for condition in conditions] == [
        "metadata.source_name",
        "metadata.source_name",
    ]
    assert [condition.match.value for condition in conditions] == [
        "复习大纲.pdf",
        "图像增强.pdf",
    ]


def test_empty_source_selection_does_not_filter_global_retrieval():
    assert build_source_filter([]) is None
