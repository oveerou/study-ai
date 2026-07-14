

from __future__ import annotations

import asyncio
import json

from langchain_core.messages import AIMessage

from ragbase.agent_planner import plan_operation
from ragbase.source_registry import SourceRecord
from ragbase.source_resolver import SourceCandidate, SourceResolution


class FakePlannerModel:
    def __init__(self, payload=None, error: Exception | None = None):
        self.payload = payload
        self.error = error
        self.messages = None

    async def ainvoke(self, messages):
        self.messages = messages
        if self.error:
            raise self.error
        return AIMessage(content=json.dumps(self.payload, ensure_ascii=False))


def _source_records():
    return [
        SourceRecord(
            source_id="src_review",
            source_name="2025-2026-2复习大纲.pdf",
            normalized_name="202520262复习大纲",
            source_type="pdf",
            session_id="session-1",
            file_hash="hash-review",
            source_path="D:/资料/2025-2026-2复习大纲.pdf",
        ),
        SourceRecord(
            source_id="src_system",
            source_name="1 基础篇-机器视觉系统.pdf",
            normalized_name="1基础篇机器视觉系统",
            source_type="pdf",
            session_id="session-1",
            file_hash="hash-system",
            source_path="D:/资料/1 基础篇-机器视觉系统.pdf",
        ),
    ]


def _resolved(record: SourceRecord, confidence: float = 1.0):
    candidate = SourceCandidate(record.source_id, record.source_name, confidence, "名称匹配")
    return SourceResolution((record.source_id,), confidence, (candidate,))


def test_new_planner_returns_four_operation_schema_and_catalog_source_ids():
    
    records = _source_records()
    model = FakePlannerModel(
        {
            "operation": "read_source",
            "source_ids": [records[0].source_id, "src_missing"],
            "query": "模型擅自改写的问题",
            "confidence": 0.92,
            "read_mode": "full_text",
            "reason": "用户要查看来源正文",
        }
    )

    plan = asyncio.run(
        plan_operation(
            model,
            question="把刚才那份资料完整展示出来",
            recent_messages=[],
            source_records=records,
            resolution=_resolved(records[0]),
            active_source_ids=[],
        )
    )

    assert plan.operation == "read_source"
    assert plan.source_ids == (records[0].source_id,)
    assert plan.query == "把刚才那份资料完整展示出来"
    assert plan.read_mode == "full_text"
    assert plan.confidence == 0.92


def test_search_plan_uses_model_rewritten_query():
    
    records = _source_records()
    model = FakePlannerModel(
        {
            "operation": "search",
            "source_ids": [records[0].source_id],
            "query": "2025-2026-2复习大纲.pdf 中图像数字化的过程是什么",
            "confidence": 0.9,
            "reason": "补全了上一轮资料指代",
        }
    )

    plan = asyncio.run(
        plan_operation(
            model,
            "它的图像数字化过程呢",
            [],
            records,
            _resolved(records[0]),
            [],
        )
    )

    assert plan.operation == "search"
    assert plan.query == "2025-2026-2复习大纲.pdf 中图像数字化的过程是什么"


def test_new_planner_cannot_invent_source_id_from_ambiguous_candidates():
    
    records = _source_records()
    candidate = SourceCandidate(records[0].source_id, records[0].source_name, 0.65, "相似名称")
    resolution = SourceResolution((), 0.65, (candidate,))
    model = FakePlannerModel(
        {
            "operation": "read_source",
            "source_ids": ["src_missing"],
            "confidence": 0.8,
            "read_mode": "overview",
            "reason": "选择候选来源",
        }
    )

    plan = asyncio.run(
        plan_operation(model, "那份大纲讲了什么", [], records, resolution, [])
    )

    assert plan.source_ids == ()
    assert "src_missing" not in plan.source_ids
    assert plan.scope == "selected"


def test_operation_model_failure_preserves_ambiguous_candidates_for_selection():
    
    records = _source_records()
    resolution = SourceResolution(
        (),
        0.65,
        (
            SourceCandidate(records[0].source_id, records[0].source_name, 0.65, "相似名称"),
            SourceCandidate(records[1].source_id, records[1].source_name, 0.64, "相似名称"),
        ),
    )

    plan = asyncio.run(
        plan_operation(
            FakePlannerModel(error=RuntimeError("model unavailable")),
            "那份基础资料讲什么",
            [],
            records,
            resolution,
            [],
        )
    )

    assert plan.operation == "search"
    assert plan.scope == "selected"
    assert plan.source_ids == ()


def test_unique_resolver_choice_wins_over_model_source_choice():
    
    records = _source_records()
    resolution = SourceResolution(
        (records[0].source_id,),
        0.75,
        (
            SourceCandidate(records[0].source_id, records[0].source_name, 0.75, "相似名称"),
            SourceCandidate(records[1].source_id, records[1].source_name, 0.7, "相似名称"),
        ),
    )
    model = FakePlannerModel(
        {
            "operation": "search",
            "source_ids": [records[1].source_id],
            "confidence": 0.9,
            "reason": "检索资料",
        }
    )

    plan = asyncio.run(plan_operation(model, "大纲里的分割算法有哪些", [], records, resolution, []))

    assert plan.source_ids == (records[0].source_id,)


def test_operation_prompt_separates_specific_questions_from_whole_source_reads():
    from ragbase.agent_planner import OPERATION_PLANNER_PROMPT

    assert "具体概念" in OPERATION_PLANNER_PROMPT
    assert "明确要求全文" in OPERATION_PLANNER_PROMPT


def test_all_scope_uses_catalog_instead_of_inheriting_active_source():
    records = _source_records()
    model = FakePlannerModel(
        {
            "operation": "search",
            "scope": "all",
            "source_ids": [],
            "confidence": 0.96,
            "reason": "需要综合当前全部来源",
        }
    )

    plan = asyncio.run(
        plan_operation(
            model,
            "比较这些材料中的两种方法",
            [],
            records,
            SourceResolution((), 0.0, ()),
            [records[1].source_id],
        )
    )

    assert plan.scope == "all"
    assert plan.source_ids == tuple(record.source_id for record in records)


def test_active_scope_keeps_followup_on_current_source():
    records = _source_records()
    model = FakePlannerModel(
        {
            "operation": "search",
            "scope": "active",
            "source_ids": [],
            "confidence": 0.95,
            "reason": "延续上一轮来源",
        }
    )

    plan = asyncio.run(
        plan_operation(
            model,
            "继续解释它",
            [],
            records,
            SourceResolution((), 0.0, ()),
            [records[1].source_id],
        )
    )

    assert plan.scope == "active"
    assert plan.source_ids == (records[1].source_id,)


def test_invalid_new_operation_falls_back_to_search_on_active_sources():
    
    records = _source_records()
    model = FakePlannerModel(
        {
            "operation": "overview",
            "source_ids": [],
            "confidence": 1.0,
            "reason": "旧动作",
        }
    )

    plan = asyncio.run(
        plan_operation(
            model,
            "继续讲",
            [],
            records,
            SourceResolution((), 0.0, ()),
            [records[1].source_id, "src_missing"],
        )
    )

    assert plan.operation == "search"
    assert plan.source_ids == (records[1].source_id,)
    assert plan.query == "继续讲"


def test_operation_model_failure_keeps_explicitly_resolved_source():
    records = _source_records()
    model = FakePlannerModel(error=RuntimeError("model unavailable"))

    plan = asyncio.run(
        plan_operation(
            model,
            "复习大纲里的图像数字化过程是什么",
            [],
            records,
            _resolved(records[0]),
            [records[1].source_id],
        )
    )

    assert plan.operation == "search"
    assert plan.scope == "selected"
    assert plan.source_ids == (records[0].source_id,)


def test_operation_validation_failure_keeps_explicitly_resolved_source():
    records = _source_records()
    model = FakePlannerModel(
        {
            "operation": "unsupported_operation",
            "scope": "active",
            "source_ids": [records[1].source_id],
        }
    )

    plan = asyncio.run(
        plan_operation(
            model,
            "复习大纲里的图像数字化过程是什么",
            [],
            records,
            _resolved(records[0]),
            [records[1].source_id],
        )
    )

    assert plan.operation == "search"
    assert plan.scope == "selected"
    assert plan.source_ids == (records[0].source_id,)
