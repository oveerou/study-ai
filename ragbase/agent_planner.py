





from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

from langchain_core.messages import HumanMessage, SystemMessage

from ragbase.planner_schema import (
    OperationPlan,
    VALID_OPERATIONS,
    VALID_READ_MODES,
    VALID_SOURCE_SCOPES,
)
from ragbase.source_registry import SourceRecord
from ragbase.source_resolver import SourceResolution

MAX_HISTORY_MESSAGES = 8
MAX_MESSAGE_CHARS = 600

OPERATION_PLANNER_PROMPT = """
你是知识库系统的行动规划器，只负责选择下一步操作，不回答用户问题。

只允许四种操作：
- chat：与资料无关的普通对话。
- list_sources：列出当前导入的来源名称和数量。
- read_source：阅读来源概览或完整正文；read_mode 只能是 overview 或 full_text。
- search：根据资料回答事实、概念、比较、步骤等问题。

选择规则：
- 问题针对具体概念、术语、方法、定义、比较、原因或步骤时，必须选择 search，即使用户同时提到了某份文件。
- 只有用户询问整份来源整体讲什么时，才选择 read_source + overview。
- 只有用户明确要求全文、原文、逐字内容或不省略输出时，才选择 read_source + full_text。

source_ids 只能逐字复制“解析器候选”中的 ID，不得编造。没有明确候选时留空。
scope 表示来源范围：selected 用于明确提到的候选来源；active 只用于明显承接上一轮的追问；all 用于综合全部资料或没有来源指代的独立资料问题；none 用于 chat 和 list_sources。
当 operation=search 时，query 必须是消解指代后的独立检索问题；其他操作的 query 原样保留当前问题。
只输出 JSON 对象，字段为 operation、scope、source_ids、query、confidence、read_mode、reason。
不要输出 Markdown 或 JSON 之外的文字。
""".strip()

async def plan_operation(
    model: Any,
    question: str,
    recent_messages: Sequence[Mapping[str, str]],
    source_records: Sequence[SourceRecord],
    resolution: SourceResolution,
    active_source_ids: Sequence[str] = (),
) -> OperationPlan:
    

    catalog_ids = {record.source_id for record in source_records}
    candidate_ids = {
        candidate.source_id
        for candidate in resolution.candidates
        if candidate.source_id in catalog_ids
    }
    planner_input = {
        "当前问题": question,
        "解析器置信度": resolution.confidence,
        "解析器候选": [
            {
                "ID": candidate.source_id,
                "名称": candidate.source_name,
                "匹配分数": candidate.score,
            }
            for candidate in resolution.candidates
            if candidate.source_id in catalog_ids
        ],
        "当前来源ID": [source_id for source_id in active_source_ids if source_id in catalog_ids],
        "最近对话": _compact_history(recent_messages),
    }
    messages = [
        SystemMessage(content=OPERATION_PLANNER_PROMPT),
        HumanMessage(content=json.dumps(planner_input, ensure_ascii=False, indent=2)),
    ]
    try:
        response = await model.ainvoke(messages)
        payload = _parse_payload(_message_text(response))
        return _validate_operation_plan(
            payload,
            question,
            source_records,
            resolution,
            candidate_ids,
            active_source_ids,
        )
    except Exception:
        return _fallback_operation_plan(
            question,
            source_records,
            active_source_ids,
            resolution,
        )


def _validate_operation_plan(
    payload: Mapping[str, Any],
    question: str,
    source_records: Sequence[SourceRecord],
    resolution: SourceResolution,
    candidate_ids: set[str],
    active_source_ids: Sequence[str],
) -> OperationPlan:
    operation = str(payload.get("operation") or "").strip()
    if operation not in VALID_OPERATIONS:
        raise ValueError("invalid planner operation")
    scope = str(payload.get("scope") or "auto").strip()
    if scope not in VALID_SOURCE_SCOPES:
        scope = "auto"

    catalog_ids = tuple(dict.fromkeys(record.source_id for record in source_records))
    catalog_set = set(catalog_ids)
    raw_source_ids = payload.get("source_ids") or []
    if isinstance(raw_source_ids, str):
        raw_source_ids = [raw_source_ids]
    selected = tuple(
        source_id
        for source_id in dict.fromkeys(str(value).strip() for value in raw_source_ids)
        if source_id in candidate_ids
    )

    resolved_ids = tuple(
        source_id
        for source_id in dict.fromkeys(resolution.source_ids)
        if source_id in catalog_set
    )
    if resolved_ids:
        selected = resolved_ids
        scope = "selected"
    elif candidate_ids and not selected:
        
        
        scope = "selected"

    read_mode = str(payload.get("read_mode") or "").strip() or None
    if operation in {"chat", "list_sources"}:
        selected = ()
        read_mode = None
        scope = "none"
    elif operation == "read_source":
        read_mode = read_mode if read_mode in VALID_READ_MODES else "overview"
    else:
        read_mode = None

    raw_query = str(payload.get("query") or "").strip()
    query = raw_query if operation == "search" and raw_query else question

    if operation in {"read_source", "search"} and scope == "all":
        selected = catalog_ids
    elif operation in {"read_source", "search"} and scope == "active":
        active = tuple(
            source_id
            for source_id in dict.fromkeys(active_source_ids)
            if source_id in catalog_set
        )
        selected = active or catalog_ids
    
    elif operation in {"read_source", "search"} and not selected and not candidate_ids:
        selected = catalog_ids
        scope = "all"

    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = min(1.0, max(0.0, confidence))
    return OperationPlan(
        operation=operation,
        source_ids=selected,
        query=query,
        confidence=confidence,
        reason=str(payload.get("reason") or "").strip(),
        read_mode=read_mode,
        scope=scope,
    )


def _fallback_operation_plan(
    question: str,
    source_records: Sequence[SourceRecord],
    active_source_ids: Sequence[str],
    resolution: SourceResolution | None = None,
) -> OperationPlan:
    catalog_ids = tuple(dict.fromkeys(record.source_id for record in source_records))
    catalog_set = set(catalog_ids)
    active = tuple(
        source_id
        for source_id in dict.fromkeys(active_source_ids)
        if source_id in catalog_set
    )
    resolved = tuple(
        source_id
        for source_id in dict.fromkeys(resolution.source_ids if resolution else ())
        if source_id in catalog_set
    )
    ambiguous_candidates = tuple(
        candidate.source_id
        for candidate in (resolution.candidates if resolution else ())
        if candidate.source_id in catalog_set
    )
    if not catalog_ids:
        return OperationPlan("chat", (), question, 0.0, "planner fallback without sources")
    if resolved:
        return OperationPlan(
            "search",
            resolved,
            question,
            0.0,
            "planner fallback on resolved sources",
            scope="selected",
        )
    if ambiguous_candidates:
        return OperationPlan(
            "search",
            (),
            question,
            0.0,
            "planner fallback awaiting source selection",
            scope="selected",
        )
    return OperationPlan(
        "search",
        active or catalog_ids,
        question,
        0.0,
        "planner fallback on available sources",
    )


def _compact_history(messages: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    
    compacted = []
    for message in messages[-MAX_HISTORY_MESSAGES:]:
        role = str(message.get("role") or "")
        content = str(message.get("content") or "")[:MAX_MESSAGE_CHARS]
        compacted.append({"role": role, "content": content})
    return compacted


def _message_text(message: Any) -> str:
    
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)


def _parse_payload(text: str) -> dict:
    
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("planner did not return JSON")
    payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("planner JSON must be an object")
    return payload
