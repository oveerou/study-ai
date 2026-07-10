from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from langchain_core.messages import HumanMessage, SystemMessage


VALID_INTENTS = {"chat", "inventory", "overview", "full_text", "grounded_qa"}
VALID_SCOPES = {"none", "selected", "all"}
CONTENT_INTENTS = {"overview", "full_text", "grounded_qa"}
MAX_HISTORY_MESSAGES = 8
MAX_MESSAGE_CHARS = 600

PLANNER_PROMPT = """
你是知识库问答系统的行动规划器，不负责回答问题。请理解用户真实语义、否定、纠正、简称、错别字、序号和跨轮指代，然后只输出一个 JSON 对象。

JSON 字段：
- intent: chat | inventory | overview | full_text | grounded_qa
- scope: none | selected | all
- source_names: 必须逐字复制“可用来源”中“名称”字段的零个或多个完整值
- standalone_question: 消解“它、这个、最后一个”等指代后的独立问题
- reason: 一句简短理由

决策原则：
1. chat 用于寒暄或与资料无关的普通对话。
2. inventory 用于询问来源名称、数量或清单，不回答资料正文。
3. overview 用于总结一个或多个来源主要讲什么。
4. full_text 只用于用户确实要查看原文、全文或不省略的内容。
5. grounded_qa 用于根据资料回答事实、概念、比较、步骤等问题。
6. 只有用户肯定且明确要求全部来源时 scope 才是 all。否定“全部/所有”表示不能选择 all。
7. 用户纠正目标来源或只补充文件名时，结合“当前来源”和“上一个内容动作”继承其未完成意图。
8. 用户明确提到名称、简称、序号、首个、末个或当前文件时，选择对应来源并使用 scope=selected。
9. 用户比较、关联或综合多个来源时，source_names 必须包含比较各方；如果一方是当前来源，也必须把当前来源和新提到的来源一起列出。
10. 当前来源只用于有指代、省略或明显延续关系的追问。完整独立的新问题代表可能已经切换主题，不能仅因为存在当前来源就继续锁定它；应根据问题与来源名称重新选择，无法确定单一来源时使用 scope=all。
11. 不得编造来源名称；不确定时 source_names 留空，由执行器使用当前来源。
12. 不要输出 Markdown、解释或 JSON 之外的任何文字。
""".strip()


@dataclass(frozen=True)
class AgentPlan:
    intent: str
    scope: str
    source_names: tuple[str, ...]
    standalone_question: str
    reason: str = ""


def requires_source_selection(plan: AgentPlan) -> bool:
    return plan.scope == "selected" and plan.intent != "chat" and not plan.source_names


def next_active_sources(current_sources: Sequence[str], plan: AgentPlan) -> list[str]:
    if plan.scope == "selected":
        return list(plan.source_names)
    if plan.scope == "all":
        return []
    return list(current_sources)


async def plan_question(
    model: Any,
    question: str,
    recent_messages: Sequence[Mapping[str, str]],
    source_names: Sequence[str],
    active_source_names: Sequence[str],
    last_content_intent: str | None,
) -> AgentPlan:
    catalog_size = len(source_names)
    planner_input = {
        "当前问题": question,
        "可用来源": [
            {
                "正序位置": index + 1,
                "倒序位置": catalog_size - index,
                "名称": name,
            }
            for index, name in enumerate(source_names)
        ],
        "当前来源": list(active_source_names),
        "上一个内容动作": last_content_intent,
        "最近对话": _compact_history(recent_messages),
    }
    messages = [
        SystemMessage(content=PLANNER_PROMPT),
        HumanMessage(content=json.dumps(planner_input, ensure_ascii=False, indent=2)),
    ]
    try:
        response = await model.ainvoke(messages)
        payload = _parse_payload(_message_text(response))
        return _validate_plan(
            payload=payload,
            question=question,
            source_names=source_names,
            active_source_names=active_source_names,
        )
    except Exception:
        return _fallback_plan(question, source_names, active_source_names)


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


def _validate_plan(
    payload: Mapping[str, Any],
    question: str,
    source_names: Sequence[str],
    active_source_names: Sequence[str],
) -> AgentPlan:
    intent = str(payload.get("intent") or "").strip()
    scope = str(payload.get("scope") or "").strip()
    if intent not in VALID_INTENTS or scope not in VALID_SCOPES:
        raise ValueError("invalid planner action")

    catalog = list(dict.fromkeys(str(name) for name in source_names))
    catalog_lookup = {name.casefold(): name for name in catalog}
    selected = []
    raw_names = payload.get("source_names") or []
    if isinstance(raw_names, str):
        raw_names = [raw_names]
    for raw_name in raw_names:
        valid_name = catalog_lookup.get(str(raw_name).strip().casefold())
        if valid_name and valid_name not in selected:
            selected.append(valid_name)

    if intent == "chat":
        scope = "none"
        selected = []
    elif scope == "all":
        selected = catalog
    elif scope == "selected" and not selected:
        selected = [name for name in active_source_names if name in catalog]
        if not selected and len(catalog) == 1:
            selected = catalog.copy()

    standalone_question = str(payload.get("standalone_question") or question).strip() or question
    return AgentPlan(
        intent=intent,
        scope=scope,
        source_names=tuple(selected),
        standalone_question=standalone_question,
        reason=str(payload.get("reason") or "").strip(),
    )


def _fallback_plan(
    question: str,
    source_names: Sequence[str],
    active_source_names: Sequence[str],
) -> AgentPlan:
    catalog = list(dict.fromkeys(str(name) for name in source_names))
    selected = [name for name in active_source_names if name in catalog]
    if not catalog:
        return AgentPlan("chat", "none", (), question, "planner fallback without sources")
    if selected:
        return AgentPlan(
            "grounded_qa",
            "selected",
            tuple(selected),
            question,
            "planner fallback on active sources",
        )
    return AgentPlan(
        "grounded_qa",
        "all",
        tuple(catalog),
        question,
        "planner fallback on all sources",
    )
