

from __future__ import annotations

from typing import Any, Mapping, Sequence

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


MAX_OVERVIEW_BATCH_CHARS = 12000
MAX_CHAT_MESSAGES = 10
MAX_CHAT_MESSAGE_CHARS = 1200


def select_source_profiles(
    source_profiles: Sequence[dict],
    source_names: Sequence[str],
) -> list[dict]:
    
    profiles_by_name = {str(profile.get("name")): profile for profile in source_profiles}
    return [profiles_by_name[name] for name in source_names if name in profiles_by_name]


async def generate_overview_answer(
    model: Any,
    question: str,
    source_profiles: Sequence[dict],
) -> str:
    
    units = _overview_units(source_profiles)
    if not units:
        return "当前资料没有可用于概览的正文。"

    summaries = []
    for batch in _bounded_batches(units, MAX_OVERVIEW_BATCH_CHARS):
        summaries.append(await _summarize_batch(model, question, batch, final=False))

    while len(summaries) > 1:
        reduced = []
        for batch in _bounded_batches(summaries, MAX_OVERVIEW_BATCH_CHARS):
            reduced.append(await _summarize_batch(model, question, batch, final=True))
        summaries = reduced
    return summaries[0]


def _overview_units(source_profiles: Sequence[dict]) -> list[str]:
    units: list[str] = []
    for profile in source_profiles:
        profile_units = list(profile.get("summary_units") or [])
        if not profile_units:
            for index, section in enumerate(profile.get("sections") or [], 1):
                page = section.get("page")
                content = str(section.get("content") or "").strip()
                if content:
                    profile_units.append(
                        f"Source: {profile.get('name')}; Page: {page if page is not None else index}\n{content}"
                    )
        for unit in profile_units:
            units.extend(_split_bounded(str(unit), MAX_OVERVIEW_BATCH_CHARS))
    return units


def _split_bounded(text: str, limit: int) -> list[str]:
    return [text[start : start + limit] for start in range(0, len(text), limit) if text[start : start + limit]]


def _bounded_batches(parts: Sequence[str], limit: int) -> list[str]:
    batches: list[str] = []
    current: list[str] = []
    current_size = 0
    separator_size = len("\n\n---\n\n")
    for part in parts:
        part = str(part).strip()
        if not part:
            continue
        addition = len(part) + (separator_size if current else 0)
        if current and current_size + addition > limit:
            batches.append("\n\n---\n\n".join(current))
            current = []
            current_size = 0
            addition = len(part)
        current.append(part)
        current_size += addition
    if current:
        batches.append("\n\n---\n\n".join(current))
    return batches


async def _summarize_batch(model: Any, question: str, context: str, final: bool) -> str:
    stage = "合并以下分批摘要" if final else "概括以下来源正文"
    messages = [
        SystemMessage(
            content=(
                "你是学习助手。只根据给定内容回答，不得混入其他文件或通用猜测。"
                "准确保留主题、结构和重点；资料不足时明确说明。不要展示内部推理过程。"
            )
        ),
        HumanMessage(content=f"用户问题：{question}\n\n任务：{stage}\n\n{context}"),
    ]
    return _message_text(await model.ainvoke(messages))


async def generate_chat_answer(
    model: Any,
    question: str,
    recent_messages: Sequence[Mapping[str, str]],
) -> str:
    
    messages = [
        SystemMessage(
            content="你是自然、准确的学习助手。当前问题不需要检索资料，请正常对话。"
        )
    ]
    for message in recent_messages[-MAX_CHAT_MESSAGES:]:
        content = str(message.get("content") or "")[:MAX_CHAT_MESSAGE_CHARS]
        if message.get("role") == "assistant":
            messages.append(AIMessage(content=content))
        elif message.get("role") == "user":
            messages.append(HumanMessage(content=content))
    messages.append(HumanMessage(content=question))
    return _message_text(await model.ainvoke(messages))


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
