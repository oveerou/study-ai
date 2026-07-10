from __future__ import annotations

from typing import Any, Mapping, Sequence

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


MAX_OVERVIEW_CONTEXT_CHARS = 30000
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
    context_parts = []
    used_chars = 0
    for profile in source_profiles:
        source_text = "\n\n".join(
            str(section.get("content") or "")
            for section in profile.get("sections") or []
        ).strip()
        remaining = MAX_OVERVIEW_CONTEXT_CHARS - used_chars
        if remaining <= 0:
            break
        source_text = source_text[:remaining]
        context_parts.append(f"来源：{profile.get('name')}\n{source_text}")
        used_chars += len(source_text)

    messages = [
        SystemMessage(
            content=(
                "你是学习助手。只根据给定来源内容回答，不得混入其他文件或通用猜测。"
                "用户要求概览时，准确概括主题、结构和重点；来源不足时明确说明。"
            )
        ),
        HumanMessage(
            content=f"用户问题：{question}\n\n已选择来源内容：\n" + "\n\n---\n\n".join(context_parts)
        ),
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
