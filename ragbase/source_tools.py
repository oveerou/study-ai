

from __future__ import annotations

import re
from collections import OrderedDict
from typing import Iterable

from langchain_core.documents import Document


INVENTORY_TERMS = (
    "我给了你什么文件",
    "给了你什么文件",
    "上传了什么文件",
    "导入了什么文件",
    "有哪些文件",
    "哪些文件",
    "文件名称",
    "文件的名称",
    "文件名字",
    "文件列表",
    "资料清单",
    "已导入资料",
)

OVERVIEW_TERMS = (
    "资料都有什么",
    "资料有什么",
    "资料里有什么",
    "资料里面有什么",
    "这些资料主要讲什么",
    "这些资料整体讲什么",
    "这些资料讲什么",
    "文件里有什么",
    "文件里面有什么",
    "内容是什么",
    "里面的内容",
    "主要内容",
    "主要讲什么",
)

FULL_TEXT_DIRECT_TERMS = (
    "全文",
    "原文",
    "完整正文",
    "全部正文",
    "完整内容",
    "所有内容",
    "完整版",
    "逐字",
    "逐页",
)

FULL_TEXT_COMPLETENESS_TERMS = (
    "全部",
    "所有",
    "不省略",
    "不要省略",
    "别省略",
    "不简略",
    "不要简略",
    "别简略",
    "不概括",
    "不要概括",
    "别概括",
    "都输出",
    "全输出",
    "都展示",
    "全展示",
    "全给我",
    "都给我",
)

FULL_TEXT_SCOPE_TERMS = (
    "内容",
    "正文",
    "文档",
    "文件",
    "资料",
    "里面",
    "里边",
)


def source_tool_route(question: str) -> str | None:
    
    text = _compact(question)
    if not text:
        return None
    if any(term in text for term in INVENTORY_TERMS):
        return "inventory"
    if any(term in text for term in FULL_TEXT_DIRECT_TERMS):
        return "full_text"
    if any(term in text for term in FULL_TEXT_COMPLETENESS_TERMS) and any(
        term in text for term in FULL_TEXT_SCOPE_TERMS
    ):
        return "full_text"
    if any(term in text for term in OVERVIEW_TERMS):
        return "overview"
    if ("文件" in text or "资料" in text or "文档" in text) and any(
        term in text for term in ("名称", "名字", "列表", "清单", "哪些", "给了", "上传", "导入")
    ):
        return "inventory"
    if ("文件" in text or "资料" in text or "文档" in text) and any(
        term in text for term in ("内容", "里面", "主要", "讲什么", "有什么", "都有啥", "都有些什么")
    ):
        return "overview"
    return None


def build_source_profiles(documents: Iterable[Document], max_chars_per_source: int = 900) -> list[dict]:
    
    grouped: OrderedDict[str, dict] = OrderedDict()
    for order, doc in enumerate(documents):
        name = str(doc.metadata.get("source_name") or doc.metadata.get("source") or "未命名来源")
        source_type = str(doc.metadata.get("source_type") or "source")
        source_id = str(doc.metadata.get("source_id") or "")
        group_key = source_id or name
        profile = grouped.setdefault(
            group_key,
            {
                "source_id": source_id,
                "name": name,
                "type": source_type,
                "sections": [],
            },
        )
        text = _normalize_full_text(doc.page_content or "")
        if text:
            profile["sections"].append(
                {
                    "page": doc.metadata.get("page"),
                    "content": text,
                    "_order": order,
                }
            )

    profiles = []
    for profile in grouped.values():
        profile["sections"].sort(key=_section_sort_key)
        for section in profile["sections"]:
            section.pop("_order", None)
        combined = _clean_text("\n".join(section["content"] for section in profile["sections"]))
        profiles.append(
            {
                "source_id": profile["source_id"],
                "name": profile["name"],
                "type": profile["type"],
                "excerpt": combined[:max_chars_per_source],
                "sections": profile["sections"],
                "summary_units": _build_summary_units(profile["name"], profile["sections"]),
            }
        )
    return profiles


def answer_source_tool(
    question: str,
    source_names: list[str],
    source_profiles: list[dict],
    route: str | None = None,
    selected_source_names: list[str] | None = None,
) -> str:
    
    route = route or source_tool_route(question)
    if route == "inventory":
        return format_source_inventory(selected_source_names or source_names)
    if route == "overview":
        overview_names = selected_source_names or source_names
        overview_profiles = _profiles_for_names(source_profiles, overview_names)
        return format_source_overview(overview_names, overview_profiles)
    if route == "full_text":
        return format_source_full_text(
            question,
            source_names,
            source_profiles,
            selected_source_names=selected_source_names,
        )
    return ""


def format_source_inventory(source_names: list[str]) -> str:
    
    if not source_names:
        return "当前还没有导入资料。"
    lines = [f"当前已导入 {len(source_names)} 个来源：", ""]
    lines.extend(f"- {name}" for name in source_names)
    return "\n".join(lines)


def format_source_overview(source_names: list[str], source_profiles: list[dict]) -> str:
    
    if not source_names:
        return "当前还没有导入资料。"

    profiles_by_name = {profile.get("name"): profile for profile in source_profiles}
    lines = [
        f"当前已导入 {len(source_names)} 个来源，按文件概览如下：",
        "",
    ]

    for name in source_names:
        profile = profiles_by_name.get(name, {})
        excerpt = _trim_sentence(profile.get("excerpt") or _infer_from_name(name), 180)
        lines.append(f"- **{name}**：{excerpt}")

    lines.extend(
        [
            "",
            "如果你要查某一份文件，可以直接问“某某文件讲什么”或“输出某某文件的正文”。",
        ]
    )
    return "\n".join(lines)


def format_source_full_text(
    question: str,
    source_names: list[str],
    source_profiles: list[dict],
    selected_source_names: list[str] | None = None,
    selected_source_ids: list[str] | None = None,
) -> str:
    
    if not source_names:
        return "当前还没有导入资料。"

    if selected_source_ids is not None:
        selected_profiles = _profiles_for_ids(source_profiles, selected_source_ids)
    elif selected_source_names is not None:
        selected_profiles = _profiles_for_names(source_profiles, selected_source_names)
    else:
        selected_profiles = _select_source_profiles(question, source_names, source_profiles)
    if not selected_profiles:
        return "当前资料没有可输出的正文。"

    lines = [f"以下为 {len(selected_profiles)} 个来源的完整正文："]
    for profile in selected_profiles:
        lines.extend(["", f"# {profile['name']}"])
        sections = sorted(profile.get("sections") or [], key=_section_sort_key)
        for index, section in enumerate(sections, 1):
            page = section.get("page")
            if page is not None:
                lines.extend(["", f"## 第 {page} 页"])
            elif len(sections) > 1:
                lines.extend(["", f"## 第 {index} 部分"])
            lines.extend(["", section.get("content", "")])
    return "\n".join(lines).strip()


def _profiles_for_names(source_profiles: list[dict], source_names: list[str]) -> list[dict]:
    
    profiles_by_name = {str(profile.get("name")): profile for profile in source_profiles}
    return [profiles_by_name[name] for name in source_names if name in profiles_by_name]


def _profiles_for_ids(source_profiles: list[dict], source_ids: list[str]) -> list[dict]:
    
    profiles_by_id = {str(profile.get("source_id") or ""): profile for profile in source_profiles}
    return [profiles_by_id[source_id] for source_id in source_ids if source_id in profiles_by_id]


def _section_sort_key(section: dict) -> tuple[int, int]:
    page = section.get("page")
    try:
        return (0, int(page)) if page is not None else (1, int(section.get("_order", 0)))
    except (TypeError, ValueError):
        return (1, int(section.get("_order", 0)))


def _build_summary_units(name: str, sections: list[dict], max_chars: int = 6000) -> list[str]:
    
    units: list[str] = []
    for index, section in enumerate(sections, 1):
        page = section.get("page")
        label = f"Source: {name}; Page: {page if page is not None else index}"
        content = str(section.get("content") or "")
        if not content:
            continue
        for start in range(0, len(content), max_chars):
            units.append(f"{label}\n{content[start:start + max_chars]}")
    return units


def _select_source_profiles(
    question: str,
    source_names: list[str],
    source_profiles: list[dict],
) -> list[dict]:
    
    profiles_by_name = {str(profile.get("name")): profile for profile in source_profiles}
    available = [profiles_by_name[name] for name in source_names if name in profiles_by_name]
    compact_question = _compact(question)
    selected = []
    for profile in available:
        stem = re.sub(r"\.[A-Za-z0-9]+$", "", str(profile.get("name") or ""))
        name_parts = re.findall(r"[A-Za-z]+|[\u4e00-\u9fff]{2,}", stem)
        if any(_compact(part) in compact_question for part in name_parts):
            selected.append(profile)
    return selected or available


def _infer_from_name(name: str) -> str:
    
    stem = re.sub(r"\.[A-Za-z0-9]+$", "", name)
    stem = re.sub(r"^\d+\s*", "", stem)
    return stem.replace("_", " ").replace("-", " ").strip() or "已导入资料"


def _trim_sentence(text: str, limit: int) -> str:
    
    text = _clean_text(text)
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _clean_text(text: str) -> str:
    
    return re.sub(r"\s+", " ", text or "").strip()


def _normalize_full_text(text: str) -> str:
    
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _compact(text: str) -> str:
    
    return re.sub(r"[\s？?。！!，,：:；;“”\"'`]+", "", text or "").lower()
