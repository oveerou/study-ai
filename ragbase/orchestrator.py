

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from langchain_core.documents import Document

from ragbase.agent_planner import plan_operation
from ragbase.agent_responses import (
    generate_chat_answer,
    generate_overview_answer,
)
from ragbase.chain import ask_question, create_chain
from ragbase.planner_schema import EvidenceCitation, OperationPlan
from ragbase.retriever import create_retriever
from ragbase.source_registry import SourceRecord
from ragbase.source_resolver import resolve_sources
from ragbase.source_tools import (
    format_source_full_text,
    format_source_inventory,
)


@dataclass(frozen=True)
class OrchestratorRuntime:
    

    model: Any
    source_records: Sequence[SourceRecord]
    source_profiles: Sequence[dict]
    active_source_ids: Sequence[str]
    chunk_documents: Sequence[Document]
    vector_store: Any
    recent_messages: Sequence[Mapping[str, str]]
    session_id: str


@dataclass(frozen=True)
class ExecutionResult:
    

    answer: str
    documents: tuple[Document, ...]
    plan: OperationPlan
    active_source_ids: tuple[str, ...]
    citations: tuple[EvidenceCitation, ...] = ()
    evidence_level: str = "none"
    missing_information: str | None = None


async def execute_question(question: str, runtime: OrchestratorRuntime) -> ExecutionResult:
    

    records = tuple(runtime.source_records)
    catalog_ids = tuple(record.source_id for record in records)
    resolution = resolve_sources(question, records, runtime.active_source_ids)
    plan = await plan_operation(
        model=runtime.model,
        question=question,
        recent_messages=runtime.recent_messages,
        source_records=records,
        resolution=resolution,
        active_source_ids=runtime.active_source_ids,
    )

    if plan.operation == "list_sources":
        answer = format_source_inventory([record.source_name for record in records])
        return _result(answer, (), plan, runtime.active_source_ids)

    if plan.operation == "chat":
        answer = await generate_chat_answer(
            model=runtime.model,
            question=plan.query or question,
            recent_messages=runtime.recent_messages,
        )
        return _result(answer, (), plan, runtime.active_source_ids)

    if plan.scope == "selected" and not plan.source_ids:
        answer = _source_selection_answer(resolution)
        return _result(answer, (), plan, runtime.active_source_ids)

    selected_ids = _selected_source_ids(
        plan.source_ids,
        runtime.active_source_ids,
        catalog_ids,
    )

    if plan.operation == "read_source":
        if plan.read_mode == "full_text":
            answer = format_source_full_text(
                plan.query or question,
                [record.source_name for record in records],
                list(runtime.source_profiles),
                selected_source_ids=list(selected_ids),
            )
        else:
            answer = await generate_overview_answer(
                model=runtime.model,
                question=plan.query or question,
                source_profiles=_profiles_for_ids(runtime.source_profiles, selected_ids),
            )
        return _result(answer, (), plan, selected_ids)

    if plan.operation != "search":
        raise ValueError(f"Unsupported operation: {plan.operation}")

    retriever = create_retriever(
        runtime.model,
        vector_store=runtime.vector_store,
        source_ids=selected_ids,
        chunk_documents=runtime.chunk_documents,
    )
    chain = create_chain(runtime.model, retriever, use_history=False)
    documents: list[Document] = []
    answer_parts: list[str] = []
    async for event in ask_question(chain, plan.query or question, runtime.session_id):
        if isinstance(event, list):
            documents.extend(event)
        elif isinstance(event, str):
            answer_parts.append(event)
    answer = "".join(answer_parts).strip()
    if not answer:
        answer = "当前资料依据不足，未能生成有证据支持的回答。"
    return _result(answer, documents, plan, selected_ids)


def _selected_source_ids(
    planned_ids: Sequence[str],
    active_ids: Sequence[str],
    catalog_ids: Sequence[str],
) -> tuple[str, ...]:
    catalog = set(catalog_ids)
    planned = tuple(source_id for source_id in dict.fromkeys(planned_ids) if source_id in catalog)
    if planned:
        return planned
    active = tuple(source_id for source_id in dict.fromkeys(active_ids) if source_id in catalog)
    return active or tuple(catalog_ids)


def _profiles_for_ids(profiles: Sequence[dict], source_ids: Sequence[str]) -> list[dict]:
    profiles_by_id = {str(profile.get("source_id") or ""): profile for profile in profiles}
    return [profiles_by_id[source_id] for source_id in source_ids if source_id in profiles_by_id]


def _result(
    answer: str,
    documents: Sequence[Document],
    plan: OperationPlan,
    active_source_ids: Sequence[str],
) -> ExecutionResult:
    citations = _citations_from_documents(documents)
    insufficient = _has_insufficient_evidence_marker(answer)
    if plan.operation == "search":
        evidence_level = "low" if insufficient or not citations else "high"
        missing_information = (
            "检索到的资料不足以支持完整回答。" if evidence_level == "low" else None
        )
    else:
        evidence_level = "none"
        missing_information = None
    return ExecutionResult(
        answer=answer,
        documents=tuple(documents),
        plan=plan,
        active_source_ids=tuple(active_source_ids),
        citations=citations,
        evidence_level=evidence_level,
        missing_information=missing_information,
    )


def _source_selection_answer(resolution: SourceResolution) -> str:
    names = tuple(dict.fromkeys(candidate.source_name for candidate in resolution.candidates))
    if not names:
        return "无法确定你指的是哪份资料，请明确资料名称后再试。"
    choices = "\n".join(f"- {name}" for name in names)
    return f"无法确定你指的是哪份资料，请选择：\n{choices}"


def _citations_from_documents(documents: Sequence[Document]) -> tuple[EvidenceCitation, ...]:
    citations: list[EvidenceCitation] = []
    seen: set[tuple[object, ...]] = set()
    for document in documents:
        metadata = document.metadata
        page_start = metadata.get("page_start", metadata.get("page"))
        page_end = metadata.get("page_end", page_start)
        citation = EvidenceCitation(
            source_id=str(metadata.get("source_id") or ""),
            source_name=str(metadata.get("source_name") or metadata.get("source") or "unknown"),
            page_start=page_start,
            page_end=page_end,
            chunk_id=str(metadata.get("chunk_id") or "unknown"),
        )
        key = (
            citation.source_id,
            citation.source_name,
            citation.page_start,
            citation.page_end,
            citation.chunk_id,
        )
        if key not in seen:
            seen.add(key)
            citations.append(citation)
    return tuple(citations)


def _has_insufficient_evidence_marker(answer: str) -> bool:
    normalized = answer.casefold()
    return any(
        marker in normalized
        for marker in (
            "当前资料依据不足",
            "资料依据不足",
            "未能生成有证据支持",
            "insufficient evidence",
        )
    )
