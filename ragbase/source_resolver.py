

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Sequence

from ragbase.source_registry import SourceRecord, normalize_source_name


@dataclass(frozen=True)
class SourceCandidate:
    

    source_id: str
    source_name: str
    score: float
    reason: str


@dataclass(frozen=True)
class SourceResolution:
    

    source_ids: tuple[str, ...]
    confidence: float
    candidates: tuple[SourceCandidate, ...]


def resolve_sources(
    query: str,
    records: Sequence[SourceRecord],
    active_source_ids: Sequence[str] = (),
) -> SourceResolution:
    

    if not records:
        return SourceResolution((), 0.0, ())

    ordinal = _ordinal_index(query)
    if ordinal is not None and 0 <= ordinal < len(records):
        candidate = _candidate(records[ordinal], 1.0, "序号指代")
        return SourceResolution((candidate.source_id,), 1.0, (candidate,))

    reference_hint = _source_reference_hint(query)
    normalized_query = _normalize_text(reference_hint or query)
    ranked = _rank_name_candidates(normalized_query, records)
    direct = [item for item in ranked if item.score >= 0.95]
    if direct:
        best_score = direct[0].score
        best = [item for item in direct if item.score == best_score]
        if len(best) == 1:
            return SourceResolution((best[0].source_id,), best[0].score, tuple(ranked[:5]))
        return SourceResolution((), 0.65, tuple(ranked[:5]))

    if ranked:
        top = ranked[0]
        runner_up_score = ranked[1].score if len(ranked) > 1 else 0.0
        if top.score >= 0.72 and top.score - runner_up_score >= 0.07:
            return SourceResolution((top.source_id,), top.score, tuple(ranked[:5]))
        if top.score >= 0.65:
            return SourceResolution((), min(0.65, top.score), tuple(ranked[:5]))

    active_records = [record for record in records if record.source_id in set(active_source_ids)]
    if _contains_source_pronoun(query) and active_records:
        candidates = tuple(_candidate(record, 0.95, "当前来源指代") for record in active_records)
        return SourceResolution(tuple(record.source_id for record in active_records), 0.95, candidates)

    return SourceResolution((), 0.0, ())


def _rank_name_candidates(query: str, records: Sequence[SourceRecord]) -> list[SourceCandidate]:
    candidates: list[tuple[int, SourceCandidate]] = []
    for index, record in enumerate(records):
        score, reason = _score_record(query, record)
        if score >= 0.65:
            candidates.append((index, _candidate(record, score, reason)))
    candidates.sort(key=lambda item: (-item[1].score, item[0]))
    return [candidate for _, candidate in candidates]


def _score_record(query: str, record: SourceRecord) -> tuple[float, str]:
    aliases = _source_aliases(record)
    direct_scores = [1.0 if alias == record.normalized_name else 0.98 for alias in aliases if alias in query]
    if direct_scores:
        return max(direct_scores), "名称匹配"

    best_similarity = max((_best_window_similarity(query, alias) for alias in aliases), default=0.0)
    longest_match = max((_longest_common_length(query, alias) for alias in aliases), default=0)
    partial_score = min(0.9, 0.55 + 0.06 * longest_match) if longest_match >= 2 else 0.0
    if best_similarity >= partial_score and best_similarity >= 0.7:
        return round(best_similarity, 4), "相似名称"
    if partial_score:
        return round(partial_score, 4), "部分名称匹配"
    return 0.0, "无匹配"


def _source_aliases(record: SourceRecord) -> tuple[str, ...]:
    stem = re.sub(
        r"\.(?:pdf|docx?|md|txt|html?|json|ya?ml|toml|py|sql)\s*$",
        "",
        unicodedata.normalize("NFKC", record.source_name).strip(),
        flags=re.IGNORECASE,
    )
    descriptive_stem = re.sub(r"^(?:\d{1,4}\s*[-_.、]?\s*)+", "", stem)
    topic_stem = re.sub(
        r"^(?:\d{1,4}\s*)?(?:入门篇|基础篇|前沿技术篇)\s*[-—_+.、]*\s*",
        "",
        stem,
    )
    topic_stem = re.sub(r"^\d{1,3}\s*", "", topic_stem)
    topic_stem = re.sub(r"\d{1,3}$", "", topic_stem)
    aliases = [
        record.normalized_name,
        normalize_source_name(descriptive_stem),
        normalize_source_name(topic_stem),
    ]
    return tuple(dict.fromkeys(alias for alias in aliases if len(alias) >= 2))


def _best_window_similarity(query: str, alias: str) -> float:
    if not query or not alias:
        return 0.0
    best = 0.0
    for width in range(max(2, len(alias) - 1), min(len(query), len(alias) + 1) + 1):
        for start in range(0, len(query) - width + 1):
            best = max(best, SequenceMatcher(None, query[start : start + width], alias).ratio())
    return best


def _longest_common_length(left: str, right: str) -> int:
    return SequenceMatcher(None, left, right).find_longest_match().size


def _normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value or "")).casefold()
    value = re.sub(r"\.(?:pdf|docx?|md|txt|html?|json|ya?ml|toml|py|sql)\b", "", value)
    return "".join(character for character in value if character.isalnum())


def _ordinal_index(query: str) -> int | None:
    match = re.search(
        r"(?:第\s*([零〇一二两三四五六七八九十百\d]+)\s*(?:个|份|篇)?(?:文件|资料|文档|来源)?|"
        r"([零〇一二两三四五六七八九十百\d]+)\s*(?:个|份|篇)(?:文件|资料|文档|来源)?)",
        query,
    )
    if not match:
        return None
    number = _parse_chinese_number(match.group(1) or match.group(2))
    return number - 1 if number and number > 0 else None


def _source_reference_hint(query: str) -> str:
    
    patterns = (
        r"([A-Za-z0-9+_.\-—\u4e00-\u9fff]{2,30}?)(?:(?:那|这)(?:一)?(?:份|个|篇)|(?:一)?(?:份|个))(?:文件|资料|文档|课件)?",
        r"([A-Za-z0-9+_.\-—\u4e00-\u9fff]{2,30}?)的?(?:文件|资料|文档|课件)(?:里|中|内)?",
    )
    for pattern in patterns:
        match = re.search(pattern, query)
        if match:
            return match.group(1)
    return ""


def _parse_chinese_number(value: str) -> int | None:
    if value.isdigit():
        return int(value)
    digits = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    if value == "十":
        return 10
    if "十" in value:
        tens, ones = value.split("十", 1)
        return (digits.get(tens, 1) * 10) + digits.get(ones, 0)
    return digits.get(value)


def _contains_source_pronoun(query: str) -> bool:
    return bool(
        re.search(
            r"(?:它|其|其中|这里|那里|这里面|那里面|这个|那个|刚才|上一个|前一个|这份|那份|该)(?:文件|资料|文档)?",
            query,
        )
    )


def _candidate(record: SourceRecord, score: float, reason: str) -> SourceCandidate:
    return SourceCandidate(record.source_id, record.source_name, score, reason)
