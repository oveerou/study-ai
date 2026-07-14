

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from langchain_core.messages import HumanMessage, SystemMessage


MAX_GRAPH_CONTEXT_CHARS = 24000
MAX_TRIPLES = 80


async def extract_knowledge_graph(model: Any, source_profiles: Sequence[dict]) -> dict:
    
    source_names = [str(profile.get("name")) for profile in source_profiles]
    context_parts = []
    used_chars = 0
    for profile in source_profiles:
        text = "\n\n".join(
            str(section.get("content") or "")
            for section in profile.get("sections") or []
        ).strip()
        remaining = MAX_GRAPH_CONTEXT_CHARS - used_chars
        if remaining <= 0:
            break
        text = text[:remaining]
        context_parts.append(f"来源：{profile.get('name')}\n{text}")
        used_chars += len(text)

    messages = [
        SystemMessage(
            content=(
                "你是知识图谱抽取器。只从给定资料提取明确出现或可直接推出的实体关系，"
                "不要补充外部知识。只输出 JSON：{\"triples\":[{\"subject\":\"\","
                "\"predicate\":\"\",\"object\":\"\",\"source_name\":\"\","
                "\"evidence\":\"\"}]}。source_name 必须逐字复制来源名称，最多 80 条。"
            )
        ),
        HumanMessage(content="\n\n---\n\n".join(context_parts)),
    ]
    extraction_mode = "model"
    error = ""
    try:
        response = await model.ainvoke(messages)
        payload = _parse_json(_message_text(response))
        triples = _normalize_model_triples(payload, source_names)
    except Exception as exc:
        extraction_mode = "fallback"
        error = str(exc)
        triples = _fallback_extract_triples(source_profiles)

    return {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(),
        "source_names": source_names,
        "extraction_mode": extraction_mode,
        "error": error,
        "triples": triples[:MAX_TRIPLES],
    }


def _normalize_model_triples(payload: dict, source_names: Sequence[str]) -> list[dict]:
    
    triples = []
    for raw in payload.get("triples") or []:
        if not isinstance(raw, dict):
            continue
        triple = {
            "subject": str(raw.get("subject") or "").strip(),
            "predicate": str(raw.get("predicate") or "").strip(),
            "object": str(raw.get("object") or "").strip(),
            "source_name": str(raw.get("source_name") or "").strip(),
            "evidence": str(raw.get("evidence") or "").strip(),
        }
        if (
            triple["subject"]
            and triple["predicate"]
            and triple["object"]
            and triple["source_name"] in source_names
        ):
            triples.append(triple)
        if len(triples) >= MAX_TRIPLES:
            break
    return triples


def _fallback_extract_triples(source_profiles: Sequence[dict]) -> list[dict]:
    
    triples: list[dict] = []
    seen: set[tuple[str, str, str, str]] = set()
    for profile in source_profiles:
        source_name = str(profile.get("name") or "").strip()
        text = "\n".join(
            str(section.get("content") or "")
            for section in profile.get("sections") or []
        )
        for statement in _candidate_statements(text):
            for triple in _triples_from_statement(statement, source_name):
                key = (
                    triple["subject"],
                    triple["predicate"],
                    triple["object"],
                    triple["source_name"],
                )
                if key in seen:
                    continue
                seen.add(key)
                triples.append(triple)
                if len(triples) >= MAX_TRIPLES:
                    return triples
    return triples


def save_knowledge_graph(graph_dir: Path, session_id: str, graph: dict) -> Path:
    
    graph_dir.mkdir(parents=True, exist_ok=True)
    safe_session_id = re.sub(r"[^A-Za-z0-9._-]", "_", session_id)
    path = graph_dir / f"{safe_session_id}.json"
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(graph, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)
    return path


def load_knowledge_graph(path: Path) -> dict:
    
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("triples"), list):
        raise ValueError(f"invalid knowledge graph file: {path}")
    return payload


def _candidate_statements(text: str) -> list[str]:
    
    normalized = re.sub(r"[ \t]+", " ", text)
    lines = []
    for raw_line in re.split(r"[\n。；;]", normalized):
        line = raw_line.strip(" \t\r\n-•·")
        line = re.sub(r"^\d+[\.、\)]\s*", "", line).strip()
        if 4 <= len(line) <= 160:
            lines.append(line)
    return lines


def _triples_from_statement(statement: str, source_name: str) -> list[dict]:
    
    patterns = [
        (r"(.{2,40}?)(?:的)?作用是(.{2,60})", "作用是"),
        (r"(.{2,40}?)(?:的)?定义是(.{2,60})", "定义是"),
        (r"(.{2,40}?)(?:主要)?包括(.{2,80})", "包括"),
        (r"(.{2,40}?)(?:主要)?用于(.{2,60})", "用于"),
        (r"(.{2,40}?)(?:可以|可)(?:用于|产生)(.{2,60})", "可用于"),
        (r"(.{2,40}?)[是为](.{2,60})", "是"),
    ]
    triples = []
    for pattern, predicate in patterns:
        match = re.search(pattern, statement)
        if not match:
            continue
        subject = _clean_entity(match.group(1))
        object_text = _clean_entity(match.group(2))
        if not subject or not object_text or subject == object_text:
            continue
        objects = [
            _clean_entity(part)
            for part in re.split(r"[、,，/和及与]", object_text)
            if _clean_entity(part)
        ] or [object_text]
        for object_ in objects[:4]:
            if object_ and object_ != subject:
                triples.append(
                    {
                        "subject": subject,
                        "predicate": predicate,
                        "object": object_,
                        "source_name": source_name,
                        "evidence": statement,
                    }
                )
        if triples:
            return triples
    return []


def _clean_entity(value: str) -> str:
    
    value = re.sub(r"\s+", " ", str(value or "")).strip(" ：:，,。.；;（）()[]【】")
    value = re.sub(r"^(其中|例如|如|以及|和|与|及)", "", value)
    return value[:50].strip()


def graph_to_interactive_html(graph: dict) -> str:
    
    triples = list(graph.get("triples") or [])

    nodes = _graph_nodes(triples)
    edges = _graph_edges(triples, nodes)
    payload = json.dumps(
        {
            "nodes": nodes,
            "edges": edges,
        },
        ensure_ascii=False,
    )
    return f"""
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<div class="kg-browser" id="kgBrowser">
  <header class="kg-topbar">
    <div>
      <strong>知识图谱浏览器</strong>
      <span>完整图谱 · 搜索、邻居聚焦、拖拽、缩放、全屏</span>
    </div>
    <div class="kg-actions">
      <button type="button" id="kgZoomIn">放大</button>
      <button type="button" id="kgZoomOut">缩小</button>
      <button type="button" id="kgFit">适应视图</button>
      <button type="button" id="kgPhysics">物理布局</button>
      <button type="button" id="kgFullscreen">全屏</button>
    </div>
  </header>
  <aside class="kg-search-panel">
    <label for="kgSearch">搜索实体或关系</label>
    <input id="kgSearch" type="search" placeholder="输入关键词..." />
    <div class="kg-summary">
      <span id="kgNodeCount">0 个实体</span>
      <span id="kgEdgeCount">0 条关系</span>
    </div>
    <div id="kgResults" class="kg-results"></div>
  </aside>
  <main class="kg-canvas-wrap">
    <div id="kgNetwork" class="kg-network"></div>
  </main>
  <aside class="kg-detail-panel">
    <h3>详情</h3>
    <div id="kgDetails" class="kg-details">点击节点或关系查看详情。点击搜索结果可定位到图中位置。</div>
  </aside>
</div>
<script>
const graphData = {payload};
const nodeById = new Map(graphData.nodes.map((node) => [node.id, node]));
const sourceEdges = new Map();
const targetEdges = new Map();
graphData.edges.forEach((edge) => {{
  sourceEdges.set(edge.source, [...(sourceEdges.get(edge.source) || []), edge]);
  targetEdges.set(edge.target, [...(targetEdges.get(edge.target) || []), edge]);
}});

function clipLabel(label, size = 22) {{
  const value = String(label || "");
  return value.length > size ? value.slice(0, size - 1) + "…" : value;
}}

function escapeHtml(value) {{
  return String(value || "").replace(/[&<>"']/g, (char) => ({{
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }}[char]));
}}

const visNodes = new vis.DataSet(graphData.nodes.map((node) => ({{
  id: node.id,
  label: clipLabel(node.label),
  title: escapeHtml(node.label),
  value: node.degree,
  x: node.x,
  y: node.y,
  color: {{
    background: node.color,
    border: "#0f172a",
    highlight: {{ background: "#f97316", border: "#7c2d12" }},
  }},
  font: {{ size: 16, face: "Segoe UI", color: "#0f172a", strokeWidth: 4, strokeColor: "#ffffff" }},
  shape: "dot",
}})));

const visEdges = new vis.DataSet(graphData.edges.map((edge) => ({{
  id: edge.id,
  from: edge.source,
  to: edge.target,
  label: edge.label,
  title: escapeHtml(`${{nodeById.get(edge.source)?.label || ""}} → ${{nodeById.get(edge.target)?.label || ""}}\\n${{edge.label}}\\n${{edge.evidence || ""}}`),
  arrows: "to",
  color: {{ color: "#94a3b8", highlight: "#f97316" }},
  font: {{ size: 13, align: "middle", color: "#334155", strokeWidth: 5, strokeColor: "#ffffff" }},
  smooth: {{ type: "dynamic" }},
}})));

const container = document.getElementById("kgNetwork");
const details = document.getElementById("kgDetails");
const results = document.getElementById("kgResults");
let physicsEnabled = true;
const network = new vis.Network(container, {{ nodes: visNodes, edges: visEdges }}, {{
  autoResize: true,
  interaction: {{
    hover: true,
    multiselect: true,
    navigationButtons: true,
    keyboard: true,
    tooltipDelay: 120,
  }},
  physics: {{
    enabled: true,
    stabilization: {{ iterations: 180, fit: true }},
    barnesHut: {{
      gravitationalConstant: -8500,
      centralGravity: 0.12,
      springLength: 180,
      springConstant: 0.035,
      damping: 0.35,
      avoidOverlap: 0.85,
    }},
  }},
  nodes: {{ borderWidth: 1.5, scaling: {{ min: 18, max: 46 }} }},
  edges: {{ width: 1.4, selectionWidth: 3 }},
}});

document.getElementById("kgNodeCount").textContent = `${{graphData.nodes.length}} 个实体`;
document.getElementById("kgEdgeCount").textContent = `${{graphData.edges.length}} 条关系`;

function neighborIds(nodeId) {{
  const ids = new Set([nodeId]);
  (sourceEdges.get(nodeId) || []).forEach((edge) => ids.add(edge.target));
  (targetEdges.get(nodeId) || []).forEach((edge) => ids.add(edge.source));
  return ids;
}}

function showNode(nodeId) {{
  const node = nodeById.get(nodeId);
  if (!node) return;
  const related = [...(sourceEdges.get(nodeId) || []), ...(targetEdges.get(nodeId) || [])];
  const neighbors = neighborIds(nodeId);
  visNodes.update(graphData.nodes.map((item) => ({{
    id: item.id,
    opacity: neighbors.has(item.id) ? 1 : 0.18,
    color: {{
      background: item.id === nodeId ? "#f97316" : item.color,
      border: item.id === nodeId ? "#7c2d12" : "#0f172a",
    }},
  }})));
  visEdges.update(graphData.edges.map((edge) => ({{
    id: edge.id,
    color: {{
      color: edge.source === nodeId || edge.target === nodeId ? "#f97316" : "rgba(148,163,184,.22)",
      highlight: "#f97316",
    }},
  }})));
  details.innerHTML = `
    <div class="kg-detail-title">${{escapeHtml(node.label)}}</div>
    <div class="kg-detail-meta">邻居聚焦：${{neighbors.size - 1}} 个关联实体 · ${{related.length}} 条关系</div>
    <div class="kg-relation-list">
      ${{related.map((edge) => `
        <button type="button" data-edge="${{edge.id}}">
          <strong>${{escapeHtml(edge.label)}}</strong>
          <span>${{escapeHtml(nodeById.get(edge.source)?.label || edge.source)}} → ${{escapeHtml(nodeById.get(edge.target)?.label || edge.target)}}</span>
        </button>
      `).join("") || "<p>没有关联关系。</p>"}}
    </div>`;
  network.selectNodes([nodeId]);
  network.focus(nodeId, {{ scale: 1.15, animation: {{ duration: 450, easingFunction: "easeInOutQuad" }} }});
}}

function showEdge(edgeId) {{
  const edge = graphData.edges.find((item) => item.id === edgeId);
  if (!edge) return;
  const source = nodeById.get(edge.source);
  const target = nodeById.get(edge.target);
  details.innerHTML = `
    <div class="kg-detail-title">${{escapeHtml(edge.label)}}</div>
    <div class="kg-detail-meta">${{escapeHtml(source?.label || edge.source)}} → ${{escapeHtml(target?.label || edge.target)}}</div>
    <p><strong>来源：</strong>${{escapeHtml(edge.sourceName)}}</p>
    <p><strong>证据：</strong>${{escapeHtml(edge.evidence || "未记录证据文本")}}</p>`;
  network.selectEdges([edgeId]);
  network.fit({{ nodes: [edge.source, edge.target], animation: {{ duration: 450, easingFunction: "easeInOutQuad" }} }});
}}

function resetStyle() {{
  visNodes.update(graphData.nodes.map((node) => ({{
    id: node.id,
    opacity: 1,
    color: {{ background: node.color, border: "#0f172a" }},
  }})));
  visEdges.update(graphData.edges.map((edge) => ({{
    id: edge.id,
    color: {{ color: "#94a3b8", highlight: "#f97316" }},
  }})));
}}

function renderResults(items) {{
  if (!items.length) {{
    results.innerHTML = '<div class="kg-empty">没有匹配结果。</div>';
    return;
  }}
  results.innerHTML = items.slice(0, 40).map((item) => `
    <button type="button" data-kind="${{item.kind}}" data-id="${{item.id}}">
      <strong>${{escapeHtml(item.title)}}</strong>
      <span>${{escapeHtml(item.subtitle)}}</span>
    </button>
  `).join("");
}}

function searchGraph(query) {{
  const keyword = query.trim().toLowerCase();
  if (!keyword) {{
    renderResults([]);
    resetStyle();
    details.innerHTML = "点击节点或关系查看详情。点击搜索结果可定位到图中位置。";
    return;
  }}
  const nodeResults = graphData.nodes
    .filter((node) => node.label.toLowerCase().includes(keyword))
    .map((node) => ({{ kind: "node", id: node.id, title: node.label, subtitle: "实体" }}));
  const edgeResults = graphData.edges
    .filter((edge) => `${{edge.label}} ${{edge.sourceName}} ${{edge.evidence}}`.toLowerCase().includes(keyword))
    .map((edge) => ({{
      kind: "edge",
      id: edge.id,
      title: edge.label,
      subtitle: `${{nodeById.get(edge.source)?.label || ""}} → ${{nodeById.get(edge.target)?.label || ""}}`,
    }}));
  renderResults([...nodeResults, ...edgeResults]);
}}

document.getElementById("kgSearch").addEventListener("input", (event) => searchGraph(event.target.value));
results.addEventListener("click", (event) => {{
  const button = event.target.closest("button");
  if (!button) return;
  if (button.dataset.kind === "node") showNode(button.dataset.id);
  if (button.dataset.kind === "edge") showEdge(button.dataset.id);
}});
details.addEventListener("click", (event) => {{
  const button = event.target.closest("button[data-edge]");
  if (button) showEdge(button.dataset.edge);
}});
network.on("click", (params) => {{
  if (params.nodes.length) return showNode(params.nodes[0]);
  if (params.edges.length) return showEdge(params.edges[0]);
  resetStyle();
}});
network.once("stabilizationIterationsDone", () => {{
  physicsEnabled = false;
  network.setOptions({{ physics: false }});
}});

document.getElementById("kgZoomIn").onclick = () => network.moveTo({{ scale: network.getScale() * 1.2 }});
document.getElementById("kgZoomOut").onclick = () => network.moveTo({{ scale: network.getScale() / 1.2 }});
document.getElementById("kgFit").onclick = () => {{ resetStyle(); network.fit({{ animation: {{ duration: 450 }} }}); }};
document.getElementById("kgPhysics").onclick = () => {{
  physicsEnabled = !physicsEnabled;
  network.setOptions({{ physics: {{ enabled: physicsEnabled }} }});
  if (physicsEnabled) network.stabilize(80);
}};
document.getElementById("kgFullscreen").onclick = () => {{
  const shell = document.getElementById("kgBrowser");
  if (!document.fullscreenElement) {{
    shell.requestFullscreen?.();
    setTimeout(() => network.fit({{ animation: true }}), 250);
  }} else {{
    document.exitFullscreen?.();
  }}
}};
setTimeout(() => network.fit({{ animation: true }}), 400);
</script>
<style>
.kg-browser {{
  height: 740px;
  display: grid;
  grid-template-columns: 280px minmax(0, 1fr) 320px;
  grid-template-rows: auto 1fr;
  border: 1px solid #d7dde8;
  border-radius: 8px;
  background: #ffffff;
  overflow: hidden;
  font-family: Inter, "Segoe UI", system-ui, sans-serif;
}}
.kg-browser:fullscreen {{
  width: 100vw;
  height: 100vh;
  background: #ffffff;
}}
.kg-topbar {{
  grid-column: 1 / 4;
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: center;
  padding: 12px 14px;
  border-bottom: 1px solid #e5e7eb;
  background: #f8fafc;
  color: #0f172a;
}}
.kg-topbar span {{
  color: #64748b;
  font-size: 13px;
  margin-left: 8px;
}}
.kg-actions {{
  display: flex;
  gap: 8px;
}}
.kg-actions button {{
  border: 1px solid #cbd5e1;
  background: #ffffff;
  border-radius: 6px;
  padding: 6px 10px;
  color: #0f172a;
  cursor: pointer;
}}
.kg-actions button:hover {{
  background: #eff6ff;
  border-color: #60a5fa;
}}
.kg-search-panel, .kg-detail-panel {{
  background: #f8fafc;
  border-right: 1px solid #e5e7eb;
  padding: 14px;
  overflow: auto;
}}
.kg-detail-panel {{
  border-right: 0;
  border-left: 1px solid #e5e7eb;
}}
.kg-search-panel label {{
  display: block;
  font-size: 13px;
  font-weight: 700;
  margin-bottom: 8px;
  color: #0f172a;
}}
#kgSearch {{
  width: 100%;
  box-sizing: border-box;
  border: 1px solid #cbd5e1;
  border-radius: 6px;
  padding: 9px 10px;
  font-size: 14px;
  outline: none;
}}
#kgSearch:focus {{
  border-color: #2563eb;
  box-shadow: 0 0 0 3px rgba(37,99,235,.12);
}}
.kg-summary {{
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  margin: 12px 0;
}}
.kg-summary span {{
  border: 1px solid #dbe3ef;
  background: #fff;
  border-radius: 999px;
  padding: 4px 8px;
  font-size: 12px;
  color: #475569;
}}
.kg-results, .kg-relation-list {{
  display: grid;
  gap: 8px;
}}
.kg-results button, .kg-relation-list button {{
  text-align: left;
  border: 1px solid #dbe3ef;
  background: #fff;
  border-radius: 6px;
  padding: 9px;
  cursor: pointer;
}}
.kg-results button:hover, .kg-relation-list button:hover {{
  border-color: #60a5fa;
  background: #eff6ff;
}}
.kg-results strong, .kg-relation-list strong {{
  display: block;
  color: #0f172a;
  font-size: 13px;
  line-height: 1.35;
}}
.kg-results span, .kg-relation-list span, .kg-detail-meta {{
  display: block;
  color: #64748b;
  font-size: 12px;
  line-height: 1.4;
  margin-top: 3px;
}}
.kg-canvas-wrap {{
  min-width: 0;
  background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
}}
.kg-network {{
  width: 100%;
  height: 100%;
}}
.kg-detail-panel h3 {{
  margin: 0 0 10px;
  color: #0f172a;
}}
.kg-details {{
  font-size: 14px;
  color: #334155;
  line-height: 1.6;
}}
.kg-detail-title {{
  font-size: 17px;
  font-weight: 800;
  color: #0f172a;
  margin-bottom: 6px;
}}
.kg-empty {{
  font-size: 14px;
  color: #64748b;
  padding: 8px 0;
}}
.vis-network:focus {{
  outline: none;
}}
.kg-browser:fullscreen .kg-network {{
  height: calc(100vh - 58px);
}}
@media (max-width: 900px) {{
  .kg-browser {{
    grid-template-columns: 1fr;
    grid-template-rows: auto auto 560px auto;
    height: auto;
  }}
  .kg-topbar, .kg-search-panel, .kg-canvas-wrap, .kg-detail-panel {{
    grid-column: 1;
  }}
  .kg-search-panel, .kg-detail-panel {{
    border: 0;
    border-bottom: 1px solid #e5e7eb;
  }}
}}
</style>
"""


def graph_to_dot(graph: dict) -> str:
    
    labels = []
    for triple in graph.get("triples") or []:
        for key in ("subject", "object"):
            label = str(triple.get(key) or "").strip()
            if label and label not in labels:
                labels.append(label)
    node_ids = {label: f"n{index}" for index, label in enumerate(labels)}
    lines = [
        "digraph KnowledgeGraph {",
        "rankdir=LR;",
        'node [shape=box, style="rounded,filled", fillcolor="#f8fafc", color="#94a3b8"];',
        'edge [color="#64748b"];',
    ]
    for label, node_id in node_ids.items():
        lines.append(f'{node_id} [label="{_dot_escape(label)}"];')
    for triple in graph.get("triples") or []:
        subject = str(triple.get("subject") or "").strip()
        object_ = str(triple.get("object") or "").strip()
        predicate = str(triple.get("predicate") or "").strip()
        if subject in node_ids and object_ in node_ids and predicate:
            lines.append(
                f'{node_ids[subject]} -> {node_ids[object_]} [label="{_dot_escape(predicate)}"];'
            )
    lines.append("}")
    return "\n".join(lines)


def _parse_json(text: str) -> dict:
    
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("knowledge graph model did not return JSON")
    payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("knowledge graph JSON must be an object")
    return payload


def _message_text(message: Any) -> str:
    
    content = getattr(message, "content", message)
    return content if isinstance(content, str) else str(content)


def _dot_escape(value: str) -> str:
    
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def _graph_nodes(triples: Sequence[dict]) -> list[dict]:
    
    labels = []
    degree: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    for triple_index, triple in enumerate(triples):
        subject = str(triple.get("subject") or "").strip()
        object_ = str(triple.get("object") or "").strip()
        if not subject or not object_:
            continue
        for label in (subject, object_):
            if label not in labels:
                labels.append(label)
            degree[label] = degree.get(label, 0) + 1
            first_seen.setdefault(label, triple_index)

    if not labels:
        return []

    labels = sorted(labels, key=lambda label: (-degree.get(label, 0), first_seen.get(label, 0), label))
    column_count = max(2, min(14, math.ceil(math.sqrt(len(labels) * 1.8))))
    spacing_x = 280
    spacing_y = 170
    margin_x = 150
    margin_y = 110
    palette = ["#2563eb", "#059669", "#dc2626", "#7c3aed", "#ea580c", "#0891b2", "#be123c"]
    nodes = []
    for index, label in enumerate(labels):
        node_degree = degree.get(label, 1)
        row = index // column_count
        col = index % column_count
        row_offset = 70 if row % 2 else 0
        nodes.append(
            {
                "id": f"n{index}",
                "label": label,
                "degree": node_degree,
                "x": round(margin_x + col * spacing_x + row_offset, 2),
                "y": round(margin_y + row * spacing_y, 2),
                "radius": min(36, 20 + node_degree * 3),
                "color": palette[index % len(palette)],
            }
        )
    return nodes


def _graph_edges(triples: Sequence[dict], nodes: Sequence[dict]) -> list[dict]:
    
    node_ids = {str(node["label"]): str(node["id"]) for node in nodes}
    edges = []
    for index, triple in enumerate(triples):
        subject = str(triple.get("subject") or "").strip()
        object_ = str(triple.get("object") or "").strip()
        predicate = str(triple.get("predicate") or "").strip()
        if subject in node_ids and object_ in node_ids and predicate:
            edges.append(
                {
                    "id": f"e{index}",
                    "source": node_ids[subject],
                    "target": node_ids[object_],
                    "label": predicate,
                    "sourceName": str(triple.get("source_name") or ""),
                    "evidence": str(triple.get("evidence") or ""),
                }
            )
    return edges
