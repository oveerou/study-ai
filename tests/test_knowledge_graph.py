

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from langchain_core.messages import AIMessage

from ragbase.knowledge_graph import (
    extract_knowledge_graph,
    graph_to_dot,
    graph_to_interactive_html,
    load_knowledge_graph,
    save_knowledge_graph,
)


class FakeGraphModel:
    def __init__(self, payload):
        self.payload = payload
        self.messages = None

    async def ainvoke(self, messages):
        self.messages = messages
        return AIMessage(content=json.dumps(self.payload, ensure_ascii=False))


class FailingGraphModel:
    async def ainvoke(self, messages):
        raise RuntimeError("connection failed")


def test_extract_knowledge_graph_validates_triples_and_sources():
    
    model = FakeGraphModel(
        {
            "triples": [
                {
                    "subject": "Canny",
                    "predicate": "用于",
                    "object": "边缘检测",
                    "source_name": "course.pdf",
                    "evidence": "Canny 算法用于检测边缘。",
                },
                {
                    "subject": "Invented",
                    "predicate": "来自",
                    "object": "Unknown",
                    "source_name": "missing.pdf",
                    "evidence": "",
                },
            ]
        }
    )
    profiles = [
        {
            "name": "course.pdf",
            "sections": [{"content": "Canny 算法用于检测边缘。"}],
        }
    ]

    graph = asyncio.run(extract_knowledge_graph(model, profiles))

    assert graph["source_names"] == ["course.pdf"]
    assert len(graph["triples"]) == 1
    assert graph["triples"][0]["object"] == "边缘检测"
    prompt = "\n".join(message.content for message in model.messages)
    assert "Canny 算法用于检测边缘" in prompt


def test_extract_knowledge_graph_falls_back_when_model_connection_fails():
    
    profiles = [
        {
            "name": "course.pdf",
            "sections": [
                {
                    "content": (
                        "1. 图像质量评价指标包括 PSNR 和 SSIM。\n"
                        "2. Canny 算法用于边缘检测。\n"
                        "3. 直方图均衡化的作用是增强图像对比度。"
                    )
                }
            ],
        }
    ]

    graph = asyncio.run(extract_knowledge_graph(FailingGraphModel(), profiles))

    assert graph["extraction_mode"] == "fallback"
    assert graph["error"]
    assert len(graph["triples"]) >= 2
    assert {triple["source_name"] for triple in graph["triples"]} == {"course.pdf"}


def test_knowledge_graph_round_trips_to_local_json(tmp_path: Path):
    
    graph = {
        "source_names": ["course.pdf"],
        "triples": [
            {
                "subject": "PSNR",
                "predicate": "评价",
                "object": "图像质量",
                "source_name": "course.pdf",
                "evidence": "",
            }
        ],
    }

    path = save_knowledge_graph(tmp_path / "graphs", "session-1", graph)

    assert path.name == "session-1.json"
    assert load_knowledge_graph(path)["triples"] == graph["triples"]


def test_graph_to_dot_escapes_labels():
    
    graph = {
        "triples": [
            {
                "subject": 'model "A"',
                "predicate": "uses",
                "object": "data",
                "source_name": "course.pdf",
            }
        ]
    }

    dot = graph_to_dot(graph)

    assert 'model \\"A\\"' in dot
    assert 'label="uses"' in dot


def test_graph_to_interactive_html_embeds_graph_browser_controls():
    
    graph = {
        "triples": [
            {"subject": "A", "predicate": "rel", "object": "B", "source_name": "one.pdf"},
            {"subject": "X", "predicate": "rel", "object": "Y", "source_name": "two.pdf"},
        ]
    }

    html = graph_to_interactive_html(graph)

    assert "vis-network" in html
    assert "放大" in html
    assert "搜索实体或关系" in html
    assert "邻居聚焦" in html
    assert "详情" in html
    assert "适应视图" in html
    assert "物理布局" in html
    assert '"label": "X"' in html
    assert '"label": "A"' in html
    assert "片段" not in html


def test_graph_to_interactive_html_can_show_all_components_and_fullscreen():
    
    graph = {
        "triples": [
            {"subject": "A", "predicate": "rel", "object": "B", "source_name": "one.pdf"},
            {"subject": "X", "predicate": "rel", "object": "Y", "source_name": "two.pdf"},
        ]
    }

    html = graph_to_interactive_html(graph)

    assert "全屏" in html
    assert "requestFullscreen" in html
    assert '"label": "A"' in html
    assert '"label": "X"' in html
