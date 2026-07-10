from __future__ import annotations

import asyncio
import json

from langchain_core.messages import AIMessage

from ragbase.agent_planner import AgentPlan, plan_question


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


def test_planner_accepts_only_exact_catalog_sources():
    model = FakePlannerModel(
        {
            "intent": "full_text",
            "scope": "selected",
            "source_names": ["复习大纲.pdf", "不存在.pdf"],
            "standalone_question": "输出复习大纲的完整正文",
            "reason": "用户选中了复习大纲",
        }
    )

    plan = asyncio.run(
        plan_question(
            model,
            question="把那份复习资料完整给我",
            recent_messages=[],
            source_names=["课程介绍.pdf", "复习大纲.pdf"],
            active_source_names=[],
            last_content_intent=None,
        )
    )

    assert plan.intent == "full_text"
    assert plan.source_names == ("复习大纲.pdf",)
    assert plan.standalone_question == "输出复习大纲的完整正文"


def test_selected_scope_inherits_active_sources_when_model_omits_names():
    model = FakePlannerModel(
        {
            "intent": "grounded_qa",
            "scope": "selected",
            "source_names": [],
            "standalone_question": "解释该文档中的高斯滤波",
            "reason": "承接当前文档",
        }
    )

    plan = asyncio.run(
        plan_question(
            model,
            question="那高斯滤波呢",
            recent_messages=[],
            source_names=["课程介绍.pdf", "复习大纲.pdf"],
            active_source_names=["复习大纲.pdf"],
            last_content_intent="grounded_qa",
        )
    )

    assert plan.source_names == ("复习大纲.pdf",)


def test_planner_prompt_caps_large_assistant_outputs():
    model = FakePlannerModel(
        {
            "intent": "chat",
            "scope": "none",
            "source_names": [],
            "standalone_question": "你好",
            "reason": "普通问候",
        }
    )
    huge_output = "开头" + ("冗长正文" * 10000) + "不应出现在规划提示末尾"

    asyncio.run(
        plan_question(
            model,
            question="继续",
            recent_messages=[{"role": "assistant", "content": huge_output}],
            source_names=["资料.pdf"],
            active_source_names=[],
            last_content_intent=None,
        )
    )

    prompt = "\n".join(str(message.content) for message in model.messages)
    assert len(prompt) < 10000
    assert "不应出现在规划提示末尾" not in prompt


def test_planner_failure_uses_active_source_for_grounded_fallback():
    model = FakePlannerModel(error=RuntimeError("model unavailable"))

    plan = asyncio.run(
        plan_question(
            model,
            question="接着解释",
            recent_messages=[],
            source_names=["课程介绍.pdf", "复习大纲.pdf"],
            active_source_names=["复习大纲.pdf"],
            last_content_intent="grounded_qa",
        )
    )

    assert plan == AgentPlan(
        intent="grounded_qa",
        scope="selected",
        source_names=("复习大纲.pdf",),
        standalone_question="接着解释",
        reason="planner fallback on active sources",
    )
