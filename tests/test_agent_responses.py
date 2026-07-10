from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessage

from ragbase.agent_responses import generate_chat_answer, generate_overview_answer, select_source_profiles


class FakeAnswerModel:
    def __init__(self, answer="ok"):
        self.answer = answer
        self.messages = None

    async def ainvoke(self, messages):
        self.messages = messages
        return AIMessage(content=self.answer)


def test_select_source_profiles_uses_validated_exact_names():
    profiles = [
        {"name": "course-notes.pdf", "sections": [{"content": "course content"}]},
        {"name": "lab-guide.pdf", "sections": [{"content": "lab content"}]},
    ]

    selected = select_source_profiles(profiles, ["lab-guide.pdf"])

    assert [profile["name"] for profile in selected] == ["lab-guide.pdf"]


def test_overview_prompt_excludes_unselected_sources():
    model = FakeAnswerModel("selected summary")
    profiles = [
        {"name": "course-notes.pdf", "sections": [{"content": "COURSE_ONLY"}]},
        {"name": "lab-guide.pdf", "sections": [{"content": "LAB_ONLY"}]},
    ]

    answer = asyncio.run(
        generate_overview_answer(
            model=model,
            question="Summarize this source.",
            source_profiles=select_source_profiles(profiles, ["lab-guide.pdf"]),
        )
    )

    prompt = "\n".join(message.content for message in model.messages)
    assert answer == "selected summary"
    assert "LAB_ONLY" in prompt
    assert "COURSE_ONLY" not in prompt


def test_chat_answer_uses_recent_dialogue_without_source_context():
    model = FakeAnswerModel("hello")

    answer = asyncio.run(
        generate_chat_answer(
            model=model,
            question="Hello",
            recent_messages=[{"role": "user", "content": "Earlier message"}],
        )
    )

    prompt = "\n".join(message.content for message in model.messages)
    assert answer == "hello"
    assert "Earlier message" in prompt
    assert "Hello" in prompt
