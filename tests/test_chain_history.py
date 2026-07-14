

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

from ragbase.chain import SYSTEM_PROMPT, create_chain, format_documents


def test_planned_grounded_chain_can_run_without_duplicating_chat_history():
    
    retriever = RunnableLambda(lambda _: [Document(page_content="grounded context")])
    llm = RunnableLambda(lambda _: AIMessage(content="answer"))

    chain = create_chain(llm, retriever, use_history=False)

    assert not isinstance(chain, RunnableWithMessageHistory)
    response = chain.invoke({"question": "standalone question"})
    assert response.content == "answer"


def test_default_chain_keeps_legacy_history_wrapper():
    
    retriever = RunnableLambda(lambda _: [Document(page_content="grounded context")])
    llm = RunnableLambda(lambda _: AIMessage(content="answer"))

    chain = create_chain(llm, retriever)

    assert isinstance(chain.bound, RunnableWithMessageHistory)


def test_grounded_context_labels_source_page_and_chunk():
    context = format_documents(
        [
            Document(
                page_content="evidence",
                metadata={
                    "source_name": "notes.pdf",
                    "page": 7,
                    "chunk_id": "chunk-42",
                },
            )
        ]
    )

    assert "notes.pdf" in context
    assert "7" in context
    assert "chunk-42" in context
    assert "evidence" in context


def test_grounded_context_displays_parent_page_range():
    context = format_documents(
        [
            Document(
                page_content="[Page 4]\nfirst\n\n[Page 5]\nsecond\n\n[Page 6]\nthird",
                metadata={
                    "source_name": "slides.pdf",
                    "page_start": 4,
                    "page_end": 6,
                    "chunk_id": "chunk-parent",
                    "context_role": "parent",
                },
            )
        ]
    )

    assert "Pages: 4-6" in context


def test_grounded_prompt_forbids_unsupported_claims_and_private_reasoning():
    assert "evidence" in SYSTEM_PROMPT.lower()
    assert "insufficient" in SYSTEM_PROMPT.lower()
    assert "chain-of-thought" not in SYSTEM_PROMPT.lower()
