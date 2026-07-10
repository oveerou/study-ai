from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

from ragbase.chain import create_chain


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
