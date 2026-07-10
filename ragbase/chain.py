import re
from operator import itemgetter
from typing import List

from langchain.schema.runnable import RunnablePassthrough
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tracers.stdout import ConsoleCallbackHandler
from langchain_core.vectorstores import VectorStoreRetriever

from ragbase.config import Config
from ragbase.session_history import get_session_history

SYSTEM_PROMPT = """
你是学习助手。请优先依据下面检索到的资料回答用户问题。
如果资料中找不到答案，明确说明“资料中没有找到”，不要编造。
回答使用用户提问的语言，结构清晰；涉及步骤、概念、对比时可以使用条目。
不要强行限制成三句话，资料能支持时要给出足够完整的解释。
资料片段按相关性排序，并用 --- 分隔。

资料:
{context}

请使用 Markdown 格式回答。
"""


def remove_links(text: str) -> str:
    url_pattern = r"https?://\S+|www\.\S+"
    return re.sub(url_pattern, "", text)


def format_documents(documents: List[Document]) -> str:
    texts = []
    for doc in documents:
        texts.append(doc.page_content)
        texts.append("---")

    return remove_links("\n".join(texts))


def create_chain(
    llm: BaseLanguageModel,
    retriever: VectorStoreRetriever,
    use_history: bool = True,
) -> Runnable:
    prompt_messages = [("system", SYSTEM_PROMPT)]
    if use_history:
        prompt_messages.append(MessagesPlaceholder("chat_history"))
    prompt_messages.append(("human", "{question}"))
    prompt = ChatPromptTemplate.from_messages(prompt_messages)

    chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question")
            | retriever.with_config({"run_name": "context_retriever"})
            | format_documents
        )
        | prompt
        | llm
    )

    if use_history:
        chain = RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history",
        )
    return chain.with_config({"run_name": "chain_answer"})


async def ask_question(chain: Runnable, question: str, session_id: str):
    async for event in chain.astream_events(
        {"question": question},
        config={
            "callbacks": [ConsoleCallbackHandler()] if Config.DEBUG else [],
            "configurable": {"session_id": session_id},
        },
        version="v2",
        include_names=["context_retriever", "chain_answer"],
    ):
        event_type = event["event"]
        if event_type == "on_retriever_end":
            yield event["data"]["output"]
        if event_type == "on_chain_stream":
            yield event["data"]["chunk"].content
