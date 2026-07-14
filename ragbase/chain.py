

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
from langchain_core.retrievers import BaseRetriever

from ragbase.config import Config
from ragbase.session_history import get_session_history

SYSTEM_PROMPT = """
你是学习助手。Use only the supplied evidence to answer the user.
If evidence is insufficient，明确回答“当前资料依据不足”，不要补充未经证据支持的断言，也不要编造。
回答使用用户提问的语言，结构清晰；涉及步骤、概念、对比时可以使用条目。
不要强行限制成三句话，资料能支持时要给出足够完整的解释。
资料片段按相关性排序，并用来源、页码和 Chunk 标签标识。只输出答案和必要引用，不展示内部推理过程。

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
        source_name = str(
            doc.metadata.get("source_name")
            or doc.metadata.get("source")
            or "unknown"
        )
        page_start = doc.metadata.get("page_start")
        page_end = doc.metadata.get("page_end")
        if page_start is not None and page_end is not None and page_start != page_end:
            page_label = f"Pages: {page_start}-{page_end}"
        else:
            page = doc.metadata.get("page")
            if page is None:
                page = page_start
            page_label = f"Page: {page if page is not None else 'unknown'}"
        chunk_id = str(doc.metadata.get("chunk_id") or "unknown")
        texts.append(
            f"[Source: {source_name} | {page_label} | Chunk: {chunk_id}]"
        )
        texts.append(doc.page_content)
        texts.append("---")

    return remove_links("\n".join(texts))


def create_chain(
    llm: BaseLanguageModel,
    retriever: BaseRetriever,
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
