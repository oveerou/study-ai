





from __future__ import annotations

import asyncio
import gc
import shutil
import uuid
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

load_dotenv()
load_dotenv(Path(__file__).resolve().parent / ".env", override=False)

from ragbase.agent_responses import select_source_profiles
from ragbase.chain import create_chain
from ragbase.config import Config
from ragbase.conversation_store import (
    delete_conversation,
    list_conversations,
    load_conversation,
    save_conversation,
)
from ragbase.ingestor import (
    Ingestor,
    load_code_dir_documents,
    load_mixed_documents,
    load_path_documents,
    load_url_documents,
)
from ragbase.knowledge_graph import (
    extract_knowledge_graph,
    graph_to_interactive_html,
    save_knowledge_graph,
)
from ragbase.model import create_llm
from ragbase.orchestrator import OrchestratorRuntime, execute_question
from ragbase.retriever import create_retriever
from ragbase.runtime import close_vector_store, reset_index_storage
from ragbase.session_history import drop_session_history
from ragbase.source_registry import attach_source_records, build_source_records
from ragbase.source_tools import build_source_profiles
from ragbase.uploader import upload_files


SOURCE_INPUT_KEY = "source_input_revision"


st.set_page_config(
    page_title="学习助手",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.4rem; padding-bottom: 7rem; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        letter-spacing: 0;
    }
    .source-pill {
        display: inline-block;
        border: 1px solid #d6dbe3;
        border-radius: 6px;
        padding: .25rem .45rem;
        margin: .15rem .2rem .15rem 0;
        font-size: .84rem;
        background: #f8fafc;
    }
    .metric-note {
        color: #64748b;
        font-size: .88rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def clear_runtime_sources() -> None:
    
    reset_index_storage(Config.Path.DATABASE_DIR)
    shutil.rmtree(Config.Path.DOCUMENTS_DIR, ignore_errors=True)
    Config.Path.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)


def release_runtime_index() -> None:
    
    st.session_state.chain = None
    vector_store = st.session_state.get("vector_store")
    st.session_state.vector_store = None
    close_vector_store(vector_store)
    gc.collect()


def reset_source_inputs() -> None:
    
    st.session_state[SOURCE_INPUT_KEY] = st.session_state.get(SOURCE_INPUT_KEY, 0) + 1


def source_widget_key(name: str) -> str:
    
    return f"{name}_{st.session_state.get(SOURCE_INPUT_KEY, 0)}"


def init_state() -> None:
    
    if SOURCE_INPUT_KEY not in st.session_state:
        st.session_state[SOURCE_INPUT_KEY] = 0
    if "runtime_sources_initialized" not in st.session_state:
        release_runtime_index()
        clear_runtime_sources()
        st.session_state.runtime_sources_initialized = True
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "你好，我会优先基于你导入的资料回答。先在左侧导入文件、网页或代码目录，然后直接提问。",
            }
        ]
    if "source_names" not in st.session_state:
        st.session_state.source_names = []
    if "source_records" not in st.session_state:
        st.session_state.source_records = []
    if "chunk_documents" not in st.session_state:
        st.session_state.chunk_documents = []
    if "source_profiles" not in st.session_state:
        st.session_state.source_profiles = []
    if "last_documents" not in st.session_state:
        st.session_state.last_documents = []
    if "chain" not in st.session_state:
        st.session_state.chain = None
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "llm" not in st.session_state:
        st.session_state.llm = None
    if "active_source_names" not in st.session_state:
        st.session_state.active_source_names = []
    if "active_source_ids" not in st.session_state:
        st.session_state.active_source_ids = []
    if "last_content_intent" not in st.session_state:
        st.session_state.last_content_intent = None
    if "viewing_history" not in st.session_state:
        st.session_state.viewing_history = False
    if "history_source_names" not in st.session_state:
        st.session_state.history_source_names = []
    if "knowledge_graph" not in st.session_state:
        st.session_state.knowledge_graph = None
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())


def persist_current_conversation() -> Path | None:
    
    if st.session_state.viewing_history:
        return None
    has_user_message = any(message.get("role") == "user" for message in st.session_state.messages)
    if not has_user_message and not st.session_state.source_names:
        return None
    return save_conversation(
        Config.Path.HISTORY_DIR,
        st.session_state.session_id,
        st.session_state.messages,
        st.session_state.source_names,
    )


def record_message(role: str, content: str) -> None:
    
    st.session_state.messages.append({"role": role, "content": content})
    persist_current_conversation()


def reset_chat() -> None:
    
    old_session_id = st.session_state.session_id
    persist_current_conversation()
    release_runtime_index()
    drop_session_history(old_session_id)
    clear_runtime_sources()
    reset_source_inputs()
    st.session_state.source_names = []
    st.session_state.source_records = []
    st.session_state.chunk_documents = []
    st.session_state.source_profiles = []
    st.session_state.last_documents = []
    st.session_state.active_source_names = []
    st.session_state.active_source_ids = []
    st.session_state.last_content_intent = None
    st.session_state.viewing_history = False
    st.session_state.history_source_names = []
    st.session_state.knowledge_graph = None
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "已开启新的空白对话。上一段对话已保存，当前资料索引已清空。",
        }
    ]


def clear_sources() -> None:
    
    persist_current_conversation()
    release_runtime_index()
    clear_runtime_sources()
    reset_source_inputs()
    st.session_state.source_names = []
    st.session_state.source_records = []
    st.session_state.chunk_documents = []
    st.session_state.source_profiles = []
    st.session_state.last_documents = []
    st.session_state.active_source_names = []
    st.session_state.active_source_ids = []
    st.session_state.last_content_intent = None
    st.session_state.viewing_history = False
    st.session_state.history_source_names = []
    st.session_state.knowledge_graph = None
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "已清空当前资料。可以重新导入文件、网页或代码目录。",
        }
    ]


def build_chain_from_documents(documents):
    
    ingestor = Ingestor()
    vector_store = ingestor.ingest_documents(documents)
    chunk_documents = list(ingestor.chunk_documents)
    try:
        llm = create_llm()
        retriever = create_retriever(
            llm,
            vector_store=vector_store,
            chunk_documents=chunk_documents,
        )
        chain = create_chain(llm, retriever)
    except Exception:
        close_vector_store(vector_store)
        raise
    return chain, vector_store, llm, chunk_documents


def index_documents(documents, success_prefix: str) -> bool:
    
    documents = [doc for doc in documents if (doc.page_content or "").strip()]
    if not documents:
        st.warning("没有解析到可索引内容。")
        return False

    source_records = build_source_records(documents, st.session_state.session_id)
    documents = attach_source_records(documents, source_records)
    names = [record.source_name for record in source_records]
    try:
        with st.spinner("正在索引资料，请稍等..."):
            release_runtime_index()
            reset_index_storage(Config.Path.DATABASE_DIR)
            chain, vector_store, llm, chunk_documents = build_chain_from_documents(documents)
            st.session_state.vector_store = vector_store
            st.session_state.chain = chain
            st.session_state.llm = llm
    except Exception as exc:
        st.error(f"索引失败：{exc}")
        return False

    st.session_state.source_names = names
    st.session_state.source_records = source_records
    st.session_state.chunk_documents = chunk_documents
    st.session_state.source_profiles = build_source_profiles(documents)
    st.session_state.active_source_ids = [source_records[0].source_id] if len(source_records) == 1 else []
    st.session_state.active_source_names = names.copy() if len(names) == 1 else []
    st.session_state.last_content_intent = None
    st.session_state.viewing_history = False
    st.session_state.history_source_names = []
    st.session_state.knowledge_graph = None
    reset_source_inputs()
    record_message("assistant", f"已导入 {len(names)} 个来源：{', '.join(names)}")
    st.success(f"{success_prefix}完成，已索引 {len(names)} 个来源。")
    return True


def load_uploaded_documents(uploaded_files):
    
    shutil.rmtree(Config.Path.DOCUMENTS_DIR, ignore_errors=True)
    file_paths = upload_files(uploaded_files, remove_old_files=False)
    documents = []
    for file_path in file_paths:
        documents.extend(load_path_documents(file_path))
    return documents


def render_source_controls() -> None:
    
    st.subheader("资料导入")
    tab_file, tab_url_github, tab_code, tab_mixed = st.tabs(
        ["文件", "URL/GitHub", "代码目录", "混合导入"]
    )

    with tab_file:
        uploaded_files = st.file_uploader(
            "上传 PDF / DOCX / MD / TXT",
            type=["pdf", "docx", "md", "txt"],
            accept_multiple_files=True,
            key=source_widget_key("uploaded_files"),
            help="支持一次导入多个文件。重新索引会刷新当前资料库。",
        )
        if st.button("索引上传文件", use_container_width=True):
            if not uploaded_files:
                st.warning("未选择文件。")
            else:
                try:
                    if index_documents(load_uploaded_documents(uploaded_files), "文件导入"):
                        st.rerun()
                except Exception as exc:
                    st.error(f"文件导入失败：{exc}")

    with tab_url_github:
        url_input = st.text_input(
            "网页 URL 或 GitHub 仓库",
            placeholder="https://example.com/article",
            key=source_widget_key("url_input"),
        )
        if st.button("索引导入 URL/GitHub", use_container_width=True):
            if not url_input.strip():
                st.warning("请输入 URL 或 GitHub 仓库。")
            else:
                try:
                    if index_documents(load_url_documents(url_input), "URL/GitHub 导入"):
                        st.rerun()
                except Exception as exc:
                    st.error(f"URL/GitHub 导入失败：{exc}")

    with tab_code:
        code_dir = st.text_input(
            "本地代码目录",
            placeholder=r"C:\path\to\your-project",
            key=source_widget_key("code_dir"),
        )
        if st.button("索引代码目录", use_container_width=True):
            if not code_dir.strip():
                st.warning("请输入代码目录。")
            else:
                try:
                    if index_documents(load_code_dir_documents(code_dir), "代码目录导入"):
                        st.rerun()
                except Exception as exc:
                    st.error(f"代码目录导入失败：{exc}")

    with tab_mixed:
        mixed_items = st.text_area(
            "混合来源",
            placeholder="每行一个来源：文件路径、目录路径、网页 URL 或 GitHub 地址",
            height=120,
            key=source_widget_key("mixed_items"),
        )
        if st.button("索引混合来源", use_container_width=True):
            items = [line.strip() for line in mixed_items.splitlines() if line.strip()]
            if not items:
                st.warning("请输入至少一个来源。")
            else:
                try:
                    if index_documents(load_mixed_documents(items), "混合导入"):
                        st.rerun()
                except Exception as exc:
                    st.error(f"混合导入失败：{exc}")


def open_history_conversation(summary: dict) -> None:
    
    persist_current_conversation()
    release_runtime_index()
    clear_runtime_sources()
    payload = load_conversation(Path(summary["path"]))
    reset_source_inputs()
    st.session_state.source_names = []
    st.session_state.source_records = []
    st.session_state.chunk_documents = []
    st.session_state.source_profiles = []
    st.session_state.active_source_names = []
    st.session_state.active_source_ids = []
    st.session_state.last_content_intent = None
    st.session_state.last_documents = []
    st.session_state.knowledge_graph = None
    st.session_state.history_source_names = list(payload.get("source_names") or [])
    st.session_state.messages = list(payload.get("messages") or [])
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.viewing_history = True


def render_history_controls() -> None:
    
    st.subheader("历史对话")
    conversations = list_conversations(Config.Path.HISTORY_DIR)
    if not conversations:
        st.caption("暂无本地历史。")
        return

    by_id = {item["session_id"]: item for item in conversations}
    selected_id = st.selectbox(
        "本地历史",
        options=list(by_id),
        format_func=lambda session_id: by_id[session_id]["title"],
        label_visibility="collapsed",
    )
    open_col, delete_col = st.columns(2)
    if open_col.button("打开", use_container_width=True, key="open_history"):
        open_history_conversation(by_id[selected_id])
        st.rerun()
    if delete_col.button("删除", use_container_width=True, key="delete_history"):
        delete_conversation(Config.Path.HISTORY_DIR, selected_id)
        st.rerun()


def render_knowledge_graph_controls() -> None:
    
    st.subheader("知识图谱")
    if not st.session_state.source_names:
        st.caption("导入资料后可生成。")
        return

    default_sources = st.session_state.active_source_names or st.session_state.source_names[:1]
    graph_sources = st.multiselect(
        "图谱来源",
        options=st.session_state.source_names,
        default=[name for name in default_sources if name in st.session_state.source_names],
        key=source_widget_key("graph_sources"),
    )
    if st.button("生成知识图谱", use_container_width=True):
        if not graph_sources:
            st.warning("请选择至少一个来源。")
            return
        profiles = select_source_profiles(st.session_state.source_profiles, graph_sources)
        model = st.session_state.llm or create_llm()
        try:
            with st.spinner("正在抽取实体关系..."):
                graph = asyncio.run(extract_knowledge_graph(model, profiles))
        except Exception as exc:
            st.error(f"知识图谱生成失败：{exc}")
            return
        if not graph.get("triples"):
            st.warning("当前资料没有抽取到明确的实体关系。")
            return
        if graph.get("extraction_mode") == "fallback":
            st.warning("模型连接失败，已使用本地规则生成基础知识图谱。")
        st.session_state.llm = model
        st.session_state.knowledge_graph = graph
        save_knowledge_graph(Config.Path.KNOWLEDGE_GRAPH_DIR, st.session_state.session_id, graph)
        st.rerun()


def render_sidebar() -> None:
    
    with st.sidebar:
        st.title("学习助手")
        st.caption("本地资料问答 · 多源导入 · 来源可追踪")
        st.divider()
        render_source_controls()

        st.divider()
        st.subheader("当前资料")
        if st.session_state.source_names:
            st.write(f"来源数：**{len(st.session_state.source_names)}**")
            st.markdown(
                " ".join(f'<span class="source-pill">{name}</span>' for name in st.session_state.source_names[:20]),
                unsafe_allow_html=True,
            )
        else:
            st.write("来源数：**0**")
            st.caption("支持文件、URL/GitHub、代码目录和混合导入。")

        st.divider()
        render_knowledge_graph_controls()

        st.divider()
        render_history_controls()

        st.divider()
        if st.button("新建空白对话", use_container_width=True):
            reset_chat()
            st.rerun()
        if st.button("清空当前资料", use_container_width=True):
            clear_sources()
            st.rerun()


def render_empty_state() -> None:
    
    st.header("知识库问答")
    st.info("左侧导入资料后，这里会进入资料问答模式。")
    st.markdown(
        """
        可以这样问：

        - 这些资料整体主要讲什么？
        - 里面有哪些重点概念？
        - 按考试复习角度帮我整理。
        - 哪些资料提到了某个主题？
        - 输出某一份资料的正文内容。
        """
    )
    st.divider()
    render_message_history()


def render_history_view() -> None:
    
    st.header("历史对话")
    if st.session_state.history_source_names:
        st.caption("当时使用的资料：" + "、".join(st.session_state.history_source_names))
    st.info("历史记录为只读。新建对话或重新导入资料后可以继续问答。")
    render_message_history()


def render_knowledge_graph_view() -> None:
    
    graph = st.session_state.knowledge_graph
    if not graph:
        return
    with st.expander("知识图谱", expanded=True):
        triples = graph.get("triples") or []
        entity_count = len(
            {
                value
                for triple in triples
                for value in (triple.get("subject"), triple.get("object"))
                if value
            }
        )
        st.markdown(
            f'<div class="metric-note">关系：{len(triples)} 条 · 实体：{entity_count} 个 · 支持搜索、聚焦、拖拽、缩放、全屏</div>',
            unsafe_allow_html=True,
        )

        if triples:
            components.html(
                graph_to_interactive_html(graph),
                height=760,
                scrolling=False,
            )
        else:
            st.info("当前知识图谱没有可展示的实体关系。")

        rows = [
            {
                "实体": triple["subject"],
                "关系": triple["predicate"],
                "目标": triple["object"],
                "来源": triple["source_name"],
            }
            for triple in triples
        ]
        st.markdown("**关系明细**")
        st.dataframe(rows, use_container_width=True, hide_index=True)


async def ask_chain(question: str, chain):
    
    model = st.session_state.llm or create_llm()
    st.session_state.llm = model
    recent_messages = st.session_state.messages[:-1]
    runtime = OrchestratorRuntime(
        model=model,
        source_records=st.session_state.source_records,
        source_profiles=st.session_state.source_profiles,
        active_source_ids=st.session_state.active_source_ids,
        chunk_documents=st.session_state.chunk_documents,
        vector_store=st.session_state.vector_store,
        recent_messages=recent_messages,
        session_id=st.session_state.session_id,
    )
    
    with st.status("执行过程", expanded=False) as status:
        status.write("正在理解问题并处理资料...")
        result = await execute_question(question, runtime)
        status.update(label="执行过程：回答完成", state="complete")
    documents = list(result.documents)
    st.session_state.active_source_ids = list(result.active_source_ids)
    names_by_id = {
        record.source_id: record.source_name
        for record in st.session_state.source_records
    }
    st.session_state.active_source_names = [
        names_by_id[source_id]
        for source_id in result.active_source_ids
        if source_id in names_by_id
    ]

    with st.chat_message("assistant"):
        st.markdown(result.answer)

        if documents:
            with st.expander("引用来源", expanded=False):
                for i, doc in enumerate(documents, 1):
                    source = (
                        doc.metadata.get("source_name")
                        or doc.metadata.get("source")
                        or doc.metadata.get("file_path")
                        or "已导入资料"
                    )
                    page = doc.metadata.get("page")
                    if page is None:
                        page = doc.metadata.get("page_start")
                    chunk_id = doc.metadata.get("chunk_id")
                    label = f"[{i}] {Path(str(source)).name}"
                    if page is not None:
                        label += f" · 第 {page} 页"
                    if chunk_id:
                        label += f" · {chunk_id}"
                    st.markdown(f"**{label}**")
                    st.write(doc.page_content)

    st.session_state.last_documents = documents
    record_message("assistant", result.answer)


def render_message_history() -> None:
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def scroll_chat_to_bottom() -> None:
    
    components.html(
        """
        <script>
        const scrollToBottom = () => {
          const doc = window.parent.document;
          const root = doc.scrollingElement || doc.documentElement || doc.body;
          if (!root) return;
          root.scrollTop = root.scrollHeight;
        };
        window.requestAnimationFrame(scrollToBottom);
        window.setTimeout(scrollToBottom, 80);
        window.setTimeout(scrollToBottom, 240);
        </script>
        """,
        height=0,
    )


def render_chat(chain) -> None:
    
    st.header("知识库问答")

    col_status, col_sources = st.columns([1, 2])
    with col_status:
        st.success("资料库已就绪")
        st.markdown(
            f'<div class="metric-note">当前来源：{len(st.session_state.source_names)} 个</div>',
            unsafe_allow_html=True,
        )
    with col_sources:
        if st.session_state.source_names:
            st.markdown("**已导入资料**")
            st.markdown(
                " ".join(f'<span class="source-pill">{name}</span>' for name in st.session_state.source_names[:20]),
                unsafe_allow_html=True,
            )

    st.divider()
    render_knowledge_graph_view()
    render_message_history()
    scroll_chat_to_bottom()

    question = st.chat_input("输入你的问题...")
    if not question:
        return

    record_message("user", question)
    with st.chat_message("user"):
        st.markdown(question)
    try:
        asyncio.run(ask_chain(question, chain))
    except Exception as exc:
        error_message = f"回答生成失败：{exc}"
        with st.chat_message("assistant"):
            st.error(error_message)
        record_message("assistant", error_message)
    scroll_chat_to_bottom()


def main() -> None:
    
    init_state()
    render_sidebar()

    if st.session_state.viewing_history:
        render_history_view()
        return

    if st.session_state.chain is None:
        render_empty_state()
        return

    render_chat(st.session_state.chain)


if __name__ == "__main__":
    main()
