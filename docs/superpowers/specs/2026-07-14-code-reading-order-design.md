# 代码阅读顺序设计

## 目标

提供一份位于项目根目录的阅读导航。学习时可按依赖和运行流程从上到下阅读，
无需重命名任何 Python 模块。

## 范围

- 在项目根目录新增 `00_CODE_READING_ORDER.md`。
- 按推荐顺序链接核心模块。
- 每个模块只说明它负责什么，以及下一步该看什么。

## 约束

- 不重命名、移动或修改任何运行中的 Python 模块。
- 不改变导入、程序行为、测试或依赖。
- 主阅读路径不包含生成目录、日志、本地数据或测试实现细节。

## 阅读顺序

按表格从上到下阅读。路径均为相对链接，在 IDE 的 Markdown 预览中可用
`Ctrl+单击` 直接打开。

| 顺序 | 文件 | 作用 |
| --- | --- | --- |
| 1 | [app.py](../../../app.py) | Streamlit 入口：页面布局、会话状态、资料导入和聊天界面。 |
| 2 | [config.py](../../../ragbase/config.py) | 集中配置模型、存储、检索数量和路径。 |
| 3 | [model.py](../../../ragbase/model.py) | 创建聊天模型、向量模型和重排模型。 |
| 4 | [uploader.py](../../../ragbase/uploader.py) | 将用户上传的文件保存到临时导入目录。 |
| 5 | [ingestor.py](../../../ragbase/ingestor.py) | 解析文件、网页和代码目录，并发起切块和 Qdrant 入库。 |
| 6 | [chunking.py](../../../ragbase/chunking.py) | 判断资料结构，按 embedding token 将文本切为 chunk。 |
| 7 | [source_registry.py](../../../ragbase/source_registry.py) | 给每份导入资料分配稳定 ID，并给页面补来源元数据。 |
| 8 | [source_resolver.py](../../../ragbase/source_resolver.py) | 将文件简称、序号、别名和指代词匹配到具体来源 ID。 |
| 9 | [planner_schema.py](../../../ragbase/planner_schema.py) | 定义操作计划和引用信息的数据结构。 |
| 10 | [agent_planner.py](../../../ragbase/agent_planner.py) | 用模型判断该闲聊、列文件、读资料，还是检索问答。 |
| 11 | [source_tools.py](../../../ragbase/source_tools.py) | 构建资料档案，并确定性地列文件或输出全文。 |
| 12 | [hybrid_retriever.py](../../../ragbase/hybrid_retriever.py) | 合并向量检索和 BM25，执行 RRF、重排和父块扩展。 |
| 13 | [retriever.py](../../../ragbase/retriever.py) | 创建检索器，配置来源过滤和重排降级。 |
| 14 | [chain.py](../../../ragbase/chain.py) | 组装基于证据回答的 Prompt，并流式生成 RAG 回答。 |
| 15 | [agent_responses.py](../../../ragbase/agent_responses.py) | 生成普通聊天回复和资料整体概览。 |
| 16 | [orchestrator.py](../../../ragbase/orchestrator.py) | 协调来源解析、规划、读取、检索和回答。 |
| 17 | [knowledge_graph.py](../../../ragbase/knowledge_graph.py) | 从资料中抽取实体关系，供知识图谱页面使用。 |
| 18 | [runtime.py](../../../ragbase/runtime.py) | 释放 Qdrant 资源、重置索引和归档会话。 |
| 19 | [conversation_store.py](../../../ragbase/conversation_store.py) | 读写本地保存的历史对话。 |
| 20 | [session_history.py](../../../ragbase/session_history.py) | 管理当前会话消息和轮数上限。 |

## 验证

- 所有链接路径都存在。
- 导航文件按名称排序时位于项目根目录 Markdown 文件的最上方。
- 不修改应用代码或测试代码。
