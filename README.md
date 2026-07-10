# Study AI

> 本地多源知识库问答助手 — 从复习资料到代码仓库，一个 Agent 搞定问答。

> 基于 LangChain + Qdrant + FastEmbed 的本地 RAG 知识库问答助手，集成 Agent 意图规划、FlashRank 重排序与知识图谱可视化。

## 项目背景

上个月期末复习，想要一个能快速整理资料、精准回答问题、自动生成知识图谱的工具，找了几个开源的感觉总是觉得回复缺点智能感，于是自己做了一个 RAG 知识库 Agent，后面又加上了网页抓取、GitHub 仓库索引、本地代码目录和多源数据渠道。

## 功能

- **多来源导入**：PDF、DOCX、Markdown、TXT、URL、GitHub 仓库、本地代码目录、混合来源。
- **智能问答**：先由 Agent 判断用户意图和文件范围，再按来源过滤检索，避免混入无关文件。
- **资料概览**：可询问"这些资料讲什么""某份文件里面有什么""输出某份资料正文"等自然问题。
- **来源追踪**：回答可展开查看引用片段和页码，杜绝幻觉。
- **知识图谱**：从选定资料中抽取实体关系三元组，生成交互式可视化。
- **历史对话**：新建对话会保存旧记录，并清空当前索引和上传缓存。

## 界面预览

### 主界面

![主界面](docs/images/01-home.png)

### 多源导入

![多源导入](docs/images/02-multisource-import.png)

### 资料已导入

![资料已导入](docs/images/03-source-ready.png)

### 资料概览回答

![资料概览回答](docs/images/04-overview-answer.png)

### 引用来源

![引用来源](docs/images/05-cited-answer.png)

### GitHub 项目导入问答

![GitHub导入问答](docs/images/github项目导入和问答.png)

### 知识图谱

![知识图谱](docs/images/知识图谱生成.png)

### 知识图谱全屏

![知识图谱全屏](docs/images/知识图谱界面.png)

## 安装

建议使用 Python 3.12。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

复制 `.env.example` 为 `.env`，填入 LLM API Key（支持 Groq、DeepSeek 等 OpenAI 兼容接口）：

```powershell
copy .env.example .env
# 编辑 .env 填入 GROQ_API_KEY（或其他兼容 OpenAI 的 Key）
```

## 启动

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501
```

打开浏览器访问：

```text
http://localhost:8501
```

## 测试

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## 技术栈

| 技术 | 用途 |
|------|------|
| **RAG** | 检索增强生成架构，文档导入 → 切分 → 向量化 → 检索 → 重排序 → 生成回答 |
| **LangChain** | LLM 应用编排框架，管理 Chain / Retriever / Agent 全链路 |
| **Qdrant** | 向量数据库，HNSW 近似最近邻索引 + Payload 条件过滤，本地磁盘持久化 |
| **FastEmbed** | 轻量级 Embedding 推理引擎，基于 ONNX Runtime 加速 |
| **BAAI/bge-small-zh-v1.5** | 中英文双语 Embedding 模型（512 维），兼顾中文检索质量 |
| **FlashRank** | 交叉编码器重排序，对初检结果做精排提升相关性 |
| **LLMChainFilter** | LLM 驱动的语义过滤，二次筛选剔除无关片段 |
| **SemanticChunker** | 语义感知文档切分，避免固定长度切割破坏上下文 |
| **Agent Planner** | LLM 驱动的意图规划器，5 种意图分类 + 来源自动选择 |
| **Groq / DeepSeek** | LLM 推理后端，OpenAI 兼容 API 接口，支持多模型切换 |
| **Knowledge Graph** | LLM 抽取实体关系三元组，生成交互式力导向图可视化 |
| **Streamlit** | 交互式 Web UI，快速搭建数据应用和对话界面 |
| **PyPDFium2** | PDF 解析引擎，支持多页提取和元数据读取 |
| **Poetry + pytest** | 依赖管理与自动化测试 |

## 项目结构

```
study-ai/
├── app.py                        # Streamlit 主界面与交互流程
├── ragbase/
│   ├── agent_planner.py          # Agent 意图规划器（5 种意图 + 来源选择）
│   ├── agent_responses.py        # 按意图分发回答生成策略
│   ├── chain.py                  # LangChain RAG Chain 组装
│   ├── config.py                 # 全局配置（路径、模型、数据库参数）
│   ├── conversation_store.py     # 本地对话历史持久化
│   ├── ingestor.py               # 多源导入器（PDF/URL/GitHub/Code/Mixed）
│   ├── knowledge_graph.py        # 知识图谱抽取与交互式可视化
│   ├── model.py                  # LLM / Embedding / Reranker 工厂
│   ├── retriever.py              # 向量检索 + 来源过滤 + 重排序
│   ├── runtime.py                # 向量存储生命周期管理
│   ├── session_history.py        # 会话历史记录管理
│   ├── source_tools.py           # 资料清单 / 全文 / 概览工具
│   └── uploader.py               # 文件上传处理
├── docs/images/                  # 界面截图
├── tests/                        # 回归测试
├── .env.example                  # 环境变量模板
├── pyproject.toml                # Poetry 项目配置
└── README.md                     # 项目说明
```

## License

[MIT](LICENSE)
