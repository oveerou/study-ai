# 从上到下阅读项目代码

不要按文件名字母顺序读。按下面顺序打开，每看完一项就往下看。

> 在 IDE 的 Markdown 预览中，按住 `Ctrl` 单击文件名可直接打开对应代码。

## 1. 程序从哪里开始

1. [app.py](app.py)  
   Streamlit 页面入口。先看 `main()`、`index_documents()`、`ask_chain()`，理解用户上传资料和提问时分别发生什么。

## 2. 配置和模型

2. [ragbase/config.py](ragbase/config.py)  
   所有模型名、API 地址、向量库路径、检索数量等配置。

3. [ragbase/model.py](ragbase/model.py)  
   创建聊天模型、Embedding 模型和 Reranker；理解项目最后实际调用的是哪个大模型。

## 3. 资料如何进入知识库

4. [ragbase/uploader.py](ragbase/uploader.py)  
   将网页上传的文件保存到临时目录。

5. [ragbase/ingestor.py](ragbase/ingestor.py)  
   解析 PDF、DOCX、文本、网页、代码目录；发起切块、向量化和 Qdrant 入库。

6. [ragbase/chunking.py](ragbase/chunking.py)  
   真正的切块规则：判断 PDF 是课件、编号大纲还是普通正文，再按 token 切成 chunk。

7. [ragbase/source_registry.py](ragbase/source_registry.py)  
   给每份资料分配 `source_id`，并把文件名、页码、来源身份附加到每个文档块。

## 4. 用户问题如何理解

8. [ragbase/source_resolver.py](ragbase/source_resolver.py)  
   识别用户说的是哪份文件，支持简称、序号、错别字和“这份/那份”等指代。

9. [ragbase/planner_schema.py](ragbase/planner_schema.py)  
   定义 Planner 返回的操作计划和引用信息的数据结构。

10. [ragbase/agent_planner.py](ragbase/agent_planner.py)  
    判断用户是闲聊、问文件清单、要概览/全文，还是要检索资料。

11. [ragbase/source_tools.py](ragbase/source_tools.py)  
    不需要检索时的确定性工具：列出文件、构建资料档案、输出全文。

## 5. 如何找资料并回答

12. [ragbase/hybrid_retriever.py](ragbase/hybrid_retriever.py)  
    核心检索：向量检索 + BM25 + RRF 融合 + Reranker 重排 + 父块扩展。

13. [ragbase/retriever.py](ragbase/retriever.py)  
    创建检索器、添加来源过滤，并在重排器不可用时降级。

14. [ragbase/chain.py](ragbase/chain.py)  
    将检索到的 chunk 组装进 Prompt，流式生成带资料依据的回答。

15. [ragbase/agent_responses.py](ragbase/agent_responses.py)  
    普通聊天和“这些资料主要讲什么”的概览回答。

16. [ragbase/orchestrator.py](ragbase/orchestrator.py)  
    总调度器：将来源解析、Planner、检索、回答串成一次完整请求。

## 6. 辅助能力和运行时

17. [ragbase/knowledge_graph.py](ragbase/knowledge_graph.py)  
    从资料抽取实体与关系，供知识图谱界面使用。

18. [ragbase/runtime.py](ragbase/runtime.py)  
    关闭 Qdrant、重置索引、归档会话，避免本地数据库文件锁。

19. [ragbase/conversation_store.py](ragbase/conversation_store.py)  
    保存和读取本地历史对话。

20. [ragbase/session_history.py](ragbase/session_history.py)  
    管理当前对话的消息记录和轮数上限。

## 最后再看

- [tests](tests)：验证代码行为，不参与应用运行。
- [tests/fixtures/rag_cases.json](tests/fixtures/rag_cases.json)：RAG 问答测试题库。
- [README.md](README.md)：项目说明、运行方式和截图。
