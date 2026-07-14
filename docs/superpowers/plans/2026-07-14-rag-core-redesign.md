# RAG Core Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Study AI resolve sources reliably, index every document within the embedding model limit, recall evidence through dense and lexical search, and execute four stable operations without phrase-specific patches.

**Architecture:** New source and chunk records become the internal contract. A deterministic source resolver narrows candidates before a four-operation planner runs. A source-aware chunking router feeds Qdrant and a local BM25 index; fused results are reranked and expanded before the existing answer chain receives them.

**Tech Stack:** Python 3.12, LangChain 0.2, Qdrant client 1.10, FastEmbed BGE small Chinese, FlashRank, Streamlit, pytest.

## Global Constraints

- Preserve all existing explanatory comments in source and test files.
- Do not revert or reformat unrelated working-tree changes.
- Keep file, URL/GitHub, code-directory, and mixed-source import channels.
- Every embedded child must contain at most 480 BGE tokenizer tokens.
- Do not add a new external service or model dependency in this phase.
- Do not expose private chain-of-thought.
- Write and run a failing test before each production behavior change.

---

### Task 1: Stable Sources And Deterministic Resolution

**Files:**
- Create: `ragbase/source_registry.py`
- Create: `ragbase/source_resolver.py`
- Create: `tests/test_source_registry.py`
- Create: `tests/test_source_resolver.py`

**Interfaces:**
- Consumes: LangChain `Document`, source paths/URLs, current `session_id`.
- Produces: `SourceRecord`, `build_source_records(documents, session_id)`, `attach_source_records(documents, records)`, `resolve_sources(query, records, active_source_ids)`.

- [ ] **Step 1: Write failing source-record tests**

```python
def test_build_source_records_assigns_stable_ids_and_session():
    docs = [Document(page_content="内容", metadata={"source": "a.pdf", "source_name": "a.pdf", "source_type": "pdf"})]
    records = build_source_records(docs, "session-1")
    assert records[0].source_id.startswith("src_")
    assert records[0].session_id == "session-1"
    assert records[0].normalized_name == "a"

def test_same_content_uses_same_source_id_across_sessions():
    first = build_source_records(docs, "one")[0]
    second = build_source_records(docs, "two")[0]
    assert first.source_id == second.source_id
    assert first.session_id != second.session_id
```

- [ ] **Step 2: Verify the tests fail because the module is absent**

Run: `.venv\Scripts\python.exe -m pytest tests/test_source_registry.py -q`

Expected: collection error for missing `ragbase.source_registry`.

- [ ] **Step 3: Implement immutable source records and metadata attachment**

```python
@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    source_name: str
    normalized_name: str
    source_type: str
    session_id: str
    file_hash: str
    source_path: str

def build_source_records(documents: Iterable[Document], session_id: str) -> list[SourceRecord]: ...
def attach_source_records(documents: Iterable[Document], records: Sequence[SourceRecord]) -> list[Document]: ...
```

- [ ] **Step 4: Run source-record tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_source_registry.py -q`

Expected: all tests pass.

- [ ] **Step 5: Write failing resolver tests**

```python
def test_resolver_handles_partial_name_typo_and_ordinal():
    assert resolve_sources("复习纲里讲了什么", records).source_ids == (outline.source_id,)
    assert resolve_sources("第二个文件", records).source_ids == (records[1].source_id,)

def test_resolver_inherits_active_source_for_pronoun_followup():
    result = resolve_sources("它里面还有什么", records, [outline.source_id])
    assert result.source_ids == (outline.source_id,)
```

- [ ] **Step 6: Verify resolver tests fail, then implement ranked candidate resolution**

Run: `.venv\Scripts\python.exe -m pytest tests/test_source_resolver.py -q`

Implementation must normalize punctuation/extensions, support exact and substring matches, ordinal phrases, `difflib.SequenceMatcher`, active-source pronouns, confidence, and ambiguous candidate output.

- [ ] **Step 7: Run Task 1 tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_source_registry.py tests/test_source_resolver.py -q`

Expected: all tests pass.

### Task 2: Source-Aware Chunking Router

**Files:**
- Create: `ragbase/chunking.py`
- Modify: `ragbase/ingestor.py`
- Create: `tests/test_chunking.py`
- Modify: `tests/test_multisource_ingestion.py`

**Interfaces:**
- Consumes: parsed LangChain `Document` pages/elements with source metadata.
- Produces: `EmbeddingTokenizer`, `ChunkingRouter.split_documents(documents)`, child `Document` values with chunk and parent metadata.

- [ ] **Step 1: Write failing tests for review outlines and slides**

```python
def test_numbered_outline_keeps_each_numbered_item_and_caps_tokens(tokenizer):
    chunks = router.split_documents([outline_page])
    assert all(tokenizer.count(chunk.page_content) <= 480 for chunk in chunks)
    assert "1. 图像质量评价指标" in "\n".join(c.page_content for c in chunks)
    assert "42. 模型部署流程" in "\n".join(c.page_content for c in chunks)

def test_slide_page_keeps_page_and_parent_window_metadata():
    chunks = router.split_documents(slide_pages)
    assert chunks[1].metadata["page_start"] == 2
    assert chunks[1].metadata["parent_content"].startswith(slide_pages[0].page_content)
```

- [ ] **Step 2: Verify chunking tests fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_chunking.py -q`

Expected: collection error for missing `ragbase.chunking`.

- [ ] **Step 3: Implement tokenizer and routing strategies**

```python
class EmbeddingTokenizer:
    def count(self, text: str) -> int: ...
    def split(self, text: str, max_tokens: int = 480, overlap_tokens: int = 50) -> list[str]: ...

class ChunkingRouter:
    def split_documents(self, documents: Sequence[Document]) -> list[Document]: ...
```

The router must detect numbered outlines, slide groups, Markdown headings, Python AST blocks, structured data, and prose. It must assign deterministic `chunk_id`, `parent_id`, `chunk_index`, `element_type`, `page_start`, `page_end`, `token_count`, and `parent_content`.

- [ ] **Step 4: Run chunking tests and fix only tested behavior**

Run: `.venv\Scripts\python.exe -m pytest tests/test_chunking.py -q`

Expected: all tests pass.

- [ ] **Step 5: Replace the unused semantic/2048-character splitter in Ingestor**

```python
self.chunking_router = ChunkingRouter()
split_documents = self.chunking_router.split_documents(source_documents)
```

- [ ] **Step 6: Verify all import channels and chunk tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_chunking.py tests/test_multisource_ingestion.py -q`

Expected: all tests pass.

### Task 3: Four-Operation Planner And Query Rewrite

**Files:**
- Create: `ragbase/planner_schema.py`
- Create: `ragbase/query_rewriter.py`
- Modify: `ragbase/agent_planner.py`
- Modify: `tests/test_agent_planner.py`
- Create: `tests/test_query_rewriter.py`

**Interfaces:**
- Consumes: question, source records, resolver result, recent messages, active source IDs.
- Produces: `AgentPlan(operation, source_ids, query, confidence, reason)`, `rewrite_query(...)`.

- [ ] **Step 1: Add failing planner-schema tests**

```python
def test_planner_accepts_only_four_operations_and_catalog_source_ids():
    plan = asyncio.run(plan_question(model, "问题", [], records, resolution, None))
    assert plan.operation == "read_source"
    assert plan.source_ids == (records[0].source_id,)

def test_model_cannot_invent_source_id():
    assert "src_missing" not in plan.source_ids
```

- [ ] **Step 2: Verify tests fail against the five-intent name-based planner**

Run: `.venv\Scripts\python.exe -m pytest tests/test_agent_planner.py -q`

Expected: failures for absent `operation` and `source_ids`.

- [ ] **Step 3: Implement planner schema and adapt planner validation**

```python
VALID_OPERATIONS = {"chat", "list_sources", "read_source", "search"}

@dataclass(frozen=True)
class AgentPlan:
    operation: str
    source_ids: tuple[str, ...]
    query: str
    confidence: float
    reason: str = ""
```

The prompt receives only resolver candidates, not unrestricted source names. Deterministic resolver output wins when confidence is high. Planner failure falls back to `search` over active or all current sources.

- [ ] **Step 4: Write failing query-rewriter tests**

```python
def test_rewrite_uses_resolved_source_without_changing_operation():
    result = asyncio.run(rewrite_query(model, "它讲了什么", records, resolution, history))
    assert records[0].source_name in result.query
    assert result.source_ids == resolution.source_ids
```

- [ ] **Step 5: Implement query rewriting and run Task 3 tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_agent_planner.py tests/test_query_rewriter.py -q`

Expected: all tests pass.

### Task 4: Hybrid Recall, Fusion, And Context Expansion

**Files:**
- Create: `ragbase/hybrid_retriever.py`
- Modify: `ragbase/retriever.py`
- Modify: `ragbase/config.py`
- Create: `tests/test_hybrid_retriever.py`
- Modify: `tests/test_source_filtered_retriever.py`

**Interfaces:**
- Consumes: vector store, indexed chunk documents, query, optional source IDs.
- Produces: LangChain-compatible `HybridRetriever` returning fused, reranked, expanded documents.

- [ ] **Step 1: Write failing BM25 and RRF tests**

```python
def test_bm25_recovers_exact_course_term():
    results = bm25_search("Dm距离", chunks, top_k=20)
    assert results[0].document.metadata["chunk_id"] == "distance"

def test_rrf_fuses_by_chunk_id_without_duplicates():
    fused = reciprocal_rank_fusion([dense, lexical])
    assert [item.document.metadata["chunk_id"] for item in fused].count("shared") == 1
```

- [ ] **Step 2: Verify hybrid tests fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_hybrid_retriever.py -q`

Expected: collection error for missing hybrid retriever.

- [ ] **Step 3: Implement local tokenization, BM25, RRF, and parent expansion**

```python
def bm25_search(query: str, documents: Sequence[Document], top_k: int = 20) -> list[RankedDocument]: ...
def reciprocal_rank_fusion(rankings: Sequence[Sequence[RankedDocument]], k: int = 60) -> list[RankedDocument]: ...
def expand_context(results: Sequence[RankedDocument], documents_by_id: Mapping[str, Document], budget_tokens: int) -> list[Document]: ...
```

- [ ] **Step 4: Increase dense candidate recall and add source-ID filters**

```python
search_kwargs = {"k": Config.Retriever.RETRIEVAL_TOP_K}
source_filter = build_source_filter(source_ids, metadata_key="metadata.source_id")
```

Use `RETRIEVAL_TOP_K=20`, `RERANK_TOP_N=5`, and a context budget in configuration.

- [ ] **Step 5: Implement `HybridRetriever` with reranker fallback**

The class retrieves dense and lexical candidates, fuses them, optionally invokes FlashRank, retains five to eight children, expands parent context, and returns LangChain documents. If reranking fails, fused order is returned.

- [ ] **Step 6: Run retrieval tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_hybrid_retriever.py tests/test_source_filtered_retriever.py tests/test_retrieval_language.py -q`

Expected: all tests pass.

### Task 5: Unified Orchestrator And Complete Read Operations

**Files:**
- Create: `ragbase/orchestrator.py`
- Modify: `ragbase/source_tools.py`
- Modify: `ragbase/agent_responses.py`
- Modify: `ragbase/chain.py`
- Modify: `app.py`
- Create: `tests/test_orchestrator.py`
- Modify: `tests/test_source_tools.py`
- Modify: `tests/test_source_input_reset.py`

**Interfaces:**
- Consumes: question, session runtime, model, source registry, profiles, chunk documents, vector store.
- Produces: `ExecutionResult(answer, documents, plan)` or an async stream adapter for Streamlit.

- [ ] **Step 1: Write failing operation-dispatch tests**

```python
def test_list_sources_bypasses_retrieval_and_model(): ...
def test_full_text_reads_selected_source_in_page_order(): ...
def test_search_uses_hybrid_retriever_with_source_ids(): ...
def test_chat_does_not_attach_source_context(): ...
```

- [ ] **Step 2: Verify orchestrator tests fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_orchestrator.py -q`

Expected: collection error for missing orchestrator.

- [ ] **Step 3: Implement operation dispatch and source-ID/name adapters**

`list_sources` reads records. `read_source` chooses inventory, paged full text, or hierarchical overview. `search` creates the hybrid retriever and grounded chain. `chat` calls the normal chat response. No branch reinterprets the user's phrase with a second keyword router.

- [ ] **Step 4: Make profiles retain all pages and hierarchical summary units**

`build_source_profiles` must preserve ordered page documents and summary units. Preview text remains capped only for display. Complete text and overview generation cannot use a fixed prefix as the entire source.

- [ ] **Step 5: Strengthen grounded context labels**

`format_documents` must include source name, page, and chunk ID before each chunk. The system prompt requires answers to stay within supplied evidence and to say when evidence is insufficient.

- [ ] **Step 6: Replace the large dispatch block in `ask_chain` with orchestrator calls**

Keep Streamlit rendering and streaming in `app.py`; move operation decisions and data access to `ragbase/orchestrator.py`. Initialize and reset `source_records` and `chunk_documents` alongside existing session state.

- [ ] **Step 7: Run operation and UI regression tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_orchestrator.py tests/test_source_tools.py tests/test_source_input_reset.py tests/test_agent_responses.py tests/test_chain_history.py -q`

Expected: all tests pass.

### Task 6: Evaluation, Real Documents, And Runtime Verification

**Files:**
- Create: `tests/test_rag_evaluation.py`
- Create: `tests/fixtures/rag_cases.json`
- Modify: `README.md`

**Interfaces:**
- Consumes: the ten course PDFs when present, otherwise synthetic equivalent pages.
- Produces: deterministic operation/source/chunk evaluation and documented architecture.

- [ ] **Step 1: Add evaluation cases with varied natural language**

```json
[
  {"question": "第二份讲的啥", "operation": "read_source", "source_index": 1},
  {"question": "复习纲里图像分割考哪些", "operation": "search", "source_contains": "复习大纲", "must_recall": ["图像分割"]},
  {"question": "刚才那个继续说", "operation": "search", "uses_active_source": true}
]
```

- [ ] **Step 2: Write failing evaluation assertions for operation, source, and recall**

Run: `.venv\Scripts\python.exe -m pytest tests/test_rag_evaluation.py -q`

Expected: failures identify any missing integration behavior.

- [ ] **Step 3: Fix integration gaps without adding phrase-specific routes**

Only shared resolver, planner validation, chunking, or retrieval logic may change. The fixture's exact question text must not be copied into production code.

- [ ] **Step 4: Run the complete suite and static checks**

Run: `.venv\Scripts\python.exe -m pytest -q`

Run: `.venv\Scripts\python.exe -m compileall -q app.py ragbase tests`

Expected: all tests pass and compileall exits zero.

- [ ] **Step 5: Start one clean Streamlit process and verify HTTP response**

Stop only `streamlit run app.py` processes for this project, start `.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501 --server.headless true`, then request `http://localhost:8501`.

Expected: HTTP 200 and exactly one Study AI Streamlit process.
