# Study AI RAG Core Redesign

## Goal

Replace phrase-specific routing and one-size-fits-all chunking with a stable,
source-aware RAG pipeline that works across paraphrases, file names, long PDF
pages, slide decks, outlines, prose, structured text, and source code.

This phase fixes four observed failures:

1. Imported sources are identified by display names instead of stable IDs.
2. Every document is split by the same 2048-character rule even though the
   embedding model accepts only about 512 tokens.
3. Dense retrieval starts with only five candidates, so reranking cannot
   recover relevant chunks that were never recalled.
4. Planning, source resolution, query rewriting, and execution are coupled,
   which encourages adding special cases for individual user phrases.

Existing explanatory comments in source and test files must be preserved.
Unrelated working-tree changes must not be reverted or reformatted.

## Scope

### Included

- Stable `SourceRecord` and `ChunkRecord` metadata.
- Deterministic source resolution before LLM planning.
- Four operations: `chat`, `list_sources`, `read_source`, and `search`.
- Independent query rewriting for follow-up questions.
- Source-specific parser and chunking strategies.
- Embedding-token hard limits and parent-child chunk relationships.
- Wider dense recall, local BM25 recall, reciprocal-rank fusion, reranking,
  deduplication, and parent/neighbor context expansion.
- Direct inventory and full-text operations that do not use vector search.
- Hierarchical source overviews that cover the whole selected source.
- Structured evidence metadata and safe insufficient-evidence behavior.
- Regression and retrieval evaluation cases based on real course PDFs.

### Deferred

- Embedding fine-tuning.
- Multi-agent orchestration.
- Qdrant server deployment and multi-user concurrency.
- Replacing every PDF parser in one change.
- Displaying private chain-of-thought.
- Knowledge-graph retrieval integration.

## Data Contracts

### SourceRecord

Each imported source receives a stable ID independent of its file name:

```text
source_id, source_name, normalized_name, source_type, session_id, file_hash,
source_path, created_at
```

`file_hash` deduplicates repeated imports. `session_id` isolates current-chat
sources without deleting unrelated files. User-facing text continues to show
`source_name`; internal filtering uses `source_id`.

### ChunkRecord

Every searchable child chunk stores:

```text
chunk_id, parent_id, source_id, source_name, source_type, page_start, page_end,
section_title, element_type, chunk_index, token_count, start_index, content
```

Parents store the larger page, section, slide window, class, or file context.
Search ranks child chunks and answer assembly expands their parents or adjacent
children as needed.

## Parsing And Chunking

`ParserRouter` converts each source into layout-aware `Document` elements.
`ChunkingRouter` chooses one strategy from source metadata and content shape.
All strategies share only final validation: non-empty content, required
metadata, deterministic IDs, and at most 480 embedding-model tokens.

### PDF classification order

1. Scanned/image-only PDF: OCR/layout extraction is required.
2. Slide deck: one slide is a child; a nearby slide window is its parent.
3. Numbered outline: numbered questions/items remain atomic.
4. Academic or multi-column PDF: split by detected sections and elements.
5. Table-heavy PDF: preserve headers and row relationships.
6. Prose PDF: heading, paragraph, Chinese sentence, punctuation, then token
   fallback.

For the current nine course slide decks, each page is normally one child and
the previous/current/next pages form parent context. A slide above 480 tokens
is split by bullets or Chinese sentences into 300-420-token children with
40-60-token overlap. Page overlap is otherwise zero.

For the two-page numbered review outline, each numbered item is atomic. Items
are packed into 200-350-token children without splitting an item. The page is
the parent. This prevents a 816-token page from being silently truncated by
the 512-token embedding model.

### Other sources

- DOCX: headings, paragraphs, lists, and tables from document structure.
- Markdown: headings first; fenced code and tables remain atomic.
- HTML: article/heading DOM structure; lists, tables, and code are atomic.
- TXT: classify numbered list, log, dialogue, or prose before splitting.
- Python: module, class, and function boundaries using `ast`.
- Other code: language-aware separators with file/class parent context.
- JSON/YAML/TOML: split by object path and preserve `key_path` metadata.
- SQL: split by statements and schema objects.

General prose children target 350-420 tokens, hard-limit at 480, and use about
50 tokens of overlap. Semantic splitting is optional only for long,
unstructured prose and is followed by the same token cap.

## Planning And Source Resolution

The planner emits one of four operations:

```json
{
  "operation": "search",
  "source_ids": ["source_003"],
  "query": "图像分割有哪些常用方法？",
  "filters": {},
  "confidence": 0.92
}
```

`SourceResolver` runs before the planner and produces candidates using exact
name, extension-free name, normalized substring, Chinese n-grams, ordinal
references, edit distance, active sources, and recent dialogue references.
An obvious single candidate is selected deterministically. The LLM may choose
only among ambiguous candidates; it cannot invent a source.

`QueryRewriter` then converts context-dependent questions into standalone
retrieval queries. It does not decide the operation and does not select files.

Operations execute as follows:

- `chat`: ordinary model conversation with no knowledge-base context.
- `list_sources`: read the source registry directly.
- `read_source`: inventory, complete text, page ranges, or hierarchical
  overview directly from stored source documents.
- `search`: source-filtered retrieval followed by grounded generation.

Legacy keyword routing is removed from the production answer path after the
new planner passes regression tests.

## Retrieval And Context Assembly

The search pipeline is:

```text
source filter
  -> dense Top-20
  + BM25 Top-20
  -> reciprocal-rank fusion
  -> FlashRank rerank
  -> retain Top-5 to Top-8
  -> parent/neighbor expansion
  -> deduplicate
  -> context budget assembly
```

BM25 is implemented over in-memory indexed chunks using existing Python
dependencies or a small local implementation; no search service is introduced
in this phase. Dense and lexical result lists are fused by stable chunk ID.
Reranking remains optional and must degrade to fused order if unavailable.

Context assembly keeps source and page boundaries, avoids duplicate parent
text, and stops at a configurable token budget. It records the evidence chunks
used for generation.

## Read Operations And Answers

Source inventories are registry reads and never call the LLM. Complete-text
requests read stored source documents in original order and support pagination;
they never use Top-K retrieval.

Overviews use map-reduce-style hierarchical summaries across all pages or
sections. No fixed character prefix may stand in for an entire long source.

Grounded answers bind claims to evidence metadata and expose:

```text
answer, citations, evidence_level, missing_information
```

Prompts require source-only claims, page citations where available, and an
explicit insufficient-evidence response. They request concise evidence notes,
not private chain-of-thought.

## Lifecycle

The current single-process local Qdrant mode remains supported. Index cleanup
must release active clients before removing storage. New-chat behavior clears
the active session's sources and in-memory index state without deleting source
files or historical conversation records.

This phase introduces an `IndexManager` boundary so Qdrant server mode and
incremental indexing can be added later without changing planner behavior.

## Testing And Evaluation

Implementation follows test-first development. Required tests include:

- Stable source IDs, session isolation, and duplicate file hashes.
- Exact, partial, misspelled, ordinal, and conversational source references.
- Planner output restricted to four operations and candidate source IDs.
- Review-outline items are not truncated and remain individually retrievable.
- Slide pages retain page metadata and dense slides stay under 480 tokens.
- Markdown/code/structured data use their respective strategies.
- Dense and BM25 candidates are fused deterministically.
- Source filters prevent cross-file leakage.
- Inventory/full text bypass retrieval; overview covers the whole source.
- Missing evidence produces a restrained answer rather than fabrication.
- Existing multi-source channels and current UI behavior remain intact.

An evaluation fixture records question, expected operation, expected sources,
and expected evidence. Initial metrics are operation accuracy, source accuracy,
Recall@K, MRR, citation accuracy, insufficient-evidence accuracy, answer
faithfulness, latency, and model usage.

## Implementation Order

1. Add data contracts, source registry, and source resolver.
2. Add parser/chunking router and token-cap tests on real document shapes.
3. Move planner to the four-operation schema and separate query rewriting.
4. Add IndexManager and parent-child payloads.
5. Add BM25, RRF, wider dense recall, reranking, and context assembly.
6. Route inventory, full text, overview, and grounded answers through the new
   orchestrator.
7. Add evaluation fixtures, timings, and final regression coverage.

Each step must keep the application runnable and preserve existing source
channels: uploaded files, URLs, local code directories, and mixed inputs.
