# Diseno: sandbox_cookbook — Evaluacion Contextual Retrieval (Anthropic)

## 1. Objetivo

Implementar un sandbox de evaluacion que replique fielmente la propuesta de Anthropic (blog + cookbook) sobre Contextual Retrieval, usando el mismo dataset (9 codebases, 248 queries), reutilizando la libreria compartida (`shared/`) de retrieval, embeddings y tipos.

El sandbox debe permitir ejecutar y comparar 4 estrategias incrementales:

| # | Estrategia | Componentes | Referencia Anthropic |
|---|---|---|---|
| 1 | `SIMPLE_VECTOR` | Embedding puro (Voyage/NIM) + cosine similarity | Baseline RAG del cookbook |
| 2 | `CONTEXTUAL_VECTOR` | Contextual Embeddings (LLM genera contexto con WHOLE_DOCUMENT) | Blog: -35% failure rate |
| 3 | `CONTEXTUAL_HYBRID` | Contextual Embeddings + Contextual BM25 + RRF | Blog: -49% failure rate |
| 4 | `CONTEXTUAL_HYBRID_RERANK` | Contextual Hybrid + Cross-encoder reranking | Blog: -67% failure rate |

Cada run produce un `EvaluationRun` comparable con los resultados del cookbook.

---

## 2. Dataset

### 2.1 Fuente

El cookbook usa dos archivos ([fuente](https://github.com/anthropics/anthropic-cookbook/tree/main/capabilities/contextual-embeddings/data)):
- `codebase_chunks.json`: 90 documentos de codigo fuente, pre-chunkeados (737 chunks total)
- `evaluation_set.jsonl`: 248 queries (306 golden chunks; 28 queries con >1 golden). Keys: `query`, `answer`, `golden_doc_uuids`, `golden_chunk_uuids`, `golden_documents`, `golden_chunks`, `meta`

### 2.2 Estructura de codebase_chunks.json

```json
[
  {
    "doc_id": "...",
    "original_uuid": "...",
    "content": "<DOCUMENTO COMPLETO>",
    "chunks": [
      {
        "chunk_id": "...",
        "original_index": 0,
        "content": "<texto del chunk>"
      }
    ]
  }
]
```

**Punto critico:** Cada documento tiene un campo `content` con el texto completo del archivo. Este es el `WHOLE_DOCUMENT` que la propuesta de Anthropic requiere para generar contexto situacional. Este sandbox lo usa siempre (resuelve DT-2).

### 2.3 Estructura de evaluation_set.jsonl

Cada linea contiene:
```json
{
  "query": "texto de la pregunta",
  "golden_chunk_uuids": [["doc_uuid", chunk_index], ...],
  "golden_documents": [
    {
      "uuid": "...",
      "content": "...",
      "chunks": [{"index": 0, "content": "..."}, ...]
    }
  ]
}
```

### 2.4 Adaptacion al modelo de datos existente

El dataset del cookbook tiene una estructura diferente a HotpotQA. No hay qrels separados ni answer text. La evaluacion se basa en si el golden chunk aparece en los top-k recuperados (Pass@k), no en metricas de generacion (F1, EM).

Mapeo propuesto:

| Concepto cookbook | Concepto shared/types.py |
|---|---|
| `doc_id` + `chunk_index` | `doc_id` en `NormalizedDocument` |
| `content` (chunk) | `NormalizedDocument.content` |
| `content` (documento completo) | Almacenado en `LoadedDataset.metadata` |
| `golden_chunk_uuids` | `NormalizedQuery.relevant_doc_ids` |
| `query` | `NormalizedQuery.query_text` |
| No hay answer text | `NormalizedQuery.expected_answer = None` |

**Decision de diseno sobre parent_content:**

Opcion A: Agregar `parent_content: Optional[str] = None` a `NormalizedDocument`.
- Pro: acceso directo desde cualquier punto del pipeline.
- Contra: infla la memoria (el documento completo se duplica N veces, una por chunk).

Opcion B: Almacenar un indice `doc_uuid -> parent_content` en `LoadedDataset.metadata`.
- Pro: un solo string por documento, O(1) lookup por doc_uuid.
- Contra: requiere pasar el indice al `ContextualRetriever`.

**Recomendacion: Opcion B.** El cookbook tiene documentos de hasta 8K tokens; con 9 documentos, almacenar 9 strings completos es trivial. Con Opcion A, se almacenarian 737 copias. La diferencia es irrelevante para este dataset pero el patron es mas correcto.

Estructura concreta:

```python
# En LoadedDataset.metadata:
{
    "parent_documents": {
        "doc_uuid_1": "texto completo del documento 1",
        "doc_uuid_2": "texto completo del documento 2",
    }
}
```

Y cada `NormalizedDocument` almacena en su `metadata`:
```python
NormalizedDocument(
    doc_id="doc_uuid_1__chunk_0",
    content="texto del chunk",
    title="nombre del archivo de codigo",
    metadata={
        "parent_doc_id": "doc_uuid_1",
        "original_index": 0,
        "original_uuid": "...",
    }
)
```

---

## 3. Arquitectura de archivos

```
├── shared/                          # Libreria compartida
│   ├── types.py                     # NormalizedQuery, LoadedDataset, EvaluationRun, Protocols
│   ├── llm.py                       # AsyncLLMService, load_embedding_model, batch_embed_queries
│   ├── config_base.py               # InfraConfig, RerankerConfig
│   ├── vector_store.py              # ChromaVectorStore
│   └── retrieval/
│       ├── core.py                  # BaseRetriever, SimpleVectorRetriever, RetrievalConfig
│       ├── hybrid_retriever.py      # BM25 + Vector + RRF (formulas classic + cookbook)
│       ├── contextual_retriever.py  # LLMContextGenerator (Mode A, prompts XML Anthropic)
│       ├── reranker.py              # CrossEncoderReranker (NVIDIARerank)
│       └── tantivy_index.py         # BM25 via Tantivy (Rust, fallback rank-bm25)
│
├── sandbox_cookbook/                 # Evaluacion Contextual Retrieval
│   ├── config.py                    # CookbookConfig
│   ├── loader.py                    # JSON/JSONL local -> LoadedDataset
│   ├── evaluator.py                 # Pipeline: 4 estrategias, Pass@k, comparison.csv
│   ├── context_cache.py             # Cache persistente de contextos LLM
│   ├── run.py                       # Entry point (--compare-all, --dry-run, -v)
│   ├── env.example
│   └── data/
│       ├── codebase_chunks.json
│       └── evaluation_set.jsonl
│
├── tests/                           # 137 tests (pytest)
│   ├── test_cookbook_config.py       # 23 tests
│   ├── test_cookbook_loader.py       # 19 tests
│   ├── test_cookbook_evaluator.py    # 29 tests
│   ├── test_contextual_mode_a.py    # 26 tests
│   ├── test_dtm4_rrf.py            # 20 tests
│   ├── test_dtm4_tantivy_edge_cases.py  # 17 tests
│   └── test_dt8_09_10_11_reranker_sort.py  # 3 tests
```

---

## 4. Componentes detallados

### 4.1 sandbox_cookbook/config.py — CookbookConfig

```python
@dataclass
class CookbookConfig:
    infra: InfraConfig
    retrieval: RetrievalConfig
    reranker: RerankerConfig

    # Dataset
    dataset_path: Path
    eval_path: Path
    results_dir: Path

    # Estrategia
    strategy: str = "SIMPLE_VECTOR"

    # Contextual Retrieval
    contextualize_model: str = ""
    contextualize_max_tokens: int = 1000
    contextualize_batch_size: int = 10
    context_position: str = "prepend"     # "prepend" (blog) | "append" (cookbook)
    contexts_cache_path: Optional[Path] = None

    # Evaluacion
    eval_k_values: List[int] = field(default_factory=lambda: [5, 10, 20])

    # Hybrid search
    semantic_weight: float = 0.8
    bm25_weight: float = 0.2
    num_chunks_to_recall: int = 150

    # Reranking
    rerank_top_n: int = 20
    rerank_oversample_factor: int = 10
    rerank_content: str = "enriched"      # "original" | "enriched" | "both"
```

### 4.2 sandbox_cookbook/loader.py — CookbookLoader

Responsabilidades:

1. Leer `codebase_chunks.json` -> `LoadedDataset` con cada chunk como `NormalizedDocument`, parent documents en metadata
2. Leer `evaluation_set.jsonl` -> `List[CookbookEvalQuery]` con golden contents
3. Validar que golden chunks existen en corpus

```python
@dataclass
class CookbookEvalQuery:
    """Query con golden chunks para evaluacion Pass@k."""
    query_id: str
    query_text: str
    golden_chunk_ids: List[str]      # doc_ids compuestos
    golden_contents: List[str]       # contenido exacto para matching

class CookbookLoader:
    def __init__(self, dataset_path: Path, eval_path: Path): ...
    def load(self) -> Tuple[LoadedDataset, List[CookbookEvalQuery]]: ...
```

**Matching por contenido vs por ID:** El cookbook evalua por contenido exacto (`retrieved_content == golden_content`). Este sandbox implementa ambos y verifica equivalencia. ID-based como primary, content-based como validacion.

### 4.3 Refactor de contextual_retriever.py — Mode A

Cambios fundamentales respecto a implementacion actual:

**a) LLMContextGenerator parametrizado:**

```python
class LLMContextGenerator:
    def __init__(
        self,
        llm_service: LLMJudgeProtocol,
        max_tokens: int = 1000,
        mode: Literal["document", "fallback"] = "document",
        document_prompt_template: Optional[str] = None,  # override prompt
        chunk_prompt_template: Optional[str] = None,
    ):
        self.mode = mode
        # Default: prompt exacto de Anthropic (XML tags)
        self.document_prompt = document_prompt_template or ANTHROPIC_DOCUMENT_PROMPT
        self.chunk_prompt = chunk_prompt_template or ANTHROPIC_CHUNK_PROMPT
```

**b) Error explicito si Mode A sin parent_content:**

```python
async def _generate_one(self, chunk_content, parent_content=None, document_title=None):
    if self.mode == "document" and not parent_content:
        raise ValueError(
            "Mode 'document' requiere parent_content. "
            "El dataset debe proporcionar documentos completos."
        )
```

**c) Prompts de Anthropic (Mode A):**

```python
ANTHROPIC_DOCUMENT_PROMPT = """
<document>
{doc_content}
</document>
"""

ANTHROPIC_CHUNK_PROMPT = """
Here is the chunk we want to situate within the whole document
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else.
"""
```

Prompts plain text se mantienen como opcion para modelos nano que no manejan XML. La seleccion se hace via config.

**d) context_position parametrizable:**

La logica de combinacion contexto+original se mueve a `LLMContextGenerator`, que recibe `context_position` en constructor. `EnrichedChunk.get_enriched_text()` existente (que hardcodea "prepend") se reemplaza: `generate_contexts_batch()` almacena directamente el texto combinado en `generated_context` segun la posicion configurada. `ContextualRetriever.index_documents()` usa `chunk.generated_context + "\n\n" + chunk.original_content` sin cambios — la combinacion ya esta resuelta.

```python
class LLMContextGenerator:
    def __init__(self, ..., context_position: str = "prepend"):
        self.context_position = context_position

    def _build_enriched_text(self, context: str, original: str) -> str:
        if self.context_position == "prepend":
            return f"{context}\n\n{original}"      # blog de Anthropic
        return f"{original}\n\n{context}"           # cookbook de Anthropic
```

El CookbookEvaluator usa `_build_enriched_text()` del generator directamente.

**e) Exponer enriched_contents para reranking:**

```python
def _swap_to_original_contents(self, result: RetrievalResult) -> RetrievalResult:
    # Guardar enriquecidos antes de swap
    result.enriched_contents = list(result.contents)
    # Swap a originales para generacion
    result.contents = [
        self._original_contents.get(doc_id, content)
        for doc_id, content in zip(result.doc_ids, result.contents)
    ]
    return result
```

### 4.4 Flujo del documento padre

```
CookbookLoader.load()
  -> LoadedDataset.metadata["parent_documents"] = {uuid: full_text}
  -> corpus[doc_id].metadata["parent_doc_id"] = uuid

CookbookEvaluator._index_documents(dataset, corpus)
  -> parent_docs = dataset.metadata["parent_documents"]
  -> documents = [
         {
             "doc_id": doc.doc_id,
             "content": doc.content,            # (PC-1) SIN get_full_text(), sin title
             "title": None,                     # chunks de codigo: title contamina embedding
             "parent_content": parent_docs[doc.metadata["parent_doc_id"]],
         }
         for doc in corpus.values()
     ]
  -> retriever.index_documents(documents)
```

### 4.5 sandbox_cookbook/evaluator.py — CookbookEvaluator

**Fases del pipeline:**

```
1. load()       -> LoadedDataset + List[CookbookEvalQuery]
2. index()      -> Indexar corpus (con/sin contextualizacion)
3. pre_embed()  -> Batch embed queries
4. retrieve()   -> Retrieval sync (vector / hybrid)
5. [rerank()]   -> Solo si CONTEXTUAL_HYBRID_RERANK
6. evaluate()   -> Pass@k sobre golden chunks
7. build_run()  -> EvaluationRun
```

**Caracteristicas del CookbookEvaluator:**

| Aspecto | CookbookEvaluator |
|---|---|
| Dataset source | JSON/JSONL local |
| Metrica principal | **Pass@k** (=Recall@k) |
| Generacion LLM | **No** (solo retrieval) |
| Evaluacion | **contenido exacto** + doc_id |
| Documento padre | **Siempre** (Mode A) |
| Estrategias | **4** (SIMPLE_VECTOR, CONTEXTUAL_VECTOR, CONTEXTUAL_HYBRID, CONTEXTUAL_HYBRID_RERANK) |

**Seleccion de estrategia:**

```python
def _create_retriever(self):
    if strategy == "SIMPLE_VECTOR":
        return SimpleVectorRetriever(...)

    elif strategy == "CONTEXTUAL_VECTOR":
        # Enrichment + busqueda vectorial pura (sin BM25)
        context_gen = LLMContextGenerator(llm, mode="document")
        inner = SimpleVectorRetriever(...)
        return ContextualRetriever(..., inner_retriever=inner)

    elif strategy in ("CONTEXTUAL_HYBRID", "CONTEXTUAL_HYBRID_RERANK"):
        # Enrichment + BM25 + Vector + RRF
        context_gen = LLMContextGenerator(llm, mode="document")
        return ContextualRetriever(...)  # default inner = HybridRetriever
```

**CONTEXTUAL_VECTOR** permite medir el impacto aislado del enriquecimiento contextual sin BM25. Se logra pasando `inner_retriever=SimpleVectorRetriever(...)` a `ContextualRetriever`.

**Reranking como paso separado del retriever:**

```python
if self.config.strategy == "CONTEXTUAL_HYBRID_RERANK":
    k_oversample = k * self.config.rerank_oversample_factor
    result = retriever.retrieve(query, top_k=k_oversample)

    # El cookbook pasa original + contexto al reranker
    if self.config.rerank_content == "enriched":
        rerank_texts = result.enriched_contents
    elif self.config.rerank_content == "both":
        rerank_texts = [
            f"{orig}\n\nContext: {enr}"
            for orig, enr in zip(result.contents, result.enriched_contents)
        ]
    else:
        rerank_texts = result.contents

    reranked = self._reranker.rerank(query, result, top_n=self.config.rerank_top_n)
    return reranked
```

### 4.6 Pass@k como metrica

**Resolucion:** Pass@k = Recall@k algebraicamente. No se crea clase `RetrievalMetrics` nueva.

**Metrica primaria (ID-based):** Se reutiliza `QueryRetrievalDetail.recall_at_k` existente. El CookbookLoader mapea `golden_chunk_uuids` a los mismos `doc_id` compuestos usados en el corpus (`uuid__chunk_index`). El set intersection de `recall_at_k` produce el mismo resultado que Pass@k.

**Validacion (content-based):** Como aseguramiento de calidad, el CookbookEvaluator ejecuta una validacion content-based en su pipeline para confirmar que el mapeo de IDs es correcto. Esta validacion NO es una metrica separada — es un assertion que aborta si detecta divergencia:

```python
# En CookbookEvaluator._validate_pass_at_k():
def _validate_pass_at_k(self, retrieval: QueryRetrievalDetail,
                         golden_contents: List[str], k: int) -> None:
    """Assertion: ID-matching y content-matching producen mismo resultado."""
    id_based = retrieval.recall_at_k.get(k, 0.0)
    # Content-based check
    found = sum(
        1 for gc in golden_contents
        if gc.strip() in {rc.strip() for rc in retrieval.retrieved_contents[:k]}
    )
    content_based = found / len(golden_contents) if golden_contents else 0.0
    assert abs(id_based - content_based) < 1e-9, (
        f"Pass@k diverge: id_based={id_based}, content_based={content_based}"
    )
```

En CSV se reporta como `pass_at_k` (renombre de `recall_at_k`).

### 4.7 Cache de contextos

Para evitar re-generar contextos LLM en cada run:

```json
{
    "model": "nvidia/nemotron-3-nano",
    "prompt_hash": "sha256_del_template",
    "contexts": {
        "doc_uuid_1__0": "This chunk describes the initial setup...",
        "doc_uuid_1__1": "This chunk covers error handling in the..."
    },
    "stats": {
        "total_generated": 737,
        "total_errors": 0,
        "generation_time_s": 312.5
    }
}
```

Invalidacion automatica si cambia modelo o prompt (hash mismatch).

### 4.8 Pesos RRF

El cookbook usa variante con pesos asimetricos y re-scoring:
```python
score += semantic_weight * (1 / (index + 1))
score += bm25_weight * (1 / (index + 1))
# Re-asigna scores 1/(new_rank+1) despues de sort
```

Formula clasica (alternativa):
```python
rrf_contribution = weight / (k + rank)   # k=60
```

**Decision:** Ambas formulas implementadas en `reciprocal_rank_fusion()` via parametro `formula`. Default `"cookbook"` para este sandbox.

---

## 5. Salida y metricas

### 5.1 Por query

```python
@dataclass
class CookbookQueryResult:
    query_id: str
    query_text: str
    pass_at_k: Dict[int, float]
    golden_chunk_ids: List[str]
    retrieved_doc_ids: List[str]
    retrieval_scores: List[float]
    retrieval_time_ms: float
    from_semantic_count: int = 0       # solo hybrid
    from_bm25_count: int = 0           # solo hybrid
    reranked: Optional[bool] = None
```

### 5.2 Agregadas

```python
{
    "pass_at_5": 0.8812,
    "pass_at_10": 0.9234,
    "pass_at_20": 0.9429,
    "avg_semantic_contribution": 0.576,
    "avg_bm25_contribution": 0.424,
    "strategy": "CONTEXTUAL_VECTOR",
    "total_chunks": 737,
    "total_queries": 248,
    "context_generation_time_s": 312.5,
    "context_cache_used": true,
}
```

### 5.3 CSV

**summary.csv:** run_id, strategy, embedding_model, pass_at_5, pass_at_10, pass_at_20, total_chunks, total_queries, execution_time_s

**detail.csv:** query_id, query_text, pass_at_5, pass_at_10, pass_at_20, n_golden, n_retrieved, retrieval_time_ms, from_semantic, from_bm25

**comparison.csv** (modo --compare-all): strategy, pass_at_5, pass_at_10, pass_at_20, failure_rate_reduction_vs_baseline

Replica la tabla del cookbook:

| Approach | Pass@5 | Pass@10 | Pass@20 |
|---|---|---|---|
| Baseline RAG | 80.92% | 87.15% | 90.06% |
| + Contextual Embeddings | 88.12% | 92.34% | 94.29% |
| + Hybrid Search (BM25) | 86.43% | 93.21% | 94.99% |
| + Reranking | 92.15% | 95.26% | 97.45% |

---

## 6. Cambios realizados en shared/

### 6.1 shared/retrieval/core.py

- Estrategias `CONTEXTUAL_VECTOR` y `CONTEXTUAL_HYBRID_RERANK` agregadas al enum
- `enriched_contents: Optional[List[str]] = None` agregado a `RetrievalResult`

### 6.2 shared/retrieval/contextual_retriever.py

- Parametro `mode` en LLMContextGenerator (document/fallback)
- Error si mode=document sin parent_content
- Prompts parametrizables (Anthropic XML vs plain text)
- `context_position` parametrizable (prepend/append)
- `enriched_contents` expuesto en resultado

### 6.3 shared/retrieval/hybrid_retriever.py

- Parametro `formula` en `reciprocal_rank_fusion()` ("classic" | "cookbook")

### 6.4 shared/llm.py

- `batch_embed_queries()` extraido como funcion standalone

---

## 7. Decisiones tomadas

| Aspecto | Decision | Rationale |
|---|---|---|
| Modelo de embedding | NVIDIA NIM | Consistente con infra. Resultados no comparables con cookbook (Voyage AI), pero comparacion relativa entre estrategias es valida. |
| Reranker | NVIDIA NIM (NVIDIARerank) | Consistente con infra. Configurable via .env. |
| BM25 backend | Tantivy (fallback rank-bm25) | Suficiente para 737 chunks. Texto enriquecido combina chunk + contexto. |
| Contenido para reranking | Parametrizable (`rerank_content`) | "enriched" (default), "original", o "both". |

---

## 8. Riesgos

| Riesgo | Mitigacion |
|---|---|
| Modelo nano genera contextos pobres vs Claude Haiku | Documentar delta. Parametrizar modelo. |
| Embeddings NIM vs Voyage dan scores incomparables | Valor en comparacion relativa entre estrategias, no absoluta. |
| Tantivy tokenization vs Elasticsearch | Marginal para 737 chunks. Documentar. |
| Prompts XML tags no efectivos con modelos no-Claude | Prompt adaptado como opcion. Parametrizar template. |
| Cache contextos invalido tras cambio modelo/prompt | Hash de modelo+prompt. Invalidacion automatica. |

---

## 9. Plan de implementacion por fases

### Fase 1: Infraestructura base (no requiere LLM)
- CookbookConfig, CookbookLoader, CookbookEvaluator (solo SIMPLE_VECTOR)
- Entry point, tests del loader y Pass@k
- Ejecutar baseline, verificar Pass@k ~80-87%

### Fase 2: Contextual Embeddings (requiere LLM)
- Refactor contextual_retriever.py con Mode A
- Cache de contextos
- Estrategia CONTEXTUAL_VECTOR
- Tests Mode A, ejecutar, verificar mejora

### Fase 3: Contextual BM25 Hybrid
- Estrategia CONTEXTUAL_HYBRID, pesos RRF configurables
- Verificar BM25 indexa texto enriquecido
- Ejecutar, verificar mejora adicional

### Fase 4: Reranking
- Estrategia CONTEXTUAL_HYBRID_RERANK con over-sampling
- Contenido para reranker: original + contexto
- Ejecutar, verificar mejora final

### Fase 5: Comparacion y reporte
- Modo --compare-all
- Generacion comparison.csv
- Documentacion resultados vs cookbook