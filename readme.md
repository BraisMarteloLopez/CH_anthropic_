# sandbox_cookbook — Contextual Retrieval (Anthropic)

Replica del experimento de **Contextual Retrieval** de Anthropic ([blog](https://www.anthropic.com/news/contextual-retrieval), [cookbook](https://platform.claude.com/cookbook/capabilities-contextual-embeddings-guide)) sobre la infraestructura de evaluacion RAG existente (RAG_P v3.2).

## Que hace

Ejecuta y compara **4 estrategias incrementales** de retrieval sobre un dataset de 90 documentos de codigo (737 chunks, 248 queries, 306 golden chunks):

| Estrategia | Descripcion | Pass@20 esperado |
|---|---|---|
| `SIMPLE_VECTOR` | Embedding cosine similarity | ~90% |
| `CONTEXTUAL_VECTOR` | Embeddings enriquecidos con contexto LLM | ~94% |
| `CONTEXTUAL_HYBRID` | Contextual embeddings + BM25 + RRF | ~95% |
| `CONTEXTUAL_HYBRID_RERANK` | Hybrid + cross-encoder reranking | ~97% |

Metricas: **Pass@k** (k=5, 10, 20) y **Failure Rate** (1 - Pass@k). Exclusivamente las del paper de Anthropic.

## Estructura del repositorio

```
_inspired_sandbox_evaluator/
  shared/                    <- libreria comun (types, retrieval, metrics, llm)
  sandbox_mteb/              <- sandbox HotpotQA (referencia + red de tests)
  sandbox_cookbook/           <- sandbox Contextual Retrieval (este proyecto)
  tests/                     <- 162 tests existentes + tests cookbook
DESIGN_sandbox_cookbook.md    <- diseno propuesto (input)
WORKPLAN.md                  <- plan de trabajo activo (fases, tareas, estado)
readme.md                    <- este archivo
```

Ambos sandboxes conviven. `sandbox_mteb` se mantiene como referencia y como red de tests para validar cambios backwards-compatible en `shared/`.

## Decisiones clave

### Dataset: replica exacta del cookbook de Anthropic

El sistema de carga de datos **NO** replica el patron MinIO/Parquet de sandbox_mteb. Se replica el sistema del cookbook de Anthropic:

- **Fuente:** Dos archivos del [cookbook de Anthropic](https://github.com/anthropics/anthropic-cookbook/tree/main/capabilities/contextual-embeddings/data): `codebase_chunks.json` (90 documentos, 737 chunks) y `evaluation_set.jsonl` (248 queries, 306 golden chunks). 28 queries tienen >1 golden chunk (Recall@k != Hit@k para esas queries).
- **Carga:** Lectura directa de JSON/JSONL local. Sin MinIO, sin Parquet, sin ETL.
- **Ubicacion:** `sandbox_cookbook/data/` (excluido de git via `.gitignore`).
- **CookbookLoader:** Lee JSON -> `LoadedDataset` con parent_documents en metadata. Lee JSONL -> queries con golden_chunk_ids mapeados a `relevant_doc_ids`. Valida que cada golden chunk existe en el corpus.

### Pass@k = Recall@k (con validacion por contenido)

Pass@k del cookbook y Recall@k del sandbox existente son **algebraicamente identicas**: `|relevantes en top-k| / |total relevantes|`. No se implementa clase `RetrievalMetrics` nueva.

- **Metrica primaria (ID-based):** Se reutiliza `recall_at_k` de `QueryRetrievalDetail`. El CookbookLoader mapea `golden_chunk_uuids` a los mismos `doc_id` compuestos del corpus (`uuid__chunk_index`). Se renombra a `pass_at_k` en CSV.
- **Validacion (content-based):** El CookbookEvaluator ejecuta un assertion que compara matching por ID vs matching por contenido exacto. Si divergen, aborta con error (indica bug en el mapeo de IDs del loader). No es una metrica separada, es una guardia de calidad.

### Solo metricas del paper

Se descartan: Hit@k, MRR, NDCG@k, F1, EM, Accuracy, Faithfulness, Answer Relevance, Context Utilization. No hay pipeline de generacion ni LLM-judge.

### Dataset incompatible con HotpotQA

El dataset del cookbook tiene estructura jerarquica (documento padre -> chunks) vs plana (HotpotQA). El loader y evaluator se escriben desde cero. La capa `shared/` (~70% del codigo util) se reutiliza intacta.

### context_position: parametrizable en LLMContextGenerator

El blog de Anthropic prepone el contexto (`context + chunk`), el cookbook lo apone (`chunk + context`). La posicion se parametriza en `LLMContextGenerator.__init__(context_position="prepend"|"append")`. La combinacion se resuelve en el generator via `_build_enriched_text()`. `EnrichedChunk.get_enriched_text()` se mantiene sin cambios (siempre "prepend") para backwards-compatibility con sandbox_mteb.

### Cache persistente compartido en --compare-all

El modo `--compare-all` ejecuta 4 estrategias secuencialmente. Las estrategias 2, 3 y 4 generan los mismos 737 contextos (mismo modelo, mismo prompt). El cache persistente en disco (JSON con invalidacion por hash modelo+prompt) se instancia una vez y se comparte entre las 3 estrategias contextuales, evitando regenerar contextos.

### Correcciones sobre el diseno original (puntos ciegos)

| ID | Problema | Solucion |
|---|---|---|
| PC-1 | `get_full_text()` antepone titulo a chunks de codigo, contaminando embeddings | Indexar con `doc.content` directo. No asignar title |
| PC-2 | Truncamiento parent a 2000 chars (6% del documento) destruye approach Anthropic | Limites configurables. Cookbook: 32000/8000 |
| PC-3 | `_swap_to_original_contents()` hardcodea strategy = CONTEXTUAL_HYBRID | Usar `self.config.strategy` |
| PC-4 | `_batch_embed_queries()` es privado de MTEBEvaluator | Extraer a `shared/llm.py` |
| PC-5 | `EVAL_K_VALUES` fija [1,3,5,10,20] | Filtrar a [5,10,20] en reporte cookbook |
| PC-6 | Dataset no debe comitearse | `.gitignore` para `sandbox_cookbook/data/` |

## Documentacion

- [`DESIGN_sandbox_cookbook.md`](DESIGN_sandbox_cookbook.md) — Diseno tecnico detallado (revisado, inconsistencias corregidas)
- [`WORKPLAN.md`](WORKPLAN.md) — Plan de trabajo con 6 fases, ~35 tareas, estado de cada una
- [`_inspired_sandbox_evaluator/README.md`](_inspired_sandbox_evaluator/README.md) — Documentacion del sistema RAG_P v3.2
- [`_inspired_sandbox_evaluator/README_TEST.md`](_inspired_sandbox_evaluator/README_TEST.md) — Infraestructura de tests

## Fuentes

- [Anthropic Blog: Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [Anthropic Cookbook: Contextual Embeddings Guide](https://platform.claude.com/cookbook/capabilities-contextual-embeddings-guide)
- [Contextual Retrieval Appendix II (PDF)](https://assets.anthropic.com/m/1632cded0a125333/original/Contextual-Retrieval-Appendix-2.pdf)
