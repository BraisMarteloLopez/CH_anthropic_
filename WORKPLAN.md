# Plan de Trabajo: sandbox_cookbook

Estado: **FASE 5 COMPLETA** | Ultima actualizacion: 2026-02-25

---

## Estructura del repositorio

```
_inspired_sandbox_evaluator/
  shared/                    <- libreria comun (reutilizada)
  sandbox_mteb/              <- sandbox HotpotQA (referencia, intacto)
  sandbox_cookbook/           <- NUEVO: sandbox Contextual Retrieval
  tests/                     <- tests existentes (162) + nuevos cookbook
```

Ambos sandboxes conviven. `sandbox_mteb` se mantiene como referencia y red de tests para validar cambios backwards-compatible en `shared/`.

---

## Fases

### Fase 0: Preparacion
**Objetivo:** Entorno estable antes de cambios.

| # | Tarea | Criterio de aceptacion | Estado |
|---|---|---|---|
| 0.1 | Verificar 162 tests existentes pasan | `pytest tests/` sin fallos | **Hecho** |
| 0.2 | Obtener dataset del cookbook de Anthropic | Archivos parseables en `sandbox_cookbook/data/` | **Hecho** |
| 0.3 | **(PC-6)** Agregar `sandbox_cookbook/data/` a `.gitignore` | Archivos de datos no se comitean | **Hecho** |

**Detalle tarea 0.2 — Fuente del dataset:**
El dataset se obtiene del [cookbook de Anthropic](https://github.com/anthropics/anthropic-cookbook/tree/main/capabilities/contextual-embeddings/data). Archivos:
- `codebase_chunks.json` — 90 documentos, 737 chunks. Keys: `doc_id`, `original_uuid`, `content`, `chunks[]` (cada chunk: `chunk_id`, `original_index`, `content`). `chunk_id` sigue formato `doc_X_chunk_Y`.
- `evaluation_set.jsonl` — 248 queries, 306 golden chunks (28 queries con >1 golden). Keys: `query`, `answer`, `golden_doc_uuids`, `golden_chunk_uuids` (cada uno: `[uuid, chunk_index]`), `golden_documents`, `golden_chunks`, `meta`.

Se descargan una vez y se colocan en `sandbox_cookbook/data/`. No se usa MinIO ni Parquet — lectura directa de JSON/JSONL local. El mapeo `golden_chunk_uuids[uuid, idx]` -> `chunk_id` del corpus es directo via `original_uuid` del documento padre.

### Fase 1: Infraestructura Base (SIMPLE_VECTOR) — Sin LLM
**Objetivo:** Baseline de retrieval vectorial puro. Verificar Pass@k ~80-87%.

| # | Tarea | Archivos | Detalle | Estado |
|---|---|---|---|---|
| 1.1 | **Agregar estrategias al enum** | `shared/retrieval/core.py` | `CONTEXTUAL_VECTOR = auto()`, `CONTEXTUAL_HYBRID_RERANK = auto()` | **Hecho** |
| 1.2 | **Agregar enriched_contents a RetrievalResult** | `shared/retrieval/core.py` | `enriched_contents: Optional[List[str]] = None` | **Hecho** |
| 1.3 | **(PC-4)** **Extraer `batch_embed_queries()` a shared/** | `shared/llm.py` | Funcion standalone. MTEBEvaluator la invoca via import | **Hecho** |
| 1.4 | **CookbookConfig** | `sandbox_cookbook/config.py` | Dataclass con: InfraConfig, RetrievalConfig, RerankerConfig, dataset_path, eval_path, results_dir, strategy, contextualize params (incluyendo truncation limits), eval_k_values=[5,10,20]. Constructor `from_env()` | **Hecho** |
| 1.5 | **CookbookLoader** | `sandbox_cookbook/loader.py` | Lee JSON -> `LoadedDataset` con parent_documents en metadata. Lee JSONL -> queries con golden_chunk_ids en `relevant_doc_ids`. Validacion: cada golden chunk existe en corpus. **(PC-1)** NO asignar title a NormalizedDocument | **Hecho** |
| 1.6 | **CookbookEvaluator (solo SIMPLE_VECTOR)** | `sandbox_cookbook/evaluator.py` | Pipeline: load -> index con `doc.content` directo **(PC-1)** -> retrieve -> Recall@k -> build_run. **(PC-5)** CSV solo k=5,10,20. Sin generacion | **Hecho** |
| 1.7 | **Entry point** | `sandbox_cookbook/run.py` | `--strategy`, `--dry-run`, `--env`, `-v` | **Hecho** |
| 1.8 | **Tests loader** | `tests/test_cookbook_loader.py` | Parseo JSON, golden chunks validados, parent_documents en metadata, title=None (19 tests) | **Hecho** |
| 1.9 | **Tests config** | `tests/test_cookbook_config.py` | from_env(), validate(), summary(), eval_k_values default [5,10,20] (23 tests) | **Hecho** |
| 1.10 | **Tests evaluator (SIMPLE_VECTOR)** | `tests/test_cookbook_evaluator.py` | Pipeline completo con mocks: load→index→retrieve→evaluate→build_run. Validacion Pass@k content-based assertion (9 tests) | **Hecho** |
| 1.11 | **`__init__.py`** | `sandbox_cookbook/__init__.py` | Paquete Python valido | **Hecho** |
| 1.12 | **`env.example`** | `sandbox_cookbook/env.example` | Template con todas las variables de entorno documentadas | **Hecho** |
| 1.13 | **Registrar dataset en DATASET_CONFIG** | `shared/types.py` | Entrada `"cookbook"` en `DATASET_CONFIG` con `type=RETRIEVAL_ONLY`, `primary_metric=None`, sin generation | **Hecho** |
| 1.14 | **Verificar tests originales pasan** | `tests/` | 198 tests passed (147 originales + 51 nuevos), 0 fallos, sin regresiones | **Hecho** |

### Fase 2: Contextual Embeddings (CONTEXTUAL_VECTOR) — Requiere LLM
**Objetivo:** Enriquecimiento contextual con documento padre. Medir mejora vs baseline.

| # | Tarea | Archivos | Detalle | Estado |
|---|---|---|---|---|
| 2.1 | **(PC-2)** **Truncation limits configurables** | `shared/retrieval/contextual_retriever.py` | `max_parent_chars`, `max_chunk_chars` en constructor. Defaults 2000/1000 (backwards-compat). Cookbook: 32000/8000 | **Hecho** |
| 2.2 | **Refactor LLMContextGenerator** | `shared/retrieval/contextual_retriever.py` | mode, document_prompt_template, chunk_prompt_template, system_prompt, context_position, _build_enriched_text(). Defaults preservan sandbox_mteb | **Hecho** |
| 2.3 | **Prompts Anthropic (XML tags)** | `shared/retrieval/contextual_retriever.py` | ANTHROPIC_DOCUMENT_PROMPT, ANTHROPIC_CHUNK_PROMPT, ANTHROPIC_SYSTEM_PROMPT | **Hecho** |
| 2.4 | **(PC-3)** **Fix strategy hardcodeada** | `shared/retrieval/contextual_retriever.py` | `result.strategy_used = self.config.strategy` | **Hecho** |
| 2.5 | **Exponer enriched_contents** | `shared/retrieval/contextual_retriever.py` | enriched_contents guardado antes de swap, + mapa _enriched_contents | **Hecho** |
| 2.6 | **Cache persistente** | `sandbox_cookbook/context_cache.py` | ContextCache: JSON en disco, invalidacion por hash modelo+prompt | **Hecho** |
| 2.7 | **Factory CONTEXTUAL_VECTOR** | `shared/retrieval/__init__.py` | get_retriever soporta CONTEXTUAL_VECTOR (inner=SimpleVector) + CONTEXTUAL_HYBRID_RERANK + **context_kwargs | **Hecho** |
| 2.8 | **Integrar en evaluator** | `sandbox_cookbook/evaluator.py` | CONTEXTUAL_VECTOR: LLM init, parent_content, Anthropic prompts, cache load/save | **Hecho** |
| 2.9 | **Tests Mode A + fixes** | `tests/test_contextual_mode_a.py` | 26 tests: truncation, mode, XML prompts, context_position, strategy (PC-3), enriched_contents, ContextCache | **Hecho** |
| 2.10 | **Verificar tests** | `tests/` | 224 passed (147 originales + 77 nuevos), 0 fallos, sin regresiones | **Hecho** |

### Fase 3: Hybrid Search (CONTEXTUAL_HYBRID)
**Objetivo:** BM25 sobre texto enriquecido + RRF. Medir mejora adicional.

| # | Tarea | Archivos | Detalle | Estado |
|---|---|---|---|---|
| 3.1 | **Formula RRF cookbook** | `shared/retrieval/hybrid_retriever.py` | Parametro `formula` en `reciprocal_rank_fusion()`. Default "classic" | **Hecho** |
| 3.2 | **Integrar en evaluator** | `sandbox_cookbook/evaluator.py` | Pesos asimetricos (semantic=0.8, bm25=0.2) | **Hecho** |
| 3.3 | **Tests RRF cookbook** | `tests/test_dtm4_rrf.py` (extension) | Formula "cookbook" produce ranking correcto (9 tests) + 3 tests evaluator hybrid | **Hecho** |

### Fase 4: Reranking (CONTEXTUAL_HYBRID_RERANK)
**Objetivo:** Cross-encoder reranking post-retrieval. Mejora final.

| # | Tarea | Archivos | Detalle | Estado |
|---|---|---|---|---|
| 4.1 | **Reranking en evaluator** | `sandbox_cookbook/evaluator.py` | Over-sample + seleccion contenido + CrossEncoderReranker + truncar | **Hecho** |
| 4.2 | **Factory CONTEXTUAL_HYBRID_RERANK** | `shared/retrieval/__init__.py` | Misma creacion que CONTEXTUAL_HYBRID (reranking en evaluator) | **Hecho** |

### Fase 5: Comparacion y Reporte
**Objetivo:** Tabla comparativa de las 4 estrategias, replicando la del cookbook.

| # | Tarea | Archivos | Detalle | Estado |
|---|---|---|---|---|
| 5.1 | **Modo --compare-all** | `sandbox_cookbook/run.py` | Ejecuta 4 estrategias secuencialmente. Cache persistente (tarea 2.6) compartido entre estrategias: contextos generados en CONTEXTUAL_VECTOR se reutilizan en CONTEXTUAL_HYBRID y CONTEXTUAL_HYBRID_RERANK sin regenerar | **Hecho** |
| 5.2 | **comparison.csv** | `sandbox_cookbook/evaluator.py` | strategy, pass_at_5/10/20, failure_rate_reduction_vs_baseline | **Hecho** |

### Fase 6: Validacion Final
**Objetivo:** Calidad y cierre.

| # | Tarea | Detalle | Estado |
|---|---|---|---|
| 6.1 | Suite completa de tests | Todos pasan (existentes + nuevos) | Pendiente |
| 6.2 | Verificar mypy | Sin errores en modulos nuevos | Pendiente |
| 6.3 | Marcar DT-2 como resuelta | Mode A implementado | Pendiente |

---

## Resumen de Cambios

### Se implementa

- 2 estrategias nuevas en enum (`CONTEXTUAL_VECTOR`, `CONTEXTUAL_HYBRID_RERANK`)
- 1 campo nuevo en `RetrievalResult` (`enriched_contents`)
- Refactor `LLMContextGenerator` (mode, prompts, context_position, truncation limits configurables)
- Fix bug strategy hardcodeada (PC-3)
- Extraer `batch_embed_queries()` a shared/llm.py (PC-4)
- Formula RRF alternativa "cookbook"
- Sandbox completo: config, loader (JSON local, sin title), evaluator (4 estrategias, indexa sin `get_full_text()`), run
- Cache persistente de contextos
- CSV con Pass@k (=Recall@k) solo para k=5,10,20, Failure Rate, % semantic, % BM25

### No se implementa (descartado)

- Clase `RetrievalMetrics` como modulo independiente — Recall@k ya existe en `QueryRetrievalDetail`. Pass@k = Recall@k algebraicamente. Content-based matching se implementa como assertion de validacion dentro del CookbookEvaluator (no como metrica separada). Ver DESIGN seccion 4.6
- Pipeline de generacion LLM — Anthropic no la usa en el cookbook
- Metricas de generacion (F1, EM, Accuracy) — No aplican
- Metricas LLM-judge (Faithfulness, Answer Relevance, Context Utilization) — No aplican
- Hit@k, MRR, NDCG@k en salida — Anthropic no las reporta
- Soporte Voyage AI / Cohere — NIM primero
- Exporter generico extendido — CSV minimalista directo en evaluator
- Title en NormalizedDocument para chunks de codigo — Contamina embeddings (PC-1)
- Truncamiento fijo 2000/1000 chars — Destruye approach Anthropic (PC-2)
- Sistema de carga MinIO/Parquet para cookbook — Se replica el sistema del cookbook Anthropic (JSON/JSONL local)
