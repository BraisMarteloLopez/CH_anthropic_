# Plan de Trabajo: sandbox_cookbook

Estado: **PRE-IMPLEMENTACION** | Ultima actualizacion: 2026-02-25

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
| 0.1 | Verificar 162 tests existentes pasan | `pytest tests/` sin fallos | Pendiente |
| 0.2 | Obtener dataset del cookbook de Anthropic | Archivos parseables en `sandbox_cookbook/data/` | Pendiente |
| 0.3 | **(PC-6)** Agregar `sandbox_cookbook/data/` a `.gitignore` | Archivos de datos no se comitean | Pendiente |

**Detalle tarea 0.2 — Fuente del dataset:**
El dataset se obtiene del propio cookbook de Anthropic. Los archivos son `codebase_chunks.json` (9 codebases, 737 chunks) y `evaluation_set.jsonl` (248 queries con golden_chunk_uuids). Se descargan una vez y se colocan en `sandbox_cookbook/data/`. No se usa MinIO ni Parquet — el sistema de carga replica exactamente el del cookbook de Anthropic (lectura directa de JSON/JSONL local).

### Fase 1: Infraestructura Base (SIMPLE_VECTOR) — Sin LLM
**Objetivo:** Baseline de retrieval vectorial puro. Verificar Pass@k ~80-87%.

| # | Tarea | Archivos | Detalle | Estado |
|---|---|---|---|---|
| 1.1 | **Agregar estrategias al enum** | `shared/retrieval/core.py` | `CONTEXTUAL_VECTOR = auto()`, `CONTEXTUAL_HYBRID_RERANK = auto()` | Pendiente |
| 1.2 | **Agregar enriched_contents a RetrievalResult** | `shared/retrieval/core.py` | `enriched_contents: Optional[List[str]] = None` | Pendiente |
| 1.3 | **(PC-4)** **Extraer `batch_embed_queries()` a shared/** | `shared/llm.py` | Funcion standalone. MTEBEvaluator la invoca via import | Pendiente |
| 1.4 | **CookbookConfig** | `sandbox_cookbook/config.py` | Dataclass con: InfraConfig, RetrievalConfig, RerankerConfig, dataset_path, eval_path, results_dir, strategy, contextualize params (incluyendo truncation limits), eval_k_values=[5,10,20]. Constructor `from_env()` | Pendiente |
| 1.5 | **CookbookLoader** | `sandbox_cookbook/loader.py` | Lee JSON -> `LoadedDataset` con parent_documents en metadata. Lee JSONL -> queries con golden_chunk_ids en `relevant_doc_ids`. Validacion: cada golden chunk existe en corpus. **(PC-1)** NO asignar title a NormalizedDocument | Pendiente |
| 1.6 | **CookbookEvaluator (solo SIMPLE_VECTOR)** | `sandbox_cookbook/evaluator.py` | Pipeline: load -> index con `doc.content` directo **(PC-1)** -> retrieve -> Recall@k -> build_run. **(PC-5)** CSV solo k=5,10,20. Sin generacion | Pendiente |
| 1.7 | **Entry point** | `sandbox_cookbook/run.py` | `--strategy`, `--dry-run`, `--env`, `-v` | Pendiente |
| 1.8 | **Tests loader** | `tests/test_cookbook_loader.py` | Parseo JSON, golden chunks validados, parent_documents en metadata, title=None | Pendiente |
| 1.9 | **Tests config** | `tests/test_cookbook_config.py` | from_env(), validate(), summary(), eval_k_values default [5,10,20] | Pendiente |
| 1.10 | **Tests evaluator (SIMPLE_VECTOR)** | `tests/test_cookbook_evaluator.py` | Pipeline completo con mocks: load→index→retrieve→evaluate→build_run. Validacion Pass@k content-based assertion | Pendiente |
| 1.11 | **`__init__.py`** | `sandbox_cookbook/__init__.py` | Paquete Python valido | Pendiente |
| 1.12 | **`env.example`** | `sandbox_cookbook/env.example` | Template con todas las variables de entorno documentadas | Pendiente |
| 1.13 | **Registrar dataset en DATASET_CONFIG** | `shared/types.py` | Entrada `"cookbook"` en `DATASET_CONFIG` con `type=RETRIEVAL_ONLY`, `primary_metric=None`, sin generation | Pendiente |
| 1.14 | **Verificar 162 tests originales pasan** | `tests/` | Sin regresiones en shared/ | Pendiente |

### Fase 2: Contextual Embeddings (CONTEXTUAL_VECTOR) — Requiere LLM
**Objetivo:** Enriquecimiento contextual con documento padre. Medir mejora vs baseline.

| # | Tarea | Archivos | Detalle | Estado |
|---|---|---|---|---|
| 2.1 | **(PC-2)** **Truncation limits configurables** | `shared/retrieval/contextual_retriever.py` | `max_parent_chars: int = 2000`, `max_chunk_chars: int = 1000`. Cookbook: 32000/8000 | Pendiente |
| 2.2 | **Refactor LLMContextGenerator** | `shared/retrieval/contextual_retriever.py` | Parametros: `mode`, `document_prompt_template`, `chunk_prompt_template`, `context_position`. Defaults preservan sandbox_mteb. `context_position` controla combinacion en `_build_enriched_text()` del generator; `EnrichedChunk.get_enriched_text()` se mantiene (backwards-compat, siempre "prepend") | Pendiente |
| 2.3 | **Prompts Anthropic (XML tags)** | `shared/retrieval/contextual_retriever.py` | Constantes `ANTHROPIC_DOCUMENT_PROMPT` y `ANTHROPIC_CHUNK_PROMPT` con `<document>` y `<chunk>` tags | Pendiente |
| 2.4 | **(PC-3)** **Fix strategy hardcodeada** | `shared/retrieval/contextual_retriever.py` | `result.strategy_used = self.config.strategy` en vez de hardcode `CONTEXTUAL_HYBRID` | Pendiente |
| 2.5 | **Exponer enriched_contents** | `shared/retrieval/contextual_retriever.py` | `result.enriched_contents = list(result.contents)` antes del swap a originales | Pendiente |
| 2.6 | **Cache persistente** | `sandbox_cookbook/context_cache.py` | JSON en disco con invalidacion por hash modelo+prompt | Pendiente |
| 2.7 | **Factory CONTEXTUAL_VECTOR** | `shared/retrieval/__init__.py` | ContextualRetriever con inner=SimpleVectorRetriever | Pendiente |
| 2.8 | **Integrar en evaluator** | `sandbox_cookbook/evaluator.py` | Pasar parent_content desde LoadedDataset.metadata["parent_documents"] | Pendiente |
| 2.9 | **Tests Mode A + fixes** | `tests/test_contextual_mode_a.py` | Error sin parent, prompts XML, context_position, enriched_contents, strategy (PC-3), truncation (PC-2) | Pendiente |
| 2.10 | **Verificar tests originales** | `tests/` | Sin regresiones | Pendiente |

### Fase 3: Hybrid Search (CONTEXTUAL_HYBRID)
**Objetivo:** BM25 sobre texto enriquecido + RRF. Medir mejora adicional.

| # | Tarea | Archivos | Detalle | Estado |
|---|---|---|---|---|
| 3.1 | **Formula RRF cookbook** | `shared/retrieval/hybrid_retriever.py` | Parametro `formula` en `reciprocal_rank_fusion()`. Default "classic" | Pendiente |
| 3.2 | **Integrar en evaluator** | `sandbox_cookbook/evaluator.py` | Pesos asimetricos (semantic=0.8, bm25=0.2) | Pendiente |
| 3.3 | **Tests RRF cookbook** | `tests/test_dtm4_rrf.py` (extension) | Formula "cookbook" produce ranking correcto | Pendiente |

### Fase 4: Reranking (CONTEXTUAL_HYBRID_RERANK)
**Objetivo:** Cross-encoder reranking post-retrieval. Mejora final.

| # | Tarea | Archivos | Detalle | Estado |
|---|---|---|---|---|
| 4.1 | **Reranking en evaluator** | `sandbox_cookbook/evaluator.py` | Over-sample + seleccion contenido + CrossEncoderReranker + truncar | Pendiente |
| 4.2 | **Factory CONTEXTUAL_HYBRID_RERANK** | `shared/retrieval/__init__.py` | Misma creacion que CONTEXTUAL_HYBRID (reranking en evaluator) | Pendiente |

### Fase 5: Comparacion y Reporte
**Objetivo:** Tabla comparativa de las 4 estrategias, replicando la del cookbook.

| # | Tarea | Archivos | Detalle | Estado |
|---|---|---|---|---|
| 5.1 | **Modo --compare-all** | `sandbox_cookbook/run.py` | Ejecuta 4 estrategias secuencialmente. Cache persistente (tarea 2.6) compartido entre estrategias: contextos generados en CONTEXTUAL_VECTOR se reutilizan en CONTEXTUAL_HYBRID y CONTEXTUAL_HYBRID_RERANK sin regenerar | Pendiente |
| 5.2 | **comparison.csv** | `sandbox_cookbook/evaluator.py` | strategy, pass_at_5/10/20, failure_rate_reduction_vs_baseline | Pendiente |

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
