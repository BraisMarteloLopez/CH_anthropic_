# Evaluacion del Diseno: sandbox_cookbook (Contextual Retrieval - Anthropic)

## Objetivo

Evaluar la viabilidad, coherencia arquitectonica y completitud del diseno propuesto en `DESIGN_sandbox_cookbook.md` para implementar un nuevo sandbox de evaluacion (`sandbox_cookbook`) que replique la propuesta de **Contextual Retrieval de Anthropic** (blog + cookbook), reutilizando la infraestructura existente del sandbox evaluador `_inspired_sandbox_evaluator` (RAG_P v3.2).

El sandbox propuesto debe permitir ejecutar y comparar **4 estrategias incrementales** de retrieval sobre un dataset de 9 codebases (737 chunks, 248 queries), midiendo el impacto de cada tecnica mediante la metrica **Pass@k**.

---

## Evaluacion del Diseno vs Sandbox Existente

### 1. Alineacion Arquitectonica

| Aspecto | Diseno (DESIGN_sandbox_cookbook.md) | Sandbox existente (RAG_P v3.2) | Evaluacion |
|---|---|---|---|
| **Patron sandbox** | `sandbox_cookbook/` con config, loader, evaluator, run | `sandbox_mteb/` con config, loader, evaluator, run | ALINEADO. Misma estructura de carpetas y separacion de responsabilidades. |
| **Libreria compartida** | Reutiliza `shared/` con cambios backwards-compatible | `shared/` con types, metrics, llm, retrieval/, report | ALINEADO. El diseno respeta la libreria compartida como nucleo. |
| **Tipos de datos** | `NormalizedDocument`, `NormalizedQuery`, `LoadedDataset` | Mismos tipos en `shared/types.py` | ALINEADO. Reutiliza tipos existentes sin romper contrato. |
| **Retriever pattern** | Patron decorador: `ContextualRetriever` wraps `SimpleVectorRetriever` o `HybridRetriever` | Mismo patron decorador ya implementado en `contextual_retriever.py` | ALINEADO. El diseno CONTEXTUAL_VECTOR propone pasar inner=SimpleVector, extension natural. |
| **Resultado final** | `EvaluationRun` comparable | `EvaluationRun` en `shared/types.py` | ALINEADO. Mismo contenedor de resultado. |

### 2. Brechas Identificadas entre Diseno y Sandbox

#### 2.1 Estrategias de Retrieval (BRECHA CRITICA)

**Estado actual:** `RetrievalStrategy` en `shared/retrieval/core.py` (linea 28-31) solo define:
- `SIMPLE_VECTOR`
- `CONTEXTUAL_HYBRID`

**Diseno requiere 4 estrategias:**
- `SIMPLE_VECTOR` (existe)
- `CONTEXTUAL_VECTOR` (NUEVA - enrichment + busqueda vectorial pura, sin BM25)
- `CONTEXTUAL_HYBRID` (existe pero necesita ajuste de formula RRF)
- `CONTEXTUAL_HYBRID_RERANK` (NUEVA - hybrid + cross-encoder reranking como paso separado)

**Impacto:** Se necesitan 2 nuevas entradas en el enum y soporte en la factory `get_retriever()` (linea 32-80 de `shared/retrieval/__init__.py`).

#### 2.2 Metrica Pass@k (BRECHA CRITICA)

**Estado actual:** `shared/metrics.py` solo tiene metricas de generacion (F1, EM, Accuracy, Faithfulness, etc.) y `QueryRetrievalDetail` calcula Hit@k, Recall@k, MRR, NDCG@k automaticamente.

**Diseno requiere:** `Pass@k` como metrica principal basada en **contenido exacto** (no doc_id). Esta metrica no existe en el sandbox actual.

**Impacto:** Nueva clase `RetrievalMetrics` en `shared/metrics.py` con metodo `pass_at_k()`.

#### 2.3 LLMContextGenerator - Mode A Obligatorio (BRECHA IMPORTANTE)

**Estado actual** (`shared/retrieval/contextual_retriever.py`, lineas 119-182):
- `_generate_one()` acepta `parent_content` opcionalmente
- Si no hay parent, usa fallback (Mode B: solo chunk + titulo)
- Trunca parent a 2000 chars (linea 145)
- Usa prompts en texto plano optimizados para modelos nano (lineas 73-86)

**Diseno requiere:**
- Mode A **obligatorio** para sandbox_cookbook (error explicito si falta parent)
- Prompts parametrizables (Anthropic XML tags vs plain text)
- `context_position` parametrizable ("prepend" vs "append")
- Exponer `enriched_contents` en `RetrievalResult` para reranking

**Impacto:** Refactor significativo del `LLMContextGenerator`. Se debe parametrizar sin romper sandbox_mteb existente.

#### 2.4 RetrievalResult - enriched_contents (BRECHA MODERADA)

**Estado actual** (`shared/retrieval/core.py`, linea 86-98): `RetrievalResult` no tiene campo `enriched_contents`.

**Diseno requiere:** `enriched_contents: Optional[List[str]] = None` para que el reranker pueda usar el texto enriquecido.

**Impacto:** Agregar campo opcional a dataclass. Backwards-compatible (default None).

#### 2.5 Formula RRF Cookbook vs Classic (BRECHA MODERADA)

**Estado actual** (`shared/retrieval/hybrid_retriever.py`, lineas 122-158):
```python
rrf_contribution = weight / (k + rank)  # Formula clasica
```

**Diseno requiere:** Formula alternativa "cookbook" con pesos asimetricos y re-scoring:
```python
score += semantic_weight * (1 / (index + 1))
score += bm25_weight * (1 / (index + 1))
# Re-asigna scores 1/(new_rank+1) despues de sort
```

**Impacto:** Parametro `formula` en `reciprocal_rank_fusion()`. Backwards-compatible con default "classic".

#### 2.6 Dataset Local vs MinIO (BRECHA MODERADA)

**Estado actual:** `sandbox_mteb/loader.py` descarga desde MinIO (Parquet).

**Diseno requiere:** `sandbox_cookbook/loader.py` carga JSON local (`codebase_chunks.json` + `evaluation_set.jsonl`). Estructura completamente diferente.

**Impacto:** Loader nuevo desde cero, pero reutiliza tipos `LoadedDataset`, `NormalizedDocument`, `NormalizedQuery`.

#### 2.7 Cache de Contextos Persistente (BRECHA MENOR)

**Estado actual:** `LLMContextGenerator` tiene cache in-memory (dict sha256 -> context, linea 112).

**Diseno requiere:** Cache persistente en JSON con invalidacion por hash de modelo+prompt.

**Impacto:** Extension del cache existente. Nuevo archivo JSON en disco.

#### 2.8 Report - Pass@k y comparison.csv (BRECHA MENOR)

**Estado actual:** `shared/report.py` exporta Hit@k, Recall@k, MRR, NDCG@k, generation scores.

**Diseno requiere:** Pass@k en summary/detail CSV + nuevo `comparison.csv` (modo --compare-all).

**Impacto:** Extension de `RunExporter` o exporter nuevo en sandbox_cookbook.

### 3. Riesgos Tecnicos Evaluados

| Riesgo | Severidad | Evaluacion |
|---|---|---|
| **Romper sandbox_mteb con cambios en shared/** | Alta | El diseno propone cambios backwards-compatible. Validar con tests existentes (162 tests). |
| **Modelo nano genera contextos pobres vs Claude Haiku** | Media | El diseno lo documenta correctamente. Parametrizar modelo mitiga. |
| **Embeddings NIM vs Voyage incomparables** | Media | El diseno acepta comparacion relativa. Decision correcta. |
| **Prompts XML no efectivos con modelos no-Claude** | Media | El diseno propone prompts parametrizables. Solucion adecuada. |
| **Truncamiento parent a 2000 chars insuficiente para docs largos** | Baja | Documentos del cookbook son codigo fuente (hasta 8K tokens). 2000 chars = ~500 tokens, puede perder contexto. Evaluar si aumentar. |
| **DT-2 (Mode A sin parent) queda resuelto** | Positivo | El diseno resuelve explicitamente DT-2 que estaba marcada como deuda tecnica abierta. |

### 4. Decisiones Pendientes (del diseno)

El diseno identifica correctamente 4 decisiones que requieren feedback:

| Decision | Recomendacion del diseno | Evaluacion |
|---|---|---|
| 7.1 Modelo embedding | C: Soportar ambos (NIM + Voyage) | Pragmatico. Implementar NIM primero, Voyage como extension futura. |
| 7.2 Reranker | C: Ambos via config | Correcto. NIM primero por consistencia con infra existente. |
| 7.3 BM25 backend | Mantener Tantivy | Correcto. 737 chunks es trivial para Tantivy. |
| 7.4 Contenido reranking | B como default (original + contexto) | Alineado con cookbook. Parametrizable es correcto. |

### 5. Calidad del Diseno - Resumen

| Criterio | Puntuacion | Justificacion |
|---|---|---|
| Completitud | 9/10 | Cubre todos los componentes necesarios. Falta detalle en manejo de errores del loader. |
| Coherencia arquitectonica | 10/10 | Respeta perfectamente la arquitectura existente. Patron sandbox replicado fielmente. |
| Backwards-compatibility | 9/10 | Todos los cambios en shared/ son backwards-compatible. Tests existentes deben pasar. |
| Viabilidad tecnica | 9/10 | Todas las propuestas son implementables con la infra existente. |
| Riesgos documentados | 8/10 | Buenos riesgos identificados. Falta analisis de performance con 737 chunks en Tantivy (trivial pero no documentado). |
| Claridad | 9/10 | Diseno muy bien estructurado y detallado. Ejemplos de codigo claros. |

**Veredicto: APROBADO para implementacion.**

---

## Plan de Trabajo Detallado

### Fase 0: Preparacion y Validacion del Entorno
**Objetivo:** Asegurar que el entorno base esta estable antes de introducir cambios.

| # | Tarea | Archivos | Criterio de aceptacion |
|---|---|---|---|
| 0.1 | Verificar que los 162 tests existentes pasan | `tests/` | `pytest tests/` sin fallos |
| 0.2 | Crear rama de trabajo | - | Branch `sandbox-cookbook` desde main |
| 0.3 | Obtener dataset del cookbook (codebase_chunks.json + evaluation_set.jsonl) | `sandbox_cookbook/data/` | Archivos presentes y parseables |

### Fase 1: Infraestructura Base (SIMPLE_VECTOR) — Sin LLM
**Objetivo:** Ejecutar baseline de retrieval vectorial puro sobre el dataset del cookbook.

| # | Tarea | Archivos afectados | Detalle |
|---|---|---|---|
| 1.1 | **Agregar estrategias al enum** | `shared/retrieval/core.py` | Agregar `CONTEXTUAL_VECTOR = auto()` y `CONTEXTUAL_HYBRID_RERANK = auto()` al enum `RetrievalStrategy` (linea 28-31) |
| 1.2 | **Agregar `enriched_contents` a RetrievalResult** | `shared/retrieval/core.py` | Campo opcional `enriched_contents: Optional[List[str]] = None` en dataclass (linea 86-98) |
| 1.3 | **Implementar metrica Pass@k** | `shared/metrics.py` | Nueva clase `RetrievalMetrics` con metodo estatico `pass_at_k(retrieved_contents, golden_contents, k) -> float`. Matching por contenido exacto (strip + compare) |
| 1.4 | **Crear `sandbox_cookbook/__init__.py`** | `sandbox_cookbook/__init__.py` | Archivo init del modulo |
| 1.5 | **Crear `sandbox_cookbook/config.py` (CookbookConfig)** | `sandbox_cookbook/config.py` | Dataclass con: InfraConfig, RetrievalConfig, RerankerConfig, dataset_path, eval_path, results_dir, strategy, contextualize_model, contextualize_max_tokens, context_position, contexts_cache_path, eval_k_values, pesos RRF, rerank params. Constructor `from_env()` |
| 1.6 | **Crear `sandbox_cookbook/loader.py` (CookbookLoader)** | `sandbox_cookbook/loader.py` | Clase que: (a) lee `codebase_chunks.json` -> `LoadedDataset` con parent_documents en metadata, (b) lee `evaluation_set.jsonl` -> `List[CookbookEvalQuery]`, (c) valida que golden chunks existen en corpus. Dataclass `CookbookEvalQuery` con query_id, query_text, golden_chunk_ids, golden_contents |
| 1.7 | **Crear `sandbox_cookbook/evaluator.py` (CookbookEvaluator) - Solo SIMPLE_VECTOR** | `sandbox_cookbook/evaluator.py` | Pipeline: load -> index -> retrieve -> evaluate Pass@k -> build_run. Solo estrategia SIMPLE_VECTOR inicialmente. Sin generacion LLM |
| 1.8 | **Crear `sandbox_cookbook/run.py` (entry point)** | `sandbox_cookbook/run.py` | Entry point con argparse: `--strategy`, `--dry-run`, `--env`, `-v`. Carga .env, construye config, ejecuta evaluator |
| 1.9 | **Crear `sandbox_cookbook/env.example`** | `sandbox_cookbook/env.example` | Plantilla con todas las variables de entorno documentadas |
| 1.10 | **Tests: loader y Pass@k** | `tests/test_cookbook_loader.py`, `tests/test_cookbook_pass_at_k.py` | Tests unitarios: (a) loader parsea JSON correctamente, golden chunks validados, (b) Pass@k calcula fraccion correcta, edge cases (vacio, sin match, match parcial) |
| 1.11 | **Verificar tests existentes no rompieron** | `tests/` | `pytest tests/` — los 162 tests originales deben seguir pasando |
| 1.12 | **Ejecutar baseline SIMPLE_VECTOR** | - | Verificar Pass@k ~80-87% (comparable con tabla del cookbook) |

### Fase 2: Contextual Embeddings (CONTEXTUAL_VECTOR) — Requiere LLM
**Objetivo:** Implementar enriquecimiento contextual con documento padre y medir mejora.

| # | Tarea | Archivos afectados | Detalle |
|---|---|---|---|
| 2.1 | **Refactor LLMContextGenerator: mode parametrizable** | `shared/retrieval/contextual_retriever.py` | Agregar parametro `mode: Literal["document", "fallback"] = "document"` al constructor. Error explicito si mode="document" sin parent_content. Mantener comportamiento actual como default para sandbox_mteb |
| 2.2 | **Prompts parametrizables (Anthropic XML vs plain text)** | `shared/retrieval/contextual_retriever.py` | Agregar `document_prompt_template` y `chunk_prompt_template` opcionales al constructor. Definir constantes `ANTHROPIC_DOCUMENT_PROMPT` (XML con `<document>` y `<chunk>`) y `ANTHROPIC_CHUNK_PROMPT`. Default: prompts existentes (plain text para nano) |
| 2.3 | **context_position parametrizable** | `shared/retrieval/contextual_retriever.py` | Parametro `context_position: str = "prepend"` en constructor o en `EnrichedChunk.get_enriched_text()`. "prepend" = contexto antes del chunk (blog Anthropic), "append" = despues (cookbook) |
| 2.4 | **Exponer enriched_contents en resultado** | `shared/retrieval/contextual_retriever.py` | En `_swap_to_original_contents()`, guardar contenidos enriquecidos en `result.enriched_contents` antes de hacer swap a originales |
| 2.5 | **Cache persistente de contextos** | `shared/retrieval/contextual_retriever.py` o `sandbox_cookbook/context_cache.py` | Cache JSON en disco: {model, prompt_hash, contexts: {doc_id: context}, stats}. Invalidacion si cambia modelo o prompt (hash mismatch). Carga al inicio, guardado al final |
| 2.6 | **Actualizar factory get_retriever() para CONTEXTUAL_VECTOR** | `shared/retrieval/__init__.py` | Nueva rama: si strategy == CONTEXTUAL_VECTOR, crear ContextualRetriever con inner=SimpleVectorRetriever (sin BM25) |
| 2.7 | **Integrar en CookbookEvaluator** | `sandbox_cookbook/evaluator.py` | Flujo documento padre: LoadedDataset.metadata["parent_documents"] -> pasar parent_content a cada documento en index_documents(). Seleccion de estrategia CONTEXTUAL_VECTOR |
| 2.8 | **Tests Mode A** | `tests/test_contextual_mode_a.py` | Tests: (a) mode="document" sin parent_content lanza ValueError, (b) prompts XML generan formato correcto, (c) context_position prepend/append, (d) enriched_contents expuestos |
| 2.9 | **Verificar tests existentes no rompieron** | `tests/` | 162 tests originales + nuevos deben pasar |
| 2.10 | **Ejecutar CONTEXTUAL_VECTOR** | - | Verificar mejora vs baseline (~88% vs ~81% en Pass@5) |

### Fase 3: Contextual BM25 Hybrid (CONTEXTUAL_HYBRID)
**Objetivo:** Agregar busqueda BM25 sobre texto enriquecido y fusionar via RRF.

| # | Tarea | Archivos afectados | Detalle |
|---|---|---|---|
| 3.1 | **Formula RRF configurable** | `shared/retrieval/hybrid_retriever.py` | Agregar parametro `formula: str = "classic"` a `reciprocal_rank_fusion()`. "classic": `weight / (k + rank)`. "cookbook": `semantic_weight * (1 / (index + 1))` con re-scoring post-sort |
| 3.2 | **Integrar en CookbookEvaluator** | `sandbox_cookbook/evaluator.py` | Seleccion CONTEXTUAL_HYBRID: ContextualRetriever con inner=HybridRetriever. Pesos RRF desde config (semantic_weight=0.8, bm25_weight=0.2). Verificar BM25 indexa texto enriquecido |
| 3.3 | **Tests RRF cookbook formula** | `tests/test_cookbook_rrf.py` o extension de `test_dtm4_rrf.py` | Tests: formula "cookbook" produce ranking diferente al "classic", pesos asimetricos funcionan |
| 3.4 | **Verificar tests existentes** | `tests/` | Todos los tests pasan |
| 3.5 | **Ejecutar CONTEXTUAL_HYBRID** | - | Verificar mejora adicional vs CONTEXTUAL_VECTOR |

### Fase 4: Reranking (CONTEXTUAL_HYBRID_RERANK)
**Objetivo:** Agregar cross-encoder reranking como paso post-retrieval.

| # | Tarea | Archivos afectados | Detalle |
|---|---|---|---|
| 4.1 | **Reranking como paso separado en evaluator** | `sandbox_cookbook/evaluator.py` | Si strategy == CONTEXTUAL_HYBRID_RERANK: (a) retrieve con k_oversample = k * rerank_oversample_factor, (b) seleccionar contenido para reranker segun config.rerank_content ("original"/"enriched"/"both"), (c) ejecutar CrossEncoderReranker.rerank(), (d) truncar a rerank_top_n |
| 4.2 | **Actualizar factory para CONTEXTUAL_HYBRID_RERANK** | `shared/retrieval/__init__.py` | Misma creacion que CONTEXTUAL_HYBRID (el reranking se aplica en el evaluator, no en el retriever) |
| 4.3 | **Tests reranking** | `tests/test_cookbook_rerank.py` | Tests: reranking reordena resultados, contenido enriched/original/both se pasa correctamente, oversample factor respetado |
| 4.4 | **Verificar tests existentes** | `tests/` | Todos los tests pasan |
| 4.5 | **Ejecutar CONTEXTUAL_HYBRID_RERANK** | - | Verificar mejora final (~92% Pass@5) |

### Fase 5: Comparacion y Reporte
**Objetivo:** Modo de comparacion multi-estrategia y documentacion de resultados.

| # | Tarea | Archivos afectados | Detalle |
|---|---|---|---|
| 5.1 | **Extender run.py con --compare-all** | `sandbox_cookbook/run.py` | Ejecuta las 4 estrategias secuencialmente, genera comparison.csv |
| 5.2 | **Generar comparison.csv** | `sandbox_cookbook/evaluator.py` o modulo nuevo | CSV con: strategy, pass_at_5, pass_at_10, pass_at_20, failure_rate_reduction_vs_baseline. Replica tabla del cookbook |
| 5.3 | **Extender summary/detail CSV para Pass@k** | `sandbox_cookbook/evaluator.py` | Summary: run_id, strategy, embedding_model, pass_at_5/10/20, total_chunks, total_queries, execution_time_s. Detail: query_id, query_text, pass_at_5/10/20, n_golden, n_retrieved, retrieval_time_ms, from_semantic, from_bm25 |
| 5.4 | **Documentar resultados vs cookbook** | `sandbox_cookbook/README.md` | Tabla comparativa de resultados obtenidos vs resultados publicados en cookbook Anthropic. Analisis de deltas (modelo, embeddings, tokenizer) |
| 5.5 | **Tests de comparacion** | `tests/test_cookbook_comparison.py` | Tests: comparison.csv tiene formato correcto, failure_rate_reduction calculada correctamente |

### Fase 6: Validacion Final y Cierre
**Objetivo:** Asegurar calidad y documentacion completa.

| # | Tarea | Archivos afectados | Detalle |
|---|---|---|---|
| 6.1 | **Ejecutar suite completa de tests** | `tests/` | Todos los tests (existentes + nuevos) pasan |
| 6.2 | **Verificar mypy** | Todos los archivos nuevos | `mypy` sin errores en modulos nuevos |
| 6.3 | **Actualizar README principal** | `_inspired_sandbox_evaluator/README.md` | Agregar seccion sobre sandbox_cookbook, estrategias nuevas, metricas Pass@k |
| 6.4 | **Marcar DT-2 como resuelta** | `_inspired_sandbox_evaluator/README.md` | DT-2 (Mode A sin parent) queda resuelta con sandbox_cookbook |

---

## Resumen de Archivos a Crear/Modificar

### Archivos NUEVOS (sandbox_cookbook/)

| Archivo | Lineas estimadas | Descripcion |
|---|---|---|
| `sandbox_cookbook/__init__.py` | ~5 | Init del modulo |
| `sandbox_cookbook/config.py` | ~100 | CookbookConfig dataclass |
| `sandbox_cookbook/loader.py` | ~150 | CookbookLoader + CookbookEvalQuery |
| `sandbox_cookbook/evaluator.py` | ~350 | CookbookEvaluator (4 estrategias) |
| `sandbox_cookbook/run.py` | ~80 | Entry point CLI |
| `sandbox_cookbook/env.example` | ~40 | Plantilla .env |

### Archivos MODIFICADOS en shared/ (backwards-compatible)

| Archivo | Cambio | Impacto en sandbox_mteb |
|---|---|---|
| `shared/retrieval/core.py` | +2 entries en enum, +1 campo en RetrievalResult | Ninguno (aditivo) |
| `shared/metrics.py` | +clase RetrievalMetrics con pass_at_k | Ninguno (clase nueva) |
| `shared/retrieval/contextual_retriever.py` | Parametros mode, prompts, context_position, enriched_contents | Ninguno si defaults preservados |
| `shared/retrieval/hybrid_retriever.py` | Parametro formula en reciprocal_rank_fusion() | Ninguno (default "classic") |
| `shared/retrieval/__init__.py` | Ramas CONTEXTUAL_VECTOR y CONTEXTUAL_HYBRID_RERANK en factory | Ninguno (ramas nuevas) |

### Tests NUEVOS

| Archivo | Tests estimados | Que cubre |
|---|---|---|
| `tests/test_cookbook_loader.py` | ~10 | Parseo JSON, validacion golden chunks |
| `tests/test_cookbook_pass_at_k.py` | ~12 | Metrica Pass@k, edge cases |
| `tests/test_contextual_mode_a.py` | ~8 | Mode A, prompts XML, context_position |
| `tests/test_cookbook_rrf.py` | ~5 | Formula RRF cookbook |
| `tests/test_cookbook_rerank.py` | ~5 | Reranking con contenido enriched/original |
| `tests/test_cookbook_comparison.py` | ~5 | Formato comparison.csv |

**Total estimado: ~45 tests nuevos** (ademas de los 162 existentes).

---

## Dependencias y Prerequisitos

1. **Dataset del cookbook:** Obtener `codebase_chunks.json` y `evaluation_set.jsonl` del repositorio de Anthropic
2. **Infraestructura NIM:** Endpoints de embedding, LLM y reranker accesibles (mismos que sandbox_mteb)
3. **Python 3.10+** con dependencias de `requirements.txt`
4. **Tantivy** instalado para BM25 (o rank-bm25 como fallback)

## Orden de Prioridad

1. **Fase 1** (base + SIMPLE_VECTOR): Establece fundacion sin riesgo de romper shared/
2. **Fase 2** (CONTEXTUAL_VECTOR): Cambio mas complejo en shared/ (refactor contextual_retriever)
3. **Fase 3** (CONTEXTUAL_HYBRID): Extension moderada (formula RRF)
4. **Fase 4** (CONTEXTUAL_HYBRID_RERANK): Extension menor (reranking en evaluator)
5. **Fase 5** (comparacion): Puramente de reporte, sin riesgo
6. **Fase 0/6** (preparacion/validacion): Bookends de calidad
