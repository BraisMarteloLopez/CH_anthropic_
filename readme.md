# Evaluacion del Diseno: sandbox_cookbook (Contextual Retrieval - Anthropic)

## Objetivo

Evaluar la viabilidad, coherencia arquitectonica y completitud del diseno propuesto en `DESIGN_sandbox_cookbook.md` para implementar un nuevo sandbox de evaluacion (`sandbox_cookbook`) que replique la propuesta de **Contextual Retrieval de Anthropic** (blog + cookbook), reutilizando la infraestructura existente del sandbox evaluador `_inspired_sandbox_evaluator` (RAG_P v3.2).

El sandbox propuesto debe permitir ejecutar y comparar **4 estrategias incrementales** de retrieval sobre un dataset de 9 codebases (737 chunks, 248 queries), midiendo el impacto de cada tecnica usando **exclusivamente las metricas del paper de Anthropic**.

---

## Analisis de Metricas: Pass@k es Recall@k

### Definicion formal

**Pass@k** segun el cookbook de Anthropic:
```python
# Por query:
query_score = chunks_found_in_top_k / len(golden_contents)
# Agregado:
pass_at_k = mean(query_scores) * 100
```

**Recall@k** ya implementado en `shared/types.py` (linea 298-302):
```python
# Por query:
recall_at_k = len(top_k_ids & expected_set) / n_relevant
# Agregado en evaluator._build_run():
avg_recall_at_k = mean(recall_at_k per query)
```

### Equivalencia algebraica

Ambas metricas computan exactamente lo mismo:

```
|items relevantes encontrados en top-k| / |total items relevantes|
```

promediado sobre todas las queries.

La **unica diferencia** es el metodo de matching:
- Pass@k: compara por **contenido textual** (strip + exact compare)
- Recall@k: compara por **doc_id** (set intersection)

Si el loader mapea correctamente `golden_chunk_uuids` a `relevant_doc_ids` (lo cual es responsabilidad del loader), ambas metricas producen **valores numericos identicos**. El propio diseno lo reconoce: *"ID-based como primary, content-based como validacion"*.

### Consecuencia

**No se necesita implementar una nueva clase `RetrievalMetrics` ni un metodo `pass_at_k()`.**

Se reutiliza `Recall@k` existente. El matching por contenido se implementa como **assertion de validacion** en el loader (verificar que cada golden_chunk_id tiene contenido coincidente en el corpus), no como metrica separada.

La metrica reportada se renombra a `pass_at_k` en los CSV de salida del sandbox_cookbook, pero el calculo subyacente es `recall_at_k`.

### Metrica derivada: Failure Rate

El blog de Anthropic reporta resultados como **failure rate** = `1 - pass_at_k`. Esto es exactamente `complement_recall_at_k`, que ya existe en `EvaluationRun` (campo `retrieval_complement_recall_at_k`).

**Tampoco se necesita implementar failure rate.** Ya existe.

---

## Metricas: Solo las del Paper de Anthropic

### Principio de simplificacion

**Solo se recojen las metricas propuestas en el blog y cookbook de Anthropic.** Todo lo demas se descarta. El sandbox_cookbook no es un benchmark general — es una replica del experimento de Anthropic.

### Metricas a RECOGER (exhaustivo)

| Metrica | Fuente Anthropic | k values | Ya existe en RAG_P | Accion |
|---|---|---|---|---|
| **Pass@k** (= Recall@k) | Cookbook, tabla principal | 5, 10, 20 | `recall_at_k` en `QueryRetrievalDetail` | Reutilizar. Renombrar en CSV a `pass_at_k` |
| **Failure Rate** (= 1 - Pass@k) | Blog, graficos principales | 20 | `complement_recall_at_k` en `EvaluationRun` | Reutilizar. Renombrar en CSV a `failure_rate` |
| **Semantic %** | Cookbook, diagnostico hybrid | - | `vector_scores` en `RetrievalResult` | Derivar de metadata RRF |
| **BM25 %** | Cookbook, diagnostico hybrid | - | `bm25_scores` en `RetrievalResult` | Derivar de metadata RRF |

### Metricas a DESCARTAR (no usadas por Anthropic)

| Metrica | Existe en RAG_P | Razon de descarte |
|---|---|---|
| Hit@k | `hit_at_k` en QueryRetrievalDetail | Anthropic no la usa. Es una version binaria inferior a Recall@k |
| MRR | `mrr` en QueryRetrievalDetail | No aparece en blog ni cookbook |
| NDCG@k | `ndcg_at_k` en QueryRetrievalDetail | No aparece en blog ni cookbook |
| F1 Score | `ReferenceBasedMetrics.f1_score` | Metrica de generacion. No hay generacion en este sandbox |
| Exact Match | `ReferenceBasedMetrics.exact_match` | Metrica de generacion. Descartada |
| Accuracy | `ReferenceBasedMetrics.accuracy` | Metrica de generacion. Descartada |
| Faithfulness | `LLMJudgeMetrics.faithfulness` | LLM-judge. No relevante. Descartada |
| Answer Relevance | `LLMJudgeMetrics.answer_relevance` | LLM-judge. Descartada |
| Context Utilization | `LLMJudgeMetrics.context_utilization` | LLM-judge. Descartada |
| Semantic Similarity | `ReferenceBasedMetrics.semantic_similarity` | No en Anthropic. Descartada |
| Generation Recall/Hit | `generation_recall/hit` | Post-rerank generation. No aplica |

### Impacto en implementacion

Esta decision **simplifica drasticamente** el evaluator:

1. **Sin pipeline de generacion:** No hay LLM de generacion, no hay `_execute_generation_async()`, no hay `GenerationResult`
2. **Sin LLM-judge:** No hay metricas sin referencia, no hay `MetricsCalculator`
3. **Sin Hit@k, MRR, NDCG@k:** No se calculan ni reportan (ya se calculan automaticamente en `QueryRetrievalDetail.__post_init__` pero se ignoran en la salida)
4. **K values fijos:** Solo k=5, 10, 20 (no la lista completa [1, 3, 5, 10, 20] del sandbox_mteb)
5. **CSV minimalista:** Solo las columnas que Anthropic reporta

---

## Analisis Estructural: Dataset HotpotQA vs Dataset Cookbook

El sandbox existente fue disenado para HotpotQA. El cookbook de Anthropic usa un dataset con estructura **fundamentalmente diferente**. Este analisis detalla las incompatibilidades y donde el diseno actual del loader/evaluator no puede reutilizarse directamente.

### Estructura de datos: comparacion lado a lado

```
HOTPOTQA (sandbox_mteb)                    COOKBOOK (sandbox_cookbook)
========================                    ========================

FUENTE: MinIO (remoto)                     FUENTE: JSON local (2 archivos)
  s3://bucket/hotpotqa/                      sandbox_cookbook/data/
    queries.parquet                            codebase_chunks.json
    corpus.parquet                             evaluation_set.jsonl
    qrels.parquet
    metadata.json

CORPUS: plano (1 nivel)                    CORPUS: jerarquico (2 niveles)
  66576 documentos Wikipedia                 9 documentos padre
  Cada doc = 1 pasaje independiente           -> 737 chunks hijos
  Sin relacion padre-hijo                    Cada chunk pertenece a un documento
  Sin documento completo                     Documento completo disponible (WHOLE_DOCUMENT)

QUERIES: con respuesta textual             QUERIES: sin respuesta textual
  query_id, text, answer, answer_type        query, golden_chunk_uuids, golden_documents
  answer_type: "text" | "label"              No hay answer text
  Evaluacion: generacion + F1/EM             Evaluacion: solo retrieval (Pass@k)

RELEVANCIA: archivo separado (qrels)       RELEVANCIA: embebida en cada query
  qrels.parquet: query_id -> doc_id          golden_chunk_uuids: [[doc_uuid, chunk_index], ...]
  Relacion simple: 1 query -> N doc_ids      golden_documents: contenido completo para validacion
  IDs directos                               IDs compuestos (doc_uuid + chunk_index)
```

### Incompatibilidades especificas con el loader existente

| Aspecto | MinIOLoader (actual) | CookbookLoader (necesario) | Reutilizable? |
|---|---|---|---|
| **Fuente de datos** | `boto3.client.get_object()` -> Parquet -> DataFrame | `open().read()` -> JSON/JSONL | NO. Totalmente diferente |
| **Parsing corpus** | `corpus_df.iterrows()` -> 1 doc = 1 fila plana | JSON con estructura jerarquica: doc -> chunks[] | NO. Estructura anidada vs plana |
| **Parsing queries** | `queries_df.iterrows()` con campos answer, answer_type, level | JSONL con campos query, golden_chunk_uuids, golden_documents | NO. Campos diferentes |
| **Mapping relevancia** | qrels.parquet separado: `{query_id: [doc_ids]}` | Embebido en query: `golden_chunk_uuids: [[uuid, idx], ...]` | NO. No hay archivo qrels |
| **Generacion doc_id** | doc_id viene directo del Parquet | doc_id **se debe construir**: `f"{doc_uuid}__{chunk_index}"` | NO. Logica de ID compuesto nueva |
| **Parent document** | No existe. `NormalizedDocument` no tiene parent | Campo `content` a nivel documento (WHOLE_DOCUMENT) | NO. Concepto inexistente en HotpotQA |
| **Cache** | Parquet local en disco (misma estructura) | No necesario (archivos ya son locales) | NO. Sin cache |
| **Validacion** | `check_connection()` a MinIO | Verificar archivos existen + golden chunks en corpus | NO. Validacion diferente |

### Incompatibilidades con el evaluator existente

| Aspecto | MTEBEvaluator (actual) | CookbookEvaluator (necesario) | Reutilizable? |
|---|---|---|---|
| **`_load_dataset()`** | `MinIOLoader(config.storage).load_dataset()` | `CookbookLoader(config.dataset_path).load()` | NO |
| **`_select_subset_dev()`** | Shuffle + gold docs garantizados + distractores | No aplica. Dataset completo siempre (737 chunks, trivial) | NO. Innecesario |
| **`_index_documents()`** | `doc.get_full_text()` = titulo + contenido | Necesita pasar **parent_content** ademas de content y title | PARCIAL. Falta parent_content |
| **`_evaluate_queries()`** | Retrieval + generacion async + metricas F1/EM/Faithfulness | Solo retrieval + Recall@k. Sin generacion | NO. Pipeline completamente diferente |
| **`_build_run()`** | Agrega Hit@k, MRR, NDCG@k, generation scores, reranker stats | Solo Recall@k (=Pass@k) para k=5,10,20 | PARCIAL. Logica de agregacion reutilizable pero campos diferentes |
| **`_batch_embed_queries()`** | REST batch via urllib | Reutilizable directamente | SI |
| **`_execute_retrieval()`** | Loop sync con retriever.retrieve_by_vector() | Reutilizable directamente | SI |
| **Reranking** | `CrossEncoderReranker` integrado en evaluate_queries | Necesita acceso a `enriched_contents` para reranker | PARCIAL |
| **Export** | `RunExporter.export()` -> JSON + 2 CSVs con metricas MTEB | CSV minimalista solo con Pass@k y Failure Rate | NO. Columnas diferentes |

### Lo que SI se reutiliza (shared/)

| Componente | Como se reutiliza |
|---|---|
| `NormalizedDocument` | Cada chunk = 1 NormalizedDocument. parent_doc_id en metadata |
| `NormalizedQuery` | query_text + relevant_doc_ids (golden_chunk_ids). expected_answer = None |
| `LoadedDataset` | Contenedor. metadata["parent_documents"] para WHOLE_DOCUMENT |
| `QueryRetrievalDetail` | Calcula recall_at_k automaticamente en __post_init__ (= Pass@k) |
| `EvaluationRun` | Contenedor de resultado. avg_recall_at_k = Pass@k |
| `SimpleVectorRetriever` | Busqueda vectorial pura. Sin cambios |
| `HybridRetriever` | BM25 + Vector + RRF. Formula RRF necesita extension |
| `ContextualRetriever` | Patron decorador. Necesita refactor para Mode A + prompts |
| `CrossEncoderReranker` | Reranking cross-encoder. Sin cambios |
| `ChromaVectorStore` | Almacen vectorial. Sin cambios |
| `TantivyIndex` | BM25 index. Sin cambios |
| `load_embedding_model()` | Carga modelo NIM. Sin cambios |
| `AsyncLLMService` | Para generacion de contextos. Sin cambios |
| `run_sync()` | Ejecutar coroutines desde sync. Sin cambios |

### Conclusion

**El loader de sandbox_mteb no es reutilizable.** Se necesita un `CookbookLoader` escrito desde cero. Las razones:

1. **Fuente diferente:** JSON local vs MinIO/Parquet remoto
2. **Estructura jerarquica:** Documentos con chunks anidados vs documentos planos
3. **IDs compuestos:** `doc_uuid__chunk_index` vs doc_id directo
4. **Relevancia embebida:** golden_chunk_uuids en cada query vs archivo qrels separado
5. **Parent document:** Concepto nuevo, no existe en HotpotQA
6. **Sin respuesta textual:** No hay expected_answer, solo retrieval

**El evaluator tampoco es reutilizable directamente** por:

1. **Sin pipeline de generacion** (toda la mitad async del evaluator sobra)
2. **Indexacion necesita parent_content** (campo adicional por documento)
3. **Metricas diferentes** (solo Recall@k vs Hit@k+MRR+NDCG@k+F1+EM+Faithfulness)
4. **Sin DEV_MODE** (dataset trivial, siempre se usa completo)
5. **Reporte diferente** (CSV minimalista vs CSV con 30+ columnas)

**Lo que si se reutiliza** es toda la capa de `shared/`: tipos de datos, retrieval pipeline (SimpleVector, Hybrid, Contextual, Reranker), vector store, BM25, embedding, LLM service. Esto es ~70% del codigo util. El loader y evaluator son ~30% que debe reescribirse.

---

## Evaluacion del Diseno vs Sandbox Existente

### 1. Alineacion Arquitectonica

| Aspecto | Diseno (DESIGN_sandbox_cookbook.md) | Sandbox existente (RAG_P v3.2) | Evaluacion |
|---|---|---|---|
| **Patron sandbox** | `sandbox_cookbook/` con config, loader, evaluator, run | `sandbox_mteb/` con config, loader, evaluator, run | ALINEADO |
| **Libreria compartida** | Reutiliza `shared/` con cambios backwards-compatible | `shared/` con types, metrics, llm, retrieval/, report | ALINEADO |
| **Tipos de datos** | `NormalizedDocument`, `NormalizedQuery`, `LoadedDataset` | Mismos tipos en `shared/types.py` | ALINEADO |
| **Retriever pattern** | Patron decorador: `ContextualRetriever` wraps inner | Mismo patron en `contextual_retriever.py` | ALINEADO |

### 2. Brechas Identificadas (revisadas tras simplificacion)

#### 2.1 Estrategias de Retrieval (BRECHA CRITICA)

`RetrievalStrategy` en `shared/retrieval/core.py` solo tiene `SIMPLE_VECTOR` y `CONTEXTUAL_HYBRID`. Faltan `CONTEXTUAL_VECTOR` y `CONTEXTUAL_HYBRID_RERANK`.

#### 2.2 ~~Metrica Pass@k~~ (ELIMINADA — ya existe como Recall@k)

~~Brecha eliminada.~~ Pass@k = Recall@k. No se necesita implementacion nueva. Solo renombrar en CSV de salida.

#### 2.3 LLMContextGenerator - Mode A Obligatorio (BRECHA IMPORTANTE)

Refactor necesario para: mode parametrizable, prompts XML Anthropic, context_position configurable, exponer enriched_contents.

#### 2.4 RetrievalResult - enriched_contents (BRECHA MODERADA)

Agregar campo `enriched_contents: Optional[List[str]] = None`.

#### 2.5 Formula RRF Cookbook (BRECHA MODERADA)

Agregar formula alternativa "cookbook": `score = weight * (1 / (rank + 1))` con re-scoring post-sort.

#### 2.6 Dataset Local (BRECHA MODERADA)

Loader nuevo para JSON local. No reutiliza MinIOLoader.

#### 2.7 Cache Persistente de Contextos (BRECHA MENOR)

Extender cache in-memory a persistencia JSON en disco.

### 3. Decisiones Pendientes

| Decision | Recomendacion |
|---|---|
| Modelo embedding | NIM primero. Voyage como extension futura |
| Reranker | NIM. Consistente con infra existente |
| BM25 backend | Tantivy. 737 chunks es trivial |
| Contenido reranking | Original + contexto (como cookbook). Parametrizable |

**Veredicto: APROBADO para implementacion.**

---

## Plan de Trabajo Detallado

### Fase 0: Preparacion
**Objetivo:** Entorno estable antes de cambios.

| # | Tarea | Criterio de aceptacion |
|---|---|---|
| 0.1 | Verificar 162 tests existentes pasan | `pytest tests/` sin fallos |
| 0.2 | Obtener dataset (codebase_chunks.json + evaluation_set.jsonl) | Archivos parseables en `sandbox_cookbook/data/` |

### Fase 1: Infraestructura Base (SIMPLE_VECTOR) — Sin LLM
**Objetivo:** Baseline de retrieval vectorial puro. Verificar Pass@k ~80-87%.

| # | Tarea | Archivos | Detalle |
|---|---|---|---|
| 1.1 | **Agregar estrategias al enum** | `shared/retrieval/core.py` | `CONTEXTUAL_VECTOR = auto()`, `CONTEXTUAL_HYBRID_RERANK = auto()` |
| 1.2 | **Agregar enriched_contents a RetrievalResult** | `shared/retrieval/core.py` | `enriched_contents: Optional[List[str]] = None` |
| 1.3 | **CookbookConfig** | `sandbox_cookbook/config.py` | Dataclass con: InfraConfig, RetrievalConfig, RerankerConfig, dataset_path, eval_path, results_dir, strategy, contextualize params, eval_k_values=[5,10,20]. Constructor `from_env()` |
| 1.4 | **CookbookLoader** | `sandbox_cookbook/loader.py` | Lee JSON -> `LoadedDataset` con parent_documents en metadata. Lee JSONL -> queries con golden_chunk_ids en `relevant_doc_ids`. Validacion: cada golden chunk existe en corpus (assertion, no metrica separada) |
| 1.5 | **CookbookEvaluator (solo SIMPLE_VECTOR)** | `sandbox_cookbook/evaluator.py` | Pipeline: load -> index -> retrieve -> **Recall@k** (k=5,10,20) -> build_run. Sin generacion. Sin Hit@k/MRR/NDCG@k en salida. CSV con columnas: strategy, pass_at_5, pass_at_10, pass_at_20, failure_rate_at_20 |
| 1.6 | **Entry point** | `sandbox_cookbook/run.py` | `--strategy`, `--dry-run`, `--env`, `-v` |
| 1.7 | **Tests loader** | `tests/test_cookbook_loader.py` | Parseo JSON, golden chunks validados, parent_documents en metadata |
| 1.8 | **Verificar 162 tests originales pasan** | `tests/` | Sin regresiones en shared/ |

### Fase 2: Contextual Embeddings (CONTEXTUAL_VECTOR) — Requiere LLM
**Objetivo:** Enriquecimiento contextual con documento padre. Medir mejora vs baseline.

| # | Tarea | Archivos | Detalle |
|---|---|---|---|
| 2.1 | **Refactor LLMContextGenerator** | `shared/retrieval/contextual_retriever.py` | Parametros: `mode` ("document"/"fallback"), `document_prompt_template`, `chunk_prompt_template`, `context_position` ("prepend"/"append"). Error explicito si mode="document" sin parent_content. Defaults preservan comportamiento actual para sandbox_mteb |
| 2.2 | **Prompts Anthropic (XML tags)** | `shared/retrieval/contextual_retriever.py` | Constantes `ANTHROPIC_DOCUMENT_PROMPT` y `ANTHROPIC_CHUNK_PROMPT` con `<document>` y `<chunk>` tags. Seleccionables via template params |
| 2.3 | **Exponer enriched_contents** | `shared/retrieval/contextual_retriever.py` | En `_swap_to_original_contents()`: `result.enriched_contents = list(result.contents)` antes del swap |
| 2.4 | **Cache persistente** | `sandbox_cookbook/context_cache.py` | JSON en disco: {model, prompt_hash, contexts, stats}. Invalidacion por hash mismatch |
| 2.5 | **Factory CONTEXTUAL_VECTOR** | `shared/retrieval/__init__.py` | ContextualRetriever con inner=SimpleVectorRetriever |
| 2.6 | **Integrar en evaluator** | `sandbox_cookbook/evaluator.py` | Pasar parent_content desde LoadedDataset.metadata["parent_documents"] |
| 2.7 | **Tests Mode A** | `tests/test_contextual_mode_a.py` | Error sin parent, prompts XML, context_position, enriched_contents |
| 2.8 | **Verificar tests originales** | `tests/` | Sin regresiones |

### Fase 3: Hybrid Search (CONTEXTUAL_HYBRID)
**Objetivo:** BM25 sobre texto enriquecido + RRF. Medir mejora adicional.

| # | Tarea | Archivos | Detalle |
|---|---|---|---|
| 3.1 | **Formula RRF cookbook** | `shared/retrieval/hybrid_retriever.py` | Parametro `formula` en `reciprocal_rank_fusion()`. "cookbook": `weight * 1/(rank+1)` + re-scoring. Default "classic" para no romper sandbox_mteb |
| 3.2 | **Integrar en evaluator** | `sandbox_cookbook/evaluator.py` | CONTEXTUAL_HYBRID con pesos asimetricos (semantic=0.8, bm25=0.2) |
| 3.3 | **Tests RRF cookbook** | `tests/test_dtm4_rrf.py` (extension) | Formula "cookbook" produce ranking correcto |

### Fase 4: Reranking (CONTEXTUAL_HYBRID_RERANK)
**Objetivo:** Cross-encoder reranking post-retrieval. Mejora final.

| # | Tarea | Archivos | Detalle |
|---|---|---|---|
| 4.1 | **Reranking en evaluator** | `sandbox_cookbook/evaluator.py` | Over-sample k * factor, seleccionar contenido para reranker (enriched/original/both segun config), ejecutar CrossEncoderReranker, truncar a top_n |
| 4.2 | **Factory CONTEXTUAL_HYBRID_RERANK** | `shared/retrieval/__init__.py` | Misma creacion que CONTEXTUAL_HYBRID (reranking en evaluator, no en retriever) |

### Fase 5: Comparacion y Reporte
**Objetivo:** Tabla comparativa de las 4 estrategias, replicando la del cookbook.

| # | Tarea | Archivos | Detalle |
|---|---|---|---|
| 5.1 | **Modo --compare-all** | `sandbox_cookbook/run.py` | Ejecuta 4 estrategias secuencialmente |
| 5.2 | **comparison.csv** | `sandbox_cookbook/evaluator.py` | Columnas: strategy, pass_at_5, pass_at_10, pass_at_20, failure_rate_reduction_vs_baseline. Replica tabla del cookbook |

### Fase 6: Validacion Final
**Objetivo:** Calidad y cierre.

| # | Tarea | Detalle |
|---|---|---|
| 6.1 | Suite completa de tests | Todos pasan (existentes + nuevos) |
| 6.2 | Verificar mypy | Sin errores en modulos nuevos |
| 6.3 | Marcar DT-2 como resuelta | Mode A implementado |

---

## Resumen de Cambios (simplificado)

### Lo que SE IMPLEMENTA

- 2 estrategias nuevas en enum (`CONTEXTUAL_VECTOR`, `CONTEXTUAL_HYBRID_RERANK`)
- 1 campo nuevo en `RetrievalResult` (`enriched_contents`)
- Refactor `LLMContextGenerator` (mode, prompts, context_position)
- Formula RRF alternativa "cookbook"
- Sandbox completo: config, loader (JSON local), evaluator (4 estrategias), run
- Cache persistente de contextos
- CSV con Pass@k (=Recall@k), Failure Rate (=1-Recall@k), % semantic, % BM25

### Lo que NO SE IMPLEMENTA (descartado)

- ~~Clase RetrievalMetrics con pass_at_k()~~ — Recall@k ya existe
- ~~Pipeline de generacion LLM~~ — Anthropic no la usa en el cookbook
- ~~Metricas de generacion (F1, EM, Accuracy)~~ — No aplican
- ~~Metricas LLM-judge (Faithfulness, Answer Relevance, Context Utilization)~~ — No aplican
- ~~Hit@k, MRR, NDCG@k en salida~~ — Anthropic no las reporta
- ~~Soporte Voyage AI / Cohere~~ — NIM primero, extension futura si necesaria
- ~~Exporter generico extendido~~ — CSV minimalista directo en evaluator

---

## Fuentes

- [Anthropic Blog: Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [Anthropic Cookbook: Contextual Embeddings Guide](https://platform.claude.com/cookbook/capabilities-contextual-embeddings-guide)
- [Contextual Retrieval Appendix II (PDF)](https://assets.anthropic.com/m/1632cded0a125333/original/Contextual-Retrieval-Appendix-2.pdf)
