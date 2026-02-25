# RAG_P v3.2

Sistema de evaluacion RAG (Retrieval-Augmented Generation) con dos sandboxes independientes:

- **sandbox_mteb**: Benchmarking MTEB/BeIR (HotpotQA) con generacion LLM + metricas
- **sandbox_cookbook**: Replica del [Contextual Retrieval cookbook de Anthropic](https://github.com/anthropics/anthropic-cookbook/tree/main/capabilities/contextual-embeddings) — solo retrieval, 4 estrategias, Pass@k

Ambos comparten `shared/` para retrieval, embeddings, tipos y config.

## Arquitectura

```
RAG_P/
├── shared/                          # Libreria compartida
│   ├── types.py                     # NormalizedQuery, LoadedDataset, EvaluationRun, Protocols
│   ├── metrics.py                   # F1, ExactMatch, Accuracy, Faithfulness (LLM-judge) [solo mteb]
│   ├── llm.py                       # AsyncLLMService, load_embedding_model, batch_embed_queries
│   ├── config_base.py               # InfraConfig, RerankerConfig, helpers _env_*
│   ├── report.py                    # RunExporter: JSON + CSV summary + CSV detail [solo mteb]
│   ├── vector_store.py              # ChromaVectorStore [usado indirectamente via retrievers]
│   ├── structured_logging.py        # Logging JSONL estructurado [solo mteb]
│   └── retrieval/
│       ├── __init__.py              # Factory get_retriever() [solo mteb]
│       ├── core.py                  # BaseRetriever, SimpleVectorRetriever, RetrievalConfig
│       ├── hybrid_retriever.py      # BM25 + Vector + RRF (formulas classic + cookbook)
│       ├── contextual_retriever.py  # LLMContextGenerator (Mode A/B, Anthropic prompts)
│       ├── reranker.py              # CrossEncoderReranker (NVIDIARerank)
│       └── tantivy_index.py         # BM25 via Tantivy (Rust, fallback rank-bm25)
│
├── sandbox_mteb/                    # Evaluacion MTEB/BeIR (HotpotQA)
│   ├── config.py                    # MTEBConfig: .env -> dataclass validada
│   ├── loader.py                    # MinIO/Parquet -> LoadedDataset
│   ├── evaluator.py                 # Pipeline: pre-embed + retrieval + gen async
│   ├── run.py                       # Entry point (--dry-run, -v)
│   └── env.example                  # Plantilla .env
│
├── sandbox_cookbook/                 # Contextual Retrieval (Anthropic cookbook)
│   ├── config.py                    # CookbookConfig: .env -> dataclass validada
│   ├── loader.py                    # JSON/JSONL local -> LoadedDataset
│   ├── evaluator.py                 # Pipeline: 4 estrategias, Pass@k, comparison.csv
│   ├── context_cache.py             # Cache persistente de contextos LLM
│   ├── run.py                       # Entry point (--compare-all, --dry-run, -v)
│   └── env.example                  # Plantilla .env
│
├── tests/                           # 253 tests (pytest)
│   ├── conftest.py                  # Mocks condicionales (solo si paquete no instalado)
│   ├── test_cookbook_*.py            # 71 tests cookbook (config, loader, evaluator)
│   ├── test_contextual_mode_a.py    # 26 tests shared/contextual_retriever
│   ├── test_dtm4_rrf.py             # 20 tests RRF (classic + cookbook)
│   ├── test_dt*.py                  # ~90 tests mteb/shared
│   └── integration/                 # Tests contra NIM + MinIO reales
│
├── pyproject.toml                   # Config pytest
├── mypy.ini                         # Config mypy
└── requirements.txt
```

## Estrategias de retrieval

| Estrategia | Indexacion | Busqueda | Reranker | Sandbox |
|---|---|---|---|---|
| `SIMPLE_VECTOR` | Embedding directo | Cosine similarity (ChromaDB) | Opcional | ambos |
| `CONTEXTUAL_VECTOR` | Enrichment LLM + embedding | Cosine similarity | No | cookbook |
| `CONTEXTUAL_HYBRID` | Enrichment LLM + embedding | BM25 (Tantivy) + Vector + RRF | Opcional | ambos |
| `CONTEXTUAL_HYBRID_RERANK` | Enrichment LLM + embedding | BM25 + Vector + RRF + cross-encoder | Si | cookbook |

Enrichment contextual: cada chunk se enriquece con un contexto generado por LLM a partir del documento padre (Mode A, prompts XML Anthropic). El texto enriquecido se indexa; el contenido original se preserva para evaluacion.

## sandbox_cookbook (Contextual Retrieval)

Replica del experimento del [cookbook de Anthropic](https://github.com/anthropics/anthropic-cookbook/tree/main/capabilities/contextual-embeddings). Dataset: 90 documentos, 737 chunks, 248 queries.

```bash
python -m sandbox_cookbook.run                                    # Una estrategia
python -m sandbox_cookbook.run --strategy CONTEXTUAL_VECTOR       # Override
python -m sandbox_cookbook.run --compare-all                      # 4 estrategias + comparison.csv
python -m sandbox_cookbook.run --compare-all --dry-run            # Validar configs
```

`--compare-all` ejecuta las 4 estrategias secuencialmente con cache compartido: los contextos LLM generados en CONTEXTUAL_VECTOR se reusan en CONTEXTUAL_HYBRID y CONTEXTUAL_HYBRID_RERANK sin regenerar.

Salida: `*_summary.csv`, `*_detail.csv` por estrategia + `comparison.csv` con Pass@k y failure rate reduction vs baseline.

## sandbox_mteb (HotpotQA)

Pipeline completo retrieval + generacion + metricas sobre HotpotQA (7405 queries, 66K docs) via MinIO/Parquet.

```
.env -> MTEBConfig -> MinIO/cache(Parquet) -> LoadedDataset
     -> shuffle(seed) -> slice(max_corpus)
     -> [enrich(LLM) si CONTEXTUAL_HYBRID]
     -> index(ChromaDB + Tantivy)
     -> pre-embed queries (batch REST NIM)
     -> retrieve(local ChromaDB + BM25 + RRF, sync)
     -> [rerank(cross-encoder) si habilitado]
     -> generate + metrics (async)
     -> EvaluationRun -> JSON + CSV
```

Queries se pre-embeben en batch via REST antes del loop de retrieval. La fase de retrieval usa vectores pre-computados con busqueda local, eliminando roundtrip REST por query. Fallback a embedding por query si el batch falla.

## Metricas

### Retrieval

Hit@K, MRR, Recall@K (K=1,3,5,10,20), NDCG@K sobre top `RETRIEVAL_K` documentos (pre-rerank).

### Generacion

Metrica primaria segun `answer_type` del query:

- `"label"` (yes/no): Accuracy (match normalizado)
- Otro (extractiva): F1 (token-overlap normalizado)

EM (Exact Match) siempre como secundaria. Faithfulness (LLM-judge) como secundaria si configurada.

### Separacion retrieval vs generacion

| Config | Metricas retrieval | Contexto generacion |
|---|---|---|
| Reranker OFF | top RETRIEVAL_K | top RETRIEVAL_K |
| Reranker ON | top RETRIEVAL_K (pre-rerank) | top RERANKER_TOP_N (post-rerank) |

### Retrieval efectivo (post-rerank)

Cuando el reranker esta activo, las metricas de retrieval (pre-rerank) pueden subestimar la calidad del contexto que recibe el LLM. El reranker opera sobre `PRE_FUSION_K` candidatos y puede promover docs de posiciones 21-150 al top de generacion.

Metricas adicionales sobre `generation_doc_ids` (post-rerank):

- `generation_recall`: fraccion de gold docs en el set de generacion.
- `generation_hit`: 1.0 si algun gold doc en el set de generacion, 0.0 si no.
- `reranker_rescue_count` (run-level): queries donde retrieval recall@K=0 pero generation_recall>0.

Sin reranker, estos campos no se emiten.

## Dataset: HotpotQA

Fuente: HotpotQA fullwiki (validation split) via HuggingFace.

| Propiedad | Valor |
|---|---|
| Queries | 7405 |
| Corpus | 66576 documentos (Wikipedia, deduplicados por titulo) |
| Qrels | 14810 (2.0 por query, solo supporting_facts) |
| Tipos de query | bridge (~80%), comparison (~20%) |
| Almacenamiento | MinIO (Parquet), `s3://lakehouse/datasets/evaluation/hotpotqa/` |
| Schema | v2.0: query_id, text, answer, answer_type, question_type, level |

**Limitacion del corpus.** 10 pasajes por query (2 gold + 8 distractores), deduplicados por titulo. Gold docs garantizados. Resultados **no comparables con benchmarks publicados** (corpus completo ~5.2M). Solo comparaciones relativas entre estrategias son validas. Corpus se baraja con seed fijo antes de `EVAL_MAX_CORPUS`.

**Retrieval multi-hop.** Cada query requiere 2 gold docs que tipicamente cubren espacios semanticos distintos (ej: una query bridge necesita un doc sobre la entidad mencionada y otro sobre la entidad preguntada). Un bi-encoder genera un unico vector por query, inherentemente mas cercano a uno de los dos temas. Esto explica el gap tipico entre recall@1 (~0.46) y recall@5 (~0.82) en corpus completo con SIMPLE_VECTOR: el primer gold doc se recupera rapido, el segundo requiere profundidad. CONTEXTUAL_HYBRID (BM25 + Vector + RRF) mitiga esto capturando coincidencias lexicas que el bi-encoder pierde (nombres propios, fechas).

## Setup y uso

```bash
pip install -r requirements.txt
cp sandbox_mteb/env.example sandbox_mteb/.env
# Editar .env con endpoints NIM y MinIO
```

```bash
python -m sandbox_mteb.run                  # Run con .env
python -m sandbox_mteb.run --dry-run        # Solo validar config
python -m sandbox_mteb.run --env /path/.env # .env alternativo
python -m sandbox_mteb.run -v               # Verbose (DEBUG)
```

## Configuracion (.env)

Referencia completa en `sandbox_mteb/env.example`. Variables criticas:

```bash
# Embedding (NIM)
EMBEDDING_MODEL_NAME=nvidia/llama-3.2-nv-embedqa-1b-v2
EMBEDDING_BASE_URL=http://<nim-embedding-host>:8000/v1
EMBEDDING_MODEL_TYPE=asymmetric       # asymmetric para NIM (input_type query/passage)
EMBEDDING_BATCH_SIZE=5                # Reducir si NIM da errores de buffer

# LLM (NIM)
LLM_BASE_URL=http://<nim-llm-host>:8000/v1
LLM_MODEL_NAME=nvidia/nemotron-3-nano
NIM_MAX_CONCURRENT_REQUESTS=32        # ATENCION: no NIM_MAX_CONCURRENT
NIM_REQUEST_TIMEOUT=120               # ATENCION: no NIM_TIMEOUT

# Retrieval
RETRIEVAL_STRATEGY=SIMPLE_VECTOR      # SIMPLE_VECTOR | CONTEXTUAL_HYBRID
RETRIEVAL_K=20                        # Docs para metricas retrieval
RETRIEVAL_PRE_FUSION_K=150            # Candidatos pre-RRF / pool reranker
RETRIEVAL_RRF_K=60                    # Parametro k de RRF (CONTEXTUAL_HYBRID)
RETRIEVAL_BM25_WEIGHT=0.5             # Peso BM25 en RRF
RETRIEVAL_VECTOR_WEIGHT=0.5           # Peso vector en RRF

# Reranker (opcional)
RERANKER_ENABLED=false
RERANKER_BASE_URL=http://<nim-reranker-host>:9000/v1
RERANKER_MODEL_NAME=nvidia/llama-3.2-nv-rerankqa-1b-v2
RERANKER_TOP_N=5                      # Docs post-rerank para generacion

# Dataset
MTEB_DATASET_NAME=hotpotqa
EVAL_MAX_QUERIES=0                    # 0 = todas
EVAL_MAX_CORPUS=0                     # 0 = todo
GENERATION_ENABLED=true
CORPUS_SHUFFLE_SEED=42                # -1 = sin shuffle (NO recomendado)

# Modo desarrollo
DEV_MODE=false                        # Subset con gold docs garantizados
DEV_QUERIES=200
DEV_CORPUS_SIZE=4000

# MinIO
MINIO_ENDPOINT=http://<minio-host>:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minio123
MINIO_BUCKET_NAME=lakehouse
S3_DATASETS_PREFIX=datasets/evaluation
```

### DEV_MODE

`DEV_MODE=true`: subset de `DEV_QUERIES` queries + gold docs garantizados en corpus + distractores hasta `DEV_CORPUS_SIZE`. Ignora `EVAL_MAX_QUERIES`/`EVAL_MAX_CORPUS`. Metricas optimistas (ratio gold/distractores ~10% vs ~0.003% en corpus completo); solo validas para comparacion relativa.

## Salida

Tres archivos por run en `data/results/`:

- `<run_id>.json`: config + metricas por query + doc_ids recuperados
- `<run_id>_summary.csv`: 1 fila, metricas agregadas
- `<run_id>_detail.csv`: N filas, una por query

## Tests

253 tests (147 originales + 106 nuevos cookbook/shared). Ejecutables con Python 3.10+.

```bash
pytest tests/                      # Todo (253 tests)
pytest tests/ -v                   # Verbose
pytest tests/ -m "not integration" # Solo unit
pytest tests/integration/ -v       # Solo integracion (requiere NIM + MinIO)
```

| Scope | Tests | Archivos |
|---|---|---|
| sandbox_cookbook | 71 | `test_cookbook_config.py`, `test_cookbook_loader.py`, `test_cookbook_evaluator.py` |
| shared/retrieval (contextual, RRF) | 46 | `test_contextual_mode_a.py`, `test_dtm4_rrf.py` |
| sandbox_mteb + shared | ~90 | `test_dt*.py`, `test_dtm*.py`, `test_metrics_*.py`, `test_format_*.py` |
| shared/tantivy | 17 | `test_dtm4_tantivy_edge_cases.py` |

**Mocking condicional:** `tests/conftest.py` solo mockea modulos de infraestructura (`boto3`, `langchain_*`, `chromadb`) si el paquete real no esta instalado.

Detalle de cobertura por archivo, decisiones de mocking y diseno de tests de integracion: ver `README_TEST.md`.

## Deuda tecnica

### Resueltas

| ID | Fix |
|---|---|
| DT-3 | Logging JSONL estructurado (`structured_logging.py`). `LOG_FORMAT=jsonl\|text`. |
| DT-4 | Tipado estricto: `mypy.ini`, `EmbeddingModelProtocol`, `LLMJudgeProtocol`, `py.typed`. 0 errores mypy. |
| DT-5 | `pre_rerank_candidate_ids` en `QueryRetrievalDetail` para trazabilidad reranker. |
| DT-6 | Eliminado truncamiento `context[:4000]` en faithfulness. Judge recibe mismo contexto que LLM de generacion. |
| DT-7 | Reranker fallback detectado: warning + contador `_rerank_failures` + metadata `reranked` en JSON/CSV. |
| DT-8 | `sorted()` explicito por `relevance_score` descendente en reranker. |
| DT-9 | `_extract_score_fallback()` reescrita: 3 patrones explicitos, elimina falsos positivos de parciales. |
| DTm-1 | `_populate_from_dataframes()` extraido en `loader.py`. ~50 lineas duplicadas eliminadas. |
| DTm-2 | `vector_store.py` usa API publica ChromaDB nativo en vez de `_store._collection` (API interna LangChain). |
| DTm-3 | `run_sync()` compatible con event loops activos (Jupyter) via `ThreadPoolExecutor`. |
| DTm-4 | 162 tests (147 unit + 15 integracion). pytest nativo, `conftest.py` centralizado, `pyproject.toml` configurado. |
| DTm-5 | Metricas secundarias fallidas: `MetricResult(value=0.0, error=...)` + warning (no desaparecen del JSON). |
| DTm-6 | Retry con backoff exponencial en `_batch_embed_queries()`. |
| DTm-7 | `_run_async` renombrado a `run_sync`. |
| DTm-8 | `failure_rate_at_k` renombrado a `complement_recall_at_k`. |
| DTm-9 | Docstring sesgo embedding en `get_full_text()`. |
| DTm-10 | `collection_name = f"eval_{run_id}"` (unico, determinista). |
| DTm-11 | `LLMMetrics.__copy__`/`__deepcopy__` crean Lock nuevo. |
| DTm-17 | Metricas de retrieval efectivo (post-rerank): `generation_recall`, `generation_hit`, `reranker_rescue_count`. 15 tests. |
| DT-2 | Mode A (documento padre) implementado en `LLMContextGenerator`: `mode="document"`, prompts XML Anthropic, truncation configurable (32000/8000 chars para cookbook vs 2000/1000 default). sandbox_cookbook usa Mode A; sandbox_mteb sigue usando Mode B (chunk + titulo). |

### Abiertas

#### Bugs (cookbook)

| ID | Descripcion | Impacto | Fix |
|---|---|---|---|
| DTc-1 | **Cache pre-fill usa key-space incorrecto.** `_load_context_cache()` hace `_make_cache_key(chunk_id, "")` pero el save path usa `_make_cache_key(chunk_content, parent_content)`. Las keys no coinciden — el cache se carga pero produce 0 hits. `--compare-all` con cache compartido regenera contextos en cada estrategia. | Bug funcional. Cache inoperante. | Cambiar pre-fill a insertar keys directamente sin re-hashear: `context_generator._cache[key] = ctx` |
| DTc-2 | Cache invalidation solo hashea `ANTHROPIC_DOCUMENT_PROMPT`, ignora `system_prompt` y `chunk_prompt`. Si cambian los otros prompts pero no el document prompt, el cache no se invalida. | Correctness. Baja probabilidad. | Hashear los 3 prompts concatenados |
| DTc-3 | `assert` en `_validate_pass_at_k()` desaparece con `python -O`. Usar excepcion propia. | Correctness con `-O`. Baja probabilidad. | Reemplazar assert por `raise` |

#### Deuda mteb (pre-existente)

| ID | Descripcion | Impacto |
|---|---|---|
| DTm-12 | Sesgo LLM-judge en faithfulness para respuestas cortas: score 0.0-0.2 incluso con F1=1.0. | Sesgo metrica. Baja prioridad. |
| DTm-13 | No-determinismo HNSW: ChromaDB 0.5-0.6 no soporta `hnsw:random_seed`. | Reproducibilidad. Baja prioridad. |
| DTm-14 | Duplicacion contenido memoria: `retrieved_contents` + `generation_contents` (~1.5GB). | Memoria. Baja prioridad. |
| DTm-15 | ETL HotpotQA no asigna `answer_type="label"` a queries comparison. | Clasificacion metrica. Baja prioridad. |
| DTm-16 | Nemotron-3-nano responde "yes" a preguntas extractivas (~10%). | Calidad generacion. Media prioridad. |

#### mypy (pre-existente)

30 errores en `shared/` (0 en `sandbox_cookbook/`). Todos son patrones de import condicional (`ChatNVIDIA = None` cuando el paquete no esta instalado) y tipos de ChromaDB. No bloquean ejecucion.

### Simplificacion de shared/

#### Modulos solo usados por un sandbox

| Modulo | Lineas | Usado por | Candidato a mover |
|---|---|---|---|
| `shared/metrics.py` | 1004 | solo mteb | Si — mover a `sandbox_mteb/metrics.py` |
| `shared/report.py` | 259 | solo mteb | Si — mover a `sandbox_mteb/report.py` |
| `shared/structured_logging.py` | 123 | solo mteb | Si — mover a `sandbox_mteb/` |
| `shared/retrieval/__init__.py` (`get_retriever`) | 127 | solo mteb | Si — factory no usada por cookbook |

`sandbox_cookbook` construye retrievers directamente en `evaluator.py` (necesita control fino sobre HybridRetriever config, cache, y over-sampling para rerank). La factory `get_retriever()` es innecesaria para el cookbook y podria eliminarse o moverse a mteb.

**Impacto de mover**: 1386 lineas (30%) saldrian de shared/. shared/ quedaria en ~3250 lineas, solo con tipos, retrieval, LLM y config — realmente compartidos.

#### Tipos no usados por cookbook

| Tipo en `shared/types.py` | Usado por |
|---|---|
| `GenerationResult` | solo mteb |
| `LLMJudgeProtocol` | solo mteb (+ `shared/retrieval/__init__.py`) |
| `get_dataset_config()` / `DATASET_CONFIG` | solo mteb (+ tiene entrada "cookbook" no usada) |

No vale la pena moverlos — `types.py` es el nucleo compartido y estos tipos no estorban.

#### Import no usado en cookbook

`MetricType` se importa en `sandbox_cookbook/evaluator.py` pero nunca se usa. Eliminar.

#### RetrievalConfig: campos mteb-only

`RetrievalConfig` tiene ~15 campos. Cookbook usa ~10. Los restantes (`bm25_language`, etc.) tienen defaults razonables y no estorban — no vale refactorizar.