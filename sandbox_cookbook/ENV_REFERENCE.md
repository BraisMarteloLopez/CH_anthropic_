# Sandbox Cookbook — Referencia de Configuracion `.env`

Guia completa de todas las variables de entorno del sandbox **Contextual Retrieval**.

---

## Tabla de contenidos

1. [Estrategia](#1-estrategia)
2. [Dataset](#2-dataset)
3. [Embedding (NIM)](#3-embedding-nim)
4. [LLM (NIM)](#4-llm-nim)
5. [Contextual Retrieval](#5-contextual-retrieval-fase-2)
6. [Hybrid Search](#6-hybrid-search-fase-3)
7. [Reranking](#7-reranking-fase-4)
8. [Requisitos por estrategia](#8-requisitos-por-estrategia)
9. [Ejemplos de ejecucion](#9-ejemplos-de-ejecucion)

---

## 1. Estrategia

| Variable | Default | Descripcion |
|---|---|---|
| `COOKBOOK_STRATEGY` | `SIMPLE_VECTOR` | Pipeline de retrieval a ejecutar |

Valores posibles (fases incrementales):

| Valor | Fase | Que hace |
|---|---|---|
| `SIMPLE_VECTOR` | 1 | Embedding puro: chunk -> vector -> cosine similarity. Baseline sin LLM. |
| `CONTEXTUAL_VECTOR` | 2 | Cada chunk se enriquece con contexto generado por un LLM antes de embebear. |
| `CONTEXTUAL_HYBRID` | 3 | Fase 2 + busqueda BM25 (keyword). Fusiona scores semantico + lexico. |
| `CONTEXTUAL_HYBRID_RERANK` | 4 | Fase 3 + reranker neural que re-ordena los candidatos finales. |

---

## 2. Dataset

| Variable | Default | Descripcion |
|---|---|---|
| `COOKBOOK_DATASET_PATH` | `sandbox_cookbook/data/codebase_chunks.json` | Corpus de chunks de codigo segmentados (JSON) |
| `COOKBOOK_EVAL_PATH` | `sandbox_cookbook/data/evaluation_set.jsonl` | Ground truth: pares (query, expected_chunks) en JSONL |
| `COOKBOOK_RESULTS_DIR` | `sandbox_cookbook/data/results` | Directorio de salida (JSONs por run, comparison.csv) |

> Todos los paths son **relativos a la raiz del proyecto**.

---

## 3. Embedding (NIM)

**Requerido para TODAS las estrategias.**

| Variable | Default | Descripcion |
|---|---|---|
| `EMBEDDING_BASE_URL` | `http://172.30.79.98:8000/v1` | Endpoint OpenAI-compatible del servicio NIM de embeddings |
| `EMBEDDING_MODEL_NAME` | `nvidia/llama-3.2-nv-embedqa-1b-v2` | Modelo servido en el NIM |
| `EMBEDDING_MODEL_TYPE` | `asymmetric` | `symmetric` (query=doc) o `asymmetric` (instrucciones distintas query/doc) |
| `EMBEDDING_BATCH_SIZE` | `5` | Chunks enviados por request HTTP al NIM |

### Notas

- El NIM de embeddings esta desplegado en `172.30.79.98:8000`.
- `EMBEDDING_BATCH_SIZE=5` es conservador. Subir si el NIM tiene buena GPU/RAM.

---

## 4. LLM (NIM)

**Requerido solo para estrategias `CONTEXTUAL_*`.**

| Variable | Default | Descripcion |
|---|---|---|
| `LLM_BASE_URL` | *(comentado)* | Endpoint NIM del LLM (OpenAI-compatible, `/v1/chat/completions`) |
| `LLM_MODEL_NAME` | *(comentado)* | Modelo usado para generar contexto de enriquecimiento por chunk |

### Ejemplo

```env
LLM_BASE_URL=http://nim-llm:8080/v1
LLM_MODEL_NAME=meta/llama-3.1-70b-instruct
```

---

## 5. Contextual Retrieval (Fase 2+)

Controla como el LLM genera el contexto que enriquece cada chunk antes de embebear.

| Variable | Default | Descripcion |
|---|---|---|
| `COOKBOOK_CONTEXTUALIZE_MODEL` | *(vacio = usa LLM_MODEL_NAME)* | Override del modelo para contextualizar |
| `COOKBOOK_CONTEXTUALIZE_MAX_TOKENS` | `1000` | Max tokens en la respuesta del LLM por chunk (~2-3 parrafos) |
| `COOKBOOK_CONTEXTUALIZE_BATCH_SIZE` | `10` | Llamadas LLM concurrentes. Subir segun capacidad del NIM |
| `COOKBOOK_CONTEXT_POSITION` | `prepend` | Donde pegar el contexto: `prepend` (antes) o `append` (despues del chunk) |
| `COOKBOOK_CONTEXTS_CACHE_PATH` | *(vacio = sin cache)* | Path al cache JSON de contextos generados |

### Cache de contextos

Variable clave para eficiencia. Si se activa:

- Los contextos generados se persisten en disco.
- En `--compare-all`, el cache se comparte automaticamente entre las 3 estrategias contextuales (se genera una vez en `CONTEXTUAL_VECTOR`, se reutiliza en `CONTEXTUAL_HYBRID` y `CONTEXTUAL_HYBRID_RERANK`).

```env
COOKBOOK_CONTEXTS_CACHE_PATH=sandbox_cookbook/data/contexts_cache.json
```

---

## 6. Hybrid Search (Fase 3+)

**Solo aplica a `CONTEXTUAL_HYBRID` y `CONTEXTUAL_HYBRID_RERANK`.**

| Variable | Default | Descripcion |
|---|---|---|
| `COOKBOOK_SEMANTIC_WEIGHT` | `0.8` | Peso del score de embedding en la fusion |
| `COOKBOOK_BM25_WEIGHT` | `0.2` | Peso del score BM25 (lexico) en la fusion |
| `COOKBOOK_NUM_CHUNKS_TO_RECALL` | `150` | Candidatos a recuperar del indice antes de fusionar/reranquear |
| `BM25_BACKEND` | `auto` | Backend BM25: `auto`, `tantivy`, `elasticsearch`, `rank_bm25` |
| `ELASTICSEARCH_HOST` | `http://localhost:9200` | URL del servidor Elasticsearch (solo si `BM25_BACKEND=elasticsearch`) |

### Backend BM25

| Valor | Motor | Requisitos |
|---|---|---|
| `auto` | Selecciona automaticamente: Tantivy > Elasticsearch > rank_bm25 | Al menos uno instalado |
| `tantivy` | Tantivy (Rust, embebido, sin servidor) | `pip install tantivy` |
| `elasticsearch` | Elasticsearch (replica cookbook Anthropic) | `pip install elasticsearch` + servidor ES corriendo |
| `rank_bm25` | rank-bm25 (Python puro, in-memory, legacy) | `pip install rank-bm25` |

Para usar Elasticsearch:

```bash
# Levantar Elasticsearch local
docker run -d --name elasticsearch -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" \
  elasticsearch:8.17.0

# Configurar en .env
BM25_BACKEND=elasticsearch
ELASTICSEARCH_HOST=http://localhost:9200
```

### Notas

- Los pesos deben sumar ~1.0. `0.8/0.2` = dominan embeddings, BM25 aporta senal lexica complementaria.
- Mas `NUM_CHUNKS_TO_RECALL` = mejor recall potencial pero mas lento.
- `elasticsearch` replica exactamente la implementacion del cookbook de Anthropic (analyzer por idioma, similarity BM25, multi_match sobre content + contextualized_content).

---

## 7. Reranking (Fase 4+)

**Solo aplica a `CONTEXTUAL_HYBRID_RERANK`.**

| Variable | Default | Descripcion |
|---|---|---|
| `RERANKER_ENABLED` | `false` | Habilitar reranker |
| `RERANKER_BASE_URL` | *(vacio)* | Endpoint NIM del modelo de reranking |
| `RERANKER_MODEL_NAME` | *(vacio)* | Modelo de reranking |
| `COOKBOOK_RERANK_TOP_N` | `20` | Resultados finales despues de reranquear |
| `COOKBOOK_RERANK_OVERSAMPLE_FACTOR` | `10` | Factor de sobre-muestreo: recupera `TOP_N x FACTOR` candidatos para el reranker |
| `COOKBOOK_RERANK_CONTENT` | `enriched` | Texto que se pasa al reranker |

### Valores de `COOKBOOK_RERANK_CONTENT`

| Valor | Que se pasa al reranker |
|---|---|
| `enriched` | Chunk + contexto LLM |
| `original` | Chunk raw (sin enriquecimiento) |
| `both` | Ambos |

### Pool del reranker

Con los defaults: `TOP_N(20) x OVERSAMPLE_FACTOR(10) = 200` candidatos entran al reranker, y solo los mejores 20 salen.

---

## 8. Requisitos por estrategia

| Estrategia | Embedding NIM | LLM NIM | Reranker NIM |
|---|:---:|:---:|:---:|
| `SIMPLE_VECTOR` | Requerido | - | - |
| `CONTEXTUAL_VECTOR` | Requerido | Requerido | - |
| `CONTEXTUAL_HYBRID` | Requerido | Requerido | - |
| `CONTEXTUAL_HYBRID_RERANK` | Requerido | Requerido | Requerido |

> `--compare-all` hace skip automatico de estrategias cuya infra no este configurada.

---

## 9. Ejemplos de ejecucion

### Estrategia unica

```bash
# Usa la estrategia definida en .env
python -m sandbox_cookbook.run

# Override de estrategia via CLI
python -m sandbox_cookbook.run --strategy SIMPLE_VECTOR
python -m sandbox_cookbook.run --strategy CONTEXTUAL_VECTOR
python -m sandbox_cookbook.run --strategy CONTEXTUAL_HYBRID
python -m sandbox_cookbook.run --strategy CONTEXTUAL_HYBRID_RERANK
```

### Comparacion completa (4 estrategias)

```bash
python -m sandbox_cookbook.run --compare-all
```

### Validacion sin ejecutar

```bash
python -m sandbox_cookbook.run --dry-run
python -m sandbox_cookbook.run --compare-all --dry-run
```

### Otras flags

```bash
python -m sandbox_cookbook.run -v                      # Logging DEBUG
python -m sandbox_cookbook.run --env /ruta/a/otro.env  # .env alternativo
```

### Flujo recomendado

```bash
# 1. Validar config
python -m sandbox_cookbook.run --dry-run

# 2. Baseline (solo embeddings)
python -m sandbox_cookbook.run --strategy SIMPLE_VECTOR

# 3. Comparacion completa (cuando LLM + reranker esten listos)
python -m sandbox_cookbook.run --compare-all
```
