# Plan de Pruebas — Contextual Retrieval Evaluation

## 1. Evaluación de la Solución

### 1.1 Estado actual

| Aspecto | Resultado |
|---|---|
| **Tests unitarios** | 137/137 pasando (1.45s) |
| **Cobertura funcional** | Config (23), Loader (19), Evaluator (29), Contextual (26), RRF (20), Tantivy (17), Reranker (3) |
| **Dependencias** | Todas instalables (`pip install -r requirements.txt`) |
| **Código** | Bien estructurado, separación limpia shared/ vs sandbox_cookbook/ |
| **Deuda técnica** | DTc-2 abierto (baja probabilidad), resto resuelto |

### 1.2 Fortalezas

- **Arquitectura modular**: `shared/retrieval/` define retrievers intercambiables via `BaseRetriever`
- **4 estrategias incrementales** que replican fielmente el paper de Anthropic
- **Mocking condicional** en `conftest.py`: tests ejecutables sin infraestructura NIM
- **Cache persistente** de contextos LLM: evita regenerar en `--compare-all`
- **Validación dual** (ID + contenido) en evaluación como safety net contra bugs de mapeo
- **CLI completa**: `--strategy`, `--compare-all`, `--dry-run`, `-v`

### 1.3 Riesgos identificados

| Riesgo | Impacto | Mitigación |
|---|---|---|
| Endpoints NIM no disponibles | Bloquea ejecución real | Validar con `--dry-run` primero |
| Modelo de embedding diferente al del paper (Voyage) | Métricas pueden diferir | Documentado, aceptable con NIM |
| DTc-2: cache invalida si cambia solo 1 de 3 prompts | Contextos stale | Borrar cache manualmente si se cambian prompts |
| Sin tests de integración con NIM real | Solo unitarios | Este plan cubre integración |

---

## 2. Plan de Pruebas por Fase

### Fase 0: Validación del entorno (sin NIM)

**Objetivo**: Confirmar que el código funciona localmente sin servicios externos.

```bash
# 1. Clonar e instalar
git clone <repo> && cd CH_anthropic_
pip install -r requirements.txt

# 2. Ejecutar tests unitarios
pytest tests/ -v
# Esperado: 137 passed

# 3. Verificar dataset
python -c "
from sandbox_cookbook.loader import CookbookLoader
ds = CookbookLoader.load(
    'sandbox_cookbook/data/codebase_chunks.json',
    'sandbox_cookbook/data/evaluation_set.jsonl'
)
print(f'Docs: {len(ds.corpus)}')
print(f'Queries: {len(ds.queries)}')
print(f'Chunks totales: {sum(1 for _ in ds.corpus.values())}')
"
# Esperado: Docs: 737, Queries: 248, Chunks totales: 737
```

**Criterio de éxito**: 137 tests pasando, dataset cargable.

---

### Fase 1: SIMPLE_VECTOR (solo embedding)

**Objetivo**: Probar la estrategia baseline con tu endpoint de embedding NIM.

**Prerequisitos**: Endpoint NIM de embedding activo.

```bash
# 1. Configurar .env
cp sandbox_cookbook/env.example sandbox_cookbook/.env
```

Editar `.env` con tus valores reales:
```env
COOKBOOK_STRATEGY=SIMPLE_VECTOR
EMBEDDING_BASE_URL=http://<tu-ip>:8000/v1
EMBEDDING_MODEL_NAME=nvidia/llama-3.2-nv-embedqa-1b-v2
EMBEDDING_MODEL_TYPE=asymmetric
EMBEDDING_BATCH_SIZE=5
COOKBOOK_DATASET_PATH=sandbox_cookbook/data/codebase_chunks.json
COOKBOOK_EVAL_PATH=sandbox_cookbook/data/evaluation_set.jsonl
COOKBOOK_RESULTS_DIR=sandbox_cookbook/data/results
```

```bash
# 2. Validar config (dry run)
python -m sandbox_cookbook.run --dry-run
# Esperado: "Config valida. No se ejecuta evaluacion."

# 3. Ejecutar evaluación
python -m sandbox_cookbook.run --strategy SIMPLE_VECTOR -v
```

**Criterio de éxito**:
- Genera `*_summary.csv` y `*_detail.csv` en `sandbox_cookbook/data/results/`
- Pass@5 > 0% (confirma que retrieval funciona)
- Referencia paper: ~81% Pass@5

**Verificación de resultados**:
```bash
cat sandbox_cookbook/data/results/*summary.csv
# Columnas esperadas: strategy, pass_at_5, pass_at_10, pass_at_20, total_queries, ...
```

---

### Fase 2: CONTEXTUAL_VECTOR (enrichment LLM)

**Objetivo**: Probar generación de contexto LLM + embedding enriquecido.

**Prerequisitos**: Endpoint NIM de embedding + endpoint NIM de LLM activos.

Añadir a `.env`:
```env
COOKBOOK_STRATEGY=CONTEXTUAL_VECTOR
LLM_BASE_URL=http://<tu-ip-llm>:8000/v1
LLM_MODEL_NAME=meta/llama-3.1-70b-instruct
COOKBOOK_CONTEXTUALIZE_MAX_TOKENS=1000
COOKBOOK_CONTEXTUALIZE_BATCH_SIZE=10
COOKBOOK_CONTEXT_POSITION=prepend
COOKBOOK_CONTEXTS_CACHE_PATH=sandbox_cookbook/data/contexts_cache.json
```

```bash
# 1. Dry run
python -m sandbox_cookbook.run --strategy CONTEXTUAL_VECTOR --dry-run

# 2. Ejecutar
python -m sandbox_cookbook.run --strategy CONTEXTUAL_VECTOR -v
```

**Qué observar**:
- Logs de generación de contexto (737 chunks a enriquecer)
- Creación de `contexts_cache.json` (cache persistente)
- Tiempo de ejecución (depende del throughput del LLM)

**Criterio de éxito**:
- Pass@5 > Pass@5 de SIMPLE_VECTOR (mejora esperada ~5-10%)
- Cache generado y reutilizable
- Referencia paper: ~88% Pass@5

---

### Fase 3: CONTEXTUAL_HYBRID (BM25 + Vector + RRF)

**Objetivo**: Probar búsqueda híbrida con fusión de rankings.

**Prerequisitos**: Mismos que Fase 2 (el cache de contextos se reutiliza).

Cambiar en `.env`:
```env
COOKBOOK_STRATEGY=CONTEXTUAL_HYBRID
COOKBOOK_SEMANTIC_WEIGHT=0.8
COOKBOOK_BM25_WEIGHT=0.2
COOKBOOK_NUM_CHUNKS_TO_RECALL=150
```

```bash
python -m sandbox_cookbook.run --strategy CONTEXTUAL_HYBRID -v
```

**Qué observar**:
- Log "Reutilizando contextos del cache" (no regenera)
- Tantivy BM25 index se construye (o fallback a rank_bm25)
- RRF fusion combina rankings

**Criterio de éxito**:
- Pass@5 comparable a Fase 2 o superior
- Referencia paper: ~86% Pass@5

---

### Fase 4: CONTEXTUAL_HYBRID_RERANK (+ cross-encoder)

**Objetivo**: Probar pipeline completo con reranking.

**Prerequisitos**: Endpoint NIM de reranker activo (además de embedding + LLM).

Añadir a `.env`:
```env
COOKBOOK_STRATEGY=CONTEXTUAL_HYBRID_RERANK
RERANKER_ENABLED=true
RERANKER_BASE_URL=http://<tu-ip-reranker>:8000/v1
RERANKER_MODEL_NAME=nvidia/nv-rerankqa-mistral-4b-v3
COOKBOOK_RERANK_TOP_N=20
COOKBOOK_RERANK_OVERSAMPLE_FACTOR=10
COOKBOOK_RERANK_CONTENT=enriched
```

```bash
python -m sandbox_cookbook.run --strategy CONTEXTUAL_HYBRID_RERANK -v
```

**Criterio de éxito**:
- Pass@5 máximo de las 4 estrategias
- Referencia paper: ~92% Pass@5

---

### Fase 5: Comparación completa (--compare-all)

**Objetivo**: Ejecutar las 4 estrategias secuencialmente y generar comparison.csv.

**Prerequisitos**: Los 3 endpoints NIM activos (embedding, LLM, reranker).

```bash
python -m sandbox_cookbook.run --compare-all -v
```

**Qué observar**:
- Ejecución secuencial de las 4 estrategias
- Cache compartido: contextos generados en CONTEXTUAL_VECTOR se reusan
- Tabla comparativa en stdout
- Archivo `comparison.csv` generado

**Criterio de éxito**:
- `comparison.csv` con 4 filas (una por estrategia)
- Tendencia esperada: SIMPLE_VECTOR < CONTEXTUAL_VECTOR ≤ CONTEXTUAL_HYBRID < CONTEXTUAL_HYBRID_RERANK
- Failure rate reduction vs baseline calculado correctamente

**Verificación**:
```bash
cat sandbox_cookbook/data/results/comparison.csv
# Esperado: strategy, pass_at_5, pass_at_10, pass_at_20, failure_rate_5, reduction_vs_baseline_5, ...
```

---

## 3. Troubleshooting

| Problema | Causa probable | Solución |
|---|---|---|
| `ConnectionError` al conectar a NIM | Endpoint caído o IP incorrecta | Verificar con `curl <EMBEDDING_BASE_URL>/v1/models` |
| `--dry-run` falla con errores de validación | Variables de .env faltantes o mal formateadas | Revisar cada campo vs `env.example` |
| Pass@k = 0% | Embedding no genera vectores útiles | Verificar modelo de embedding compatible |
| Contextos no se cachean | `COOKBOOK_CONTEXTS_CACHE_PATH` no definido | Añadir a .env o usar `--compare-all` (lo setea automáticamente) |
| `ImportError: tantivy` | Tantivy no instalado | `pip install tantivy` o usa fallback rank_bm25 (automático) |
| Resultados muy diferentes al paper | Modelo distinto a Voyage/Claude | Esperado: NIM usa modelos distintos, las tendencias deben mantenerse |

---

## 4. Checklist resumen

- [ ] **Fase 0**: `pytest tests/` → 137 passed
- [ ] **Fase 0**: Dataset cargable (737 chunks, 248 queries)
- [ ] **Fase 1**: SIMPLE_VECTOR genera CSVs con métricas
- [ ] **Fase 2**: CONTEXTUAL_VECTOR mejora sobre baseline + cache generado
- [ ] **Fase 3**: CONTEXTUAL_HYBRID ejecuta con BM25 + RRF
- [ ] **Fase 4**: CONTEXTUAL_HYBRID_RERANK alcanza mejor Pass@5
- [ ] **Fase 5**: `--compare-all` genera `comparison.csv` con 4 estrategias
- [ ] Tendencia ascendente confirmada: cada estrategia mejora o mantiene Pass@k

---

## 5. Servicios NIM necesarios

| Servicio | Modelo sugerido | Puerto típico | Requerido desde |
|---|---|---|---|
| **Embedding** | `nvidia/llama-3.2-nv-embedqa-1b-v2` | 8000 | Fase 1 |
| **LLM** | `meta/llama-3.1-70b-instruct` | 8001 | Fase 2 |
| **Reranker** | `nvidia/nv-rerankqa-mistral-4b-v3` | 8002 | Fase 4 |

> **Nota**: Si solo tienes el endpoint de embedding, puedes ejecutar Fase 0 y Fase 1 completas. Las fases 2-5 requieren LLM, y la fase 4-5 requiere además reranker.
