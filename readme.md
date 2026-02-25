# sandbox_cookbook — Contextual Retrieval (Anthropic)

Replica del experimento de **Contextual Retrieval** de Anthropic ([blog](https://www.anthropic.com/news/contextual-retrieval), [cookbook](https://platform.claude.com/cookbook/capabilities-contextual-embeddings-guide)) sobre la infraestructura de evaluacion RAG existente (RAG_P v3.2).

## Que hace

Ejecuta y compara **4 estrategias incrementales** de retrieval sobre un dataset de 9 codebases (737 chunks, 248 queries):

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

### Pass@k = Recall@k

Pass@k del cookbook y Recall@k del sandbox existente son **algebraicamente identicas**: `|relevantes en top-k| / |total relevantes|`. No se implementa metrica nueva. Se reutiliza `recall_at_k` de `QueryRetrievalDetail` y se renombra en CSV.

### Solo metricas del paper

Se descartan: Hit@k, MRR, NDCG@k, F1, EM, Accuracy, Faithfulness, Answer Relevance, Context Utilization. No hay pipeline de generacion ni LLM-judge.

### Dataset incompatible con HotpotQA

El dataset del cookbook tiene estructura jerarquica (documento padre -> chunks) vs plana (HotpotQA). El loader y evaluator se escriben desde cero. La capa `shared/` (~70% del codigo util) se reutiliza intacta.

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

- [`DESIGN_sandbox_cookbook.md`](DESIGN_sandbox_cookbook.md) — Diseno propuesto (evaluado y aprobado)
- [`WORKPLAN.md`](WORKPLAN.md) — Plan de trabajo con 6 fases, ~30 tareas, estado de cada una
- [`_inspired_sandbox_evaluator/README.md`](_inspired_sandbox_evaluator/README.md) — Documentacion del sistema RAG_P v3.2
- [`_inspired_sandbox_evaluator/README_TEST.md`](_inspired_sandbox_evaluator/README_TEST.md) — Infraestructura de tests

## Fuentes

- [Anthropic Blog: Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [Anthropic Cookbook: Contextual Embeddings Guide](https://platform.claude.com/cookbook/capabilities-contextual-embeddings-guide)
- [Contextual Retrieval Appendix II (PDF)](https://assets.anthropic.com/m/1632cded0a125333/original/Contextual-Retrieval-Appendix-2.pdf)
