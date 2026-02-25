# Contextual Retrieval — Evaluacion RAG

Replica del experimento de [Contextual Retrieval de Anthropic](https://www.anthropic.com/news/contextual-retrieval) sobre infraestructura NIM. Compara 4 estrategias incrementales de retrieval sobre un dataset de 90 documentos de codigo (737 chunks, 248 queries).

## Arquitectura

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
│   ├── config.py                    # CookbookConfig: .env -> dataclass validada
│   ├── loader.py                    # JSON/JSONL local -> LoadedDataset
│   ├── evaluator.py                 # Pipeline: 4 estrategias, Pass@k, comparison.csv
│   ├── context_cache.py             # Cache persistente de contextos LLM
│   ├── run.py                       # Entry point (--compare-all, --dry-run, -v)
│   └── env.example                  # Plantilla .env
│
├── tests/                           # 137 tests (pytest)
│   ├── conftest.py                  # Mocks condicionales
│   ├── test_cookbook_config.py       # 23 tests
│   ├── test_cookbook_loader.py       # 19 tests
│   ├── test_cookbook_evaluator.py    # 29 tests
│   ├── test_contextual_mode_a.py    # 26 tests (LLMContextGenerator, Mode A)
│   ├── test_dtm4_rrf.py            # 20 tests (RRF classic + cookbook)
│   ├── test_dtm4_tantivy_edge_cases.py  # 17 tests (BM25/Tantivy)
│   └── test_dt8_09_10_11_reranker_sort.py  # 3 tests (CrossEncoderReranker)
│
├── docs/                            # Documentacion historica
│   ├── DESIGN.md                    # Especificacion de diseno original
│   └── WORKPLAN.md                  # Plan de trabajo (6 fases, COMPLETO)
│
├── pyproject.toml
├── mypy.ini
└── requirements.txt
```

## Estrategias de retrieval

| Estrategia | Indexacion | Busqueda | Reranker |
|---|---|---|---|
| `SIMPLE_VECTOR` | Embedding directo | Cosine similarity (ChromaDB) | No |
| `CONTEXTUAL_VECTOR` | Enrichment LLM + embedding | Cosine similarity | No |
| `CONTEXTUAL_HYBRID` | Enrichment LLM + embedding | BM25 (Tantivy) + Vector + RRF | No |
| `CONTEXTUAL_HYBRID_RERANK` | Enrichment LLM + embedding | BM25 + Vector + RRF + cross-encoder | Si |

Enrichment contextual: cada chunk se enriquece con un contexto generado por LLM a partir del documento padre (Mode A, prompts XML Anthropic). El texto enriquecido se indexa; el contenido original se preserva para evaluacion.

## Uso

```bash
pip install -r requirements.txt
cp sandbox_cookbook/env.example sandbox_cookbook/.env
# Editar .env con endpoints NIM
```

```bash
python -m sandbox_cookbook.run                                    # Una estrategia
python -m sandbox_cookbook.run --strategy CONTEXTUAL_VECTOR       # Override
python -m sandbox_cookbook.run --compare-all                      # 4 estrategias + comparison.csv
python -m sandbox_cookbook.run --compare-all --dry-run            # Validar configs
python -m sandbox_cookbook.run -v                                 # Verbose (DEBUG)
```

`--compare-all` ejecuta las 4 estrategias secuencialmente con cache compartido: los contextos LLM generados en CONTEXTUAL_VECTOR se reusan en CONTEXTUAL_HYBRID y CONTEXTUAL_HYBRID_RERANK sin regenerar.

Salida: `*_summary.csv`, `*_detail.csv` por estrategia + `comparison.csv` con Pass@k y failure rate reduction vs baseline.

## Metricas

**Pass@k** (k=5, 10, 20) y **Failure Rate** (1 - Pass@k). Pass@k = Recall@k algebraicamente.

Validacion content-based: el evaluator compara matching por ID vs matching por contenido exacto. Si divergen, aborta (indica bug en mapeo de IDs).

## Dataset

Fuente: [cookbook de Anthropic](https://github.com/anthropics/anthropic-cookbook/tree/main/capabilities/contextual-embeddings/data).

| Propiedad | Valor |
|---|---|
| Documentos | 90 (codigo fuente) |
| Chunks | 737 |
| Queries | 248 (306 golden chunks, 28 queries con >1 golden) |
| Formato | JSON/JSONL local |

## Decisiones clave

- **Dataset**: Replica exacta de los archivos del cookbook. Carga JSON/JSONL local, sin MinIO ni Parquet.
- **Solo metricas del paper**: Sin pipeline de generacion ni LLM-judge.
- **context_position**: Parametrizable (`"prepend"` blog / `"append"` cookbook) en `LLMContextGenerator`.
- **Cache persistente**: JSON en disco con invalidacion por hash modelo+prompt.

## Tests

137 tests. Ejecutables con Python 3.10+.

```bash
pytest tests/           # Todo
pytest tests/ -v        # Verbose
```

**Mocking condicional:** `conftest.py` mockea `langchain_*` y `chromadb` solo si no estan instalados.

## Deuda tecnica

| ID | Descripcion | Estado |
|---|---|---|
| DTc-1 | Cache pre-fill key-space mismatch | Resuelto |
| DTc-2 | Cache invalidation incompleta (solo hashea 1 de 3 prompts) | Abierto (baja probabilidad) |
| DTc-3 | assert en _validate_pass_at_k | Resuelto |

## Fuentes

- [Anthropic Blog: Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [Anthropic Cookbook: Contextual Embeddings Guide](https://platform.claude.com/cookbook/capabilities-contextual-embeddings-guide)
- [Contextual Retrieval Appendix II (PDF)](https://assets.anthropic.com/m/1632cded0a125333/original/Contextual-Retrieval-Appendix-2.pdf)
