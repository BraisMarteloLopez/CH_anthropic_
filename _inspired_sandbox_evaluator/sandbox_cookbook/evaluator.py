"""
CookbookEvaluator: evaluacion Contextual Retrieval (Anthropic).

Pipeline:
    load -> index -> pre-embed queries -> retrieve -> evaluate Pass@k -> build_run

Estrategias soportadas:
  - SIMPLE_VECTOR: embedding puro (Fase 1)
  - CONTEXTUAL_VECTOR: enrichment LLM + vector (Fase 2)
  - CONTEXTUAL_HYBRID / CONTEXTUAL_HYBRID_RERANK: Fases 3-4

Diferencias clave con MTEBEvaluator:
  - Dataset: JSON/JSONL local (no MinIO/Parquet)
  - Metrica: Pass@k = Recall@k (solo k=5,10,20)
  - Sin generacion LLM
  - Indexa con doc.content (no get_full_text(), PC-1)
  - Pasa parent_content desde metadata["parent_documents"]
  - Validacion content-based como assertion de calidad
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from shared.types import (
    DatasetType,
    EvaluationRun,
    EvaluationStatus,
    LoadedDataset,
    MetricType,
    NormalizedDocument,
    QueryEvaluationResult,
    QueryRetrievalDetail,
)
from shared.llm import AsyncLLMService, batch_embed_queries, load_embedding_model
from shared.retrieval.core import (
    BaseRetriever,
    RetrievalConfig,
    RetrievalResult,
    RetrievalStrategy,
    SimpleVectorRetriever,
)
from shared.retrieval.contextual_retriever import (
    ContextualRetriever,
    LLMContextGenerator,
    ANTHROPIC_DOCUMENT_PROMPT,
    ANTHROPIC_CHUNK_PROMPT,
    ANTHROPIC_SYSTEM_PROMPT,
)

from .config import CookbookConfig
from .context_cache import ContextCache
from .loader import CookbookEvalQuery, CookbookLoader

logger = logging.getLogger(__name__)


class CookbookEvaluator:
    """
    Evaluador para el sandbox Contextual Retrieval.

    Estrategias:
      - SIMPLE_VECTOR: embedding puro (Fase 1)
      - CONTEXTUAL_VECTOR: enrichment + vector puro (Fase 2)
      - CONTEXTUAL_HYBRID: enrichment + BM25 + RRF (Fase 3)
      - CONTEXTUAL_HYBRID_RERANK: + cross-encoder reranking (Fase 4)
    """

    def __init__(self, config: CookbookConfig):
        self.config = config
        self._embedding_model = None
        self._llm_service = None
        self._retriever: Optional[BaseRetriever] = None
        self._context_cache: Optional[ContextCache] = None

    def run(self) -> EvaluationRun:
        """Ejecuta evaluacion completa."""
        start_time = time.time()
        strategy = self.config.strategy
        run_id = f"cookbook_{strategy}_{time.strftime('%Y%m%d_%H%M%S')}"

        logger.info("=" * 60)
        logger.info("COOKBOOK EVALUATION RUN")
        logger.info(f"  Run ID:     {run_id}")
        logger.info(f"  Strategy:   {strategy}")
        logger.info(f"  Eval K:     {self.config.eval_k_values}")
        logger.info("=" * 60)

        # 1. Inicializar embedding model
        self._init_components()

        # 2. Cargar dataset
        dataset, eval_queries = self._load_dataset()

        # 3. Indexar corpus
        corpus_size = self._index_documents(dataset)

        # 4. Evaluar queries
        query_results = self._evaluate_queries(
            dataset, eval_queries
        )

        # 5. Construir EvaluationRun
        elapsed = time.time() - start_time
        run = self._build_run(
            run_id, dataset, query_results, elapsed, corpus_size
        )

        # 6. Exportar CSV
        self._export_csv(run)

        logger.info("=" * 60)
        logger.info("RUN COMPLETADO")
        for k in self.config.eval_k_values:
            pass_at_k = run.avg_recall_at_k.get(k, 0.0)
            logger.info(f"  Pass@{k}: {pass_at_k:.4f} ({pass_at_k*100:.2f}%)")
        logger.info(f"  Tiempo: {elapsed:.1f}s")
        logger.info("=" * 60)

        return run

    # -----------------------------------------------------------------
    # INICIALIZACION
    # -----------------------------------------------------------------

    def _init_components(self) -> None:
        """Inicializa embedding model y LLM (si necesario)."""
        logger.info("Inicializando componentes...")
        self._embedding_model = load_embedding_model(
            base_url=self.config.infra.embedding_base_url,
            model_name=self.config.infra.embedding_model_name,
            model_type=self.config.infra.embedding_model_type,
        )
        logger.info(
            f"  Embedding: {self.config.infra.embedding_model_name} "
            f"({self.config.infra.embedding_model_type})"
        )

        # LLM requerido para estrategias contextuales
        needs_llm = self.config.strategy in (
            "CONTEXTUAL_VECTOR",
            "CONTEXTUAL_HYBRID",
            "CONTEXTUAL_HYBRID_RERANK",
        )
        if needs_llm:
            self._llm_service = AsyncLLMService(
                base_url=self.config.infra.llm_base_url,
                model_name=self.config.infra.llm_model_name,
                max_concurrent=self.config.infra.nim_max_concurrent,
                timeout_seconds=self.config.infra.nim_timeout,
                max_retries=self.config.infra.nim_max_retries,
            )
            logger.info(f"  LLM: {self.config.infra.llm_model_name}")

    # -----------------------------------------------------------------
    # DATASET
    # -----------------------------------------------------------------

    def _load_dataset(self):
        """Carga dataset via CookbookLoader."""
        logger.info("Cargando dataset...")
        loader = CookbookLoader(
            dataset_path=self.config.dataset_path,
            eval_path=self.config.eval_path,
        )
        return loader.load()

    # -----------------------------------------------------------------
    # INDEXACION
    # -----------------------------------------------------------------

    def _index_documents(self, dataset: LoadedDataset) -> int:
        """
        Indexa el corpus en el retriever.

        PC-1: usa doc.content directo, sin get_full_text(), sin title.
        Para estrategias CONTEXTUAL_*: pasa parent_content desde metadata.
        """
        logger.info("Indexando corpus...")

        strategy = self.config.get_strategy()
        parent_docs = dataset.metadata.get("parent_documents", {})

        retrieval_config = RetrievalConfig(
            strategy=strategy,
            retrieval_k=max(self.config.eval_k_values),
            hnsw_num_threads=self.config.retrieval.hnsw_num_threads,
            context_max_tokens=self.config.contextualize_max_tokens,
            context_batch_size=self.config.contextualize_batch_size,
        )

        if strategy == RetrievalStrategy.SIMPLE_VECTOR:
            self._retriever = SimpleVectorRetriever(
                config=retrieval_config,
                embedding_model=self._embedding_model,
                collection_name="cookbook_corpus",
                embedding_batch_size=self.config.infra.embedding_batch_size,
            )

        elif strategy == RetrievalStrategy.CONTEXTUAL_VECTOR:
            context_generator = LLMContextGenerator(
                llm_service=self._llm_service,
                max_tokens=self.config.contextualize_max_tokens,
                mode="document",
                document_prompt_template=ANTHROPIC_DOCUMENT_PROMPT,
                chunk_prompt_template=ANTHROPIC_CHUNK_PROMPT,
                system_prompt=ANTHROPIC_SYSTEM_PROMPT,
                context_position=self.config.context_position,
                max_parent_chars=32000,
                max_chunk_chars=8000,
            )

            # Cargar cache persistente si configurado
            self._load_context_cache(context_generator)

            inner = SimpleVectorRetriever(
                config=retrieval_config,
                embedding_model=self._embedding_model,
                collection_name="cookbook_corpus",
                embedding_batch_size=self.config.infra.embedding_batch_size,
            )
            self._retriever = ContextualRetriever(
                config=retrieval_config,
                embedding_model=self._embedding_model,
                context_generator=context_generator,
                inner_retriever=inner,
                collection_name="cookbook_corpus",
                embedding_batch_size=self.config.infra.embedding_batch_size,
            )

        else:
            raise NotImplementedError(
                f"Estrategia {strategy.name} sera implementada en fases posteriores"
            )

        # Construir lista de documentos para indexacion
        documents = []
        for doc in dataset.corpus.values():
            doc_dict: Dict[str, Any] = {
                "doc_id": doc.doc_id,
                "content": doc.content,  # PC-1: sin get_full_text()
                "title": "",  # PC-1: sin title
            }
            # Para estrategias contextuales: agregar parent_content
            if strategy != RetrievalStrategy.SIMPLE_VECTOR:
                parent_uuid = doc.metadata.get("parent_doc_id", "")
                doc_dict["parent_content"] = parent_docs.get(parent_uuid, "")
            documents.append(doc_dict)

        t0 = time.time()
        success = self._retriever.index_documents(documents)
        if not success:
            raise RuntimeError("Error indexando documentos")
        indexing_time = time.time() - t0

        # Guardar cache persistente si procede
        if strategy != RetrievalStrategy.SIMPLE_VECTOR:
            self._save_context_cache(indexing_time)

        logger.info(f"  Indexados {len(documents)} chunks en {indexing_time:.1f}s")
        return len(documents)

    def _load_context_cache(self, context_generator: LLMContextGenerator) -> None:
        """Carga cache persistente y pre-llena el cache in-memory del generator."""
        if not self.config.contexts_cache_path:
            return

        self._context_cache = ContextCache(self.config.contexts_cache_path)
        prompt_template = ANTHROPIC_DOCUMENT_PROMPT

        if self._context_cache.load(
            model_name=self.config.infra.llm_model_name,
            prompt_template=prompt_template,
        ):
            # Pre-llenar cache in-memory del generator
            for chunk_id, ctx in self._context_cache.get_all().items():
                context_generator._cache[
                    context_generator._make_cache_key(chunk_id, "")
                ] = ctx
            logger.info(
                f"  Cache pre-llenado: {self._context_cache.size} contextos"
            )

    def _save_context_cache(self, generation_time_s: float) -> None:
        """Guarda cache persistente con contextos nuevos."""
        if not self._context_cache:
            return

        # Extraer contextos del retriever
        if isinstance(self._retriever, ContextualRetriever):
            gen = self._retriever.context_generator
            for key, ctx in gen._cache.items():
                self._context_cache.put(key, ctx)

        self._context_cache.save(generation_time_s=generation_time_s)

    # -----------------------------------------------------------------
    # EVALUACION
    # -----------------------------------------------------------------

    def _evaluate_queries(
        self,
        dataset: LoadedDataset,
        eval_queries: List[CookbookEvalQuery],
    ) -> List[QueryEvaluationResult]:
        """
        Pipeline de evaluacion:
          0. Pre-embed queries en batch
          1. Retrieval sync
          2. Calcular Pass@k (= Recall@k) + validacion content-based
        """
        n = len(eval_queries)
        k_max = max(self.config.eval_k_values)

        # Fase 0: Pre-embed queries
        query_texts = [eq.query_text for eq in eval_queries]
        query_vectors = batch_embed_queries(
            query_texts=query_texts,
            base_url=self.config.infra.embedding_base_url,
            model_name=self.config.infra.embedding_model_name,
            model_type=self.config.infra.embedding_model_type,
            batch_size=self.config.infra.embedding_batch_size,
        )
        use_preembed = len(query_vectors) == n
        if not use_preembed:
            logger.warning(
                "Pre-embed fallido. Usando retrieval con embedding por query."
            )

        # Fase 1: Retrieval
        logger.info(
            f"  Retrieval: {n} queries "
            f"({'pre-embed' if use_preembed else 'per-query'})..."
        )
        t0 = time.time()
        results: List[QueryEvaluationResult] = []

        for i, eq in enumerate(eval_queries):
            # Retrieve
            if use_preembed:
                rr = self._retriever.retrieve_by_vector(
                    eq.query_text, query_vectors[i], top_k=k_max
                )
            else:
                rr = self._retriever.retrieve(eq.query_text, top_k=k_max)

            # Construir QueryRetrievalDetail (calcula recall_at_k automaticamente)
            detail = QueryRetrievalDetail(
                retrieved_doc_ids=rr.doc_ids,
                retrieved_contents=rr.contents,
                retrieval_scores=rr.scores,
                expected_doc_ids=eq.golden_chunk_ids,
                retrieval_time_ms=rr.retrieval_time_ms,
            )

            # Validacion content-based
            self._validate_pass_at_k(detail, eq.golden_contents, k_max)

            results.append(QueryEvaluationResult(
                query_id=eq.query_id,
                query_text=eq.query_text,
                dataset_name="cookbook",
                dataset_type=DatasetType.RETRIEVAL_ONLY,
                retrieval=detail,
                status=EvaluationStatus.COMPLETED,
            ))

        logger.info(f"  Retrieval completado en {time.time() - t0:.1f}s")
        return results

    def _validate_pass_at_k(
        self,
        retrieval: QueryRetrievalDetail,
        golden_contents: List[str],
        k: int,
    ) -> None:
        """
        Assertion: ID-matching y content-matching producen mismo resultado.

        Si divergen, indica bug en el mapeo de IDs del loader.
        """
        if not golden_contents:
            return

        id_based = retrieval.recall_at_k.get(k, 0.0)

        retrieved_set = {
            rc.strip()
            for rc in retrieval.retrieved_contents[:k]
        }
        found = sum(
            1 for gc in golden_contents
            if gc.strip() in retrieved_set
        )
        content_based = found / len(golden_contents)

        assert abs(id_based - content_based) < 1e-9, (
            f"Pass@{k} diverge: id_based={id_based}, "
            f"content_based={content_based}. "
            f"Bug en mapeo de IDs del loader."
        )

    # -----------------------------------------------------------------
    # BUILD RUN
    # -----------------------------------------------------------------

    def _build_run(
        self,
        run_id: str,
        dataset: LoadedDataset,
        query_results: List[QueryEvaluationResult],
        elapsed_seconds: float,
        indexed_corpus_size: int = 0,
    ) -> EvaluationRun:
        """Construye EvaluationRun a partir de resultados."""
        completed = [
            qr for qr in query_results
            if qr.status == EvaluationStatus.COMPLETED
        ]

        avg_recall: Dict[int, float] = {}
        if completed:
            nc = len(completed)
            recall_sums: Dict[int, float] = {}
            for qr in completed:
                for k, v in qr.retrieval.recall_at_k.items():
                    recall_sums[k] = recall_sums.get(k, 0.0) + v
            avg_recall = {k: v / nc for k, v in recall_sums.items()}

        config_snapshot = {
            "strategy": self.config.strategy,
            "eval_k_values": self.config.eval_k_values,
            "embedding_model": self.config.infra.embedding_model_name,
            "embedding_model_type": self.config.infra.embedding_model_type,
            "corpus_indexed": indexed_corpus_size,
        }

        return EvaluationRun(
            run_id=run_id,
            dataset_name="cookbook",
            embedding_model=self.config.infra.embedding_model_name,
            retrieval_strategy=self.config.strategy,
            config_snapshot=config_snapshot,
            num_queries_total=len(query_results),
            num_queries_evaluated=len(completed),
            num_queries_failed=len(query_results) - len(completed),
            total_documents=indexed_corpus_size,
            avg_recall_at_k=avg_recall,
            retrieval_complement_recall_at_k={
                k: 1.0 - v for k, v in avg_recall.items()
            },
            query_results=query_results,
            execution_time_seconds=elapsed_seconds,
            status=EvaluationStatus.COMPLETED,
        )

    # -----------------------------------------------------------------
    # CSV EXPORT
    # -----------------------------------------------------------------

    def _export_csv(self, run: EvaluationRun) -> None:
        """Exporta summary y detail CSV con Pass@k (k=5,10,20)."""
        self.config.ensure_directories()
        k_values = self.config.eval_k_values

        # Summary CSV
        summary_path = self.config.results_dir / f"{run.run_id}_summary.csv"
        row = {
            "run_id": run.run_id,
            "strategy": run.retrieval_strategy,
            "embedding_model": run.embedding_model,
            "queries_evaluated": run.num_queries_evaluated,
            "total_chunks": run.total_documents,
        }
        for k in k_values:
            pass_at_k = run.avg_recall_at_k.get(k, 0.0)
            row[f"pass_at_{k}"] = round(pass_at_k, 4)
            row[f"failure_rate_at_{k}"] = round(1.0 - pass_at_k, 4)
        row["execution_time_s"] = round(run.execution_time_seconds, 2)

        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writeheader()
            writer.writerow(row)
        logger.info(f"  Summary CSV: {summary_path}")

        # Detail CSV
        detail_path = self.config.results_dir / f"{run.run_id}_detail.csv"
        fieldnames = [
            "query_id", "query_text",
        ]
        for k in k_values:
            fieldnames.append(f"pass_at_{k}")
        fieldnames.extend([
            "n_golden", "n_retrieved", "retrieval_time_ms",
        ])

        with open(detail_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for qr in run.query_results:
                row = {
                    "query_id": qr.query_id,
                    "query_text": qr.query_text[:200],
                }
                for k in k_values:
                    row[f"pass_at_{k}"] = round(
                        qr.retrieval.recall_at_k.get(k, 0.0), 4
                    )
                row["n_golden"] = len(qr.retrieval.expected_doc_ids)
                row["n_retrieved"] = len(qr.retrieval.retrieved_doc_ids)
                row["retrieval_time_ms"] = round(
                    qr.retrieval.retrieval_time_ms, 1
                )
                writer.writerow(row)
        logger.info(f"  Detail CSV: {detail_path}")


__all__ = ["CookbookEvaluator"]
