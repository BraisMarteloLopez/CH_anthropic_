"""
Modulo: Hybrid Retriever
Descripcion: Busqueda hibrida Vector + BM25 + RRF.

Ubicacion: shared/retrieval/hybrid_retriever.py

v3: Tantivy como backend BM25 primario, rank_bm25 como fallback.
"""

from __future__ import annotations

import logging
import re
import time
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from shared.types import EmbeddingModelProtocol

from .core import (
    BaseRetriever,
    RetrievalConfig,
    RetrievalResult,
    RetrievalStrategy,
)

logger = logging.getLogger(__name__)

# --- Tantivy (preferido) ---
try:
    from .tantivy_index import TantivyIndex, HAS_TANTIVY
except ImportError:
    HAS_TANTIVY = False
    TantivyIndex = None  # type: ignore

# --- Elasticsearch (replica cookbook Anthropic) ---
try:
    from .elasticsearch_index import ElasticsearchBM25Index, HAS_ELASTICSEARCH
except ImportError:
    HAS_ELASTICSEARCH = False
    ElasticsearchBM25Index = None  # type: ignore

# --- rank_bm25 (fallback) ---
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    BM25Okapi = None


# =============================================================================
# LEGACY: BM25Index (rank_bm25, solo como fallback)
# =============================================================================

class BM25Index:
    """
    Wrapper sobre rank-bm25 (Python puro, in-memory).
    LEGACY: usar solo si Tantivy no esta disponible.
    """

    def __init__(self, language: str = "en"):
        if not HAS_BM25:
            raise ImportError("rank-bm25 no instalado: pip install rank-bm25")

        self._language = language
        self._bm25: Optional[BM25Okapi] = None
        self._doc_ids: List[str] = []
        self._doc_contents: List[str] = []
        self._tokenized_corpus: List[List[str]] = []

    def build_index(self, documents: List[Dict[str, Any]]) -> int:
        self._doc_ids = []
        self._doc_contents = []
        self._tokenized_corpus = []

        for doc in documents:
            content = doc.get("content", "")
            self._doc_ids.append(doc.get("doc_id", ""))
            self._doc_contents.append(content)
            # Simple tokenization (lowercase + split)
            tokens = content.lower().split()
            self._tokenized_corpus.append(tokens)

        if self._tokenized_corpus:
            self._bm25 = BM25Okapi(self._tokenized_corpus)

        logger.debug(f"BM25Index (legacy): {len(self._doc_ids)} documentos")
        return len(self._doc_ids)

    def search(
        self, query: str, top_k: int = 20
    ) -> List[Tuple[str, str, float]]:
        if self._bm25 is None or not self._doc_ids:
            return []

        query_tokens = query.lower().split()
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        indexed_scores = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )

        results = []
        for idx, score in indexed_scores[:top_k]:
            if score > 0:
                results.append(
                    (self._doc_ids[idx], self._doc_contents[idx], float(score))
                )
        return results

    def clear(self) -> None:
        self._bm25 = None
        self._doc_ids.clear()
        self._doc_contents.clear()
        self._tokenized_corpus.clear()

    @property
    def size(self) -> int:
        return len(self._doc_ids)


# =============================================================================
# RECIPROCAL RANK FUSION (RRF)
# =============================================================================

def reciprocal_rank_fusion(
    rankings: List[List[Tuple[str, float]]],
    weights: Optional[List[float]] = None,
    k: int = 60,
    top_n: int = 10,
    formula: str = "classic",
) -> List[Tuple[str, float]]:
    """
    Fusiona multiples rankings usando RRF.

    Formulas disponibles:
      - "classic": RRF_score(d) = SUM weight_i / (k + rank_i(d))
        k=60 por defecto. Formula original de Cormack et al.
      - "cookbook": score(d) = SUM weight_i / (rank_i(d))
        Equivalente a k=0. Despues del sort, re-asigna scores
        como 1/(new_rank+1) para normalizar. Variante de Anthropic cookbook.
    """
    if not rankings:
        return []

    num_rankings = len(rankings)

    if weights is None or len(weights) != num_rankings:
        if weights is not None and len(weights) != num_rankings:
            logger.warning(
                f"weights ({len(weights)}) != rankings ({num_rankings}). "
                "Pesos iguales."
            )
        weights = [1.0 / num_rankings] * num_rankings

    rrf_scores: Dict[str, float] = {}

    if formula == "cookbook":
        # Cookbook variant: k=0, weighted rank fusion
        for ranking_idx, ranking in enumerate(rankings):
            weight = weights[ranking_idx]
            for rank, (doc_id, _score) in enumerate(ranking, start=1):
                rrf_contribution = weight / rank
                rrf_scores[doc_id] = (
                    rrf_scores.get(doc_id, 0.0) + rrf_contribution
                )

        sorted_results = sorted(
            rrf_scores.items(), key=lambda x: x[1], reverse=True
        )

        # Re-score: 1/(new_rank+1) for normalized final scores
        reranked = [
            (doc_id, 1.0 / (new_rank + 1))
            for new_rank, (doc_id, _) in enumerate(sorted_results)
        ]
        return reranked[:top_n]
    else:
        # Classic RRF: weight / (k + rank)
        for ranking_idx, ranking in enumerate(rankings):
            weight = weights[ranking_idx]
            for rank, (doc_id, _score) in enumerate(ranking, start=1):
                rrf_contribution = weight / (k + rank)
                rrf_scores[doc_id] = (
                    rrf_scores.get(doc_id, 0.0) + rrf_contribution
                )

        sorted_results = sorted(
            rrf_scores.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_results[:top_n]


# =============================================================================
# HYBRID RETRIEVER
# =============================================================================

class HybridRetriever(BaseRetriever):
    """Retriever hibrido: Vector + BM25 (Tantivy) + RRF."""

    def __init__(
        self,
        config: RetrievalConfig,
        embedding_model: EmbeddingModelProtocol,
        collection_name: Optional[str] = None,
        embedding_batch_size: int = 0,
    ):
        super().__init__(config)

        self.embedding_model = embedding_model

        from .core import SimpleVectorRetriever

        self._vector_retriever = SimpleVectorRetriever(
            config,
            embedding_model,
            collection_name,
            embedding_batch_size=embedding_batch_size,
        )

        # Seleccion de backend BM25
        self._bm25_index, self._bm25_backend = self._init_bm25_backend(config)

        self._doc_map: Dict[str, str] = {}

    @staticmethod
    def _init_bm25_backend(config: RetrievalConfig):
        """
        Selecciona backend BM25 segun config.bm25_backend.

        Valores:
          - "auto": Tantivy > Elasticsearch > rank_bm25 (prioridad)
          - "tantivy": fuerza Tantivy
          - "elasticsearch": fuerza Elasticsearch
          - "rank_bm25": fuerza rank-bm25 (legacy)

        Returns:
            Tupla (bm25_index, backend_name)
        """
        backend = config.bm25_backend.lower().strip()
        lang = config.bm25_language

        if backend == "elasticsearch":
            if not HAS_ELASTICSEARCH:
                raise ImportError(
                    "BM25_BACKEND=elasticsearch pero elasticsearch no instalado: "
                    "pip install elasticsearch"
                )
            idx = ElasticsearchBM25Index(
                host=config.elasticsearch_host,
                language=lang,
            )
            logger.info(
                f"HybridRetriever: BM25 backend = Elasticsearch "
                f"(host={config.elasticsearch_host}, lang={lang})"
            )
            return idx, "elasticsearch"

        if backend == "tantivy":
            if not HAS_TANTIVY:
                raise ImportError(
                    "BM25_BACKEND=tantivy pero tantivy no instalado: "
                    "pip install tantivy"
                )
            idx = TantivyIndex(language=lang)
            logger.info(
                f"HybridRetriever: BM25 backend = Tantivy (lang={lang})"
            )
            return idx, "tantivy"

        if backend == "rank_bm25":
            if not HAS_BM25:
                raise ImportError(
                    "BM25_BACKEND=rank_bm25 pero rank-bm25 no instalado: "
                    "pip install rank-bm25"
                )
            idx = BM25Index(language=lang)
            logger.info("HybridRetriever: BM25 backend = rank_bm25 (legacy)")
            return idx, "rank_bm25"

        # auto: prioridad Tantivy > Elasticsearch > rank_bm25
        if HAS_TANTIVY:
            idx = TantivyIndex(language=lang)
            logger.info(
                f"HybridRetriever: BM25 backend = Tantivy [auto] (lang={lang})"
            )
            return idx, "tantivy"

        if HAS_ELASTICSEARCH:
            idx = ElasticsearchBM25Index(
                host=config.elasticsearch_host,
                language=lang,
            )
            logger.info(
                f"HybridRetriever: BM25 backend = Elasticsearch [auto] "
                f"(host={config.elasticsearch_host}, lang={lang})"
            )
            return idx, "elasticsearch"

        if HAS_BM25:
            idx = BM25Index(language=lang)
            logger.warning(
                "HybridRetriever: usando rank_bm25 [auto] (legacy, in-memory)"
            )
            return idx, "rank_bm25"

        raise ImportError(
            "Se requiere tantivy (pip install tantivy), "
            "elasticsearch (pip install elasticsearch), o "
            "rank-bm25 (pip install rank-bm25) para HybridRetriever"
        )

    def index_documents(
        self,
        documents: List[Dict[str, Any]],
        collection_name: Optional[str] = None,
    ) -> bool:
        if not documents:
            logger.warning("index_documents llamado con lista vacia")
            return False

        start_time = time.perf_counter()

        for doc in documents:
            self._doc_map[doc.get("doc_id", "")] = doc.get("content", "")

        vector_ok = self._vector_retriever.index_documents(
            documents, collection_name
        )
        bm25_count = self._bm25_index.build_index(documents)
        bm25_ok = bm25_count > 0

        self._is_indexed = vector_ok and bm25_ok
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            f"HybridRetriever: indexacion {elapsed_ms:.0f}ms. "
            f"Vector: {'OK' if vector_ok else 'FAIL'}, BM25: {bm25_count} docs"
        )
        return self._is_indexed

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        k = top_k or self.config.retrieval_k
        pre_k = self.config.pre_fusion_k
        start_time = time.perf_counter()

        # Vector search
        vector_result = self._vector_retriever.retrieve(query, top_k=pre_k)
        vector_ranking = list(
            zip(vector_result.doc_ids, vector_result.scores)
        )

        # BM25 search
        bm25_results = self._bm25_index.search(query, top_k=pre_k)
        bm25_ranking = [
            (doc_id, score)
            for doc_id, _content, score in bm25_results
        ]

        return self._fuse_and_build_result(
            vector_ranking, bm25_ranking, vector_result,
            k, pre_k, start_time,
        )

    def retrieve_by_vector(
        self,
        query_text: str,
        query_vector: List[float],
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        """
        Retrieval hibrido con vector pre-computado.

        Componente vectorial: usa query_vector directamente (sin llamada NIM).
        Componente BM25: usa query_text (busqueda local Tantivy).
        """
        k = top_k or self.config.retrieval_k
        pre_k = self.config.pre_fusion_k
        start_time = time.perf_counter()

        # Vector search con vector pre-computado
        vector_result = self._vector_retriever.retrieve_by_vector(
            query_text, query_vector, top_k=pre_k
        )
        vector_ranking = list(
            zip(vector_result.doc_ids, vector_result.scores)
        )

        # BM25 search (local, no necesita embedding)
        bm25_results = self._bm25_index.search(query_text, top_k=pre_k)
        bm25_ranking = [
            (doc_id, score)
            for doc_id, _content, score in bm25_results
        ]

        return self._fuse_and_build_result(
            vector_ranking, bm25_ranking, vector_result,
            k, pre_k, start_time,
        )

    def _fuse_and_build_result(
        self,
        vector_ranking: List[Tuple[str, float]],
        bm25_ranking: List[Tuple[str, float]],
        vector_result: RetrievalResult,
        k: int,
        pre_k: int,
        start_time: float,
    ) -> RetrievalResult:
        """Fusiona rankings via RRF y construye RetrievalResult."""
        # RRF fusion
        fused = reciprocal_rank_fusion(
            rankings=[vector_ranking, bm25_ranking],
            weights=[self.config.vector_weight, self.config.bm25_weight],
            k=self.config.rrf_k,
            top_n=k,
            formula=self.config.rrf_formula,
        )

        doc_ids = []
        contents = []
        scores = []
        vector_scores_map = dict(vector_ranking)
        bm25_scores_map = dict(bm25_ranking)
        per_doc_vector = []
        per_doc_bm25 = []

        for doc_id, rrf_score in fused:
            doc_ids.append(doc_id)
            scores.append(rrf_score)

            content = self._doc_map.get(doc_id, "")
            if not content:
                for vid, vc in zip(
                    vector_result.doc_ids, vector_result.contents
                ):
                    if vid == doc_id:
                        content = vc
                        break
            contents.append(content)

            per_doc_vector.append(vector_scores_map.get(doc_id, 0.0))
            per_doc_bm25.append(bm25_scores_map.get(doc_id, 0.0))

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return RetrievalResult(
            doc_ids=doc_ids,
            contents=contents,
            scores=scores,
            vector_scores=per_doc_vector,
            bm25_scores=per_doc_bm25,
            retrieval_time_ms=elapsed_ms,
            strategy_used=RetrievalStrategy.CONTEXTUAL_HYBRID,
            metadata={
                "rrf_k": self.config.rrf_k,
                "rrf_formula": self.config.rrf_formula,
                "pre_fusion_k": pre_k,
                "vector_weight": self.config.vector_weight,
                "bm25_weight": self.config.bm25_weight,
                "vector_candidates": len(vector_ranking),
                "bm25_candidates": len(bm25_ranking),
                "bm25_backend": self._bm25_backend,
            },
        )

    def clear_index(self) -> None:
        self._vector_retriever.clear_index()
        self._bm25_index.clear()
        self._doc_map.clear()
        self._is_indexed = False
        logger.debug("HybridRetriever: indices limpiados")


__all__ = [
    "HybridRetriever",
    "BM25Index",
    "reciprocal_rank_fusion",
    "HAS_BM25",
    "HAS_TANTIVY",
    "HAS_ELASTICSEARCH",
]
