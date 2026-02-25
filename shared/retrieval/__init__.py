"""
Retrieval strategies for RAG evaluation.

Estrategias soportadas:
  - SIMPLE_VECTOR: embedding search puro via ChromaDB
  - CONTEXTUAL_VECTOR: enriquecimiento LLM + busqueda vectorial pura
  - CONTEXTUAL_HYBRID: enriquecimiento LLM + BM25+Vector+RRF
  - CONTEXTUAL_HYBRID_RERANK: CONTEXTUAL_HYBRID + cross-encoder reranking
"""

from .core import (
    RetrievalStrategy,
    RetrievalConfig,
    RetrievalResult,
    BaseRetriever,
    SimpleVectorRetriever,
)
from .hybrid_retriever import HybridRetriever
from .contextual_retriever import (
    ContextualRetriever,
    LLMContextGenerator,
    EnrichedChunk,
)

__all__ = [
    "RetrievalStrategy",
    "RetrievalConfig",
    "RetrievalResult",
    "BaseRetriever",
    "SimpleVectorRetriever",
    "HybridRetriever",
    "ContextualRetriever",
    "LLMContextGenerator",
    "EnrichedChunk",
]
