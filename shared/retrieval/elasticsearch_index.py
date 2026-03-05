"""
Modulo: Elasticsearch BM25 Index
Descripcion: Indice BM25 basado en Elasticsearch, replica del cookbook oficial
             de Anthropic (platform.claude.com/cookbook/capabilities-contextual-embeddings-guide).

Ubicacion: shared/retrieval/elasticsearch_index.py

Requiere:
  - pip install elasticsearch
  - Elasticsearch corriendo (ej: docker run -d -p 9200:9200 ...)

API compatible con TantivyIndex / BM25Index para swap transparente en HybridRetriever.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from elasticsearch import Elasticsearch
    HAS_ELASTICSEARCH = True
except ImportError:
    HAS_ELASTICSEARCH = False
    Elasticsearch = None  # type: ignore


# Mapeo de codigos de idioma a analyzers de Elasticsearch
_ES_ANALYZERS = {
    "en": "english",
    "es": "spanish",
    "de": "german",
    "fr": "french",
    "it": "italian",
    "pt": "portuguese",
    "ru": "russian",
    "nl": "dutch",
    "sv": "swedish",
    "fi": "finnish",
    "da": "danish",
    "hu": "hungarian",
    "ro": "romanian",
    "tr": "turkish",
    "ar": "arabic",
}


class ElasticsearchBM25Index:
    """
    Indice BM25 basado en Elasticsearch.

    Replica la implementacion del cookbook oficial de Anthropic:
    - Analyzer por idioma (default: english)
    - Similarity BM25 (default de Elasticsearch)
    - multi_match sobre content y contextualized_content

    API compatible con TantivyIndex / BM25Index:
        build_index(documents) -> int
        search(query, top_k) -> List[Tuple[doc_id, content, score]]
        clear() -> None
        size -> int
    """

    def __init__(
        self,
        host: str = "http://localhost:9200",
        index_name: str = "contextual_bm25_index",
        language: str = "en",
    ):
        if not HAS_ELASTICSEARCH:
            raise ImportError(
                "elasticsearch no instalado: pip install elasticsearch"
            )

        self._host = host
        self._index_name = index_name
        self._language = language
        self._analyzer = _ES_ANALYZERS.get(language, "english")
        self._doc_count = 0

        self._es = Elasticsearch(host)

        # Verificar conexion
        if not self._es.ping():
            raise ConnectionError(
                f"No se pudo conectar a Elasticsearch en {host}. "
                "Asegurate de que el servicio esta corriendo."
            )

        logger.debug(
            f"ElasticsearchBM25Index: host={host}, "
            f"index={index_name}, analyzer={self._analyzer}"
        )

    def _create_index(self) -> None:
        """Crea el indice con settings del cookbook de Anthropic."""
        if self._es.indices.exists(index=self._index_name):
            self._es.indices.delete(index=self._index_name)

        index_settings = {
            "settings": {
                "analysis": {
                    "analyzer": {
                        "default": {"type": self._analyzer},
                    },
                },
                "similarity": {
                    "default": {"type": "BM25"},
                },
                "index.queries.cache.enabled": False,
            },
            "mappings": {
                "properties": {
                    "content": {
                        "type": "text",
                        "analyzer": self._analyzer,
                    },
                    "contextualized_content": {
                        "type": "text",
                        "analyzer": self._analyzer,
                    },
                    "doc_id": {
                        "type": "keyword",
                        "index": False,
                    },
                    "title": {
                        "type": "keyword",
                        "index": False,
                    },
                },
            },
        }

        self._es.indices.create(index=self._index_name, body=index_settings)
        logger.debug(f"Indice '{self._index_name}' creado")

    def build_index(self, documents: List[Dict[str, Any]]) -> int:
        """
        Indexa documentos en Elasticsearch.

        Args:
            documents: Lista de dicts con keys: doc_id, content, title (opcional).

        Returns:
            Numero de documentos indexados.
        """
        if not documents:
            return 0

        self._create_index()

        operations = []
        for doc in documents:
            doc_id = doc.get("doc_id", "")
            operations.append({"index": {"_index": self._index_name}})
            operations.append({
                "doc_id": doc_id,
                "content": doc.get("content", ""),
                "contextualized_content": doc.get("contextualized_content", ""),
                "title": doc.get("title", ""),
            })

        if operations:
            self._es.bulk(operations=operations, refresh=True)

        self._doc_count = len(documents)
        logger.debug(
            f"ElasticsearchBM25Index: {self._doc_count} documentos indexados"
        )
        return self._doc_count

    def search(
        self, query: str, top_k: int = 50
    ) -> List[Tuple[str, str, float]]:
        """
        Busqueda BM25 via Elasticsearch.

        Args:
            query: Texto de busqueda.
            top_k: Numero maximo de resultados.

        Returns:
            Lista de (doc_id, content, score) ordenada por score descendente.
        """
        if not query or not query.strip():
            return []

        if self._doc_count == 0:
            return []

        try:
            self._es.indices.refresh(index=self._index_name)

            response = self._es.search(
                index=self._index_name,
                query={
                    "multi_match": {
                        "query": query,
                        "fields": ["content", "contextualized_content"],
                    },
                },
                size=top_k,
            )

            results = []
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                results.append((
                    source.get("doc_id", ""),
                    source.get("content", ""),
                    float(hit["_score"]),
                ))

            return results

        except Exception as e:
            logger.warning(f"ElasticsearchBM25Index search error: {e}")
            return []

    def clear(self) -> None:
        """Elimina el indice de Elasticsearch."""
        try:
            if self._es.indices.exists(index=self._index_name):
                self._es.indices.delete(index=self._index_name)
        except Exception as e:
            logger.warning(f"Error eliminando indice: {e}")
        self._doc_count = 0

    @property
    def size(self) -> int:
        return self._doc_count


__all__ = ["ElasticsearchBM25Index", "HAS_ELASTICSEARCH"]
