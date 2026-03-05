"""
Tests para ElasticsearchBM25Index.

Cobertura:
  - Constructor: verificacion de import, analyzer por idioma
  - build_index con lista vacia retorna 0
  - search con query vacia retorna []
  - search con doc_count=0 retorna []
  - Integracion con Elasticsearch real (skip si no disponible)
  - Seleccion de backend en HybridRetriever via _init_bm25_backend

Usa mocks para tests unitarios (sin Elasticsearch real).
Tests de integracion marcados con @pytest.mark.elasticsearch.
"""
import pytest
from unittest.mock import patch, MagicMock

from shared.retrieval.elasticsearch_index import (
    ElasticsearchBM25Index,
    HAS_ELASTICSEARCH,
    _ES_ANALYZERS,
)


# =================================================================
# Tests de mapeo de idiomas (Python puro, sin ES)
# =================================================================

def test_es_analyzers_mapping():
    """Verifica mapeo de idiomas a analyzers de ES."""
    assert _ES_ANALYZERS["en"] == "english"
    assert _ES_ANALYZERS["es"] == "spanish"
    assert _ES_ANALYZERS["fr"] == "french"
    assert _ES_ANALYZERS["de"] == "german"


def test_es_analyzers_unknown_language_defaults_to_english():
    """Idioma desconocido usa 'english' como fallback."""
    assert _ES_ANALYZERS.get("xx", "english") == "english"


# =================================================================
# Tests unitarios con mock de Elasticsearch client
# =================================================================

@patch("shared.retrieval.elasticsearch_index.HAS_ELASTICSEARCH", True)
class TestElasticsearchBM25IndexUnit:
    """Tests unitarios con mock del cliente ES."""

    def _make_index(self, language="en"):
        """Helper: crea indice con ES mockeado."""
        with patch("shared.retrieval.elasticsearch_index.Elasticsearch") as MockES:
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            MockES.return_value = mock_client
            idx = ElasticsearchBM25Index(
                host="http://localhost:9200",
                language=language,
            )
            return idx, mock_client

    def test_constructor_sets_analyzer(self):
        """Constructor configura analyzer segun idioma."""
        idx, _ = self._make_index(language="es")
        assert idx._analyzer == "spanish"

    def test_constructor_unknown_lang_defaults_english(self):
        """Idioma desconocido usa analyzer english."""
        idx, _ = self._make_index(language="xx")
        assert idx._analyzer == "english"

    def test_build_index_empty_returns_zero(self):
        """build_index con lista vacia retorna 0."""
        idx, _ = self._make_index()
        result = idx.build_index([])
        assert result == 0
        assert idx.size == 0

    def test_build_index_calls_bulk(self):
        """build_index llama bulk con los documentos."""
        idx, mock_client = self._make_index()
        mock_client.indices.exists.return_value = False

        docs = [
            {"doc_id": "d1", "content": "hello world", "title": "Test"},
            {"doc_id": "d2", "content": "foo bar"},
        ]
        count = idx.build_index(docs)

        assert count == 2
        assert idx.size == 2
        mock_client.bulk.assert_called_once()
        mock_client.indices.create.assert_called_once()

    def test_search_empty_query_returns_empty(self):
        """Query vacia retorna lista vacia."""
        idx, _ = self._make_index()
        assert idx.search("", top_k=10) == []
        assert idx.search("   ", top_k=10) == []

    def test_search_zero_docs_returns_empty(self):
        """Sin documentos indexados retorna lista vacia."""
        idx, _ = self._make_index()
        assert idx._doc_count == 0
        assert idx.search("test query", top_k=10) == []

    def test_search_returns_results(self):
        """search parsea respuesta de ES correctamente."""
        idx, mock_client = self._make_index()
        idx._doc_count = 2

        mock_client.indices.refresh.return_value = None
        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_score": 5.5,
                        "_source": {
                            "doc_id": "d1",
                            "content": "hello world",
                        },
                    },
                    {
                        "_score": 3.2,
                        "_source": {
                            "doc_id": "d2",
                            "content": "foo bar",
                        },
                    },
                ],
            },
        }

        results = idx.search("hello", top_k=10)
        assert len(results) == 2
        assert results[0] == ("d1", "hello world", 5.5)
        assert results[1] == ("d2", "foo bar", 3.2)

    def test_search_exception_returns_empty(self):
        """Error en search retorna lista vacia."""
        idx, mock_client = self._make_index()
        idx._doc_count = 1
        mock_client.indices.refresh.side_effect = Exception("connection lost")

        results = idx.search("test", top_k=10)
        assert results == []

    def test_clear_deletes_index(self):
        """clear() elimina el indice y resetea doc_count."""
        idx, mock_client = self._make_index()
        idx._doc_count = 5
        mock_client.indices.exists.return_value = True

        idx.clear()

        assert idx.size == 0
        mock_client.indices.delete.assert_called_once()

    def test_size_property(self):
        """size refleja doc_count."""
        idx, _ = self._make_index()
        assert idx.size == 0
        idx._doc_count = 42
        assert idx.size == 42


# =================================================================
# Tests de seleccion de backend en HybridRetriever
# =================================================================

class TestBM25BackendSelection:
    """Tests para _init_bm25_backend."""

    def test_explicit_elasticsearch_backend(self):
        """BM25_BACKEND=elasticsearch selecciona ElasticsearchBM25Index."""
        from shared.retrieval.core import RetrievalConfig

        config = RetrievalConfig(
            bm25_backend="elasticsearch",
            elasticsearch_host="http://localhost:9200",
        )

        with patch("shared.retrieval.hybrid_retriever.HAS_ELASTICSEARCH", True), \
             patch("shared.retrieval.hybrid_retriever.ElasticsearchBM25Index") as MockES:
            mock_instance = MagicMock()
            MockES.return_value = mock_instance

            from shared.retrieval.hybrid_retriever import HybridRetriever
            idx, name = HybridRetriever._init_bm25_backend(config)

            assert name == "elasticsearch"
            MockES.assert_called_once_with(
                host="http://localhost:9200",
                language="en",
            )

    def test_explicit_tantivy_backend(self):
        """BM25_BACKEND=tantivy selecciona TantivyIndex."""
        from shared.retrieval.core import RetrievalConfig

        config = RetrievalConfig(bm25_backend="tantivy")

        with patch("shared.retrieval.hybrid_retriever.HAS_TANTIVY", True), \
             patch("shared.retrieval.hybrid_retriever.TantivyIndex") as MockTantivy:
            mock_instance = MagicMock()
            MockTantivy.return_value = mock_instance

            from shared.retrieval.hybrid_retriever import HybridRetriever
            idx, name = HybridRetriever._init_bm25_backend(config)

            assert name == "tantivy"

    def test_explicit_rank_bm25_backend(self):
        """BM25_BACKEND=rank_bm25 selecciona BM25Index."""
        from shared.retrieval.core import RetrievalConfig

        config = RetrievalConfig(bm25_backend="rank_bm25")

        with patch("shared.retrieval.hybrid_retriever.HAS_BM25", True), \
             patch("shared.retrieval.hybrid_retriever.BM25Index") as MockBM25:
            mock_instance = MagicMock()
            MockBM25.return_value = mock_instance

            from shared.retrieval.hybrid_retriever import HybridRetriever
            idx, name = HybridRetriever._init_bm25_backend(config)

            assert name == "rank_bm25"

    def test_elasticsearch_not_installed_raises(self):
        """BM25_BACKEND=elasticsearch sin libreria lanza ImportError."""
        from shared.retrieval.core import RetrievalConfig

        config = RetrievalConfig(bm25_backend="elasticsearch")

        with patch("shared.retrieval.hybrid_retriever.HAS_ELASTICSEARCH", False):
            from shared.retrieval.hybrid_retriever import HybridRetriever
            with pytest.raises(ImportError, match="elasticsearch"):
                HybridRetriever._init_bm25_backend(config)

    def test_auto_falls_through_to_available(self):
        """BM25_BACKEND=auto selecciona el primer backend disponible."""
        from shared.retrieval.core import RetrievalConfig

        config = RetrievalConfig(bm25_backend="auto")

        # Solo rank_bm25 disponible
        with patch("shared.retrieval.hybrid_retriever.HAS_TANTIVY", False), \
             patch("shared.retrieval.hybrid_retriever.HAS_ELASTICSEARCH", False), \
             patch("shared.retrieval.hybrid_retriever.HAS_BM25", True), \
             patch("shared.retrieval.hybrid_retriever.BM25Index") as MockBM25:
            MockBM25.return_value = MagicMock()

            from shared.retrieval.hybrid_retriever import HybridRetriever
            idx, name = HybridRetriever._init_bm25_backend(config)

            assert name == "rank_bm25"

    def test_auto_no_backend_raises(self):
        """BM25_BACKEND=auto sin ningun backend lanza ImportError."""
        from shared.retrieval.core import RetrievalConfig

        config = RetrievalConfig(bm25_backend="auto")

        with patch("shared.retrieval.hybrid_retriever.HAS_TANTIVY", False), \
             patch("shared.retrieval.hybrid_retriever.HAS_ELASTICSEARCH", False), \
             patch("shared.retrieval.hybrid_retriever.HAS_BM25", False):
            from shared.retrieval.hybrid_retriever import HybridRetriever
            with pytest.raises(ImportError):
                HybridRetriever._init_bm25_backend(config)


# =================================================================
# Tests de integracion (requieren Elasticsearch real)
# =================================================================

def _elasticsearch_available():
    """Verifica si hay un Elasticsearch accesible."""
    if not HAS_ELASTICSEARCH:
        return False
    try:
        from elasticsearch import Elasticsearch
        es = Elasticsearch("http://localhost:9200")
        return es.ping()
    except Exception:
        return False


@pytest.mark.skipif(
    not _elasticsearch_available(),
    reason="Elasticsearch no disponible en localhost:9200",
)
class TestElasticsearchBM25Integration:
    """Tests de integracion con Elasticsearch real."""

    def test_index_and_search(self):
        """Indexar y buscar documentos."""
        idx = ElasticsearchBM25Index(
            index_name="test_integration_bm25",
            language="en",
        )
        try:
            docs = [
                {"doc_id": "d1", "content": "artificial intelligence machine learning", "title": "AI"},
                {"doc_id": "d2", "content": "natural language processing text analysis", "title": "NLP"},
                {"doc_id": "d3", "content": "computer vision image recognition deep learning", "title": "CV"},
            ]
            count = idx.build_index(docs)
            assert count == 3

            results = idx.search("artificial intelligence", top_k=2)
            assert len(results) > 0
            assert results[0][0] == "d1"
        finally:
            idx.clear()

    def test_search_with_contextualized_content(self):
        """Busca sobre content y contextualized_content."""
        idx = ElasticsearchBM25Index(
            index_name="test_integration_contextual",
            language="en",
        )
        try:
            docs = [
                {
                    "doc_id": "d1",
                    "content": "simple function code",
                    "contextualized_content": "This chunk implements authentication logic for user login",
                },
                {
                    "doc_id": "d2",
                    "content": "database connection pool",
                    "contextualized_content": "This chunk manages PostgreSQL connections",
                },
            ]
            count = idx.build_index(docs)
            assert count == 2

            # Buscar por termino en contextualized_content
            results = idx.search("authentication login", top_k=5)
            assert len(results) > 0
            assert results[0][0] == "d1"
        finally:
            idx.clear()

    def test_rebuild_replaces_index(self):
        """build_index dos veces reemplaza el indice anterior."""
        idx = ElasticsearchBM25Index(
            index_name="test_integration_rebuild",
            language="en",
        )
        try:
            docs1 = [{"doc_id": "d1", "content": "first version"}]
            idx.build_index(docs1)

            docs2 = [
                {"doc_id": "d2", "content": "second version new"},
                {"doc_id": "d3", "content": "third document"},
            ]
            count = idx.build_index(docs2)
            assert count == 2
            assert idx.size == 2

            results = idx.search("second version", top_k=5)
            doc_ids = [r[0] for r in results]
            assert "d2" in doc_ids
            assert "d1" not in doc_ids
        finally:
            idx.clear()
