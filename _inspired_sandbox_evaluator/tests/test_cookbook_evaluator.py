"""
Tests para CookbookEvaluator (SIMPLE_VECTOR).

Valida pipeline completo con mocks:
  load -> index -> retrieve -> evaluate -> build_run

- Pass@k content-based assertion
- CSV con k=5,10,20
- EvaluationRun correctamente construido
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from sandbox_cookbook.config import CookbookConfig
from sandbox_cookbook.evaluator import CookbookEvaluator
from sandbox_cookbook.loader import CookbookEvalQuery
from shared.types import (
    DatasetType,
    EvaluationRun,
    EvaluationStatus,
    LoadedDataset,
    NormalizedDocument,
    NormalizedQuery,
    QueryRetrievalDetail,
)
from shared.retrieval.core import RetrievalResult, RetrievalStrategy


# =============================================================================
# FIXTURES
# =============================================================================

def _make_test_dataset():
    """Construye LoadedDataset y CookbookEvalQuery list para tests."""
    corpus = {
        "doc_1_chunk_0": NormalizedDocument(
            doc_id="doc_1_chunk_0",
            content="function setup() { return config; }",
            title=None,
            metadata={"parent_doc_id": "uuid-aaa", "original_index": 0},
        ),
        "doc_1_chunk_1": NormalizedDocument(
            doc_id="doc_1_chunk_1",
            content="function teardown() { cleanup(); }",
            title=None,
            metadata={"parent_doc_id": "uuid-aaa", "original_index": 1},
        ),
        "doc_2_chunk_0": NormalizedDocument(
            doc_id="doc_2_chunk_0",
            content="class Handler { handle(event) {} }",
            title=None,
            metadata={"parent_doc_id": "uuid-bbb", "original_index": 0},
        ),
    }

    queries = [
        NormalizedQuery(
            query_id="q_0",
            query_text="What does setup do?",
            relevant_doc_ids=["doc_1_chunk_0"],
        ),
        NormalizedQuery(
            query_id="q_1",
            query_text="How does Handler work?",
            relevant_doc_ids=["doc_2_chunk_0"],
        ),
    ]

    eval_queries = [
        CookbookEvalQuery(
            query_id="q_0",
            query_text="What does setup do?",
            golden_chunk_ids=["doc_1_chunk_0"],
            golden_contents=["function setup() { return config; }"],
        ),
        CookbookEvalQuery(
            query_id="q_1",
            query_text="How does Handler work?",
            golden_chunk_ids=["doc_2_chunk_0"],
            golden_contents=["class Handler { handle(event) {} }"],
        ),
    ]

    dataset = LoadedDataset(
        name="cookbook",
        dataset_type=DatasetType.RETRIEVAL_ONLY,
        queries=queries,
        corpus=corpus,
        total_queries=2,
        total_corpus=3,
        load_status="success",
        metadata={
            "parent_documents": {
                "uuid-aaa": "Full doc 1",
                "uuid-bbb": "Full doc 2",
            },
        },
    )

    return dataset, eval_queries


class MockRetriever:
    """Mock retriever que retorna resultados predefinidos."""

    def __init__(self, corpus):
        self.corpus = corpus
        self._is_indexed = False

    def index_documents(self, documents, collection_name=None):
        self._is_indexed = True
        return True

    def retrieve(self, query: str, top_k: Optional[int] = None):
        # Retorna todos los docs del corpus
        doc_ids = list(self.corpus.keys())[:top_k or 20]
        contents = [self.corpus[did].content for did in doc_ids]
        scores = [1.0 / (i + 1) for i in range(len(doc_ids))]
        return RetrievalResult(
            doc_ids=doc_ids,
            contents=contents,
            scores=scores,
            strategy_used=RetrievalStrategy.SIMPLE_VECTOR,
            retrieval_time_ms=1.0,
        )

    def retrieve_by_vector(self, query_text, query_vector, top_k=None):
        return self.retrieve(query_text, top_k)

    @property
    def is_indexed(self):
        return self._is_indexed


# =============================================================================
# TESTS
# =============================================================================

class TestCookbookEvaluatorPipeline:
    """Tests del pipeline completo con mocks."""

    def _make_evaluator(self, tmp_path):
        """Crea evaluator con config apuntando a tmp_path."""
        config = CookbookConfig(
            dataset_path=tmp_path / "corpus.json",
            eval_path=tmp_path / "eval.jsonl",
            results_dir=tmp_path / "results",
            strategy="SIMPLE_VECTOR",
            eval_k_values=[5, 10, 20],
        )
        config.infra.embedding_base_url = "http://test:8080/v1"
        config.infra.embedding_model_name = "test-model"
        config.infra.embedding_batch_size = 50

        evaluator = CookbookEvaluator(config)
        return evaluator, config

    def test_evaluate_queries_produces_results(self, tmp_path):
        """Pipeline de evaluacion produce QueryEvaluationResult por query."""
        dataset, eval_queries = _make_test_dataset()
        evaluator, config = self._make_evaluator(tmp_path)

        # Inject mock retriever
        evaluator._retriever = MockRetriever(dataset.corpus)

        # Mock batch_embed_queries to return empty (triggers per-query fallback)
        with patch("sandbox_cookbook.evaluator.batch_embed_queries", return_value=[]):
            results = evaluator._evaluate_queries(dataset, eval_queries)

        assert len(results) == 2
        for r in results:
            assert r.status == EvaluationStatus.COMPLETED
            assert r.dataset_name == "cookbook"
            assert r.dataset_type == DatasetType.RETRIEVAL_ONLY

    def test_recall_at_k_calculated(self, tmp_path):
        """recall_at_k (=pass_at_k) se calcula correctamente."""
        dataset, eval_queries = _make_test_dataset()
        evaluator, config = self._make_evaluator(tmp_path)
        evaluator._retriever = MockRetriever(dataset.corpus)

        with patch("sandbox_cookbook.evaluator.batch_embed_queries", return_value=[]):
            results = evaluator._evaluate_queries(dataset, eval_queries)

        # MockRetriever retorna todos los docs, golden esta en top-k -> recall=1.0
        for r in results:
            assert r.retrieval.recall_at_k[20] == 1.0

    def test_build_run_aggregation(self, tmp_path):
        """_build_run agrega recall_at_k correctamente."""
        dataset, eval_queries = _make_test_dataset()
        evaluator, config = self._make_evaluator(tmp_path)
        evaluator._retriever = MockRetriever(dataset.corpus)

        with patch("sandbox_cookbook.evaluator.batch_embed_queries", return_value=[]):
            results = evaluator._evaluate_queries(dataset, eval_queries)

        run = evaluator._build_run(
            "test_run", dataset, results, 1.0, len(dataset.corpus)
        )

        assert run.run_id == "test_run"
        assert run.dataset_name == "cookbook"
        assert run.retrieval_strategy == "SIMPLE_VECTOR"
        assert run.num_queries_evaluated == 2
        assert run.num_queries_failed == 0
        assert run.total_documents == 3
        assert run.status == EvaluationStatus.COMPLETED

        # avg_recall_at_k[20] = 1.0 (both queries have golden in top-20)
        assert abs(run.avg_recall_at_k.get(20, 0.0) - 1.0) < 1e-9

    def test_export_csv(self, tmp_path):
        """CSV se genera con las columnas correctas."""
        dataset, eval_queries = _make_test_dataset()
        evaluator, config = self._make_evaluator(tmp_path)
        evaluator._retriever = MockRetriever(dataset.corpus)

        with patch("sandbox_cookbook.evaluator.batch_embed_queries", return_value=[]):
            results = evaluator._evaluate_queries(dataset, eval_queries)

        run = evaluator._build_run(
            "test_csv", dataset, results, 1.0, len(dataset.corpus)
        )
        evaluator._export_csv(run)

        # Check summary CSV exists
        summary_path = tmp_path / "results" / "test_csv_summary.csv"
        assert summary_path.exists(), f"Summary CSV not found at {summary_path}"

        # Check detail CSV exists
        detail_path = tmp_path / "results" / "test_csv_detail.csv"
        assert detail_path.exists()

        # Read and verify summary CSV content
        import csv
        with open(summary_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        row = rows[0]
        assert row["strategy"] == "SIMPLE_VECTOR"
        assert "pass_at_5" in row
        assert "pass_at_10" in row
        assert "pass_at_20" in row
        assert "failure_rate_at_20" in row

        # Read and verify detail CSV content
        with open(detail_path) as f:
            reader = csv.DictReader(f)
            detail_rows = list(reader)
        assert len(detail_rows) == 2
        assert "pass_at_5" in detail_rows[0]
        assert "n_golden" in detail_rows[0]


class TestCookbookEvaluatorValidation:
    """Tests de validacion content-based."""

    def test_validate_pass_at_k_matching(self):
        """No assertion error when ID and content matching agree."""
        evaluator = CookbookEvaluator.__new__(CookbookEvaluator)

        detail = QueryRetrievalDetail(
            retrieved_doc_ids=["doc_1_chunk_0", "doc_1_chunk_1"],
            retrieved_contents=["function setup() {", "return config;"],
            retrieval_scores=[0.9, 0.8],
            expected_doc_ids=["doc_1_chunk_0"],
        )
        golden_contents = ["function setup() {"]

        # Should not raise
        evaluator._validate_pass_at_k(detail, golden_contents, k=20)

    def test_validate_pass_at_k_divergence_raises(self):
        """AssertionError when ID-based and content-based pass@k diverge."""
        evaluator = CookbookEvaluator.__new__(CookbookEvaluator)

        # ID-based: doc_1_chunk_0 is in top-20, so recall=1.0
        # Content-based: golden content doesn't match any retrieved content
        detail = QueryRetrievalDetail(
            retrieved_doc_ids=["doc_1_chunk_0"],
            retrieved_contents=["DIFFERENT CONTENT"],
            retrieval_scores=[0.9],
            expected_doc_ids=["doc_1_chunk_0"],
        )
        golden_contents = ["function setup() {"]

        with pytest.raises(AssertionError, match="Pass@"):
            evaluator._validate_pass_at_k(detail, golden_contents, k=20)

    def test_validate_empty_golden_is_ok(self):
        """Empty golden_contents skips validation."""
        evaluator = CookbookEvaluator.__new__(CookbookEvaluator)

        detail = QueryRetrievalDetail(
            retrieved_doc_ids=["doc_1_chunk_0"],
            retrieved_contents=["anything"],
            retrieval_scores=[0.9],
            expected_doc_ids=[],
        )

        # Should not raise
        evaluator._validate_pass_at_k(detail, [], k=20)


class TestCookbookEvaluatorIndexing:
    """Tests de indexacion."""

    def test_index_uses_content_not_full_text(self, tmp_path):
        """PC-1: indexa con doc.content, no get_full_text()."""
        dataset, eval_queries = _make_test_dataset()
        evaluator, config = self._make_evaluator(tmp_path)

        # Mock SimpleVectorRetriever to capture indexed documents
        indexed_docs = []

        class CapturingRetriever:
            def __init__(self):
                self._is_indexed = False

            def index_documents(self, documents, collection_name=None):
                indexed_docs.extend(documents)
                self._is_indexed = True
                return True

            @property
            def is_indexed(self):
                return self._is_indexed

        with patch("sandbox_cookbook.evaluator.SimpleVectorRetriever") as mock_cls:
            mock_cls.return_value = CapturingRetriever()
            evaluator._init_components = lambda: None  # skip real init
            evaluator._embedding_model = MagicMock()
            evaluator._index_documents(dataset)

        # Verify documents use content directly, not get_full_text()
        for doc in indexed_docs:
            assert doc["title"] == "", f"Title should be empty, got: {doc['title']}"
            # Content should be the raw chunk content, not "title\n\ncontent"
            corpus_doc = dataset.corpus.get(doc["doc_id"])
            if corpus_doc:
                assert doc["content"] == corpus_doc.content

    def _make_evaluator(self, tmp_path):
        config = CookbookConfig(
            dataset_path=tmp_path / "corpus.json",
            eval_path=tmp_path / "eval.jsonl",
            results_dir=tmp_path / "results",
            strategy="SIMPLE_VECTOR",
        )
        config.infra.embedding_base_url = "http://test:8080/v1"
        config.infra.embedding_model_name = "test-model"
        return CookbookEvaluator(config), config

    def test_unsupported_strategy_raises(self, tmp_path):
        """Estrategias no implementadas lanzan NotImplementedError."""
        dataset, _ = _make_test_dataset()
        config = CookbookConfig(
            dataset_path=tmp_path / "corpus.json",
            eval_path=tmp_path / "eval.jsonl",
            strategy="CONTEXTUAL_HYBRID",  # Fase 3, no implementada aun
        )
        evaluator = CookbookEvaluator(config)
        evaluator._embedding_model = MagicMock()

        with pytest.raises(NotImplementedError):
            evaluator._index_documents(dataset)


# =============================================================================
# STANDALONE
# =============================================================================

if __name__ == "__main__":
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    for cls in [
        TestCookbookEvaluatorPipeline,
        TestCookbookEvaluatorValidation,
        TestCookbookEvaluatorIndexing,
    ]:
        instance = cls()
        for name in dir(instance):
            if name.startswith("test_"):
                print(f"  {cls.__name__}.{name}...", end=" ")
                try:
                    method = getattr(instance, name)
                    import inspect
                    sig = inspect.signature(method)
                    if "tmp_path" in sig.parameters:
                        method(tmp)
                    else:
                        method()
                    print("OK")
                except Exception as e:
                    print(f"FAIL: {e}")
    print("Done!")
