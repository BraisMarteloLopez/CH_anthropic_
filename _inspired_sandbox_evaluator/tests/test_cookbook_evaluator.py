"""
Tests para CookbookEvaluator (SIMPLE_VECTOR).

Valida pipeline completo con mocks:
  load -> index -> retrieve -> evaluate -> build_run

- Pass@k content-based assertion
- CSV con k=5,10,20
- EvaluationRun correctamente construido
"""

import json
import sys
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
from shared.retrieval.core import RetrievalConfig, RetrievalResult, RetrievalStrategy
from shared.retrieval.hybrid_retriever import HybridRetriever


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
        """ValueError when ID-based and content-based pass@k diverge."""
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

        with pytest.raises(ValueError, match="Pass@"):
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


class TestCookbookEvaluatorHybrid:
    """Tests para CONTEXTUAL_HYBRID strategy (Fase 3)."""

    def test_contextual_hybrid_creates_hybrid_inner(self, tmp_path):
        """CONTEXTUAL_HYBRID crea HybridRetriever como inner retriever."""
        dataset, _ = _make_test_dataset()
        config = CookbookConfig(
            dataset_path=tmp_path / "corpus.json",
            eval_path=tmp_path / "eval.jsonl",
            strategy="CONTEXTUAL_HYBRID",
            semantic_weight=0.8,
            bm25_weight=0.2,
        )
        config.infra.embedding_base_url = "http://test:8080/v1"
        config.infra.embedding_model_name = "test-model"
        config.infra.llm_base_url = "http://test:8081/v1"
        config.infra.llm_model_name = "test-llm"

        evaluator = CookbookEvaluator(config)
        evaluator._embedding_model = MagicMock()
        evaluator._llm_service = MagicMock()

        # Capture what gets created
        from shared.retrieval.contextual_retriever import ContextualRetriever
        from shared.retrieval.hybrid_retriever import HybridRetriever

        created_retrievers = []

        original_init = ContextualRetriever.__init__

        def capture_init(self_inner, **kwargs):
            created_retrievers.append(kwargs.get("inner_retriever"))
            # Don't actually initialize (avoids ChromaDB/Tantivy deps)
            raise RuntimeError("Captured — stopping init")

        with patch.object(ContextualRetriever, "__init__", capture_init):
            try:
                evaluator._index_documents(dataset)
            except RuntimeError as e:
                if "Captured" not in str(e):
                    raise

        assert len(created_retrievers) == 1
        assert isinstance(created_retrievers[0], HybridRetriever)

    def test_contextual_hybrid_uses_cookbook_rrf(self, tmp_path):
        """CONTEXTUAL_HYBRID uses cookbook RRF formula with asymmetric weights."""
        dataset, _ = _make_test_dataset()
        config = CookbookConfig(
            dataset_path=tmp_path / "corpus.json",
            eval_path=tmp_path / "eval.jsonl",
            strategy="CONTEXTUAL_HYBRID",
            semantic_weight=0.8,
            bm25_weight=0.2,
            num_chunks_to_recall=150,
        )
        config.infra.embedding_base_url = "http://test:8080/v1"
        config.infra.embedding_model_name = "test-model"
        config.infra.llm_base_url = "http://test:8081/v1"
        config.infra.llm_model_name = "test-llm"

        evaluator = CookbookEvaluator(config)
        evaluator._embedding_model = MagicMock()
        evaluator._llm_service = MagicMock()

        # Capture the HybridRetriever config
        captured_configs = []

        original_hybrid_init = HybridRetriever.__init__

        def capture_hybrid_init(self_inner, config, *args, **kwargs):
            captured_configs.append(config)
            raise RuntimeError("Captured — stopping init")

        with patch(
            "sandbox_cookbook.evaluator.HybridRetriever.__init__",
            capture_hybrid_init,
        ):
            try:
                evaluator._index_documents(dataset)
            except RuntimeError as e:
                if "Captured" not in str(e):
                    raise

        assert len(captured_configs) == 1
        hybrid_cfg = captured_configs[0]
        assert hybrid_cfg.vector_weight == 0.8
        assert hybrid_cfg.bm25_weight == 0.2
        assert hybrid_cfg.rrf_formula == "cookbook"
        assert hybrid_cfg.pre_fusion_k == 150

    def test_contextual_hybrid_needs_llm_in_config_validation(self, tmp_path):
        """CONTEXTUAL_HYBRID requires LLM config in validation."""
        config = CookbookConfig(
            dataset_path=tmp_path / "corpus.json",
            eval_path=tmp_path / "eval.jsonl",
            strategy="CONTEXTUAL_HYBRID",
        )
        config.infra.embedding_base_url = "http://test:8080/v1"
        config.infra.embedding_model_name = "test-model"
        # LLM not configured
        config.infra.llm_base_url = ""
        config.infra.llm_model_name = ""

        errors = config.validate()
        assert any("LLM_BASE_URL" in e for e in errors), (
            f"Expected LLM_BASE_URL error, got: {errors}"
        )
        assert any("LLM_MODEL_NAME" in e for e in errors), (
            f"Expected LLM_MODEL_NAME error, got: {errors}"
        )


class TestCookbookEvaluatorRerank:
    """Tests para CONTEXTUAL_HYBRID_RERANK strategy (Fase 4)."""

    def test_rerank_uses_same_setup_as_hybrid(self, tmp_path):
        """CONTEXTUAL_HYBRID_RERANK uses same retriever as CONTEXTUAL_HYBRID."""
        dataset, _ = _make_test_dataset()
        config = CookbookConfig(
            dataset_path=tmp_path / "corpus.json",
            eval_path=tmp_path / "eval.jsonl",
            strategy="CONTEXTUAL_HYBRID_RERANK",
            semantic_weight=0.8,
            bm25_weight=0.2,
        )
        config.infra.embedding_base_url = "http://test:8080/v1"
        config.infra.embedding_model_name = "test-model"
        config.infra.llm_base_url = "http://test:8081/v1"
        config.infra.llm_model_name = "test-llm"
        config.reranker.base_url = "http://test:8082/v1"
        config.reranker.model_name = "test-reranker"

        evaluator = CookbookEvaluator(config)
        evaluator._embedding_model = MagicMock()
        evaluator._llm_service = MagicMock()

        # Capture HybridRetriever creation
        captured_configs = []

        def capture_hybrid_init(self_inner, config, *args, **kwargs):
            captured_configs.append(config)
            raise RuntimeError("Captured — stopping init")

        with patch(
            "sandbox_cookbook.evaluator.HybridRetriever.__init__",
            capture_hybrid_init,
        ):
            try:
                evaluator._index_documents(dataset)
            except RuntimeError as e:
                if "Captured" not in str(e):
                    raise

        assert len(captured_configs) == 1
        hybrid_cfg = captured_configs[0]
        assert hybrid_cfg.rrf_formula == "cookbook"
        assert hybrid_cfg.vector_weight == 0.8
        assert hybrid_cfg.bm25_weight == 0.2

    def test_rerank_oversamples_retrieval_k(self, tmp_path):
        """CONTEXTUAL_HYBRID_RERANK over-samples retrieval_k."""
        dataset, _ = _make_test_dataset()
        config = CookbookConfig(
            dataset_path=tmp_path / "corpus.json",
            eval_path=tmp_path / "eval.jsonl",
            strategy="CONTEXTUAL_HYBRID_RERANK",
            rerank_top_n=20,
            rerank_oversample_factor=10,
        )
        config.infra.embedding_base_url = "http://test:8080/v1"
        config.infra.embedding_model_name = "test-model"
        config.infra.llm_base_url = "http://test:8081/v1"
        config.infra.llm_model_name = "test-llm"
        config.reranker.base_url = "http://test:8082/v1"
        config.reranker.model_name = "test-reranker"

        evaluator = CookbookEvaluator(config)
        evaluator._embedding_model = MagicMock()
        evaluator._llm_service = MagicMock()

        # Capture the retrieval_k
        captured_configs = []

        def capture_hybrid_init(self_inner, config, *args, **kwargs):
            captured_configs.append(config)
            raise RuntimeError("Captured — stopping init")

        with patch(
            "sandbox_cookbook.evaluator.HybridRetriever.__init__",
            capture_hybrid_init,
        ):
            try:
                evaluator._index_documents(dataset)
            except RuntimeError as e:
                if "Captured" not in str(e):
                    raise

        assert len(captured_configs) == 1
        # retrieval_k = rerank_top_n * oversample_factor = 20 * 10 = 200
        assert captured_configs[0].retrieval_k == 200

    def test_rerank_result_enriched_content(self):
        """_rerank_result selects enriched content for reranker."""
        config = CookbookConfig(
            rerank_content="enriched",
            rerank_top_n=2,
        )
        evaluator = CookbookEvaluator(config)

        # Mock reranker
        mock_reranker = MagicMock()
        # Reranker returns reranked result
        mock_reranker.rerank.return_value = RetrievalResult(
            doc_ids=["d2", "d1"],
            contents=["enriched_d2", "enriched_d1"],
            scores=[0.9, 0.8],
            retrieval_time_ms=5.0,
        )
        evaluator._reranker = mock_reranker

        rr = RetrievalResult(
            doc_ids=["d1", "d2", "d3"],
            contents=["original_d1", "original_d2", "original_d3"],
            scores=[0.7, 0.6, 0.5],
            enriched_contents=["enriched_d1", "enriched_d2", "enriched_d3"],
            retrieval_time_ms=10.0,
        )

        result = evaluator._rerank_result("test query", rr)

        # Reranker called with enriched contents
        call_args = mock_reranker.rerank.call_args
        rerank_input = call_args[0][1]  # second positional arg
        assert rerank_input.contents == ["enriched_d1", "enriched_d2", "enriched_d3"]

        # Result has original contents restored
        assert result.contents == ["original_d2", "original_d1"]

    def test_rerank_result_original_content(self):
        """_rerank_result selects original content when configured."""
        config = CookbookConfig(
            rerank_content="original",
            rerank_top_n=2,
        )
        evaluator = CookbookEvaluator(config)

        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = RetrievalResult(
            doc_ids=["d1"],
            contents=["original_d1"],
            scores=[0.9],
            retrieval_time_ms=5.0,
        )
        evaluator._reranker = mock_reranker

        rr = RetrievalResult(
            doc_ids=["d1", "d2"],
            contents=["original_d1", "original_d2"],
            scores=[0.7, 0.6],
            enriched_contents=["enriched_d1", "enriched_d2"],
            retrieval_time_ms=10.0,
        )

        evaluator._rerank_result("test query", rr)

        # Reranker called with original contents
        call_args = mock_reranker.rerank.call_args
        rerank_input = call_args[0][1]
        assert rerank_input.contents == ["original_d1", "original_d2"]

    def test_rerank_result_both_content(self):
        """_rerank_result combines original + enriched when content='both'."""
        config = CookbookConfig(
            rerank_content="both",
            rerank_top_n=2,
        )
        evaluator = CookbookEvaluator(config)

        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = RetrievalResult(
            doc_ids=["d1"],
            contents=["combined"],
            scores=[0.9],
            retrieval_time_ms=5.0,
        )
        evaluator._reranker = mock_reranker

        rr = RetrievalResult(
            doc_ids=["d1"],
            contents=["original_d1"],
            scores=[0.7],
            enriched_contents=["enriched_d1"],
            retrieval_time_ms=10.0,
        )

        evaluator._rerank_result("test query", rr)

        # Reranker called with combined text
        call_args = mock_reranker.rerank.call_args
        rerank_input = call_args[0][1]
        assert rerank_input.contents == ["original_d1\n\nContext: enriched_d1"]

    def test_rerank_config_validation(self, tmp_path):
        """CONTEXTUAL_HYBRID_RERANK requires reranker config."""
        config = CookbookConfig(
            dataset_path=tmp_path / "corpus.json",
            eval_path=tmp_path / "eval.jsonl",
            strategy="CONTEXTUAL_HYBRID_RERANK",
        )
        config.infra.embedding_base_url = "http://test:8080/v1"
        config.infra.embedding_model_name = "test-model"
        config.infra.llm_base_url = "http://test:8081/v1"
        config.infra.llm_model_name = "test-llm"
        # Reranker not configured
        config.reranker.base_url = ""
        config.reranker.model_name = ""

        errors = config.validate()
        assert any("RERANKER_BASE_URL" in e for e in errors), (
            f"Expected RERANKER_BASE_URL error, got: {errors}"
        )
        assert any("RERANKER_MODEL_NAME" in e for e in errors), (
            f"Expected RERANKER_MODEL_NAME error, got: {errors}"
        )

    def test_rerank_content_validation(self, tmp_path):
        """Invalid rerank_content value is caught by validation."""
        config = CookbookConfig(
            dataset_path=tmp_path / "corpus.json",
            eval_path=tmp_path / "eval.jsonl",
            strategy="CONTEXTUAL_HYBRID_RERANK",
            rerank_content="invalid_value",
        )
        config.infra.embedding_base_url = "http://test:8080/v1"
        config.infra.embedding_model_name = "test-model"
        config.infra.llm_base_url = "http://test:8081/v1"
        config.infra.llm_model_name = "test-llm"
        config.reranker.base_url = "http://test:8082/v1"
        config.reranker.model_name = "test-reranker"

        errors = config.validate()
        assert any("COOKBOOK_RERANK_CONTENT" in e for e in errors), (
            f"Expected COOKBOOK_RERANK_CONTENT error, got: {errors}"
        )


class TestComparisonCsv:
    """Tests para export_comparison_csv (Fase 5)."""

    def _make_run(self, strategy: str, recall_at_k: Dict[int, float]) -> EvaluationRun:
        """Helper para crear un EvaluationRun con recall_at_k dado."""
        return EvaluationRun(
            run_id=f"test_{strategy}",
            dataset_name="cookbook",
            embedding_model="test-model",
            retrieval_strategy=strategy,
            config_snapshot={"strategy": strategy},
            num_queries_total=10,
            num_queries_evaluated=10,
            num_queries_failed=0,
            total_documents=100,
            avg_recall_at_k=recall_at_k,
            retrieval_complement_recall_at_k={
                k: 1.0 - v for k, v in recall_at_k.items()
            },
            query_results=[],
            execution_time_seconds=1.0,
            status=EvaluationStatus.COMPLETED,
        )

    def test_comparison_csv_generated(self, tmp_path):
        """comparison.csv se genera con las columnas correctas."""
        from sandbox_cookbook.evaluator import export_comparison_csv

        runs = [
            self._make_run("SIMPLE_VECTOR", {5: 0.80, 10: 0.85, 20: 0.90}),
            self._make_run("CONTEXTUAL_VECTOR", {5: 0.85, 10: 0.90, 20: 0.95}),
        ]

        path = export_comparison_csv(runs, tmp_path, [5, 10, 20])

        assert path.exists()
        assert path.name == "comparison.csv"

        import csv
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["strategy"] == "SIMPLE_VECTOR"
        assert rows[1]["strategy"] == "CONTEXTUAL_VECTOR"

        # Check all columns present
        for k in [5, 10, 20]:
            assert f"pass_at_{k}" in rows[0]
            assert f"failure_rate_at_{k}" in rows[0]
            assert f"failure_rate_reduction_at_{k}" in rows[0]

    def test_comparison_csv_pass_at_k_values(self, tmp_path):
        """Pass@k y failure_rate correctos en CSV."""
        from sandbox_cookbook.evaluator import export_comparison_csv

        runs = [
            self._make_run("SIMPLE_VECTOR", {5: 0.80, 10: 0.90, 20: 0.95}),
        ]

        path = export_comparison_csv(runs, tmp_path, [5, 10, 20])

        import csv
        with open(path) as f:
            rows = list(csv.DictReader(f))

        row = rows[0]
        assert float(row["pass_at_5"]) == 0.80
        assert float(row["pass_at_10"]) == 0.90
        assert float(row["pass_at_20"]) == 0.95
        assert float(row["failure_rate_at_5"]) == 0.20
        assert float(row["failure_rate_at_10"]) == 0.10
        assert float(row["failure_rate_at_20"]) == 0.05

    def test_comparison_csv_failure_rate_reduction(self, tmp_path):
        """failure_rate_reduction calcula correctamente vs baseline."""
        from sandbox_cookbook.evaluator import export_comparison_csv

        # Baseline: failure_rate@5 = 0.20, @10 = 0.10
        # Contextual: failure_rate@5 = 0.10, @10 = 0.05
        # Reduction@5 = (0.20 - 0.10) / 0.20 * 100 = 50.0%
        # Reduction@10 = (0.10 - 0.05) / 0.10 * 100 = 50.0%
        runs = [
            self._make_run("SIMPLE_VECTOR", {5: 0.80, 10: 0.90}),
            self._make_run("CONTEXTUAL_VECTOR", {5: 0.90, 10: 0.95}),
        ]

        path = export_comparison_csv(runs, tmp_path, [5, 10])

        import csv
        with open(path) as f:
            rows = list(csv.DictReader(f))

        # Baseline has 0% reduction
        assert float(rows[0]["failure_rate_reduction_at_5"]) == 0.0  # baseline vs itself

        # Wait — baseline_failure@5 = 0.20, baseline_failure@5 = 0.20
        # reduction = (0.20 - 0.20) / 0.20 * 100 = 0.0 — correct

        # Contextual reduction
        assert float(rows[1]["failure_rate_reduction_at_5"]) == 50.0
        assert float(rows[1]["failure_rate_reduction_at_10"]) == 50.0

    def test_comparison_csv_baseline_zero_failure(self, tmp_path):
        """Si baseline tiene 0% failure, reduction es 0 (no divide por cero)."""
        from sandbox_cookbook.evaluator import export_comparison_csv

        runs = [
            self._make_run("SIMPLE_VECTOR", {5: 1.0}),  # 0% failure
            self._make_run("CONTEXTUAL_VECTOR", {5: 1.0}),
        ]

        path = export_comparison_csv(runs, tmp_path, [5])

        import csv
        with open(path) as f:
            rows = list(csv.DictReader(f))

        # No division por cero
        assert float(rows[1]["failure_rate_reduction_at_5"]) == 0.0

    def test_comparison_csv_no_baseline(self, tmp_path):
        """Si no hay SIMPLE_VECTOR, reduction es 0."""
        from sandbox_cookbook.evaluator import export_comparison_csv

        runs = [
            self._make_run("CONTEXTUAL_VECTOR", {5: 0.90}),
            self._make_run("CONTEXTUAL_HYBRID", {5: 0.95}),
        ]

        path = export_comparison_csv(runs, tmp_path, [5])

        import csv
        with open(path) as f:
            rows = list(csv.DictReader(f))

        assert float(rows[0]["failure_rate_reduction_at_5"]) == 0.0
        assert float(rows[1]["failure_rate_reduction_at_5"]) == 0.0

    def test_comparison_csv_four_strategies(self, tmp_path):
        """Comparison con las 4 estrategias genera 4 filas."""
        from sandbox_cookbook.evaluator import export_comparison_csv

        runs = [
            self._make_run("SIMPLE_VECTOR", {5: 0.80, 10: 0.87, 20: 0.90}),
            self._make_run("CONTEXTUAL_VECTOR", {5: 0.85, 10: 0.90, 20: 0.95}),
            self._make_run("CONTEXTUAL_HYBRID", {5: 0.90, 10: 0.93, 20: 0.97}),
            self._make_run("CONTEXTUAL_HYBRID_RERANK", {5: 0.95, 10: 0.96, 20: 0.98}),
        ]

        path = export_comparison_csv(runs, tmp_path, [5, 10, 20])

        import csv
        with open(path) as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 4
        strategies = [r["strategy"] for r in rows]
        assert strategies == [
            "SIMPLE_VECTOR",
            "CONTEXTUAL_VECTOR",
            "CONTEXTUAL_HYBRID",
            "CONTEXTUAL_HYBRID_RERANK",
        ]

        # Verify reduction increases with each strategy
        r5_values = [float(r["failure_rate_reduction_at_5"]) for r in rows]
        assert r5_values[0] == 0.0  # baseline
        assert r5_values[1] > 0  # better than baseline
        assert r5_values[2] > r5_values[1]  # even better
        assert r5_values[3] > r5_values[2]  # best

    def test_comparison_csv_creates_directory(self, tmp_path):
        """export_comparison_csv crea directorio si no existe."""
        from sandbox_cookbook.evaluator import export_comparison_csv

        deep_dir = tmp_path / "a" / "b" / "c"
        runs = [
            self._make_run("SIMPLE_VECTOR", {5: 0.80}),
            self._make_run("CONTEXTUAL_VECTOR", {5: 0.90}),
        ]

        path = export_comparison_csv(runs, deep_dir, [5])
        assert path.exists()
        assert path.parent == deep_dir


class TestRunCompareAll:
    """Tests para run_compare_all (Fase 5)."""

    def test_compare_all_dry_run(self, tmp_path, capsys):
        """--compare-all --dry-run valida configs sin ejecutar."""
        from sandbox_cookbook.run import run_compare_all

        config = CookbookConfig(
            dataset_path=tmp_path / "corpus.json",
            eval_path=tmp_path / "eval.jsonl",
            results_dir=tmp_path / "results",
        )
        config.infra.embedding_base_url = "http://test:8080/v1"
        config.infra.embedding_model_name = "test-model"

        result = run_compare_all(config, dry_run=True)

        assert result == 0
        captured = capsys.readouterr()
        assert "COMPARE-ALL: Dry Run" in captured.out
        assert "SIMPLE_VECTOR" in captured.out
        # Contextual strategies need LLM, so they'll show SKIP
        assert "CONTEXTUAL_VECTOR" in captured.out

    def test_compare_all_sets_shared_cache(self, tmp_path):
        """--compare-all establece cache path compartido automaticamente."""
        from sandbox_cookbook.run import run_compare_all

        config = CookbookConfig(
            dataset_path=tmp_path / "corpus.json",
            eval_path=tmp_path / "eval.jsonl",
            results_dir=tmp_path / "results",
            contexts_cache_path=None,  # No cache configured
        )
        config.infra.embedding_base_url = "http://test:8080/v1"
        config.infra.embedding_model_name = "test-model"

        # Dry run to check cache is set without running evaluations
        run_compare_all(config, dry_run=True)

        # Original config not mutated (replace creates new)
        assert config.contexts_cache_path is None

    def test_compare_all_strategy_and_compare_mutually_exclusive(self):
        """--strategy y --compare-all son mutuamente exclusivos."""
        from sandbox_cookbook.run import parse_args

        with pytest.raises(SystemExit):
            sys.argv = [
                "run.py",
                "--strategy", "SIMPLE_VECTOR",
                "--compare-all",
            ]
            parse_args()


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

    def test_invalid_strategy_raises(self, tmp_path):
        """Estrategia invalida lanza KeyError en get_strategy()."""
        config = CookbookConfig(
            dataset_path=tmp_path / "corpus.json",
            eval_path=tmp_path / "eval.jsonl",
            strategy="NONEXISTENT_STRATEGY",
        )
        with pytest.raises(KeyError):
            config.get_strategy()


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
