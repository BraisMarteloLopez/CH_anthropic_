"""
Tests para CookbookConfig.

Valida:
  - from_env() carga parametros correctamente
  - validate() detecta errores de configuracion
  - summary() produce string legible
  - eval_k_values default [5, 10, 20]
  - get_strategy() convierte string a enum
"""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from sandbox_cookbook.config import CookbookConfig
from shared.retrieval.core import RetrievalStrategy


class TestCookbookConfigDefaults:
    """Tests de valores por defecto."""

    def test_default_strategy(self):
        config = CookbookConfig()
        assert config.strategy == "SIMPLE_VECTOR"

    def test_default_eval_k_values(self):
        config = CookbookConfig()
        assert config.eval_k_values == [5, 10, 20]

    def test_default_context_position(self):
        config = CookbookConfig()
        assert config.context_position == "prepend"

    def test_default_hybrid_weights(self):
        config = CookbookConfig()
        assert config.semantic_weight == 0.8
        assert config.bm25_weight == 0.2

    def test_default_rerank_content(self):
        config = CookbookConfig()
        assert config.rerank_content == "enriched"


class TestCookbookConfigGetStrategy:
    """Tests de get_strategy()."""

    def test_simple_vector(self):
        config = CookbookConfig(strategy="SIMPLE_VECTOR")
        assert config.get_strategy() == RetrievalStrategy.SIMPLE_VECTOR

    def test_contextual_vector(self):
        config = CookbookConfig(strategy="CONTEXTUAL_VECTOR")
        assert config.get_strategy() == RetrievalStrategy.CONTEXTUAL_VECTOR

    def test_contextual_hybrid(self):
        config = CookbookConfig(strategy="CONTEXTUAL_HYBRID")
        assert config.get_strategy() == RetrievalStrategy.CONTEXTUAL_HYBRID

    def test_contextual_hybrid_rerank(self):
        config = CookbookConfig(strategy="CONTEXTUAL_HYBRID_RERANK")
        assert config.get_strategy() == RetrievalStrategy.CONTEXTUAL_HYBRID_RERANK

    def test_invalid_strategy_raises(self):
        config = CookbookConfig(strategy="INVALID")
        with pytest.raises(KeyError):
            config.get_strategy()


class TestCookbookConfigValidate:
    """Tests de validate()."""

    def test_valid_config_no_errors(self, tmp_path):
        # Crear archivos de dataset dummy
        (tmp_path / "corpus.json").write_text("[]")
        (tmp_path / "eval.jsonl").write_text("")

        config = CookbookConfig(
            dataset_path=tmp_path / "corpus.json",
            eval_path=tmp_path / "eval.jsonl",
            strategy="SIMPLE_VECTOR",
        )
        config.infra.embedding_base_url = "http://localhost:8080/v1"
        config.infra.embedding_model_name = "test-model"

        errors = config.validate()
        assert len(errors) == 0, f"Unexpected errors: {errors}"

    def test_missing_dataset_path(self, tmp_path):
        config = CookbookConfig(
            dataset_path=tmp_path / "nonexistent.json",
            eval_path=tmp_path / "nonexistent.jsonl",
        )
        config.infra.embedding_base_url = "http://localhost:8080/v1"
        config.infra.embedding_model_name = "test-model"

        errors = config.validate()
        assert any("Dataset no encontrado" in e for e in errors)
        assert any("Evaluation set no encontrado" in e for e in errors)

    def test_invalid_strategy(self, tmp_path):
        (tmp_path / "corpus.json").write_text("[]")
        (tmp_path / "eval.jsonl").write_text("")

        config = CookbookConfig(
            dataset_path=tmp_path / "corpus.json",
            eval_path=tmp_path / "eval.jsonl",
            strategy="INVALID_STRATEGY",
        )
        config.infra.embedding_base_url = "http://localhost:8080/v1"
        config.infra.embedding_model_name = "test-model"

        errors = config.validate()
        assert any("no valida" in e for e in errors)

    def test_missing_embedding_url(self, tmp_path):
        (tmp_path / "corpus.json").write_text("[]")
        (tmp_path / "eval.jsonl").write_text("")

        config = CookbookConfig(
            dataset_path=tmp_path / "corpus.json",
            eval_path=tmp_path / "eval.jsonl",
        )
        errors = config.validate()
        assert any("EMBEDDING_BASE_URL" in e for e in errors)

    def test_contextual_requires_llm(self, tmp_path):
        (tmp_path / "corpus.json").write_text("[]")
        (tmp_path / "eval.jsonl").write_text("")

        config = CookbookConfig(
            dataset_path=tmp_path / "corpus.json",
            eval_path=tmp_path / "eval.jsonl",
            strategy="CONTEXTUAL_VECTOR",
        )
        config.infra.embedding_base_url = "http://localhost:8080/v1"
        config.infra.embedding_model_name = "test-model"

        errors = config.validate()
        assert any("LLM_BASE_URL" in e for e in errors)

    def test_invalid_context_position(self, tmp_path):
        (tmp_path / "corpus.json").write_text("[]")
        (tmp_path / "eval.jsonl").write_text("")

        config = CookbookConfig(
            dataset_path=tmp_path / "corpus.json",
            eval_path=tmp_path / "eval.jsonl",
            context_position="invalid",
        )
        config.infra.embedding_base_url = "http://localhost:8080/v1"
        config.infra.embedding_model_name = "test-model"

        errors = config.validate()
        assert any("COOKBOOK_CONTEXT_POSITION" in e for e in errors)

    def test_empty_eval_k_values(self, tmp_path):
        (tmp_path / "corpus.json").write_text("[]")
        (tmp_path / "eval.jsonl").write_text("")

        config = CookbookConfig(
            dataset_path=tmp_path / "corpus.json",
            eval_path=tmp_path / "eval.jsonl",
            eval_k_values=[],
        )
        config.infra.embedding_base_url = "http://localhost:8080/v1"
        config.infra.embedding_model_name = "test-model"

        errors = config.validate()
        assert any("eval_k_values" in e for e in errors)


class TestCookbookConfigSummary:
    """Tests de summary()."""

    def test_summary_contains_strategy(self):
        config = CookbookConfig(strategy="SIMPLE_VECTOR")
        summary = config.summary()
        assert "SIMPLE_VECTOR" in summary

    def test_summary_contains_eval_k(self):
        config = CookbookConfig()
        summary = config.summary()
        assert "[5, 10, 20]" in summary

    def test_summary_hybrid_shows_weights(self):
        config = CookbookConfig(strategy="CONTEXTUAL_HYBRID")
        summary = config.summary()
        assert "semantic=0.8" in summary
        assert "bm25=0.2" in summary

    def test_summary_rerank_shows_config(self):
        config = CookbookConfig(strategy="CONTEXTUAL_HYBRID_RERANK")
        summary = config.summary()
        assert "top_n=20" in summary
        assert "oversample=10x" in summary


class TestCookbookConfigFromEnv:
    """Tests de from_env()."""

    def test_from_env_reads_strategy(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text(
            "COOKBOOK_STRATEGY=CONTEXTUAL_VECTOR\n"
            "EMBEDDING_BASE_URL=http://test:8080/v1\n"
            "EMBEDDING_MODEL_NAME=test-model\n"
        )

        config = CookbookConfig.from_env(str(env_file))
        assert config.strategy == "CONTEXTUAL_VECTOR"

    def test_from_env_reads_weights(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text(
            "COOKBOOK_SEMANTIC_WEIGHT=0.7\n"
            "COOKBOOK_BM25_WEIGHT=0.3\n"
        )

        config = CookbookConfig.from_env(str(env_file))
        assert abs(config.semantic_weight - 0.7) < 1e-10
        assert abs(config.bm25_weight - 0.3) < 1e-10


# =============================================================================
# STANDALONE
# =============================================================================

if __name__ == "__main__":
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    for cls in [
        TestCookbookConfigDefaults,
        TestCookbookConfigGetStrategy,
        TestCookbookConfigValidate,
        TestCookbookConfigSummary,
        TestCookbookConfigFromEnv,
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
                except Exception as e:
                    print(f"FAIL: {e}")
                    continue
                print("OK")
    print("All tests passed!")
