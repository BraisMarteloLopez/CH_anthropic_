"""
Tests para refactor de LLMContextGenerator y ContextualRetriever (Fase 2).

Valida:
  - (PC-2) Truncation limits configurables (max_parent_chars, max_chunk_chars)
  - mode="document" requiere parent_content (ValueError si falta)
  - mode="fallback" tolera ausencia de parent
  - Prompts Anthropic XML tags vs plain text
  - context_position="prepend" vs "append"
  - _build_enriched_text() combina correctamente
  - (PC-3) strategy_used = config.strategy (no hardcode)
  - enriched_contents expuesto en RetrievalResult
  - ContextCache persistente (load/save/invalidation)
"""

import asyncio
import json
import pytest
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

from shared.retrieval.contextual_retriever import (
    LLMContextGenerator,
    ContextualRetriever,
    EnrichedChunk,
    ANTHROPIC_DOCUMENT_PROMPT,
    ANTHROPIC_CHUNK_PROMPT,
    ANTHROPIC_SYSTEM_PROMPT,
    DOCUMENT_CONTEXT_PROMPT,
    FALLBACK_CONTEXT_PROMPT,
    CONTEXT_SYSTEM_PROMPT,
)
from shared.retrieval.core import (
    RetrievalConfig,
    RetrievalResult,
    RetrievalStrategy,
)


def _run(coro):
    """Helper para ejecutar coroutines en tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


class MockLLM:
    """Mock LLM que captura prompts y retorna respuesta fija."""

    def __init__(self, response: str = "Generated context."):
        self.response = response
        self.calls: List[Dict] = []

    async def invoke_async(self, prompt, system_prompt=None, max_tokens=1000):
        self.calls.append({
            "prompt": prompt,
            "system_prompt": system_prompt,
            "max_tokens": max_tokens,
        })
        return self.response


# =============================================================================
# Tests PC-2: Truncation limits configurables
# =============================================================================

class TestTruncationLimits:
    """(PC-2) Truncation limits configurables en LLMContextGenerator."""

    def test_default_truncation_limits(self):
        llm = MockLLM()
        gen = LLMContextGenerator(llm_service=llm)
        assert gen.max_parent_chars == 2000
        assert gen.max_chunk_chars == 1000

    def test_custom_truncation_limits(self):
        llm = MockLLM()
        gen = LLMContextGenerator(
            llm_service=llm,
            max_parent_chars=32000,
            max_chunk_chars=8000,
        )
        assert gen.max_parent_chars == 32000
        assert gen.max_chunk_chars == 8000

    def test_parent_truncated_to_limit(self):
        """Parent content se trunca a max_parent_chars."""
        llm = MockLLM()
        gen = LLMContextGenerator(
            llm_service=llm,
            max_parent_chars=50,
            max_chunk_chars=20,
        )

        parent = "A" * 100
        chunk = "B" * 50

        _run(gen._generate_one(chunk, parent_content=parent))

        prompt = llm.calls[0]["prompt"]
        # El parent en el prompt debe estar truncado a 50 chars
        assert "A" * 50 in prompt
        assert "A" * 51 not in prompt
        # El chunk en el prompt debe estar truncado a 20 chars
        assert "B" * 20 in prompt
        assert "B" * 21 not in prompt

    def test_chunk_truncated_to_limit(self):
        """Chunk content se trunca a max_chunk_chars."""
        llm = MockLLM()
        gen = LLMContextGenerator(
            llm_service=llm,
            max_chunk_chars=30,
            mode="fallback",
        )

        chunk = "C" * 100
        _run(gen._generate_one(chunk))

        prompt = llm.calls[0]["prompt"]
        assert "C" * 30 in prompt
        assert "C" * 31 not in prompt


# =============================================================================
# Tests Mode: document vs fallback
# =============================================================================

class TestModeDocumentFallback:
    """mode='document' requiere parent, mode='fallback' tolera ausencia."""

    def test_mode_document_with_parent_ok(self):
        llm = MockLLM()
        gen = LLMContextGenerator(llm_service=llm, mode="document")

        result = _run(gen._generate_one(
            "chunk text", parent_content="parent text"
        ))
        assert result == "Generated context."
        assert gen._total_with_parent == 1

    def test_mode_document_without_parent_raises(self):
        llm = MockLLM()
        gen = LLMContextGenerator(llm_service=llm, mode="document")

        with pytest.raises(ValueError, match="Mode 'document' requiere parent_content"):
            _run(gen._generate_one("chunk text"))

    def test_mode_fallback_without_parent_ok(self):
        llm = MockLLM()
        gen = LLMContextGenerator(llm_service=llm, mode="fallback")

        result = _run(gen._generate_one("chunk text"))
        assert result == "Generated context."
        assert gen._total_fallback == 1

    def test_mode_fallback_with_parent_uses_parent(self):
        llm = MockLLM()
        gen = LLMContextGenerator(llm_service=llm, mode="fallback")

        _run(gen._generate_one("chunk", parent_content="parent"))
        assert gen._total_with_parent == 1
        assert gen._total_fallback == 0

    def test_default_mode_is_fallback(self):
        llm = MockLLM()
        gen = LLMContextGenerator(llm_service=llm)
        assert gen.mode == "fallback"


# =============================================================================
# Tests: Prompts Anthropic XML vs plain text
# =============================================================================

class TestPromptsAnthropicXML:
    """Prompts XML tags de Anthropic vs plain text."""

    def test_default_prompts_are_plain_text(self):
        llm = MockLLM()
        gen = LLMContextGenerator(llm_service=llm)

        _run(gen._generate_one("chunk", parent_content="parent"))
        prompt = llm.calls[0]["prompt"]
        # Default prompts use "Document:\n" header (plain text)
        assert "Document:" in prompt
        assert "<document>" not in prompt

    def test_anthropic_xml_prompts(self):
        llm = MockLLM()
        gen = LLMContextGenerator(
            llm_service=llm,
            document_prompt_template=ANTHROPIC_DOCUMENT_PROMPT,
            chunk_prompt_template=ANTHROPIC_CHUNK_PROMPT,
            system_prompt=ANTHROPIC_SYSTEM_PROMPT,
        )

        _run(gen._generate_one("chunk text", parent_content="parent text"))
        prompt = llm.calls[0]["prompt"]
        assert "<document>" in prompt
        assert "</document>" in prompt
        assert "<chunk>" in prompt
        assert "</chunk>" in prompt

        sys_prompt = llm.calls[0]["system_prompt"]
        assert sys_prompt == ANTHROPIC_SYSTEM_PROMPT

    def test_custom_prompt_template(self):
        llm = MockLLM()
        custom = "DOC: {doc_content}\nCHUNK: {chunk_content}\nAnswer:"
        gen = LLMContextGenerator(
            llm_service=llm,
            document_prompt_template=custom,
        )

        _run(gen._generate_one("my chunk", parent_content="my doc"))
        prompt = llm.calls[0]["prompt"]
        assert "DOC: my doc" in prompt
        assert "CHUNK: my chunk" in prompt


# =============================================================================
# Tests: context_position
# =============================================================================

class TestContextPosition:
    """context_position='prepend' vs 'append'."""

    def test_build_enriched_text_prepend(self):
        llm = MockLLM()
        gen = LLMContextGenerator(llm_service=llm, context_position="prepend")

        result = gen._build_enriched_text("CONTEXT", "ORIGINAL")
        assert result == "CONTEXT\n\nORIGINAL"

    def test_build_enriched_text_append(self):
        llm = MockLLM()
        gen = LLMContextGenerator(llm_service=llm, context_position="append")

        result = gen._build_enriched_text("CONTEXT", "ORIGINAL")
        assert result == "ORIGINAL\n\nCONTEXT"

    def test_default_position_is_prepend(self):
        llm = MockLLM()
        gen = LLMContextGenerator(llm_service=llm)
        assert gen.context_position == "prepend"

    def test_enriched_chunk_get_enriched_text_is_always_prepend(self):
        """EnrichedChunk.get_enriched_text() siempre prepend (backwards-compat)."""
        chunk = EnrichedChunk(
            chunk_id="c1",
            original_content="ORIGINAL",
            generated_context="CONTEXT",
        )
        assert chunk.get_enriched_text() == "CONTEXT\n\nORIGINAL"


# =============================================================================
# Tests PC-3: strategy_used = config.strategy
# =============================================================================

class TestStrategyUsed:
    """(PC-3) strategy_used = self.config.strategy, no hardcode."""

    def _make_contextual_retriever(self, strategy: RetrievalStrategy):
        config = RetrievalConfig(strategy=strategy)
        llm = MockLLM()
        gen = LLMContextGenerator(llm_service=llm, mode="fallback")

        mock_inner = MagicMock()
        mock_inner.retrieve.return_value = RetrievalResult(
            doc_ids=["d1"], contents=["content1"], scores=[0.9],
            strategy_used=RetrievalStrategy.SIMPLE_VECTOR,
        )

        retriever = ContextualRetriever(
            config=config,
            embedding_model=MagicMock(),
            context_generator=gen,
            inner_retriever=mock_inner,
        )
        # Pre-populate original contents map
        retriever._original_contents["d1"] = "original1"
        retriever._enriched_contents["d1"] = "enriched1"
        return retriever

    def test_contextual_vector_strategy_used(self):
        retriever = self._make_contextual_retriever(RetrievalStrategy.CONTEXTUAL_VECTOR)
        result = retriever.retrieve("test query")
        assert result.strategy_used == RetrievalStrategy.CONTEXTUAL_VECTOR

    def test_contextual_hybrid_strategy_used(self):
        retriever = self._make_contextual_retriever(RetrievalStrategy.CONTEXTUAL_HYBRID)
        result = retriever.retrieve("test query")
        assert result.strategy_used == RetrievalStrategy.CONTEXTUAL_HYBRID

    def test_contextual_hybrid_rerank_strategy_used(self):
        retriever = self._make_contextual_retriever(RetrievalStrategy.CONTEXTUAL_HYBRID_RERANK)
        result = retriever.retrieve("test query")
        assert result.strategy_used == RetrievalStrategy.CONTEXTUAL_HYBRID_RERANK


# =============================================================================
# Tests: enriched_contents expuesto
# =============================================================================

class TestEnrichedContents:
    """enriched_contents expuesto en RetrievalResult antes de swap."""

    def test_enriched_contents_preserved(self):
        config = RetrievalConfig(strategy=RetrievalStrategy.CONTEXTUAL_VECTOR)
        llm = MockLLM()
        gen = LLMContextGenerator(llm_service=llm)

        mock_inner = MagicMock()
        mock_inner.retrieve.return_value = RetrievalResult(
            doc_ids=["d1", "d2"],
            contents=["enriched_d1", "enriched_d2"],
            scores=[0.9, 0.8],
        )

        retriever = ContextualRetriever(
            config=config,
            embedding_model=MagicMock(),
            context_generator=gen,
            inner_retriever=mock_inner,
        )
        retriever._original_contents = {"d1": "original_d1", "d2": "original_d2"}
        retriever._enriched_contents = {"d1": "enriched_d1", "d2": "enriched_d2"}

        result = retriever.retrieve("query")

        # contents swapped to original
        assert result.contents == ["original_d1", "original_d2"]
        # enriched_contents preserved
        assert result.enriched_contents == ["enriched_d1", "enriched_d2"]

    def test_enriched_contents_none_for_simple_vector(self):
        """SimpleVectorRetriever no tiene enriched_contents."""
        result = RetrievalResult(
            doc_ids=["d1"], contents=["c1"], scores=[0.9]
        )
        assert result.enriched_contents is None


# =============================================================================
# Tests: ContextCache
# =============================================================================

class TestContextCache:
    """Tests para cache persistente de contextos."""

    def test_save_and_load(self, tmp_path):
        from sandbox_cookbook.context_cache import ContextCache

        cache_path = tmp_path / "cache.json"
        cache = ContextCache(cache_path)
        cache.load(model_name="test-model", prompt_template="test prompt")

        cache.put("chunk_1", "context for chunk 1")
        cache.put("chunk_2", "context for chunk 2")
        cache.save(generation_time_s=5.0)

        assert cache_path.exists()

        # Load in new instance
        cache2 = ContextCache(cache_path)
        valid = cache2.load(model_name="test-model", prompt_template="test prompt")
        assert valid is True
        assert cache2.size == 2
        assert cache2.get("chunk_1") == "context for chunk 1"

    def test_invalidation_on_model_change(self, tmp_path):
        from sandbox_cookbook.context_cache import ContextCache

        cache_path = tmp_path / "cache.json"
        cache = ContextCache(cache_path)
        cache.load(model_name="model-a", prompt_template="prompt")
        cache.put("chunk_1", "ctx")
        cache.save()

        cache2 = ContextCache(cache_path)
        valid = cache2.load(model_name="model-b", prompt_template="prompt")
        assert valid is False
        assert cache2.size == 0

    def test_invalidation_on_prompt_change(self, tmp_path):
        from sandbox_cookbook.context_cache import ContextCache

        cache_path = tmp_path / "cache.json"
        cache = ContextCache(cache_path)
        cache.load(model_name="model", prompt_template="prompt-v1")
        cache.put("chunk_1", "ctx")
        cache.save()

        cache2 = ContextCache(cache_path)
        valid = cache2.load(model_name="model", prompt_template="prompt-v2")
        assert valid is False

    def test_missing_cache_file(self, tmp_path):
        from sandbox_cookbook.context_cache import ContextCache

        cache_path = tmp_path / "nonexistent.json"
        cache = ContextCache(cache_path)
        valid = cache.load(model_name="m", prompt_template="p")
        assert valid is False

    def test_corrupted_cache_file(self, tmp_path):
        from sandbox_cookbook.context_cache import ContextCache

        cache_path = tmp_path / "cache.json"
        cache_path.write_text("not valid json {{{")

        cache = ContextCache(cache_path)
        valid = cache.load(model_name="m", prompt_template="p")
        assert valid is False


# =============================================================================
# STANDALONE
# =============================================================================

if __name__ == "__main__":
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    for cls in [
        TestTruncationLimits,
        TestModeDocumentFallback,
        TestPromptsAnthropicXML,
        TestContextPosition,
        TestStrategyUsed,
        TestEnrichedContents,
        TestContextCache,
    ]:
        instance = cls()
        for name in dir(instance):
            if name.startswith("test_"):
                print(f"  {cls.__name__}.{name}...", end=" ")
                try:
                    import inspect
                    method = getattr(instance, name)
                    sig = inspect.signature(method)
                    if "tmp_path" in sig.parameters:
                        method(tmp)
                    else:
                        method()
                    print("OK")
                except Exception as e:
                    print(f"FAIL: {e}")
    print("Done!")
