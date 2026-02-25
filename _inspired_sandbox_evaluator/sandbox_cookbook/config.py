"""
Configuracion para sandbox Cookbook (Contextual Retrieval).

Toda la parametrizacion viene del .env. El entry point (run.py)
construye CookbookConfig.from_env() una sola vez.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from shared.config_base import (
    InfraConfig,
    RerankerConfig,
    _env,
    _env_int,
    _env_float,
    _env_path,
    load_dotenv_file,
)
from shared.retrieval.core import RetrievalConfig, RetrievalStrategy


@dataclass
class CookbookConfig:
    """
    Configuracion completa para un run de evaluacion Contextual Retrieval.

    Composicion de sub-configs de shared/ + config especifico del sandbox.
    Se construye una sola vez en el entry point via from_env().
    """
    infra: InfraConfig = field(default_factory=InfraConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)

    # Dataset (JSON/JSONL local, replica del cookbook de Anthropic)
    dataset_path: Path = Path("sandbox_cookbook/data/codebase_chunks.json")
    eval_path: Path = Path("sandbox_cookbook/data/evaluation_set.jsonl")
    results_dir: Path = Path("sandbox_cookbook/data/results")

    # Estrategia
    strategy: str = "SIMPLE_VECTOR"

    # Contextual Retrieval (Fase 2+)
    contextualize_model: str = ""
    contextualize_max_tokens: int = 1000
    contextualize_batch_size: int = 10
    context_position: str = "prepend"
    contexts_cache_path: Optional[Path] = None

    # Evaluacion: solo k=5, 10, 20 (PC-5)
    eval_k_values: List[int] = field(default_factory=lambda: [5, 10, 20])

    # Hybrid search (Fase 3+)
    semantic_weight: float = 0.8
    bm25_weight: float = 0.2
    num_chunks_to_recall: int = 150

    # Reranking (Fase 4+)
    rerank_top_n: int = 20
    rerank_oversample_factor: int = 10
    rerank_content: str = "enriched"

    @classmethod
    def from_env(cls, env_path: str = ".env") -> "CookbookConfig":
        """Construye config completa desde .env."""
        load_dotenv_file(env_path)

        strategy_str = _env("COOKBOOK_STRATEGY", "SIMPLE_VECTOR")

        return cls(
            infra=InfraConfig.from_env(),
            retrieval=RetrievalConfig.from_env(),
            reranker=RerankerConfig.from_env(),
            dataset_path=_env_path("COOKBOOK_DATASET_PATH", "sandbox_cookbook/data/codebase_chunks.json"),
            eval_path=_env_path("COOKBOOK_EVAL_PATH", "sandbox_cookbook/data/evaluation_set.jsonl"),
            results_dir=_env_path("COOKBOOK_RESULTS_DIR", "sandbox_cookbook/data/results"),
            strategy=strategy_str,
            contextualize_model=_env("COOKBOOK_CONTEXTUALIZE_MODEL", ""),
            contextualize_max_tokens=_env_int("COOKBOOK_CONTEXTUALIZE_MAX_TOKENS", 1000),
            contextualize_batch_size=_env_int("COOKBOOK_CONTEXTUALIZE_BATCH_SIZE", 10),
            context_position=_env("COOKBOOK_CONTEXT_POSITION", "prepend"),
            contexts_cache_path=(
                _env_path("COOKBOOK_CONTEXTS_CACHE_PATH", "")
                if _env("COOKBOOK_CONTEXTS_CACHE_PATH", "")
                else None
            ),
            semantic_weight=_env_float("COOKBOOK_SEMANTIC_WEIGHT", 0.8),
            bm25_weight=_env_float("COOKBOOK_BM25_WEIGHT", 0.2),
            num_chunks_to_recall=_env_int("COOKBOOK_NUM_CHUNKS_TO_RECALL", 150),
            rerank_top_n=_env_int("COOKBOOK_RERANK_TOP_N", 20),
            rerank_oversample_factor=_env_int("COOKBOOK_RERANK_OVERSAMPLE_FACTOR", 10),
            rerank_content=_env("COOKBOOK_RERANK_CONTENT", "enriched"),
        )

    def get_strategy(self) -> RetrievalStrategy:
        """Convierte string a RetrievalStrategy enum."""
        return RetrievalStrategy[self.strategy]

    def validate(self) -> List[str]:
        """Valida la configuracion. Retorna lista de errores (vacia = OK)."""
        errors = []

        # Dataset
        if not self.dataset_path.exists():
            errors.append(f"Dataset no encontrado: {self.dataset_path}")
        if not self.eval_path.exists():
            errors.append(f"Evaluation set no encontrado: {self.eval_path}")

        # Estrategia
        VALID_STRATEGIES = {
            "SIMPLE_VECTOR",
            "CONTEXTUAL_VECTOR",
            "CONTEXTUAL_HYBRID",
            "CONTEXTUAL_HYBRID_RERANK",
        }
        if self.strategy not in VALID_STRATEGIES:
            errors.append(
                f"COOKBOOK_STRATEGY='{self.strategy}' no valida. "
                f"Valores: {', '.join(sorted(VALID_STRATEGIES))}"
            )

        # Embedding requerido siempre
        if not self.infra.embedding_base_url:
            errors.append("EMBEDDING_BASE_URL no configurado")
        if not self.infra.embedding_model_name:
            errors.append("EMBEDDING_MODEL_NAME no configurado")

        # LLM requerido para estrategias contextuales
        needs_llm = self.strategy in (
            "CONTEXTUAL_VECTOR",
            "CONTEXTUAL_HYBRID",
            "CONTEXTUAL_HYBRID_RERANK",
        )
        if needs_llm:
            if not self.infra.llm_base_url:
                errors.append(
                    f"{self.strategy} requiere LLM_BASE_URL para "
                    "generar contextos de enriquecimiento"
                )
            if not self.infra.llm_model_name:
                errors.append(f"{self.strategy} requiere LLM_MODEL_NAME")

        # context_position
        if self.context_position not in ("prepend", "append"):
            errors.append(
                f"COOKBOOK_CONTEXT_POSITION='{self.context_position}' no valido. "
                "Usar 'prepend' o 'append'"
            )

        # eval_k_values
        if not self.eval_k_values:
            errors.append("eval_k_values no puede estar vacio")

        return errors

    def ensure_directories(self) -> None:
        """Crea directorios necesarios."""
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def summary(self) -> str:
        """Resumen legible de la configuracion."""
        lines = [
            "=== Cookbook Sandbox Config ===",
            f"  Strategy:     {self.strategy}",
            f"  Dataset:      {self.dataset_path}",
            f"  Eval set:     {self.eval_path}",
            f"  Embedding:    {self.infra.embedding_model_name} ({self.infra.embedding_model_type})",
            f"  Eval K:       {self.eval_k_values}",
            f"  Results:      {self.results_dir}",
        ]
        if self.strategy != "SIMPLE_VECTOR":
            lines.append(f"  Context pos:  {self.context_position}")
        if self.strategy in ("CONTEXTUAL_HYBRID", "CONTEXTUAL_HYBRID_RERANK"):
            lines.append(
                f"  Hybrid:       semantic={self.semantic_weight}, "
                f"bm25={self.bm25_weight}"
            )
        if self.strategy == "CONTEXTUAL_HYBRID_RERANK":
            lines.append(
                f"  Rerank:       top_n={self.rerank_top_n}, "
                f"oversample={self.rerank_oversample_factor}x, "
                f"content={self.rerank_content}"
            )
        return "\n".join(lines)


__all__ = ["CookbookConfig"]
