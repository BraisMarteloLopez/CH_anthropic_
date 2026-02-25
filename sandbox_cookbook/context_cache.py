"""
Cache persistente de contextos generados por LLM.

Evita re-generar contextos en cada run. Invalidacion automatica
si cambia modelo o prompt template (hash mismatch).

Formato JSON:
{
    "model": "nvidia/nemotron-3-nano",
    "prompt_hash": "sha256 del template",
    "contexts": {
        "doc_1_chunk_0": "This chunk describes...",
        ...
    },
    "stats": {
        "total_generated": 737,
        "total_errors": 0,
        "generation_time_s": 312.5
    }
}
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ContextCache:
    """
    Cache persistente de contextos LLM en disco.

    Uso:
        cache = ContextCache(Path("contexts_cache.json"))
        cache.load(model_name="llama-70b", prompt_template="...")

        # Si cache valido (mismo modelo y prompt):
        ctx = cache.get("doc_1_chunk_0")

        # Despues de generar nuevos contextos:
        cache.put("doc_1_chunk_0", "This chunk describes...")
        cache.save(generation_time_s=312.5)
    """

    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self._contexts: Dict[str, str] = {}
        self._model: str = ""
        self._prompt_hash: str = ""
        self._is_valid = False

    def load(self, model_name: str, prompt_template: str) -> bool:
        """
        Carga cache desde disco si existe y es valido.

        Returns:
            True si cache cargado y valido (mismo modelo + prompt).
            False si no existe, invalido, o error.
        """
        self._model = model_name
        self._prompt_hash = self._hash_prompt(prompt_template)
        self._is_valid = False

        if not self.cache_path.exists():
            logger.info(f"  Cache no encontrado: {self.cache_path}")
            return False

        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Validar modelo y prompt
            if data.get("model") != model_name:
                logger.info(
                    f"  Cache invalidado: modelo cambio "
                    f"({data.get('model')} -> {model_name})"
                )
                return False

            if data.get("prompt_hash") != self._prompt_hash:
                logger.info("  Cache invalidado: prompt template cambio")
                return False

            self._contexts = data.get("contexts", {})
            self._is_valid = True
            logger.info(
                f"  Cache cargado: {len(self._contexts)} contextos "
                f"desde {self.cache_path}"
            )
            return True

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"  Error leyendo cache: {e}. Regenerando.")
            return False

    def save(self, generation_time_s: float = 0.0) -> None:
        """Persiste cache a disco."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "model": self._model,
            "prompt_hash": self._prompt_hash,
            "contexts": self._contexts,
            "stats": {
                "total_cached": len(self._contexts),
                "generation_time_s": round(generation_time_s, 1),
            },
        }

        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(
            f"  Cache guardado: {len(self._contexts)} contextos "
            f"en {self.cache_path}"
        )

    def get(self, chunk_id: str) -> Optional[str]:
        """Retorna contexto cacheado o None."""
        return self._contexts.get(chunk_id)

    def put(self, chunk_id: str, context: str) -> None:
        """Almacena contexto en cache (en memoria)."""
        self._contexts[chunk_id] = context

    def get_all(self) -> Dict[str, str]:
        """Retorna todos los contextos cacheados."""
        return dict(self._contexts)

    @property
    def is_valid(self) -> bool:
        return self._is_valid

    @property
    def size(self) -> int:
        return len(self._contexts)

    @staticmethod
    def _hash_prompt(template: str) -> str:
        return hashlib.sha256(template.encode("utf-8")).hexdigest()[:16]


__all__ = ["ContextCache"]
