"""
Configuracion compartida de pytest.

Mockea modulos de infraestructura (langchain_*, chromadb) si no estan
instalados. Permite ejecutar unit tests sin dependencias pesadas.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_INFRA_MODULES = [
    "langchain_nvidia_ai_endpoints",
    "langchain_core",
    "langchain_core.messages",
    "langchain_core.documents",
    "langchain_core.embeddings",
    "langchain_chroma",
    "chromadb",
]

for mod in _INFRA_MODULES:
    if mod not in sys.modules:
        try:
            __import__(mod)
        except ImportError:
            sys.modules[mod] = MagicMock()
