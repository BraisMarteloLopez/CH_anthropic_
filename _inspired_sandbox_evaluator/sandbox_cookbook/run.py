#!/usr/bin/env python3
"""
Entry point para sandbox Cookbook (Contextual Retrieval).

Uso:
    python -m sandbox_cookbook.run                              # Run con .env
    python -m sandbox_cookbook.run --strategy SIMPLE_VECTOR     # Strategy override
    python -m sandbox_cookbook.run --dry-run                    # Solo valida config
    python -m sandbox_cookbook.run --env /path/.env             # .env alternativo
    python -m sandbox_cookbook.run -v                           # Logging verbose
"""

import argparse
import logging
import sys
from pathlib import Path

# Asegurar que el proyecto raiz esta en sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from sandbox_cookbook.config import CookbookConfig
from sandbox_cookbook.evaluator import CookbookEvaluator


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cookbook Contextual Retrieval Evaluation"
    )
    sandbox_dir = Path(__file__).resolve().parent
    default_env = str(sandbox_dir / ".env")

    parser.add_argument(
        "--env",
        default=default_env,
        help=f"Ruta al archivo .env (default: {default_env})",
    )
    parser.add_argument(
        "--strategy",
        default=None,
        choices=[
            "SIMPLE_VECTOR",
            "CONTEXTUAL_VECTOR",
            "CONTEXTUAL_HYBRID",
            "CONTEXTUAL_HYBRID_RERANK",
        ],
        help="Override de estrategia (default: valor en .env)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo validar config y mostrar resumen",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Logging verbose (DEBUG)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger("sandbox_cookbook")

    # 1. Construir config
    env_path = Path(args.env)
    if not env_path.exists():
        logger.error(f"Archivo .env no encontrado: {env_path.resolve()}")
        logger.error("Copiar env.example a .env y completar valores.")
        return 1

    config = CookbookConfig.from_env(str(env_path))

    # Override de estrategia via CLI
    if args.strategy:
        config.strategy = args.strategy

    # 2. Validar
    errors = config.validate()
    if errors:
        logger.error("Errores de configuracion:")
        for e in errors:
            logger.error(f"  - {e}")
        return 1

    # 3. Mostrar resumen
    print(config.summary())

    # 4. Dry run?
    if args.dry_run:
        print("\n[DRY RUN] Config valida. No se ejecuta evaluacion.")
        return 0

    # 5. Ejecutar
    evaluator = CookbookEvaluator(config)
    run_result = evaluator.run()

    print(f"\nResultados en: {config.results_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
