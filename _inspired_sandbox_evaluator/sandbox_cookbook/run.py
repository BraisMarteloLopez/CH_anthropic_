#!/usr/bin/env python3
"""
Entry point para sandbox Cookbook (Contextual Retrieval).

Uso:
    python -m sandbox_cookbook.run                              # Run con .env
    python -m sandbox_cookbook.run --strategy SIMPLE_VECTOR     # Strategy override
    python -m sandbox_cookbook.run --compare-all                # 4 estrategias + comparison.csv
    python -m sandbox_cookbook.run --dry-run                    # Solo valida config
    python -m sandbox_cookbook.run --env /path/.env             # .env alternativo
    python -m sandbox_cookbook.run -v                           # Logging verbose
"""

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import List

# Asegurar que el proyecto raiz esta en sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from sandbox_cookbook.config import CookbookConfig
from sandbox_cookbook.evaluator import CookbookEvaluator, export_comparison_csv

logger = logging.getLogger("sandbox_cookbook")

COMPARE_ALL_STRATEGIES = [
    "SIMPLE_VECTOR",
    "CONTEXTUAL_VECTOR",
    "CONTEXTUAL_HYBRID",
    "CONTEXTUAL_HYBRID_RERANK",
]


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

    strategy_group = parser.add_mutually_exclusive_group()
    strategy_group.add_argument(
        "--strategy",
        default=None,
        choices=COMPARE_ALL_STRATEGIES,
        help="Override de estrategia (default: valor en .env)",
    )
    strategy_group.add_argument(
        "--compare-all",
        action="store_true",
        help="Ejecutar las 4 estrategias y generar comparison.csv",
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


def run_compare_all(base_config: CookbookConfig, dry_run: bool = False) -> int:
    """
    Ejecuta las 4 estrategias secuencialmente y genera comparison.csv.

    Cache persistente compartido: contextos generados en CONTEXTUAL_VECTOR
    se reutilizan en CONTEXTUAL_HYBRID y CONTEXTUAL_HYBRID_RERANK.
    """
    # Asegurar cache path compartido para estrategias contextuales
    if not base_config.contexts_cache_path:
        base_config = replace(
            base_config,
            contexts_cache_path=base_config.results_dir / "contexts_cache.json",
        )

    base_config.ensure_directories()

    if dry_run:
        print("=== COMPARE-ALL: Dry Run ===\n")
        for strategy in COMPARE_ALL_STRATEGIES:
            cfg = replace(base_config, strategy=strategy)
            errors = cfg.validate()
            status = "OK" if not errors else f"SKIP ({'; '.join(errors)})"
            print(f"  {strategy}: {status}")
        print("\n[DRY RUN] No se ejecuta evaluacion.")
        return 0

    print("=" * 60)
    print("  COMPARE-ALL: 4 estrategias")
    print("=" * 60)

    from shared.types import EvaluationRun

    runs: List[EvaluationRun] = []
    skipped: List[str] = []

    for strategy in COMPARE_ALL_STRATEGIES:
        cfg = replace(base_config, strategy=strategy)
        errors = cfg.validate()
        if errors:
            logger.warning(
                f"Skipping {strategy}: {'; '.join(errors)}"
            )
            skipped.append(strategy)
            continue

        print(f"\n{'=' * 60}")
        print(f"  Strategy: {strategy}")
        print(f"{'=' * 60}")

        evaluator = CookbookEvaluator(cfg)
        run = evaluator.run()
        runs.append(run)

    # Generar comparison.csv
    if len(runs) >= 2:
        comparison_path = export_comparison_csv(
            runs, base_config.results_dir, base_config.eval_k_values,
        )
        print(f"\n{'=' * 60}")
        print("  COMPARISON SUMMARY")
        print(f"{'=' * 60}")
        for run in runs:
            pass_at_k_strs = [
                f"Pass@{k}: {run.avg_recall_at_k.get(k, 0.0)*100:.2f}%"
                for k in base_config.eval_k_values
            ]
            print(f"  {run.retrieval_strategy:35s} {', '.join(pass_at_k_strs)}")
        if skipped:
            for s in skipped:
                print(f"  {s:35s} SKIPPED")
        print(f"\n  Comparison CSV: {comparison_path}")
    elif runs:
        logger.warning(
            "Solo 1 estrategia ejecutada. "
            "comparison.csv requiere al menos 2."
        )
    else:
        logger.error("Ninguna estrategia pudo ejecutarse.")
        return 1

    print(f"\nResultados en: {base_config.results_dir}")
    return 0


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    # 1. Construir config
    env_path = Path(args.env)
    if not env_path.exists():
        logger.error(f"Archivo .env no encontrado: {env_path.resolve()}")
        logger.error("Copiar env.example a .env y completar valores.")
        return 1

    config = CookbookConfig.from_env(str(env_path))

    # 2. --compare-all mode
    if args.compare_all:
        return run_compare_all(config, dry_run=args.dry_run)

    # Override de estrategia via CLI
    if args.strategy:
        config.strategy = args.strategy

    # 3. Validar
    errors = config.validate()
    if errors:
        logger.error("Errores de configuracion:")
        for e in errors:
            logger.error(f"  - {e}")
        return 1

    # 4. Mostrar resumen
    print(config.summary())

    # 5. Dry run?
    if args.dry_run:
        print("\n[DRY RUN] Config valida. No se ejecuta evaluacion.")
        return 0

    # 6. Ejecutar
    evaluator = CookbookEvaluator(config)
    run_result = evaluator.run()

    print(f"\nResultados en: {config.results_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
