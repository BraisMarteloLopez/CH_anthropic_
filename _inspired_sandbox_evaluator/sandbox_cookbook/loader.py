"""
CookbookLoader: carga dataset local del cookbook de Anthropic.

Lee directamente JSON/JSONL local (no MinIO, no Parquet).
Fuente: https://github.com/anthropics/anthropic-cookbook/tree/main/capabilities/contextual-embeddings/data

Archivos:
  - codebase_chunks.json: 90 documentos, 737 chunks
  - evaluation_set.jsonl: 248 queries, 306 golden chunks
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from shared.types import (
    DatasetType,
    LoadedDataset,
    NormalizedDocument,
    NormalizedQuery,
)

logger = logging.getLogger(__name__)


@dataclass
class CookbookEvalQuery:
    """Query con golden chunks para evaluacion Pass@k."""
    query_id: str
    query_text: str
    golden_chunk_ids: List[str]
    golden_contents: List[str]


class CookbookLoader:
    """
    Carga el dataset del cookbook de Anthropic desde archivos locales.

    Mapeo:
      - Cada chunk -> NormalizedDocument(doc_id=chunk_id, content=chunk_content)
      - Parent documents -> LoadedDataset.metadata["parent_documents"]
      - golden_chunk_uuids[uuid, idx] -> chunk_id via uuid_to_doc_id index
    """

    def __init__(self, dataset_path: Path, eval_path: Path):
        self.dataset_path = dataset_path
        self.eval_path = eval_path

    def load(self) -> Tuple[LoadedDataset, List[CookbookEvalQuery]]:
        """
        Carga corpus y queries del cookbook.

        Returns:
            Tuple de (LoadedDataset con corpus y queries, lista de CookbookEvalQuery
            con golden_contents para validacion content-based).
        """
        # 1. Cargar corpus
        corpus, parent_documents, uuid_chunk_index = self._load_corpus()

        # 2. Cargar evaluation queries
        eval_queries, normalized_queries = self._load_eval_queries(uuid_chunk_index)

        # 3. Validar que todos los golden chunks existen en corpus
        self._validate_golden_chunks(eval_queries, corpus)

        # 4. Construir LoadedDataset
        dataset = LoadedDataset(
            name="cookbook",
            dataset_type=DatasetType.RETRIEVAL_ONLY,
            queries=normalized_queries,
            corpus=corpus,
            total_queries=len(normalized_queries),
            total_corpus=len(corpus),
            load_status="success",
            metadata={
                "parent_documents": parent_documents,
            },
        )

        logger.info(
            f"Dataset cargado: {len(corpus)} chunks, "
            f"{len(normalized_queries)} queries, "
            f"{len(parent_documents)} parent documents"
        )

        return dataset, eval_queries

    def _load_corpus(
        self,
    ) -> Tuple[
        Dict[str, NormalizedDocument],
        Dict[str, str],
        Dict[str, Dict[int, str]],
    ]:
        """
        Lee codebase_chunks.json.

        Returns:
            Tuple de:
              - corpus: Dict[chunk_id, NormalizedDocument]
              - parent_documents: Dict[original_uuid, full_text]
              - uuid_chunk_index: Dict[uuid, Dict[chunk_index, chunk_id]]
                para mapeo golden_chunk_uuids -> chunk_id del corpus
        """
        logger.info(f"Cargando corpus desde {self.dataset_path}...")

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            raw_documents = json.load(f)

        corpus: Dict[str, NormalizedDocument] = {}
        parent_documents: Dict[str, str] = {}
        # uuid -> {chunk_index -> chunk_id}
        uuid_chunk_index: Dict[str, Dict[int, str]] = {}

        for doc in raw_documents:
            doc_id_str = str(doc["doc_id"])
            original_uuid = doc["original_uuid"]
            parent_content = doc["content"]

            # Almacenar parent document
            parent_documents[original_uuid] = parent_content
            uuid_chunk_index[original_uuid] = {}

            # Cada chunk -> NormalizedDocument
            for chunk in doc["chunks"]:
                chunk_id = chunk["chunk_id"]  # ej: "doc_1_chunk_0"
                original_index = chunk["original_index"]

                corpus[chunk_id] = NormalizedDocument(
                    doc_id=chunk_id,
                    content=chunk["content"],
                    title=None,  # PC-1: no title para chunks de codigo
                    metadata={
                        "parent_doc_id": original_uuid,
                        "original_index": original_index,
                        "doc_id_str": doc_id_str,
                    },
                )

                uuid_chunk_index[original_uuid][original_index] = chunk_id

        logger.info(
            f"  Corpus: {len(raw_documents)} documentos, "
            f"{len(corpus)} chunks"
        )
        return corpus, parent_documents, uuid_chunk_index

    def _load_eval_queries(
        self,
        uuid_chunk_index: Dict[str, Dict[int, str]],
    ) -> Tuple[List[CookbookEvalQuery], List[NormalizedQuery]]:
        """
        Lee evaluation_set.jsonl.

        Mapea golden_chunk_uuids[uuid, chunk_index] -> chunk_id del corpus
        usando el indice uuid_chunk_index construido en _load_corpus().

        Returns:
            Tuple de (CookbookEvalQuery list, NormalizedQuery list).
        """
        logger.info(f"Cargando evaluation set desde {self.eval_path}...")

        eval_queries: List[CookbookEvalQuery] = []
        normalized_queries: List[NormalizedQuery] = []

        with open(self.eval_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                entry = json.loads(line)
                query_id = f"q_{i}"
                query_text = entry["query"]

                # Mapear golden_chunk_uuids -> chunk_ids del corpus
                golden_chunk_ids: List[str] = []
                for uuid, chunk_index in entry["golden_chunk_uuids"]:
                    chunk_map = uuid_chunk_index.get(uuid)
                    if chunk_map is None:
                        logger.warning(
                            f"  Query {query_id}: UUID '{uuid[:20]}...' "
                            "no encontrado en corpus. Ignorando golden chunk."
                        )
                        continue
                    chunk_id = chunk_map.get(chunk_index)
                    if chunk_id is None:
                        logger.warning(
                            f"  Query {query_id}: chunk_index {chunk_index} "
                            f"no encontrado para UUID '{uuid[:20]}...'. "
                            "Ignorando golden chunk."
                        )
                        continue
                    golden_chunk_ids.append(chunk_id)

                # Extraer golden_contents para validacion content-based
                golden_contents: List[str] = []
                for gc in entry.get("golden_chunks", []):
                    golden_contents.append(gc["content"])

                eval_queries.append(CookbookEvalQuery(
                    query_id=query_id,
                    query_text=query_text,
                    golden_chunk_ids=golden_chunk_ids,
                    golden_contents=golden_contents,
                ))

                normalized_queries.append(NormalizedQuery(
                    query_id=query_id,
                    query_text=query_text,
                    relevant_doc_ids=golden_chunk_ids,
                    expected_answer=entry.get("answer"),
                    metadata={
                        "golden_doc_uuids": entry.get("golden_doc_uuids", []),
                    },
                ))

        logger.info(
            f"  Queries: {len(eval_queries)}, "
            f"Golden chunks totales: {sum(len(q.golden_chunk_ids) for q in eval_queries)}"
        )
        return eval_queries, normalized_queries

    def _validate_golden_chunks(
        self,
        eval_queries: List[CookbookEvalQuery],
        corpus: Dict[str, NormalizedDocument],
    ) -> None:
        """Valida que todos los golden_chunk_ids existen en el corpus."""
        missing = []
        for eq in eval_queries:
            for chunk_id in eq.golden_chunk_ids:
                if chunk_id not in corpus:
                    missing.append((eq.query_id, chunk_id))

        if missing:
            sample = missing[:5]
            raise ValueError(
                f"{len(missing)} golden chunks no encontrados en corpus. "
                f"Primeros 5: {sample}. "
                "Verificar mapeo uuid->doc_id->chunk_id en el loader."
            )

        total_golden = sum(len(eq.golden_chunk_ids) for eq in eval_queries)
        logger.info(
            f"  Validacion: {total_golden} golden chunks verificados en corpus"
        )


__all__ = ["CookbookLoader", "CookbookEvalQuery"]
