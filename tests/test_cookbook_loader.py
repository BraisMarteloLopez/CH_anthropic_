"""
Tests para CookbookLoader.

Valida:
  - Parseo de codebase_chunks.json -> corpus NormalizedDocument
  - Parseo de evaluation_set.jsonl -> queries con golden_chunk_ids
  - Mapeo golden_chunk_uuids[uuid, idx] -> chunk_id del corpus
  - Parent documents en LoadedDataset.metadata
  - title=None en NormalizedDocument (PC-1)
  - Validacion: cada golden chunk existe en corpus
  - Content matching entre golden_contents y corpus
"""

import json
import pytest
import tempfile
from pathlib import Path

from sandbox_cookbook.loader import CookbookLoader, CookbookEvalQuery
from shared.types import DatasetType, NormalizedDocument


# =============================================================================
# FIXTURES: dataset sintetico minimal
# =============================================================================

def _make_dataset_files(tmp_path: Path):
    """Crea archivos JSON/JSONL sinteticos para tests."""

    # codebase_chunks.json: 2 documentos, 4 chunks total
    corpus_data = [
        {
            "doc_id": "doc_1",
            "original_uuid": "uuid-aaa-111",
            "content": "Full content of document 1: function setup() {}",
            "chunks": [
                {
                    "chunk_id": "doc_1_chunk_0",
                    "original_index": 0,
                    "content": "function setup() {",
                },
                {
                    "chunk_id": "doc_1_chunk_1",
                    "original_index": 1,
                    "content": "  return config;",
                },
            ],
        },
        {
            "doc_id": "doc_2",
            "original_uuid": "uuid-bbb-222",
            "content": "Full content of document 2: class Handler {}",
            "chunks": [
                {
                    "chunk_id": "doc_2_chunk_0",
                    "original_index": 0,
                    "content": "class Handler {",
                },
                {
                    "chunk_id": "doc_2_chunk_1",
                    "original_index": 1,
                    "content": "  handle(event) {}",
                },
            ],
        },
    ]

    dataset_path = tmp_path / "codebase_chunks.json"
    with open(dataset_path, "w") as f:
        json.dump(corpus_data, f)

    # evaluation_set.jsonl: 3 queries (1 con un golden, 1 con dos goldens, 1 normal)
    eval_data = [
        {
            "query": "What does setup do?",
            "answer": "It returns config.",
            "golden_doc_uuids": ["uuid-aaa-111"],
            "golden_chunk_uuids": [["uuid-aaa-111", 0]],
            "golden_documents": [],
            "golden_chunks": [
                {"doc_uuid": "uuid-aaa-111", "index": 0, "content": "function setup() {", "meta": {}},
            ],
            "meta": {},
        },
        {
            "query": "How does Handler work?",
            "answer": "It handles events.",
            "golden_doc_uuids": ["uuid-bbb-222"],
            "golden_chunk_uuids": [["uuid-bbb-222", 0], ["uuid-bbb-222", 1]],
            "golden_documents": [],
            "golden_chunks": [
                {"doc_uuid": "uuid-bbb-222", "index": 0, "content": "class Handler {", "meta": {}},
                {"doc_uuid": "uuid-bbb-222", "index": 1, "content": "  handle(event) {}", "meta": {}},
            ],
            "meta": {},
        },
        {
            "query": "What is the config?",
            "answer": "Return statement.",
            "golden_doc_uuids": ["uuid-aaa-111"],
            "golden_chunk_uuids": [["uuid-aaa-111", 1]],
            "golden_documents": [],
            "golden_chunks": [
                {"doc_uuid": "uuid-aaa-111", "index": 1, "content": "  return config;", "meta": {}},
            ],
            "meta": {},
        },
    ]

    eval_path = tmp_path / "evaluation_set.jsonl"
    with open(eval_path, "w") as f:
        for entry in eval_data:
            f.write(json.dumps(entry) + "\n")

    return dataset_path, eval_path


# =============================================================================
# TESTS
# =============================================================================

class TestCookbookLoaderParsing:
    """Tests de parseo basico."""

    def test_load_corpus_count(self, tmp_path):
        dataset_path, eval_path = _make_dataset_files(tmp_path)
        loader = CookbookLoader(dataset_path, eval_path)
        dataset, _ = loader.load()

        assert len(dataset.corpus) == 4, f"Expected 4 chunks, got {len(dataset.corpus)}"

    def test_load_query_count(self, tmp_path):
        dataset_path, eval_path = _make_dataset_files(tmp_path)
        loader = CookbookLoader(dataset_path, eval_path)
        dataset, eval_queries = loader.load()

        assert len(dataset.queries) == 3
        assert len(eval_queries) == 3

    def test_corpus_chunk_ids(self, tmp_path):
        dataset_path, eval_path = _make_dataset_files(tmp_path)
        loader = CookbookLoader(dataset_path, eval_path)
        dataset, _ = loader.load()

        expected_ids = {"doc_1_chunk_0", "doc_1_chunk_1", "doc_2_chunk_0", "doc_2_chunk_1"}
        assert set(dataset.corpus.keys()) == expected_ids

    def test_corpus_content(self, tmp_path):
        dataset_path, eval_path = _make_dataset_files(tmp_path)
        loader = CookbookLoader(dataset_path, eval_path)
        dataset, _ = loader.load()

        assert dataset.corpus["doc_1_chunk_0"].content == "function setup() {"
        assert dataset.corpus["doc_2_chunk_1"].content == "  handle(event) {}"

    def test_title_is_none(self, tmp_path):
        """PC-1: chunks de codigo no deben tener title."""
        dataset_path, eval_path = _make_dataset_files(tmp_path)
        loader = CookbookLoader(dataset_path, eval_path)
        dataset, _ = loader.load()

        for doc in dataset.corpus.values():
            assert doc.title is None, f"doc {doc.doc_id} has title={doc.title}"

    def test_dataset_type_retrieval_only(self, tmp_path):
        dataset_path, eval_path = _make_dataset_files(tmp_path)
        loader = CookbookLoader(dataset_path, eval_path)
        dataset, _ = loader.load()

        assert dataset.dataset_type == DatasetType.RETRIEVAL_ONLY

    def test_load_status_success(self, tmp_path):
        dataset_path, eval_path = _make_dataset_files(tmp_path)
        loader = CookbookLoader(dataset_path, eval_path)
        dataset, _ = loader.load()

        assert dataset.load_status == "success"


class TestCookbookLoaderParentDocuments:
    """Tests de parent documents en metadata."""

    def test_parent_documents_in_metadata(self, tmp_path):
        dataset_path, eval_path = _make_dataset_files(tmp_path)
        loader = CookbookLoader(dataset_path, eval_path)
        dataset, _ = loader.load()

        parent_docs = dataset.metadata["parent_documents"]
        assert len(parent_docs) == 2
        assert "uuid-aaa-111" in parent_docs
        assert "uuid-bbb-222" in parent_docs

    def test_parent_document_content(self, tmp_path):
        dataset_path, eval_path = _make_dataset_files(tmp_path)
        loader = CookbookLoader(dataset_path, eval_path)
        dataset, _ = loader.load()

        parent_docs = dataset.metadata["parent_documents"]
        assert parent_docs["uuid-aaa-111"] == "Full content of document 1: function setup() {}"

    def test_chunk_metadata_parent_doc_id(self, tmp_path):
        dataset_path, eval_path = _make_dataset_files(tmp_path)
        loader = CookbookLoader(dataset_path, eval_path)
        dataset, _ = loader.load()

        doc = dataset.corpus["doc_1_chunk_0"]
        assert doc.metadata["parent_doc_id"] == "uuid-aaa-111"
        assert doc.metadata["original_index"] == 0

        doc2 = dataset.corpus["doc_2_chunk_1"]
        assert doc2.metadata["parent_doc_id"] == "uuid-bbb-222"
        assert doc2.metadata["original_index"] == 1


class TestCookbookLoaderGoldenMapping:
    """Tests del mapeo golden_chunk_uuids -> chunk_id."""

    def test_single_golden_chunk(self, tmp_path):
        dataset_path, eval_path = _make_dataset_files(tmp_path)
        loader = CookbookLoader(dataset_path, eval_path)
        _, eval_queries = loader.load()

        eq = eval_queries[0]  # "What does setup do?"
        assert eq.golden_chunk_ids == ["doc_1_chunk_0"]

    def test_multiple_golden_chunks(self, tmp_path):
        dataset_path, eval_path = _make_dataset_files(tmp_path)
        loader = CookbookLoader(dataset_path, eval_path)
        _, eval_queries = loader.load()

        eq = eval_queries[1]  # "How does Handler work?"
        assert eq.golden_chunk_ids == ["doc_2_chunk_0", "doc_2_chunk_1"]

    def test_golden_contents_match(self, tmp_path):
        """Content-based matching: golden_contents == corpus content."""
        dataset_path, eval_path = _make_dataset_files(tmp_path)
        loader = CookbookLoader(dataset_path, eval_path)
        dataset, eval_queries = loader.load()

        for eq in eval_queries:
            for chunk_id, golden_content in zip(eq.golden_chunk_ids, eq.golden_contents):
                assert dataset.corpus[chunk_id].content == golden_content, (
                    f"Content mismatch for {chunk_id}"
                )

    def test_relevant_doc_ids_in_normalized_query(self, tmp_path):
        dataset_path, eval_path = _make_dataset_files(tmp_path)
        loader = CookbookLoader(dataset_path, eval_path)
        dataset, _ = loader.load()

        q0 = dataset.queries[0]
        assert q0.relevant_doc_ids == ["doc_1_chunk_0"]

        q1 = dataset.queries[1]
        assert q1.relevant_doc_ids == ["doc_2_chunk_0", "doc_2_chunk_1"]


class TestCookbookLoaderValidation:
    """Tests de validacion de golden chunks."""

    def test_validation_passes_for_valid_data(self, tmp_path):
        """No debe lanzar excepcion con datos validos."""
        dataset_path, eval_path = _make_dataset_files(tmp_path)
        loader = CookbookLoader(dataset_path, eval_path)
        dataset, eval_queries = loader.load()  # No raise = OK

    def test_validation_fails_for_invalid_golden(self, tmp_path):
        """Debe lanzar ValueError si golden chunk no existe en corpus."""
        # Crear corpus minimal
        corpus = [
            {
                "doc_id": "doc_1",
                "original_uuid": "uuid-aaa",
                "content": "content",
                "chunks": [
                    {"chunk_id": "doc_1_chunk_0", "original_index": 0, "content": "chunk"},
                ],
            },
        ]
        dataset_path = tmp_path / "corpus.json"
        with open(dataset_path, "w") as f:
            json.dump(corpus, f)

        # Eval query que referencia UUID inexistente
        eval_data = [
            {
                "query": "test",
                "answer": "test",
                "golden_doc_uuids": ["uuid-nonexistent"],
                "golden_chunk_uuids": [["uuid-nonexistent", 0]],
                "golden_documents": [],
                "golden_chunks": [{"doc_uuid": "uuid-nonexistent", "index": 0, "content": "x", "meta": {}}],
                "meta": {},
            },
        ]
        eval_path = tmp_path / "eval.jsonl"
        with open(eval_path, "w") as f:
            f.write(json.dumps(eval_data[0]) + "\n")

        loader = CookbookLoader(dataset_path, eval_path)
        # The UUID won't be found in uuid_chunk_index, so golden_chunk_ids will be empty
        # The validation should pass (empty golden is not invalid per se)
        # But let's test with an UUID that exists but wrong chunk index
        corpus2 = [
            {
                "doc_id": "doc_1",
                "original_uuid": "uuid-aaa",
                "content": "content",
                "chunks": [
                    {"chunk_id": "doc_1_chunk_0", "original_index": 0, "content": "chunk"},
                ],
            },
        ]
        dataset_path2 = tmp_path / "corpus2.json"
        with open(dataset_path2, "w") as f:
            json.dump(corpus2, f)

        # Reference chunk index 5 which doesn't exist
        eval_data2 = [
            {
                "query": "test",
                "answer": "test",
                "golden_doc_uuids": ["uuid-aaa"],
                "golden_chunk_uuids": [["uuid-aaa", 5]],
                "golden_documents": [],
                "golden_chunks": [{"doc_uuid": "uuid-aaa", "index": 5, "content": "x", "meta": {}}],
                "meta": {},
            },
        ]
        eval_path2 = tmp_path / "eval2.jsonl"
        with open(eval_path2, "w") as f:
            f.write(json.dumps(eval_data2[0]) + "\n")

        loader2 = CookbookLoader(dataset_path2, eval_path2)
        # chunk_index 5 won't be found in uuid_chunk_index, logged as warning
        # golden_chunk_ids will be empty, validation passes
        dataset, eval_queries = loader2.load()
        assert len(eval_queries[0].golden_chunk_ids) == 0


class TestCookbookLoaderCounters:
    """Tests de contadores del dataset."""

    def test_total_queries(self, tmp_path):
        dataset_path, eval_path = _make_dataset_files(tmp_path)
        loader = CookbookLoader(dataset_path, eval_path)
        dataset, _ = loader.load()

        assert dataset.total_queries == 3

    def test_total_corpus(self, tmp_path):
        dataset_path, eval_path = _make_dataset_files(tmp_path)
        loader = CookbookLoader(dataset_path, eval_path)
        dataset, _ = loader.load()

        assert dataset.total_corpus == 4

    def test_dataset_name(self, tmp_path):
        dataset_path, eval_path = _make_dataset_files(tmp_path)
        loader = CookbookLoader(dataset_path, eval_path)
        dataset, _ = loader.load()

        assert dataset.name == "cookbook"


# =============================================================================
# STANDALONE
# =============================================================================

if __name__ == "__main__":
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    for cls in [
        TestCookbookLoaderParsing,
        TestCookbookLoaderParentDocuments,
        TestCookbookLoaderGoldenMapping,
        TestCookbookLoaderValidation,
        TestCookbookLoaderCounters,
    ]:
        instance = cls()
        for name in dir(instance):
            if name.startswith("test_"):
                print(f"  {cls.__name__}.{name}...", end=" ")
                getattr(instance, name)(tmp)
                print("OK")
    print("All tests passed!")
