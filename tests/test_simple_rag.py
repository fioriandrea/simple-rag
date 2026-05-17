import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("simple_rag", REPO_ROOT / "simple_rag.py")
assert SPEC is not None
simple_rag = importlib.util.module_from_spec(SPEC)
sys.modules["simple_rag"] = simple_rag
assert SPEC.loader is not None
SPEC.loader.exec_module(simple_rag)


class CharacterTokenizer:
    def tokenize(self, text):
        return [ord(char) for char in text]

    def detokenize(self, tokens):
        return "".join(chr(token) for token in tokens)


class FakeEmbeddingModel:
    model_path = "fake.gguf"
    n_batch = 10

    def embed(self, input, **_kwargs):
        return [[float(len(text)), 1.0] for text in input]


class FailingEmbeddingModel(FakeEmbeddingModel):
    def embed(self, input, **_kwargs):
        if any(text == "abcdefghij" for text in input):
            raise RuntimeError("embedding failed")
        return super().embed(input, **_kwargs)


@pytest.fixture
def fake_model():
    return FakeEmbeddingModel()


@pytest.fixture
def corpus(tmp_path):
    docs1 = tmp_path / "docs1"
    docs2 = tmp_path / "docs2"
    docs1.mkdir()
    docs2.mkdir()
    (docs1 / "a.txt").write_text("qwertyuiop0123456789")
    (docs1 / "empty.txt").write_text("")
    (docs2 / "b.txt").write_text("abcdefghijABCDEFGHIJ")
    (docs2 / "c.txt").write_text("klmnopqrstKLMNOPQRST")
    return docs1, docs2


def collection_records(dbpath):
    client = simple_rag.DB.make_db_client(dbpath)
    try:
        collection = client.get_collection(name="docs")
        data = collection.get(include=["documents", "metadatas"])
        records = zip(data["ids"], data["documents"], data["metadatas"])
        return sorted(records, key=lambda item: int(item[0]))
    finally:
        client.close()


def write_test_files(dbpath, paths, fake_model, append=False):
    with simple_rag.DB(dbpath, fake_model, exists_ok=append) as db:
        db.write_files(
            [str(path) for path in paths], CharacterTokenizer(), overlap_perc=0
        )


def test_split_tokens_uses_context_window():
    tokens = CharacterTokenizer().tokenize("qwertyuiop0123456789")
    chunks = simple_rag.split_tokens(tokens, context_window=10, overlap_perc=0)

    assert [CharacterTokenizer().detokenize(chunk) for chunk in chunks] == [
        "qwertyuiop",
        "0123456789",
    ]


def test_split_tokens_uses_sliding_overlap():
    text = "".join(chr(ord("A") + i % 26) for i in range(100))
    tokens = CharacterTokenizer().tokenize(text)

    chunks = list(simple_rag.split_tokens(tokens, context_window=10, overlap_perc=0.2))

    chunk_size = 10
    overlap = int(chunk_size * 0.2)
    step = chunk_size - overlap
    assert chunks == [
        CharacterTokenizer().tokenize(text[start : start + chunk_size])
        for start in range(0, 100, step)
    ]


def test_files_to_splits_handles_empty_text(tmp_path):
    filepath = tmp_path / "empty.txt"
    filepath.write_text("")

    splitsiter = simple_rag.files_to_splits(
        filepaths=[str(filepath)],
        tokenizer=CharacterTokenizer(),
        context_window=4,
        overlap_perc=0,
    )

    assert [list(splits) for splits in splitsiter] == [[]]


def test_read_file_list_skips_empty_lines(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    file_list = tmp_path / "files.txt"
    file_list.write_text(f"{a}\n\n{b}\n")

    assert simple_rag.read_file_list(file_list) == [a, b]


def test_read_file_list_preserves_filename_whitespace(tmp_path):
    file_list = tmp_path / "files.txt"
    file_list.write_text("name with trailing space \n")

    assert simple_rag.read_file_list(file_list) == [Path("name with trailing space ")]


def test_read_file_list_supports_nul_separators(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    file_list = tmp_path / "files.txt"
    file_list.write_text(f"{a}\0{b}\0")

    assert simple_rag.read_file_list(file_list, from0=True) == [a, b]


def test_db_write_files_creates_and_appends_documents(tmp_path, corpus, fake_model):
    docs1, docs2 = corpus
    dbpath = tmp_path / "db"

    write_test_files(dbpath, [docs1 / "a.txt", docs1 / "empty.txt"], fake_model)
    write_test_files(
        dbpath, [docs2 / "b.txt", docs2 / "c.txt"], fake_model, append=True
    )

    records = collection_records(dbpath)

    assert [
        (item_id, metadata["file"], document) for item_id, document, metadata in records
    ] == [
        ("0", str(docs1 / "a.txt"), "qwertyuiop"),
        ("1", str(docs1 / "a.txt"), "0123456789"),
        ("2", str(docs2 / "b.txt"), "abcdefghij"),
        ("3", str(docs2 / "b.txt"), "ABCDEFGHIJ"),
        ("4", str(docs2 / "c.txt"), "klmnopqrst"),
        ("5", str(docs2 / "c.txt"), "KLMNOPQRST"),
    ]


def test_db_write_documents_dumps_current_and_remaining_on_failure(tmp_path, corpus):
    docs1, docs2 = corpus
    dbpath = tmp_path / "db"
    filepaths = [docs1 / "a.txt", docs2 / "b.txt", docs2 / "c.txt"]
    splitsiter = simple_rag.files_to_splits(
        filepaths=filepaths,
        tokenizer=CharacterTokenizer(),
        context_window=10,
        overlap_perc=0,
    )

    with simple_rag.DB(dbpath, FailingEmbeddingModel(), exists_ok=False) as db:
        current_dump = db.current_file_dump_path
        remaining_dump = db.remaining_files_dump_path
        with pytest.raises(RuntimeError, match="embedding failed"):
            db.write_documents(filepaths, splitsiter)

    assert simple_rag.read_file_list(current_dump, from0=True) == [docs2 / "b.txt"]
    assert simple_rag.read_file_list(remaining_dump, from0=True) == [docs2 / "c.txt"]
    assert [
        (metadata["file"], document)
        for _item_id, document, metadata in collection_records(dbpath)
    ] == [
        (str(docs1 / "a.txt"), "qwertyuiop"),
        (str(docs1 / "a.txt"), "0123456789"),
    ]


def test_db_stores_model_path_in_collection_metadata(tmp_path, fake_model):
    dbpath = tmp_path / "db"

    write_test_files(dbpath, [], fake_model)

    with simple_rag.DB(dbpath, fake_model) as db:
        assert db.collection.metadata["model_path"] == "fake.gguf"


def test_db_get_model_path_reads_collection_metadata(tmp_path, fake_model):
    dbpath = tmp_path / "db"

    write_test_files(dbpath, [], fake_model)

    assert simple_rag.DB.get_model_path(dbpath) == Path("fake.gguf")


def test_db_refuses_existing_path_without_append(tmp_path, fake_model):
    dbpath = tmp_path / "db"

    write_test_files(dbpath, [], fake_model)

    with pytest.raises(FileExistsError):
        write_test_files(dbpath, [], fake_model)


def test_db_delete_files_removes_matching_chunks(tmp_path, corpus, fake_model):
    docs1, docs2 = corpus
    dbpath = tmp_path / "db"

    write_test_files(
        dbpath, [docs1 / "a.txt", docs2 / "b.txt", docs2 / "c.txt"], fake_model
    )

    with simple_rag.DB(dbpath, fake_model) as db:
        db.delete_files([docs1 / "a.txt", docs2 / "c.txt"])

    records = collection_records(dbpath)

    assert [
        (metadata["file"], document) for _item_id, document, metadata in records
    ] == [
        (str(docs2 / "b.txt"), "abcdefghij"),
        (str(docs2 / "b.txt"), "ABCDEFGHIJ"),
    ]
