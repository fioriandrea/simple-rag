import argparse
import asyncio
import json
import os
from pathlib import Path

import pytest
import simple_rag


class CharacterTokenizer:
    def tokenize(self, text):
        return [ord(char) for char in text]

    def detokenize(self, tokens):
        return "".join(chr(token) for token in tokens)


class FakeEmbeddingModel:
    model_path = "fake.gguf"
    n_batch = 10

    def embed(self, input, **_kwargs):
        return [self.embed_text(text) for text in input]

    def embed_text(self, text):
        values = [float(ord(char)) for char in text[: self.n_batch]]
        return values + [0.0] * (self.n_batch - len(values))


class FailingEmbeddingModel(FakeEmbeddingModel):
    def embed(self, input, **_kwargs):
        if any(text == "abcdefghij" for text in input):
            raise RuntimeError("embedding failed")
        return super().embed(input, **_kwargs)


class FakeMCP:
    def __init__(self):
        self.tools = []
        self.transport = None

    def tool(self, description):
        def decorator(func):
            self.tools.append((description, func))
            return func

        return decorator

    def run(self, transport):
        self.transport = transport


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
        db.write_files(paths, CharacterTokenizer(), overlap_perc=0)


def dbgen_args(dbpath, **overrides):
    args = argparse.Namespace(
        db=dbpath,
        append=False,
        resume=False,
        sync_indexed=False,
        files=None,
        files_from=None,
        from0=False,
        glob=None,
        overlap_perc=0,
    )
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


def query_args(dbpath, **overrides):
    args = argparse.Namespace(
        db=dbpath,
        query="query",
        k=1,
        files_only=False,
        json=False,
    )
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


def mcp_args(dbpath, **overrides):
    args = argparse.Namespace(
        db=dbpath,
        description="test docs",
        description_file=None,
    )
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


def run_test_dbgen(args, fake_model):
    simple_rag.run_dbgen(args, fake_model, CharacterTokenizer())


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


def test_read_file_list_preserves_filename_whitespace(tmp_path):
    file_list = tmp_path / "files.txt"
    file_list.write_text("name with trailing space \n")

    assert simple_rag.read_file_list(file_list) == [Path("name with trailing space ")]


def test_read_file_list_supports_nul_separators(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    file_list = tmp_path / "files.txt"
    file_list.write_text(f"{a}\0{b}\0\0")

    assert simple_rag.read_file_list(file_list, from0=True) == [a, b]


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


def test_db_get_model_path_reads_collection_metadata(tmp_path, fake_model):
    dbpath = tmp_path / "db"

    write_test_files(dbpath, [], fake_model)

    assert simple_rag.DB.get_model_path(dbpath) == Path("fake.gguf")


def test_db_refuses_existing_path_without_append(tmp_path, fake_model):
    dbpath = tmp_path / "db"

    write_test_files(dbpath, [], fake_model)

    with pytest.raises(FileExistsError):
        write_test_files(dbpath, [], fake_model)


def test_query_command_prints_json(tmp_path, corpus, fake_model, capsys):
    docs1, _docs2 = corpus
    filepath = docs1 / "a.txt"
    dbpath = tmp_path / "db"
    write_test_files(dbpath, [filepath], fake_model)

    simple_rag.run_query(
        query_args(dbpath, query="qwertyuiop", k=1, json=True),
        fake_model,
    )

    assert json.loads(capsys.readouterr().out) == [
        {
            "file": str(filepath),
            "text": "qwertyuiop",
        }
    ]


def test_delete_command_removes_files(tmp_path, corpus, fake_model):
    docs1, docs2 = corpus
    dbpath = tmp_path / "db"
    write_test_files(dbpath, [docs1 / "a.txt", docs2 / "b.txt"], fake_model)

    simple_rag.run_delete(
        argparse.Namespace(db=dbpath, files=[docs1 / "a.txt"]), fake_model
    )

    assert [
        (metadata["file"], document)
        for _item_id, document, metadata in collection_records(dbpath)
    ] == [
        (str(docs2 / "b.txt"), "abcdefghij"),
        (str(docs2 / "b.txt"), "ABCDEFGHIJ"),
    ]


def test_mcp_command_registers_query_tool(tmp_path, corpus, fake_model):
    docs1, _docs2 = corpus
    dbpath = tmp_path / "db"
    server = FakeMCP()
    write_test_files(dbpath, [docs1 / "a.txt"], fake_model)

    simple_rag.run_mcp(mcp_args(dbpath), fake_model, server)

    assert server.transport == "streamable-http"
    assert len(server.tools) == 1
    _description, tool = server.tools[0]
    assert asyncio.run(tool("qwerty", 1, files_only=True)) == str(docs1 / "a.txt")


def test_dbgen_files_from_reads_file_list(tmp_path, corpus, fake_model):
    docs1, _docs2 = corpus
    dbpath = tmp_path / "db"
    file_list = tmp_path / "files.txt"
    file_list.write_text(f"\n{docs1 / 'a.txt'}\n\n")

    run_test_dbgen(
        dbgen_args(dbpath, files_from=file_list),
        fake_model,
    )

    assert [
        (metadata["file"], document)
        for _item_id, document, metadata in collection_records(dbpath)
    ] == [
        (str(docs1 / "a.txt"), "qwertyuiop"),
        (str(docs1 / "a.txt"), "0123456789"),
    ]


def test_dbgen_resume_rewrites_current_then_remaining(tmp_path, corpus, fake_model):
    docs1, docs2 = corpus
    dbpath = tmp_path / "db"

    write_test_files(dbpath, [docs1 / "a.txt", docs2 / "b.txt"], fake_model)
    with simple_rag.DB(dbpath, fake_model) as db:
        simple_rag.write_file_list(db.current_file_dump_path, [docs2 / "b.txt"])
        simple_rag.write_file_list(db.remaining_files_dump_path, [docs2 / "c.txt"])

    run_test_dbgen(
        dbgen_args(dbpath, resume=True),
        fake_model,
    )

    with simple_rag.DB(dbpath, fake_model) as db:
        assert not db.current_file_dump_path.exists()
        assert not db.remaining_files_dump_path.exists()
    assert [
        (metadata["file"], document)
        for _item_id, document, metadata in collection_records(dbpath)
    ] == [
        (str(docs1 / "a.txt"), "qwertyuiop"),
        (str(docs1 / "a.txt"), "0123456789"),
        (str(docs2 / "b.txt"), "abcdefghij"),
        (str(docs2 / "b.txt"), "ABCDEFGHIJ"),
        (str(docs2 / "c.txt"), "klmnopqrst"),
        (str(docs2 / "c.txt"), "KLMNOPQRST"),
    ]


def test_dbgen_sync_indexed_removes_missing_and_rewrites_modified(
    tmp_path, corpus, fake_model
):
    docs1, docs2 = corpus
    dbpath = tmp_path / "db"

    write_test_files(dbpath, [docs1 / "a.txt", docs2 / "b.txt"], fake_model)
    stored_mtime = (docs2 / "b.txt").stat().st_mtime
    (docs1 / "a.txt").unlink()
    (docs2 / "b.txt").write_text("updatedTXT")
    os.utime(docs2 / "b.txt", (stored_mtime + 10, stored_mtime + 10))

    run_test_dbgen(
        dbgen_args(dbpath, sync_indexed=True),
        fake_model,
    )

    assert [
        (metadata["file"], document)
        for _item_id, document, metadata in collection_records(dbpath)
    ] == [
        (str(docs2 / "b.txt"), "updatedTXT"),
    ]
