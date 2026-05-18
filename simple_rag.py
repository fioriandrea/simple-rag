# Copyright (C) Andrea Fiori

# This file is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# For a full copy of the GNU General Public License
# see <http://www.gnu.org/licenses/>.

import argparse
import glob
import itertools
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Optional, Protocol

import chromadb
from llama_cpp import Llama
from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Tokenizer(Protocol):
    def tokenize(self, text: str) -> list[int]: ...
    def detokenize(self, tokens: list[int]) -> str: ...


class LlamaTokenizer(Tokenizer):
    def __init__(self, model):
        self.model = model

    def tokenize(self, text: str) -> list[int]:
        return self.model.tokenize(bytes(text, "utf-8", errors="ignore"))

    def detokenize(self, tokens: list[int]) -> str:
        return str(self.model.detokenize(tokens), "utf-8", errors="ignore")


def split_tokens(
    tokens: list[int], context_window: int, overlap_perc: float
) -> Iterator[list[int]]:
    assert context_window > 0
    assert 0 <= overlap_perc < 1
    overlap = int(context_window * overlap_perc)
    i = 0
    while i < len(tokens):
        yield tokens[i : i + context_window]
        i += context_window - overlap


def read_file_list(filepath: Path, from0: bool = False) -> list[Path]:
    with open(filepath, "r") as f:
        sep = "\0" if from0 else "\n"
        return [Path(item) for item in f.read().split(sep) if item]


def write_file_list(filepath: Path, filepaths: list[Path]):
    with open(filepath, "w") as f:
        f.write("\0".join(str(p) for p in filepaths))


def files_to_splits(
    filepaths: list[Path],
    tokenizer: Tokenizer,
    context_window: int,
    overlap_perc: float,
) -> Iterator[Iterator[str]]:
    for filepath in filepaths:
        with open(filepath, "r", errors="ignore") as f:
            text = f.read()
        tokens = tokenizer.tokenize(text)
        yield (
            tokenizer.detokenize(t)
            for t in split_tokens(tokens, context_window, overlap_perc)
        )


def make_embedding_model(
    model_path: Path, n_batch: int, n_ubatch: Optional[int] = None
) -> Llama:
    assert n_batch is not None
    if n_ubatch is None:
        n_ubatch = n_batch
    return Llama(
        model_path=str(model_path),
        embedding=True,
        n_ctx=0,
        n_batch=n_batch,
        n_ubatch=n_ubatch,
        verbose=False,
    )


class DB:
    def __init__(self, dbpath: Path, model, mode: str = "r"):
        self.dbpath = dbpath
        assert mode in {"r", "x"}
        if mode == "r" and not self.dbpath.exists():
            raise FileNotFoundError(f"{dbpath} does not exist")
        if mode == "x" and self.dbpath.exists():
            raise FileExistsError(f"{dbpath} already exists")
        self.client = DB.make_db_client(dbpath)
        self.max_batch_size = self.client.get_max_batch_size()
        self.collection = self.client.get_or_create_collection(
            name="docs",
            metadata={"model_path": model.model_path},
        )
        self.model = model
        assert self.collection.metadata["model_path"] == self.model.model_path

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    @staticmethod
    def make_db_client(dbpath: Path):
        return chromadb.PersistentClient(
            path=dbpath,
            settings=chromadb.config.Settings(anonymized_telemetry=False),
        )

    def indexed_files_with_metadata(self) -> list[dict]:
        data = self.collection.get(include=["metadatas"])
        res = []
        seen = set()
        for meta in data["metadatas"]:
            filepath = meta["file"]
            if filepath in seen:
                continue
            seen.add(filepath)
            res.append(
                {
                    **meta,
                    "file": Path(filepath),
                }
            )
        return res

    @property
    def current_file_dump_path(self) -> Path:
        return self.dbpath / "current-file-dump"

    @property
    def remaining_files_dump_path(self) -> Path:
        return self.dbpath / "remaining-files-dump"

    @staticmethod
    def get_model_path(dbpath):
        if not Path(dbpath).exists():
            raise FileNotFoundError(f"{dbpath} does not exist")
        client = DB.make_db_client(dbpath)
        try:
            collection = client.get_collection(name="docs")
            model_path = (collection.metadata or {}).get("model_path")
            if not model_path:
                raise ValueError(f"database {dbpath} has no model_path metadata")
            return Path(model_path)
        finally:
            client.close()

    @property
    def count(self):
        return self.collection.count()

    def write_files(
        self,
        filepaths: list[Path],
        tokenizer: Tokenizer,
        overlap_perc: float,
    ):
        splitsiter = files_to_splits(
            filepaths=filepaths,
            tokenizer=tokenizer,
            context_window=self.model.n_batch,
            overlap_perc=overlap_perc,
        )
        self.write_documents(filepaths, splitsiter)

    def write_document(
        self,
        filepath: Path,
        splits: Iterator[str],
    ):
        mtime = filepath.stat().st_mtime
        for batch in itertools.batched(splits, n=self.max_batch_size):
            logger.info(
                f"writing batch of size {len(batch)}"
                f" from {filepath} to {self.dbpath}"
            )
            ids = [str(k) for k in range(self.count, self.count + len(batch))]
            # truncate=True because splitter may off by one token length.
            # This is because sometimes:
            # len(tokenize(detokenize(tokenize(t)))) == len(tokenize(t)) + 1
            embeddings = self.model.embed(batch, truncate=True)
            metadatas = [{"file": str(filepath), "mtime": mtime}] * len(batch)
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=list(batch),
                metadatas=metadatas,
            )

    def dump_current_file(self, filepath: Path):
        try:
            write_file_list(self.current_file_dump_path, [filepath])
        except Exception:
            logger.exception(
                f"unable to write current file dump {self.current_file_dump_path} "
                f"for {filepath}"
            )

    def dump_remaining_files(self, filepaths: list[Path]):
        try:
            write_file_list(self.remaining_files_dump_path, filepaths)
        except Exception:
            logger.exception(
                f"unable to write remaining files dump {self.remaining_files_dump_path}"
            )

    def write_documents(
        self,
        filepaths: list[Path],
        splitsiter: Iterator[Iterator[str]],
    ):
        splitsiter = iter(splitsiter)
        for i, filepath in enumerate(filepaths):
            try:
                splits = next(splitsiter)
                logger.info(
                    f"writing file [{i + 1} / {len(filepaths)}] "
                    f"{filepath} to {self.dbpath}"
                )
                self.write_document(filepath, splits)
                logger.info(
                    f"database {self.dbpath} updated "
                    f"(now with {self.count} records)"
                )
            except:
                self.dump_current_file(filepaths[i])
                self.dump_remaining_files(filepaths[i + 1 :])
                raise

    def resume_writing_files(
        self,
        tokenizer: Tokenizer,
        overlap_perc: float,
    ):
        current = read_file_list(self.current_file_dump_path, from0=True)
        remaining = read_file_list(self.remaining_files_dump_path, from0=True)
        self.delete_files(current)
        self.write_files(current + remaining, tokenizer, overlap_perc)
        self.current_file_dump_path.unlink(missing_ok=True)
        self.remaining_files_dump_path.unlink(missing_ok=True)

    def sync_indexed_files(
        self,
        tokenizer: Tokenizer,
        overlap_perc: float,
    ):
        files = self.indexed_files_with_metadata()
        todelete = []
        tosync = []
        for f in files:
            if not f["file"].exists():
                todelete.append(f["file"])
            elif f["file"].stat().st_mtime > f["mtime"]:
                tosync.append(f["file"])
        self.delete_files(todelete + tosync)
        self.write_files(tosync, tokenizer, overlap_perc)

    def delete_files(self, filepaths: list[Path]):
        if len(filepaths) == 1:
            self.collection.delete(where={"file": str(filepaths[0])})
        elif len(filepaths) > 1:
            self.collection.delete(where={"$or": [{"file": str(p)} for p in filepaths]})
        for filepath in filepaths:
            logger.info(f"deleted file {filepath} from {self.dbpath}")

    def query(self, query: str, k: int = 5):
        assert k > 0
        results = self.collection.query(
            query_embeddings=self.model.embed([query], truncate=False),
            n_results=k,
        )
        texts = results["documents"][0]
        files = [x["file"] for x in results["metadatas"][0]]
        return [
            {"text": text, "file": filepath} for text, filepath in zip(texts, files)
        ]

    @staticmethod
    def pretty_report_query_results(queryres):
        res = []
        for item in queryres:
            if isinstance(item, str):
                res.append(item)
            else:
                res.append(f"File: {item['file']}")
                res.append("")
                res.append(item["text"])
                res.append("")
                res.append("-" * 10)
                res.append("")
        return "\n".join(res)


def run_dbgen(args, model, tokenizer: Tokenizer):
    mode = "x"
    if args.append or args.resume or args.sync_indexed:
        mode = "r"
    with DB(args.db, model, mode=mode) as db:
        if args.resume:
            db.resume_writing_files(tokenizer, args.overlap_perc)
        elif args.sync_indexed:
            db.sync_indexed_files(tokenizer, args.overlap_perc)
        else:
            if args.files:
                paths = args.files
            elif args.files_from:
                paths = read_file_list(args.files_from, from0=args.from0)
            elif args.glob:
                paths = [Path(p) for p in glob.iglob(args.glob, recursive=True)]
            else:
                assert False, "Unreachable"
            if args.append:
                db.delete_files(paths)
            db.write_files(paths, tokenizer, args.overlap_perc)


def run_query(args, model):
    with DB(args.db, model) as db:
        res = db.query(args.query, args.k)
        if args.files_only:
            res = [x["file"] for x in res]
        if args.json:
            print(json.dumps(res, indent=4))
        else:
            print(DB.pretty_report_query_results(res))


def run_delete(args, model):
    with DB(args.db, model) as db:
        db.delete_files(args.files)


def run_mcp(args, model, server):
    description = args.description
    if args.description_file is not None:
        description = args.description_file.read_text()
    if description is None:
        raise ValueError(
            "either --description or --description-file is required when launching MCP server"
        )

    query_db_description = f"""
Search this vector database when the user needs information from its documents.

Database contents:
{description}

Args:
    query: Natural language search string.
    k: Number of relevant chunks to return.
    files_only: Return only files containing matching chunks.
Returns:
    Text report of matching chunks with file paths, or one matching file path per line.
"""

    @server.tool(description=query_db_description)
    async def mcp_query_db(query: str, k: int, files_only: bool = False) -> str:
        with DB(args.db, model) as db:
            queryres = db.query(query, k)
            if files_only:
                queryres = [x["file"] for x in queryres]
            return DB.pretty_report_query_results(queryres)

    server.run(transport=args.transport)


def main():

    def cmd_dbgen(args):
        logger.info(
            f"using llama config n_batch={args.n_batch}, n_ubatch={args.n_ubatch}"
        )
        model_path = args.model_path
        if model_path is None:
            if not args.append and not args.resume and not args.sync_indexed:
                raise ValueError(
                    "--model-path is required when creating a new database"
                )
            model_path = DB.get_model_path(args.db)
        model = make_embedding_model(
            model_path,
            n_batch=args.n_batch,
            n_ubatch=args.n_ubatch,
        )
        run_dbgen(args, model, LlamaTokenizer(model))

    def cmd_query(args):
        logger.info(
            f"using llama config n_batch={args.n_batch}, n_ubatch={args.n_ubatch}"
        )
        model = make_embedding_model(
            DB.get_model_path(args.db),
            n_batch=args.n_batch,
            n_ubatch=args.n_ubatch,
        )
        run_query(args, model)

    def cmd_delete(args):
        logger.info(
            f"using llama config n_batch={args.n_batch}, n_ubatch={args.n_ubatch}"
        )
        model = make_embedding_model(
            DB.get_model_path(args.db),
            n_batch=args.n_batch,
            n_ubatch=args.n_ubatch,
        )
        run_delete(args, model)

    def cmd_mcp(args):
        logger.info(
            f"using llama config n_batch={args.n_batch}, n_ubatch={args.n_ubatch}"
        )
        model = make_embedding_model(
            DB.get_model_path(args.db),
            n_batch=args.n_batch,
            n_ubatch=args.n_ubatch,
        )
        server = FastMCP(
            "simple-rag-vectordb",
            host=args.host,
            port=args.port,
        )
        run_mcp(args, model, server)

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_parser(name):
        return subparsers.add_parser(
            name,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

    def add_common_args(subparser):
        subparser.add_argument(
            "--db",
            type=Path,
            required=True,
            help="Path of the ChromaDB database",
        )
        subparser.add_argument(
            "--n-batch",
            type=int,
            required=False,
            default=512,
            help="n_batch value for llama.cpp",
        )
        subparser.add_argument(
            "--n-ubatch",
            type=int,
            required=False,
            default=None,
            help="n_ubatch value for llama.cpp. Defaults to n_batch",
        )

    dbgen = add_parser("dbgen")
    add_common_args(dbgen)
    dbgen.add_argument(
        "--model-path",
        type=Path,
        required=False,
        help="Embedding model file to use. Required unless --append reads it from the DB metadata",
    )
    dbgen_files = dbgen.add_mutually_exclusive_group(required=True)
    dbgen_files.add_argument(
        "--files",
        type=Path,
        nargs="+",
        help="Files to scan",
    )
    dbgen_files.add_argument(
        "--files-from",
        type=Path,
        help="File containing paths to scan, one per line",
    )
    dbgen_files.add_argument(
        "--glob",
        type=str,
        help="Glob for the files to scan (e.g. ./docs/**/*.txt)",
    )
    dbgen_files.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume an interrupted dbgen run",
    )
    dbgen_files.add_argument(
        "--sync-indexed",
        action="store_true",
        default=False,
        help="Sync indexed files: delete missing files and re-index changed files",
    )
    dbgen.add_argument(
        "--from0",
        "-0",
        action="store_true",
        default=False,
        help="Read --files-from entries separated by NUL instead of newline",
    )
    dbgen.add_argument(
        "--overlap-perc",
        type=float,
        default=0.2,
        help="Overlapping percentage (from min of 0 to max of 1 exclusive)",
    )
    dbgen.add_argument(
        "--append",
        action="store_true",
        default=False,
        help="Add documents to an existing database instead of requiring a new path",
    )
    dbgen.set_defaults(func=cmd_dbgen)

    query = add_parser("query")
    add_common_args(query)
    query.add_argument(
        "--query", type=str, required=True, help="Query to run against the database"
    )
    query.add_argument("--k", type=int, required=False, default=5, help="Top K matches")
    query.add_argument(
        "--files-only",
        action="store_true",
        default=False,
        help="Output file names only",
    )
    query.add_argument(
        "--json", action="store_true", default=False, help="Output in JSON format"
    )
    query.set_defaults(func=cmd_query)

    delete = add_parser("delete")
    add_common_args(delete)
    delete.add_argument(
        "--files",
        type=Path,
        nargs="+",
        required=True,
        help="Files whose chunks should be deleted from the database",
    )
    delete.set_defaults(func=cmd_delete)

    mcpcmd = add_parser("mcp")
    add_common_args(mcpcmd)
    mcpcmd.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host interface for the MCP HTTP server",
    )
    mcpcmd.add_argument(
        "--port",
        type=int,
        default=9182,
        help="Port for the MCP HTTP server",
    )
    mcpcmd.add_argument(
        "--transport",
        choices=["streamable-http", "stdio", "sse"],
        default="streamable-http",
        help="MCP transport protocol",
    )
    description_group = mcpcmd.add_mutually_exclusive_group(required=True)
    description_group.add_argument(
        "--description",
        type=str,
        default=None,
        help="Description of what this vector database contains",
    )
    description_group.add_argument(
        "--description-file",
        type=Path,
        default=None,
        help="File containing a description of what this vector database contains",
    )
    mcpcmd.set_defaults(func=cmd_mcp)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
