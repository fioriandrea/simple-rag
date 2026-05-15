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
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

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

    def tokenize(self, text):
        return self.model.tokenize(bytes(text, "utf-8", errors="ignore"))

    def detokenize(self, tokens):
        return str(self.model.detokenize(tokens), "utf-8", errors="ignore")


@dataclass(frozen=True)
class Doc:
    splits: list[str]
    filepath: str


def split_tokens(tokens, context_window, overlap_perc):
    assert context_window > 0
    assert 0 <= overlap_perc < 1
    overlap = int(context_window * overlap_perc)
    i = 0
    while i < len(tokens):
        yield tokens[i : i + context_window]
        i += context_window - overlap


def files_to_docs(filepaths, tokenizer, context_window, overlap_perc):
    res = []
    for filepath in filepaths:
        with open(filepath, "r", errors="ignore") as f:
            text = f.read()
        tokens = tokenizer.tokenize(text)
        res.append(
            Doc(
                splits=[
                    tokenizer.detokenize(s)
                    for s in split_tokens(tokens, context_window, overlap_perc)
                ],
                filepath=filepath,
            )
        )
    return res


def make_embedding_model(model_path, n_batch, n_ubatch=None):
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
    def __init__(self, dbpath, model, exists_ok=True):
        self.dbpath = Path(dbpath)
        if not exists_ok and self.dbpath.exists():
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
    def make_db_client(dbpath):
        return chromadb.PersistentClient(
            path=dbpath,
            settings=chromadb.config.Settings(anonymized_telemetry=False),
        )

    @staticmethod
    def get_model_path(dbpath):
        client = DB.make_db_client(dbpath)
        try:
            collection = client.get_collection(name="docs")
            model_path = (collection.metadata or {}).get("model_path")
            if not model_path:
                raise ValueError(f"database {dbpath} has no model_path metadata")
            return Path(model_path)
        finally:
            client.close()

    def write_docs(self, docs):
        n = len(docs)
        count = self.collection.count()
        for i, doc in enumerate(docs):
            splits = doc.splits
            filepath = doc.filepath
            logger.info(f"writing file [{i + 1} / {n}] " f"{filepath} to {self.dbpath}")
            if not splits:
                logger.info(
                    f"database {self.dbpath} unchanged " f"(no records from {filepath})"
                )
                continue
            self.collection.add(
                ids=[str(k) for k in range(count, count + len(splits))],
                # truncate=True because splitter may off by one token length.
                # This is because sometimes:
                # len(tokenize(detokenize(tokenize(t)))) == len(tokenize(t)) + 1
                embeddings=self.model.embed(splits, truncate=True),
                documents=splits,
                metadatas=[{"file": filepath}] * len(splits),
            )
            count += len(splits)
            logger.info(
                f"database {self.dbpath} updated "
                f"(now with {count} records from {filepath})"
            )

    def delete_files(self, filepaths):
        for filepath in filepaths:
            logger.info(f"deleting file {filepath} from {self.dbpath}")
            self.collection.delete(where={"file": str(filepath)})

    def query(self, query, k=5):
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


def main():

    def cmd_dbgen(args):
        logger.info(
            f"using llama config n_batch={args.n_batch}, n_ubatch={args.n_ubatch}"
        )
        model_path = args.model_path
        if model_path is None:
            if not args.append:
                raise ValueError(
                    "--model-path is required when creating a new database"
                )
            model_path = DB.get_model_path(args.db)
        model = make_embedding_model(
            model_path,
            n_batch=args.n_batch,
            n_ubatch=args.n_ubatch,
        )
        with DB(args.db, model, exists_ok=args.append) as db:
            tokenizer = LlamaTokenizer(model)
            filepaths = glob.glob(args.glob, recursive=True)
            docs = files_to_docs(
                filepaths=filepaths,
                tokenizer=tokenizer,
                context_window=model.n_batch,
                overlap_perc=args.overlap_perc,
            )
            db.write_docs(docs)

    def cmd_query(args):
        logger.info(
            f"using llama config n_batch={args.n_batch}, n_ubatch={args.n_ubatch}"
        )
        model = make_embedding_model(
            DB.get_model_path(args.db),
            n_batch=args.n_batch,
            n_ubatch=args.n_ubatch,
        )
        with DB(args.db, model) as db:
            res = db.query(args.query, args.k)
            if args.files_only:
                res = [x["file"] for x in res]
            if args.json:
                print(json.dumps(res, indent=4))
            else:
                print(DB.pretty_report_query_results(res))

    def cmd_delete(args):
        logger.info(
            f"using llama config n_batch={args.n_batch}, n_ubatch={args.n_ubatch}"
        )
        model = make_embedding_model(
            DB.get_model_path(args.db),
            n_batch=args.n_batch,
            n_ubatch=args.n_ubatch,
        )
        with DB(args.db, model) as db:
            db.delete_files(args.files)

    def cmd_mcp(args):
        logger.info(
            f"using llama config n_batch={args.n_batch}, n_ubatch={args.n_ubatch}"
        )
        description = args.description
        if args.description_file is not None:
            description = args.description_file.read_text()
        if description is None:
            raise ValueError(
                "either --description or --description-file is required when launching MCP server"
            )

        model = make_embedding_model(
            DB.get_model_path(args.db),
            n_batch=args.n_batch,
            n_ubatch=args.n_ubatch,
        )
        mcp = FastMCP(
            "simple-rag-vectordb",
            host=args.host,
            port=args.port,
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

        @mcp.tool(description=query_db_description)
        async def mcp_query_db(query: str, k: int, files_only: bool = False) -> str:
            with DB(args.db, model) as db:
                queryres = db.query(query, k)
                if files_only:
                    queryres = [x["file"] for x in queryres]
                return DB.pretty_report_query_results(queryres)

        mcp.run(transport="streamable-http")

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
    dbgen.add_argument(
        "--glob",
        type=str,
        required=True,
        help="Glob for the files to scan (e.g. ./docs/**/*.txt)",
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
