"""Run the standalone local Hash API service."""

from __future__ import annotations

import argparse

import uvicorn

from server.hash_api.app import create_app
from server.hash_api.client import HashCliClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument("--binary", default="hashapi-cli")
    parser.add_argument("--timeout", default=60.0, type=float)
    parser.add_argument("--max-concurrency", default=1, type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    client = HashCliClient(
        binary=args.binary,
        timeout_seconds=args.timeout,
        max_concurrency=args.max_concurrency,
    )
    uvicorn.run(create_app(client), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
