from __future__ import annotations

import argparse
import logging
import sys

from rag_pipeline.config import get_settings
from rag_pipeline.logging_config import configure_logging
from rag_pipeline.pipeline import RagPipeline

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Small local-first RAG pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("ingest", help="Load documents and build the vector index")

    ask_parser = subparsers.add_parser("ask", help="Ask one question")
    ask_parser.add_argument("query", help="Question to answer")

    chat_parser = subparsers.add_parser("chat", help="Start an interactive QA session")
    chat_parser.add_argument(
        "--exit-command",
        default="exit",
        help="Command that ends the interactive session",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)

    try:
        pipeline = RagPipeline(settings)

        if args.command == "ingest":
            indexed = pipeline.ingest()
            print(f"Indexed {indexed} chunks into {settings.index_dir}")
            return 0

        if args.command == "ask":
            result = pipeline.answer(args.query)
            print_answer(result.answer, result.sources)
            return 0

        if args.command == "chat":
            return run_chat(pipeline, args.exit_command)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as exc:
        logger.exception("Application error: %s", exc)
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    parser.print_help()
    return 1


def run_chat(pipeline: RagPipeline, exit_command: str) -> int:
    print(f"Interactive mode. Type '{exit_command}' to quit.")
    while True:
        query = input("\nQuestion> ").strip()
        if not query:
            continue
        if query.lower() == exit_command.lower():
            return 0

        result = pipeline.answer(query)
        print_answer(result.answer, result.sources)


def print_answer(answer: str, sources: list[object]) -> None:
    print("\nAnswer:")
    print(answer)
    print("\nSources:")
    if not sources:
        print("- No sources returned")
        return

    for item in sources:
        filename = item.chunk.metadata.get("filename", item.chunk.source_path)
        chunk_index = item.chunk.metadata.get("chunk_index", "?")
        print(f"- {filename} | chunk {chunk_index} | score={item.score:.4f}")


if __name__ == "__main__":
    raise SystemExit(main())

