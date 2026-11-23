"""Command line interface for JFK Reader."""
import argparse
import os
import sys
from pathlib import Path
from .core import build_index, answer
from .download import collect


def main(argv=None):
    parser = argparse.ArgumentParser(prog="jfk-reader", description="Explore JFK archival PDFs with cited AI answers")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("download", help="Download PDFs linked from the 2025 release listing")
    p.add_argument("--output", type=Path, default=Path("data/pdfs"))
    p.add_argument("--max-pages", type=int, default=3)
    p.add_argument("--delay", type=float, default=1.0)
    p = sub.add_parser("index", help="Extract PDF text and create a local vector index")
    p.add_argument("--pdf-dir", type=Path, default=Path("data/pdfs"))
    p.add_argument("--index-dir", type=Path, default=Path("data/index"))
    p.add_argument("--ocr", action="store_true", help="OCR pages with little extractable text (requires Tesseract)")
    p = sub.add_parser("ask", help="Ask a question using indexed documents")
    p.add_argument("question", nargs="?", help="Omit for interactive mode")
    p.add_argument("--index-dir", type=Path, default=Path("data/index"))
    p.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args(argv)
    try:
        if args.command == "download":
            print(f"Downloaded {collect(args.output, args.max_pages, args.delay)} new PDFs")
            return 0
        from dotenv import load_dotenv
        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            parser.error("OPENAI_API_KEY is not set. See .env.example")
        from openai import OpenAI
        client = OpenAI()
        if args.command == "index":
            count = build_index(args.pdf_dir, args.index_dir, client, use_ocr=args.ocr)
            print(f"Indexed {count} chunks in {args.index_dir}")
            return 0
        def ask_once(question):
            result, sources = answer(question, args.index_dir, client, top_k=args.top_k)
            print(f"\n{result}\n\nRetrieved sources:")
            for i, source in enumerate(sources, 1):
                print(f"  [S{i}] {source['file']} — page {source['page']} (similarity {source['score']:.3f})")
            print()
        if args.question:
            ask_once(args.question)
        else:
            print("JFK Reader interactive mode. Type 'exit' to quit.")
            while True:
                try:
                    question = input("JFK Reader > ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if question.lower() in {"exit", "quit", ":q"}:
                    break
                if question:
                    try:
                        ask_once(question)
                    except Exception as exc:
                        print(f"Question failed: {exc}", file=sys.stderr)
        return 0
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
