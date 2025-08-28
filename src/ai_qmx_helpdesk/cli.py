"""
cli.py - Command-line interface for RAG database operations
2025-08-26

This module provides a comprehensive CLI for RAG database operations including
initialization, document ingestion, embedding generation, and querying.
Uses TOML configuration with precedence and supports multiple embedding providers.

Example usage:
    python -m ai_qmx_helpdesk.cli ragdb init --db rag.db
    python -m ai_qmx_helpdesk.cli ragdb ingest --db rag.db --dir ./data --glob "**/*.txt"
    python -m ai_qmx_helpdesk.cli ragdb embed --db rag.db --provider toy
    python -m ai_qmx_helpdesk.cli ragdb query --db rag.db --q "What is this?" --k 5
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import tomllib
except ImportError:
    # Python < 3.11 fallback
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        tomllib = None  # type: ignore[assignment]

from . import rag_db


# STEP_1: Setup logging and constants
_logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "data_dir": "~/sw/qmx_helpdesk/data/",
    "db": {"path": "rag.db", "backend": "faiss"},
    "split": {"chunk_size": 1000, "chunk_overlap": 200},
    "embed": {"provider": "toy", "model": "default", "batch_size": 64},
    "retriever": {"k": 5},
}


class RagCLI:
    """RAG database command-line interface with TOML configuration support."""

    def __init__(self, cfg_dict: Optional[Dict[str, Any]] = None) -> None:
        """Initialize CLI with configuration management."""
        # STEP_2: Initialize logging
        logging.basicConfig(
            level=logging.WARNING,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self._logger = logging.getLogger(self.__class__.__name__)

        # STEP_3: Load configuration with precedence
        self.cfg = self._load_config(cfg_dict or {})

    def _load_config(self, override_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Load configuration with precedence: defaults → files → overrides."""
        cfg = DEFAULT_CONFIG.copy()

        # Load from ./ragdb.toml
        local_config_path = Path("./ragdb.toml")
        if local_config_path.exists():
            cfg.update(self._load_toml_file(local_config_path))

        # Load from ~/.qmx_helpdesk/ragdb.toml
        home_config_path = Path.home() / ".qmx_helpdesk" / "ragdb.toml"
        if home_config_path.exists():
            cfg.update(self._load_toml_file(home_config_path))

        # Apply overrides
        cfg.update(override_cfg)

        # Expand tilde in paths
        if "data_dir" in cfg:
            cfg["data_dir"] = str(Path(str(cfg["data_dir"])).expanduser())
        if "db" in cfg and isinstance(cfg["db"], dict) and "path" in cfg["db"]:
            cfg["db"]["path"] = str(Path(str(cfg["db"]["path"])).expanduser())

        return cfg

    def _load_toml_file(self, path: Path) -> Dict[str, Any]:
        """Load TOML configuration file."""
        if tomllib is None:
            self._logger.warning("TOML support not available, skipping %s", path)
            return {}

        try:
            with open(path, "rb") as f:
                return tomllib.load(f)
        except Exception as e:
            self._logger.warning("Failed to load TOML config %s: %s", path, e)
            return {}

    def _expand_env_vars(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Expand environment variables in configuration."""
        # Check for API keys in environment
        if "OPENAI_API_KEY" in os.environ:
            if "openai" not in cfg:
                cfg["openai"] = {}
            cfg["openai"]["api_key"] = os.environ["OPENAI_API_KEY"]

        if "ANTHROPIC_API_KEY" in os.environ:
            if "anthropic" not in cfg:
                cfg["anthropic"] = {}
            cfg["anthropic"]["api_key"] = os.environ["ANTHROPIC_API_KEY"]

        if "HUGGINGFACEHUB_API_TOKEN" in os.environ:
            if "huggingface" not in cfg:
                cfg["huggingface"] = {}
            cfg["huggingface"]["api_token"] = os.environ["HUGGINGFACEHUB_API_TOKEN"]

        return cfg

    def cmd_init(self, args: argparse.Namespace) -> int:
        """Initialize RAG database."""
        db_path = args.db or self.cfg["db"]["path"]
        print(f"[STEP_1] Initializing database: {db_path}")

        try:
            rag_db.init_db(db_path)
            print("[STEP_2] Database initialized successfully")
            return 0
        except Exception as e:
            print(f"Error: Failed to initialize database: {e}", file=sys.stderr)
            return 1

    def cmd_ingest(self, args: argparse.Namespace) -> int:
        """Ingest documents into RAG database."""
        db_path = args.db or self.cfg["db"]["path"]
        # Validate required arguments - don't use invalid config defaults
        if not args.dir:
            print("❌ Error: --dir is required", file=sys.stderr)
            print("💡 Usage: qmx ragdb ingest --db mydocs.db --dir ./documents", file=sys.stderr)
            return 1

        data_dir = args.dir
        if not Path(data_dir).exists():
            print(f"❌ Error: Directory not found: {data_dir}", file=sys.stderr)
            return 1
        glob_pattern = args.glob or "**/*.{txt,pdf}"

        print(f"[STEP_1] Ingesting documents from: {data_dir}")
        print(f"[STEP_2] Using pattern: {glob_pattern}")

        try:
            # Check if database exists
            if not Path(db_path).exists():
                print(
                    f"Error: Database {db_path} does not exist. Run 'init' first.", file=sys.stderr
                )
                return 1

            # Ingest documents
            rag_db.ingest_path(db_path, data_dir, glob_pattern)
            print("[STEP_3] Document ingestion completed")
            return 0
        except Exception as e:
            print(f"Error: Failed to ingest documents: {e}", file=sys.stderr)
            return 1

    def cmd_embed(self, args: argparse.Namespace) -> int:
        """Generate embeddings for ingested documents."""
        db_path = args.db or self.cfg["db"]["path"]
        provider = args.provider or self.cfg["embed"]["provider"]
        model = args.model or self.cfg["embed"].get("model", "default")
        batch_size = args.batch_size or self.cfg["embed"]["batch_size"]
        backend = args.backend or self.cfg["db"]["backend"]

        # Check for existing embeddings and warn about overwrite
        try:
            import sqlite3

            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT provider, model, total_vectors FROM embeddings_info ORDER BY created_at DESC LIMIT 1"
                )
                existing = cursor.fetchone()
                if existing:
                    print("⚠️  Warning: Embeddings already exist!")
                    print(
                        f"   Existing: {existing['provider']} ({existing['model']}) - {existing['total_vectors']} vectors"
                    )
                    print(f"   New: {provider} ({model})")
                    if existing["provider"] != provider or existing["model"] != model:
                        print(
                            "   🔄 This will REPLACE existing embeddings with different provider/model"
                        )
                    else:
                        print("   🔄 This will REBUILD embeddings with same provider/model")
        except Exception:
            pass

        print(f"[STEP_1] Generating embeddings with provider: {provider}")
        print(f"[STEP_2] Using model: {model}, batch size: {batch_size}")

        try:
            # Check if database exists
            if not Path(db_path).exists():
                print(
                    f"Error: Database {db_path} does not exist. Run 'init' first.", file=sys.stderr
                )
                return 1

            # Build embeddings
            embed_cfg = {"provider": provider, "model": model, "batch_size": batch_size}
            db_cfg = {"backend": backend}
            full_cfg = {"embed": embed_cfg, "db": db_cfg}
            rag_db.build_embeddings(db_path, full_cfg)
            print("[STEP_3] Embedding generation completed")
            return 0
        except Exception as e:
            print(f"Error: Failed to generate embeddings: {e}", file=sys.stderr)
            return 1

    def cmd_query(self, args: argparse.Namespace) -> int:
        """Query the RAG database."""
        db_path = args.db or self.cfg["db"]["path"]
        query = args.q
        k = args.k or self.cfg["retriever"]["k"]

        # Get stored provider from database and validate
        stored_provider = None
        try:
            import sqlite3

            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT provider, model FROM embeddings_info ORDER BY created_at DESC LIMIT 1"
                )
                stored_info = cursor.fetchone()
                if stored_info:
                    stored_provider = stored_info["provider"]
                    stored_model = stored_info["model"]
        except Exception:
            pass

        # Validate provider compatibility
        if args.provider:
            if stored_provider and args.provider != stored_provider:
                print("❌ Error: Provider mismatch!", file=sys.stderr)
                print(
                    f"   Database embedded with: {stored_provider} ({stored_model})",
                    file=sys.stderr,
                )
                print(f"   You specified: {args.provider}", file=sys.stderr)
                print("💡 Either:", file=sys.stderr)
                print(
                    f"   • Remove --provider to use stored provider: {stored_provider}",
                    file=sys.stderr,
                )
                print(
                    f"   • Re-embed with: qmx ragdb embed --db {db_path} --provider {args.provider}",
                    file=sys.stderr,
                )
                return 1
            provider = args.provider
        else:
            if stored_provider:
                provider = stored_provider
                print(f"🔧 Using stored provider: {stored_provider} ({stored_model})")
            else:
                print("❌ Error: No embeddings found in database!", file=sys.stderr)
                print(
                    f"💡 First run: qmx ragdb embed --db {db_path} --provider <provider>",
                    file=sys.stderr,
                )
                return 1

        if not query:
            print("Error: Query text is required", file=sys.stderr)
            return 1

        if not args.json:
            print(f"[STEP_1] Searching for: {query}")

        try:
            # Check if database exists
            if not Path(db_path).exists():
                print(
                    f"Error: Database {db_path} does not exist. Run 'init' first.", file=sys.stderr
                )
                return 1

            # Search documents with embedding provider config
            search_cfg = {"embed": {"provider": provider}}
            results = rag_db.search(db_path, query, k=k, cfg=search_cfg)

            if args.json:
                # JSON output
                output = {
                    "query": query,
                    "k": k,
                    "results": [
                        {
                            "rank": i + 1,
                            "score": float(score),
                            "path": "unknown",  # Search function doesn't return path info
                            "content": text[:200] + "..." if len(text) > 200 else text,
                        }
                        for i, (text, score) in enumerate(results)
                    ],
                }
                if args.show_chunks:
                    for i, (text, score) in enumerate(results):
                        output["results"][i]["full_content"] = text

                print(json.dumps(output, indent=2))
            else:
                # Table output
                print(f"[STEP_2] Found {len(results)} results:")
                print()
                print(f"{'Rank':<4} {'Score':<8} {'Path':<50}")
                print("-" * 70)

                for i, (text, score) in enumerate(results):
                    rank = i + 1
                    path = "text_chunk"  # Search function doesn't return path info
                    print(f"{rank:<4} {score:<8.4f} {path:<50}")

                if args.show_paths or args.show_chunks:
                    print("\nDetails:")
                    for i, (text, score) in enumerate(results):
                        print(f"\n--- Result {i + 1} ---")
                        if args.show_paths:
                            print("Path: text_chunk")
                        if args.show_chunks:
                            print(f"Content: {text}")

            return 0
        except Exception as e:
            print(f"Error: Failed to query database: {e}", file=sys.stderr)
            return 1

    def cmd_inspect(self, args: argparse.Namespace) -> int:
        """Inspect database status or query results."""
        # If no query provided, show database status
        if not args.q:
            return self.cmd_sanity_check(args)

        # If sanity flag is used, run sanity check
        if args.sanity:
            return self.cmd_sanity_check(args)

        # Set detailed output flags for query inspection
        args.show_paths = True
        args.show_chunks = True
        return self.cmd_query(args)

    def cmd_sanity_check(self, args: argparse.Namespace) -> int:
        """Run comprehensive database sanity check."""
        import sqlite3
        from pathlib import Path

        db_path = args.db or "rag.db"
        db_file = Path(db_path)

        if not db_file.exists():
            print(f"Error: Database {db_path} does not exist. Run 'init' first.", file=sys.stderr)
            return 1

        print(f"🔬 Database Sanity Check: {db_path}")
        print("=" * 60)

        try:
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Check documents table
                cursor.execute("SELECT COUNT(*) as count FROM documents")
                doc_count = cursor.fetchone()["count"]
                print(f"📄 Documents: {doc_count}")

                if doc_count > 0:
                    cursor.execute("""
                        SELECT path, mtime, bytes, 
                               datetime(mtime, 'unixepoch') as modified_date
                        FROM documents 
                        ORDER BY mtime DESC 
                        LIMIT 5
                    """)
                    print("\n   Recent documents:")
                    for row in cursor.fetchall():
                        size_kb = row["bytes"] / 1024
                        print(
                            f"   • {Path(row['path']).name} ({size_kb:.1f}KB, {row['modified_date']})"
                        )

                # Check chunks table
                cursor.execute("SELECT COUNT(*) as count FROM chunks")
                chunk_count = cursor.fetchone()["count"]
                print(f"\n📝 Chunks: {chunk_count}")

                if chunk_count > 0:
                    cursor.execute("""
                        SELECT doc_id, COUNT(*) as chunk_count, 
                               AVG(LENGTH(text)) as avg_length
                        FROM chunks 
                        GROUP BY doc_id
                        ORDER BY chunk_count DESC
                        LIMIT 5
                    """)
                    print("\n   Chunking breakdown:")
                    for row in cursor.fetchall():
                        print(
                            f"   • Doc {row['doc_id']}: {row['chunk_count']} chunks (avg {row['avg_length']:.0f} chars)"
                        )

                    # Show chunk length distribution
                    cursor.execute("""
                        SELECT 
                            CASE 
                                WHEN LENGTH(text) < 500 THEN 'Small (<500)'
                                WHEN LENGTH(text) < 1000 THEN 'Medium (500-1000)'
                                WHEN LENGTH(text) < 2000 THEN 'Large (1000-2000)'
                                ELSE 'XLarge (>2000)'
                            END as size_category,
                            COUNT(*) as count
                        FROM chunks
                        GROUP BY size_category
                        ORDER BY count DESC
                    """)
                    print("\n   Chunk size distribution:")
                    for row in cursor.fetchall():
                        print(f"   • {row['size_category']}: {row['count']} chunks")

                # Check if embeddings exist
                db_stem = Path(db_path).stem
                index_dir = Path(db_path).parent / f"{db_stem}_index"

                # Check embedding provider info
                cursor.execute("SELECT * FROM embeddings_info ORDER BY created_at DESC LIMIT 1")
                embed_info = cursor.fetchone()

                print("\n🧮 Vector Embeddings:")
                if embed_info:
                    print(f"   📊 Provider: {embed_info['provider']} ({embed_info['model']})")
                    if embed_info["dimensions"]:
                        print(f"   📐 Dimensions: {embed_info['dimensions']}")
                    print(f"   🔢 Vectors: {embed_info['total_vectors']}")
                    created = embed_info["created_at"].split(".")[0]  # Remove microseconds
                    print(f"   📅 Created: {created}")

                    # Check if index files exist
                    if index_dir.exists():
                        faiss_file = index_dir / "index.faiss"
                        pkl_file = index_dir / "index.pkl"
                        if faiss_file.exists() and pkl_file.exists():
                            faiss_size = faiss_file.stat().st_size / 1024
                            print(f"   ✅ Index files: {faiss_size:.1f}KB at {index_dir}")
                        else:
                            print(f"   ⚠️  Index directory exists but files missing: {index_dir}")
                    else:
                        print(f"   ❌ Index directory missing: {index_dir}")
                else:
                    print("   ❌ No embeddings found. Run 'embed' command first.")
                    if index_dir.exists():
                        print(f"   ⚠️  Old index directory exists: {index_dir}")

        except Exception as e:
            print(f"Error during sanity check: {e}", file=sys.stderr)
            return 1

        print("\n✅ Sanity check completed!")
        return 0

    def cmd_tune(self, args: argparse.Namespace) -> int:
        """Tune RAG parameters (stub implementation)."""
        print("[STEP_1] RAG parameter tuning (feature stub)")
        print("This feature will integrate scripts/rag_tune.py when available")
        return 0


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for RAG CLI."""
    parser = argparse.ArgumentParser(
        prog="qmx-helpdesk",
        description="🤖 QMX Helpdesk RAG Database CLI\n\nIngest documents, generate embeddings, and query with semantic search.",
        epilog="Complete Workflow:\n"
        "  # 1. Initialize database\n"
        "  qmx ragdb init --db my_docs.db\n\n"
        "  # 2. Ingest documents\n"
        "  qmx ragdb ingest --db my_docs.db --dir ./data\n\n"
        "  # 3. Generate embeddings\n"
        "  qmx ragdb embed --db my_docs.db --provider toy\n\n"
        "  # 4. Query documents\n"
        "  qmx ragdb query --db my_docs.db --q 'question' --provider toy\n\n"
        "Command Help:\n"
        "  qmx ragdb -h              # All ragdb commands\n"
        "  qmx ragdb <command> -h    # Specific command help\n\n"
        "Development (uninstalled):\n"
        "  PYTHONPATH=src python -m ai_qmx_helpdesk ragdb <command>\n\n"
        "⚠️  IMPORTANT: All commands need 'ragdb' before the operation!\n"
        "   ❌ Wrong: qmx init\n"
        "   ✅ Right: qmx ragdb init",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Add debug flag before subparsers
    parser.add_argument(
        "--debug-exec", action="store_true", help="Show execution method and package location"
    )

    subparsers = parser.add_subparsers(
        dest="command", help="Available command groups", metavar="{ragdb}"
    )

    # ragdb subcommand group
    ragdb_parser = subparsers.add_parser(
        "ragdb",
        help="📚 RAG database operations (init, ingest, embed, query)",
        description="RAG (Retrieval-Augmented Generation) database operations\n\n"
        "Workflow: init → ingest → embed → query",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ragdb_subparsers = ragdb_parser.add_subparsers(
        dest="ragdb_command",
        help="RAG database commands",
        metavar="{init,ingest,embed,query,inspect,tune}",
    )

    # init command
    init_parser = ragdb_subparsers.add_parser(
        "init",
        help="🗄️  Initialize new RAG database",
        description="Create a new SQLite database and vector index directory for RAG operations.",
    )
    init_parser.add_argument("--db", help="Database path (default: rag.db)", metavar="PATH")

    # ingest command
    ingest_parser = ragdb_subparsers.add_parser(
        "ingest",
        help="📄 Ingest documents into database",
        description="Load and chunk text documents from files into the database.\n\n"
        "Supports: .txt, .pdf files (PDFs processed page-by-page)\n"
        "Recursively searches subdirectories with **/ patterns.\n\n"
        "Supported glob patterns:\n"
        "  *.txt                - Text files in current dir\n"
        "  *.pdf                - PDF files in current dir  \n"
        "  **/*.txt             - Text files in all subdirs\n"
        "  **/*.pdf             - PDF files in all subdirs\n"
        "  **/*.{txt,pdf}       - Both types, all subdirs (default)\n\n"
        "Examples:\n"
        "  ingest --dir ./docs --glob '*.pdf'        # PDFs only\n"
        "  ingest --dir ./data --glob '**/*.txt'     # All txt files",
    )
    ingest_parser.add_argument("--db", help="Database path", metavar="PATH")
    ingest_parser.add_argument("--dir", help="Directory to scan for documents", metavar="PATH")
    ingest_parser.add_argument(
        "--glob", help="File pattern (default: **/*.{txt,pdf})", metavar="PATTERN"
    )

    # embed command
    embed_parser = ragdb_subparsers.add_parser(
        "embed",
        help="🧮 Generate vector embeddings",
        description="Convert text chunks to vector embeddings for semantic search.\n\n"
        "Providers:\n"
        "  toy         - Fast offline embeddings (for testing)\n"
        "  huggingface - High quality open-source models\n"
        "  openai      - OpenAI's embedding models (requires API key)",
    )
    embed_parser.add_argument("--db", help="Database path", metavar="PATH")
    embed_parser.add_argument(
        "--provider",
        choices=["toy", "huggingface", "openai"],
        help="Embedding provider (default: toy)",
        metavar="PROVIDER",
    )
    embed_parser.add_argument("--model", help="Model name (provider-specific)", metavar="MODEL")
    embed_parser.add_argument(
        "--batch-size", type=int, help="Batch size (default: 64)", metavar="N"
    )
    embed_parser.add_argument(
        "--backend",
        choices=["faiss", "chroma"],
        help="Vector store (default: faiss)",
        metavar="BACKEND",
    )

    # query command
    query_parser = ragdb_subparsers.add_parser(
        "query",
        help="🔍 Search documents with natural language",
        description="Query the database using semantic search.\n\n"
        "Examples:\n"
        "  query --q 'civil war' --k 3\n"
        "  query --q 'government policy' --json --provider toy",
    )
    query_parser.add_argument("--db", help="Database path", metavar="PATH")
    query_parser.add_argument(
        "--q", required=True, help="Query text (natural language)", metavar="TEXT"
    )
    query_parser.add_argument("--k", type=int, help="Number of results (default: 5)", metavar="N")
    query_parser.add_argument(
        "--provider",
        choices=["toy", "huggingface", "openai"],
        help="Must match embedding provider",
        metavar="PROVIDER",
    )
    query_parser.add_argument("--json", action="store_true", help="Output as JSON instead of table")
    query_parser.add_argument("--show-paths", action="store_true", help="Show source file paths")
    query_parser.add_argument("--show-chunks", action="store_true", help="Show full text content")

    # inspect command
    inspect_parser = ragdb_subparsers.add_parser(
        "inspect",
        help="🔬 Inspect database status or query results",
        description="Show database status (default) or run detailed query inspection with full content and metadata.",
    )
    inspect_parser.add_argument("--db", help="Database path", metavar="PATH")
    inspect_parser.add_argument(
        "--q", help="Query text (optional - shows database status if omitted)", metavar="TEXT"
    )
    inspect_parser.add_argument("--k", type=int, help="Number of results (default: 5)", metavar="N")
    inspect_parser.add_argument("--json", action="store_true", help="Output as JSON")
    inspect_parser.add_argument(
        "--sanity",
        action="store_true",
        help="Run database sanity check (shows docs, chunks, embeddings)",
    )

    # tune command
    tune_parser = ragdb_subparsers.add_parser(
        "tune",
        help="⚙️  Tune RAG parameters (coming soon)",
        description="Optimize chunking, embedding, and retrieval parameters.\n\n"
        "This feature will integrate with parameter tuning scripts.",
    )
    tune_parser.add_argument("--db", help="Database path", metavar="PATH")

    return parser


def main() -> int:
    """Main CLI entry point."""
    import sys
    import os

    parser = create_parser()

    # Check for common mistakes and provide helpful errors BEFORE parsing
    if len(sys.argv) > 1 and sys.argv[1] not in ["--debug-exec", "-h", "--help"]:
        common_commands = ["init", "ingest", "embed", "query", "inspect", "tune"]
        if sys.argv[1] in common_commands:
            print(f"❌ Error: '{sys.argv[1]}' is not a top-level command.", file=sys.stderr)
            print(f"💡 Did you mean: qmx ragdb {sys.argv[1]} ?", file=sys.stderr)
            print(f"📖 For help: qmx ragdb {sys.argv[1]} -h", file=sys.stderr)
            return 1

    args = parser.parse_args()

    # Handle debug flag
    if hasattr(args, "debug_exec") and args.debug_exec:
        import ai_qmx_helpdesk

        if sys.argv[0].endswith("qmx"):
            print("🔧 Execution: Installed CLI command (qmx)")
            print(f"📍 Script: {sys.argv[0]}")
        elif "__main__.py" in sys.argv[0] or "-m" in sys.argv:
            print("🔧 Execution: Python module (python -m ai_qmx_helpdesk)")
            print(f"📍 PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
        else:
            print("🔧 Execution: Unknown method")
            print(f"📍 sys.argv[0]: {sys.argv[0]}")
        print(f"📦 Package: {ai_qmx_helpdesk.__file__}")
        print(f"🔖 Version: {ai_qmx_helpdesk.__version__}")
        return 0

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "ragdb":
        if not args.ragdb_command:
            parser.print_help()
            return 1

        # Initialize CLI
        cli = RagCLI()

        # Route to appropriate command
        command_map = {
            "init": cli.cmd_init,
            "ingest": cli.cmd_ingest,
            "embed": cli.cmd_embed,
            "query": cli.cmd_query,
            "inspect": cli.cmd_inspect,
            "tune": cli.cmd_tune,
        }

        if args.ragdb_command in command_map:
            return command_map[args.ragdb_command](args)
        else:
            print(f"Unknown ragdb command: {args.ragdb_command}", file=sys.stderr)
            return 1

    print(f"Unknown command: {args.command}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
