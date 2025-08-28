"""
test_cli_ragdb.py - Comprehensive tests for RAG database CLI
2025-08-26

This module provides comprehensive tests for the CLI functionality
including all subcommands, configuration loading, and output formatting.
Uses toy providers to avoid external dependencies and network calls.

Example usage:
    pytest tests/test_cli_ragdb.py -v
    pytest tests/test_cli_ragdb.py::TestRagCLI::test_init_command
"""

import json
import tempfile
from io import StringIO
from pathlib import Path
from typing import Any
from unittest.mock import patch, MagicMock

import pytest

from ai_qmx_helpdesk.cli import RagCLI, create_parser, main


class TestRagCLI:
    """Test suite for RAG CLI functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.db_path = self.temp_path / "test.db"
        self.data_dir = self.temp_path / "data"
        self.data_dir.mkdir(exist_ok=True)

        # Create test text file
        self.test_txt = self.data_dir / "test.txt"
        self.test_txt.write_text("This is a test document for RAG testing.")

    def teardown_method(self) -> None:
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def test_cli_initialization(self) -> None:
        """Test CLI initialization and configuration loading."""
        cli = RagCLI()
        assert cli.cfg is not None
        assert "data_dir" in cli.cfg
        assert "db" in cli.cfg
        assert "embed" in cli.cfg

    def test_config_override(self) -> None:
        """Test configuration override mechanism."""
        override_cfg = {"embed": {"provider": "test", "batch_size": 128}}
        cli = RagCLI(override_cfg)
        assert cli.cfg["embed"]["provider"] == "test"
        assert cli.cfg["embed"]["batch_size"] == 128

    @patch("ai_qmx_helpdesk.rag_db.init_db")
    def test_init_command_success(self, mock_init_db: Any) -> None:
        """Test successful database initialization."""
        cli = RagCLI()

        # Create mock args
        args = MagicMock()
        args.db = str(self.db_path)

        result = cli.cmd_init(args)

        assert result == 0
        mock_init_db.assert_called_once_with(str(self.db_path))

    @patch("ai_qmx_helpdesk.rag_db.init_db")
    def test_init_command_failure(self, mock_init_db: Any) -> None:
        """Test database initialization failure."""
        mock_init_db.side_effect = Exception("Database error")
        cli = RagCLI()

        args = MagicMock()
        args.db = str(self.db_path)

        result = cli.cmd_init(args)
        assert result == 1

    @patch("ai_qmx_helpdesk.rag_db.ingest_path")
    def test_ingest_command_success(self, mock_ingest: Any) -> None:
        """Test successful document ingestion."""
        # Create database file
        self.db_path.touch()

        cli = RagCLI()
        args = MagicMock()
        args.db = str(self.db_path)
        args.dir = str(self.data_dir)
        args.glob = "**/*.txt"

        result = cli.cmd_ingest(args)

        assert result == 0
        mock_ingest.assert_called_once_with(str(self.db_path), str(self.data_dir), "**/*.txt")

    def test_ingest_command_no_db(self) -> None:
        """Test ingestion with missing database."""
        cli = RagCLI()
        args = MagicMock()
        args.db = str(self.db_path)  # Database doesn't exist
        args.dir = str(self.data_dir)
        args.glob = "**/*.txt"

        result = cli.cmd_ingest(args)
        assert result == 1

    @patch("ai_qmx_helpdesk.rag_db.build_embeddings")
    def test_embed_command_success(self, mock_build_embeddings: Any) -> None:
        """Test successful embedding generation."""
        # Create database file
        self.db_path.touch()

        cli = RagCLI()
        args = MagicMock()
        args.db = str(self.db_path)
        args.provider = "toy"
        args.model = "test"
        args.batch_size = 32
        args.backend = "faiss"

        result = cli.cmd_embed(args)

        assert result == 0
        mock_build_embeddings.assert_called_once()

    @patch("ai_qmx_helpdesk.rag_db.search")
    def test_query_command_table_output(self, mock_search: Any) -> None:
        """Test query command with table output."""
        # Create database file
        self.db_path.touch()

        # Mock search results
        mock_doc = MagicMock()
        mock_doc.page_content = "Test content"
        mock_doc.metadata = {"source": "test.txt", "score": 0.85}
        mock_search.return_value = [mock_doc]

        cli = RagCLI()
        args = MagicMock()
        args.db = str(self.db_path)
        args.q = "test query"
        args.k = 5
        args.json = False
        args.show_paths = False
        args.show_chunks = False

        result = cli.cmd_query(args)

        assert result == 0
        mock_search.assert_called_once_with(str(self.db_path), "test query", k=5)

    @patch("ai_qmx_helpdesk.rag_db.search")
    def test_query_command_json_output(self, mock_search: Any) -> None:
        """Test query command with JSON output."""
        # Create database file
        self.db_path.touch()

        # Mock search results
        mock_doc = MagicMock()
        mock_doc.page_content = "Test content for JSON output"
        mock_doc.metadata = {"source": "test.txt", "score": 0.95}
        mock_search.return_value = [mock_doc]

        cli = RagCLI()
        args = MagicMock()
        args.db = str(self.db_path)
        args.q = "json test"
        args.k = 3
        args.json = True
        args.show_chunks = False

        # Capture stdout
        with patch("sys.stdout", new=StringIO()) as mock_stdout:
            result = cli.cmd_query(args)
            output = mock_stdout.getvalue()

        assert result == 0

        # Verify JSON output
        json_data = json.loads(output)
        assert json_data["query"] == "json test"
        assert json_data["k"] == 3
        assert len(json_data["results"]) == 1
        assert json_data["results"][0]["rank"] == 1
        assert json_data["results"][0]["path"] == "test.txt"

    def test_query_command_no_query(self) -> None:
        """Test query command without query text."""
        cli = RagCLI()
        args = MagicMock()
        args.db = str(self.db_path)
        args.q = ""  # Empty query
        args.k = 5

        result = cli.cmd_query(args)
        assert result == 1

    def test_inspect_command(self) -> None:
        """Test inspect command (alias for detailed query)."""
        # Create database file
        self.db_path.touch()

        cli = RagCLI()
        args = MagicMock()
        args.db = str(self.db_path)
        args.q = "inspect test"
        args.k = 5
        args.json = False

        with patch.object(cli, "cmd_query", return_value=0) as mock_query:
            result = cli.cmd_inspect(args)

            assert result == 0
            mock_query.assert_called_once_with(args)
            # Verify detailed output flags were set
            assert args.show_paths is True
            assert args.show_chunks is True

    def test_tune_command(self) -> None:
        """Test tune command (stub implementation)."""
        cli = RagCLI()
        args = MagicMock()

        result = cli.cmd_tune(args)
        assert result == 0


class TestCLIParser:
    """Test suite for CLI argument parsing."""

    def test_parser_creation(self) -> None:
        """Test argument parser creation."""
        parser = create_parser()
        assert parser is not None
        assert parser.prog == "qmx-helpdesk"

    def test_parse_init_command(self) -> None:
        """Test parsing init command."""
        parser = create_parser()
        args = parser.parse_args(["ragdb", "init", "--db", "test.db"])

        assert args.command == "ragdb"
        assert args.ragdb_command == "init"
        assert args.db == "test.db"

    def test_parse_ingest_command(self) -> None:
        """Test parsing ingest command."""
        parser = create_parser()
        args = parser.parse_args(
            ["ragdb", "ingest", "--db", "test.db", "--dir", "./data", "--glob", "**/*.pdf"]
        )

        assert args.command == "ragdb"
        assert args.ragdb_command == "ingest"
        assert args.db == "test.db"
        assert args.dir == "./data"
        assert args.glob == "**/*.pdf"

    def test_parse_embed_command(self) -> None:
        """Test parsing embed command."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "ragdb",
                "embed",
                "--db",
                "test.db",
                "--provider",
                "toy",
                "--model",
                "test-model",
                "--batch-size",
                "128",
                "--backend",
                "chroma",
            ]
        )

        assert args.command == "ragdb"
        assert args.ragdb_command == "embed"
        assert args.provider == "toy"
        assert args.model == "test-model"
        assert args.batch_size == 128
        assert args.backend == "chroma"

    def test_parse_query_command(self) -> None:
        """Test parsing query command."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "ragdb",
                "query",
                "--db",
                "test.db",
                "--q",
                "test query",
                "--k",
                "10",
                "--json",
                "--show-paths",
                "--show-chunks",
            ]
        )

        assert args.command == "ragdb"
        assert args.ragdb_command == "query"
        assert args.q == "test query"
        assert args.k == 10
        assert args.json is True
        assert getattr(args, "show_paths") is True
        assert getattr(args, "show_chunks") is True


class TestCLIIntegration:
    """Integration tests for the full CLI workflow."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.db_path = self.temp_path / "integration.db"
        self.data_dir = self.temp_path / "data"
        self.data_dir.mkdir(exist_ok=True)

        # Create test text file
        self.test_txt = self.data_dir / "integration.txt"
        self.test_txt.write_text("Integration test document with sample content for testing.")

    def teardown_method(self) -> None:
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    @patch("ai_qmx_helpdesk.rag_db.init_db")
    @patch("ai_qmx_helpdesk.rag_db.ingest_path")
    @patch("ai_qmx_helpdesk.rag_db.build_embeddings")
    @patch("ai_qmx_helpdesk.rag_db.search")
    def test_full_pipeline_with_toy_providers(
        self, mock_search: Any, mock_build_embeddings: Any, mock_ingest: Any, mock_init: Any
    ) -> None:
        """Test complete RAG pipeline using CLI commands."""
        # Mock search results for final query
        mock_doc = MagicMock()
        mock_doc.page_content = "Integration test content"
        mock_doc.metadata = {"source": "integration.txt", "score": 0.92}
        mock_search.return_value = [mock_doc]

        cli = RagCLI()

        # Test init
        args_init = MagicMock()
        args_init.db = str(self.db_path)
        result = cli.cmd_init(args_init)
        assert result == 0
        mock_init.assert_called_once()

        # Create database file for subsequent commands
        self.db_path.touch()

        # Test ingest
        args_ingest = MagicMock()
        args_ingest.db = str(self.db_path)
        args_ingest.dir = str(self.data_dir)
        args_ingest.glob = "**/*.txt"
        result = cli.cmd_ingest(args_ingest)
        assert result == 0
        mock_ingest.assert_called_once()

        # Test embed
        args_embed = MagicMock()
        args_embed.db = str(self.db_path)
        args_embed.provider = "toy"
        args_embed.model = "test"
        args_embed.batch_size = 64
        args_embed.backend = "faiss"
        result = cli.cmd_embed(args_embed)
        assert result == 0
        mock_build_embeddings.assert_called_once()

        # Test query with JSON output
        args_query = MagicMock()
        args_query.db = str(self.db_path)
        args_query.q = "integration test"
        args_query.k = 3
        args_query.json = True
        args_query.show_chunks = False

        with patch("sys.stdout", new=StringIO()) as mock_stdout:
            result = cli.cmd_query(args_query)
            output = mock_stdout.getvalue()

        assert result == 0
        mock_search.assert_called_once_with(str(self.db_path), "integration test", k=3)

        # Verify JSON structure
        json_data = json.loads(output)
        assert "query" in json_data
        assert "results" in json_data
        assert len(json_data["results"]) == 1
        assert json_data["results"][0]["rank"] == 1


class TestCLIMain:
    """Test suite for main entry point."""

    def test_main_no_args(self) -> None:
        """Test main function with no arguments."""
        with patch("sys.argv", ["cli.py"]):
            with patch("sys.stdout", new=StringIO()):
                result = main()
                assert result == 1

    def test_main_help(self) -> None:
        """Test main function with help."""
        with patch("sys.argv", ["cli.py", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    @patch("ai_qmx_helpdesk.rag_db.init_db")
    def test_main_init_command(self, mock_init: Any) -> None:
        """Test main function with init command."""
        with patch("sys.argv", ["cli.py", "ragdb", "init", "--db", "test.db"]):
            result = main()
            assert result == 0
            mock_init.assert_called_once_with("test.db")

    def test_main_unknown_command(self) -> None:
        """Test main function with unknown command."""
        with patch("sys.argv", ["cli.py", "unknown"]):
            with patch("sys.stderr", new=StringIO()):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 2
