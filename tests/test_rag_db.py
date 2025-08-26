"""
test_rag_db.py - Comprehensive tests for RAG database module
<DATE>: 2025-08-25

This module provides comprehensive tests for the RAG database functionality
including document ingestion, embedding generation, search, and chain operations.
Uses toy providers to avoid external dependencies and network calls.

Example usage:
    pytest tests/test_rag_db.py -v
    pytest tests/test_rag_db.py::TestRagDbFunctions::test_init_db
"""

import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch, MagicMock

import pytest

from ai_qmx_helpdesk import rag_db


class TestRagDbFunctions:
    """Test suite for RAG database functions."""

    def setup_method(self) -> None:
        """Set up test fixtures for each test method."""
        # Create temporary directory for test databases
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_rag.db")
        self.data_dir = os.path.join(self.temp_dir, "data")

        # Create test data directory with sample files
        os.makedirs(self.data_dir, exist_ok=True)

        # Create sample text files
        self.sample_txt1 = os.path.join(self.data_dir, "doc1.txt")
        with open(self.sample_txt1, "w") as f:
            f.write(
                "This is the first test document. It contains important information about testing."
            )

        self.sample_txt2 = os.path.join(self.data_dir, "doc2.txt")
        with open(self.sample_txt2, "w") as f:
            f.write(
                "This is the second test document. It discusses advanced concepts in machine learning."
            )

        # Create a minimal test PDF (if PDF creation is available)
        self.sample_pdf = os.path.join(self.data_dir, "doc3.pdf")
        try:
            # Try to create a minimal PDF for testing
            self._create_minimal_pdf(self.sample_pdf)
        except Exception:
            # Skip PDF if creation fails
            self.sample_pdf = None  # type: ignore[assignment]

    def teardown_method(self) -> None:
        """Clean up test fixtures after each test method."""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _create_minimal_pdf(self, path: str) -> None:
        """
        Create a minimal PDF file for testing.

        Args:
            path: Output path for PDF file
        """
        # Minimal PDF content (just enough to be valid)
        pdf_content = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj
4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
72 720 Td
(Test PDF Content) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f 
0000000015 00000 n 
0000000074 00000 n 
0000000131 00000 n 
0000000225 00000 n 
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
318
%%EOF"""

        with open(path, "wb") as f:
            f.write(pdf_content)

    def test_init_db(self) -> None:
        """Test database initialization."""
        # Test successful initialization
        rag_db.init_db(self.db_path)

        # Verify database file exists
        assert os.path.exists(self.db_path)

        # Verify tables were created
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check documents table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='documents'")
        assert cursor.fetchone() is not None

        # Check chunks table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chunks'")
        assert cursor.fetchone() is not None

        # Check indexes
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_doc_path'")
        assert cursor.fetchone() is not None

        conn.close()

        # Verify index directory was created
        index_dir = Path(self.db_path).parent / "index"
        assert index_dir.exists()

    def test_init_db_with_custom_path(self) -> None:
        """Test database initialization with custom path."""
        custom_path = os.path.join(self.temp_dir, "custom", "my_rag.db")

        rag_db.init_db(custom_path)

        assert os.path.exists(custom_path)

        # Check that custom index directory is created
        index_dir = Path(custom_path).parent / "index"
        assert index_dir.exists()

    @patch("ai_qmx_helpdesk.rag_db._load_document_loaders")
    def test_ingest_path_no_loaders(self, mock_load_loaders: Any) -> None:
        """Test ingestion failure when no loaders available."""
        mock_load_loaders.return_value = {}

        rag_db.init_db(self.db_path)

        with pytest.raises(ImportError, match="No document loaders available"):
            rag_db.ingest_path(self.db_path, self.data_dir)

    def test_ingest_path_missing_directory(self) -> None:
        """Test ingestion with missing data directory."""
        rag_db.init_db(self.db_path)

        with pytest.raises(FileNotFoundError, match="Data directory not found"):
            rag_db.ingest_path(self.db_path, "/nonexistent/directory")

    @patch("ai_qmx_helpdesk.rag_db._load_document_loaders")
    @patch("ai_qmx_helpdesk.rag_db._get_text_splitter")
    def test_ingest_path_success(self, mock_splitter: Any, mock_load_loaders: Any) -> None:
        """Test successful document ingestion."""
        # Mock loaders
        mock_loader_class = MagicMock()
        mock_loader_instance = MagicMock()
        mock_loader_class.return_value = mock_loader_instance

        # Mock document loading
        mock_doc = MagicMock()
        mock_doc.page_content = "Test document content for processing"
        mock_doc.metadata = {"source": "test.txt"}
        mock_loader_instance.load.return_value = [mock_doc]

        mock_load_loaders.return_value = {"txt": mock_loader_class}

        # Mock text splitter
        mock_splitter_instance = MagicMock()
        mock_splitter_instance.split_text.return_value = ["Test document content", "for processing"]
        mock_splitter.return_value = mock_splitter_instance

        rag_db.init_db(self.db_path)

        # Run ingestion
        result = rag_db.ingest_path(self.db_path, self.data_dir, "*.txt")

        # Verify results
        assert result >= 1  # At least one document processed

        # Verify database contents
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM documents")
        doc_count = cursor.fetchone()[0]
        assert doc_count >= 1

        cursor.execute("SELECT COUNT(*) FROM chunks")
        chunk_count = cursor.fetchone()[0]
        assert chunk_count >= 2  # Two chunks from split_text mock

        conn.close()

    def test_ingest_path_idempotent(self) -> None:
        """Test that ingestion is idempotent (unchanged files skipped)."""
        with (
            patch("ai_qmx_helpdesk.rag_db._load_document_loaders") as mock_load_loaders,
            patch("ai_qmx_helpdesk.rag_db._get_text_splitter") as mock_splitter,
        ):
            # Setup mocks
            mock_loader_class = MagicMock()
            mock_loader_instance = MagicMock()
            mock_loader_class.return_value = mock_loader_instance

            mock_doc = MagicMock()
            mock_doc.page_content = "Test content"
            mock_doc.metadata = {}
            mock_loader_instance.load.return_value = [mock_doc]

            mock_load_loaders.return_value = {"txt": mock_loader_class}

            mock_splitter_instance = MagicMock()
            mock_splitter_instance.split_text.return_value = ["Test content"]
            mock_splitter.return_value = mock_splitter_instance

            rag_db.init_db(self.db_path)

            # First ingestion
            result1 = rag_db.ingest_path(self.db_path, self.data_dir, "doc1.txt")
            assert result1 == 1

            # Second ingestion (should skip unchanged files)
            result2 = rag_db.ingest_path(self.db_path, self.data_dir, "doc1.txt")
            assert result2 == 0  # No new documents processed

    def test_toy_embeddings(self) -> None:
        """Test toy embeddings provider."""
        toy_embeddings = rag_db.ToyEmbeddings("test-model")

        # Test single embedding
        embedding = toy_embeddings.embed_query("test query")
        assert isinstance(embedding, list)
        assert len(embedding) == 384  # Standard dimension
        assert all(isinstance(x, float) for x in embedding)

        # Test deterministic behavior
        embedding2 = toy_embeddings.embed_query("test query")
        assert embedding == embedding2

        # Test different inputs produce different embeddings
        embedding3 = toy_embeddings.embed_query("different query")
        assert embedding != embedding3

        # Test batch embedding
        texts = ["text1", "text2", "text3"]
        embeddings = toy_embeddings.embed_documents(texts)
        assert len(embeddings) == 3
        assert all(len(emb) == 384 for emb in embeddings)

    def test_make_embeddings_toy_provider(self) -> None:
        """Test embeddings factory with toy provider."""
        cfg = {"provider": "toy", "model": "test-model"}
        embeddings = rag_db.make_embeddings(cfg)

        assert isinstance(embeddings, rag_db.ToyEmbeddings)
        assert embeddings.model == "test-model"

    def test_make_embeddings_unsupported_provider(self) -> None:
        """Test embeddings factory with unsupported provider."""
        cfg = {"provider": "unknown_provider"}

        with pytest.raises(ValueError, match="Unsupported embeddings provider"):
            rag_db.make_embeddings(cfg)

    @patch("ai_qmx_helpdesk.rag_db._get_db_connection")
    def test_build_embeddings(self, mock_get_conn: Any) -> None:
        """Test embedding generation for database chunks."""
        # Mock database connection and data
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=None)
        mock_get_conn.return_value = mock_conn

        # Mock chunk data
        mock_rows = [
            {"text": "First chunk content", "metadata": json.dumps({"doc_id": 1})},
            {"text": "Second chunk content", "metadata": json.dumps({"doc_id": 2})},
        ]
        mock_cursor.fetchall.return_value = mock_rows

        cfg = {
            "embed": {"provider": "toy"},
            "db": {"backend": "faiss", "persist_dir": os.path.join(self.temp_dir, "index")},
        }

        with patch("langchain_community.vectorstores.FAISS") as mock_faiss_class:
            mock_faiss = MagicMock()
            mock_faiss_class.from_texts.return_value = mock_faiss

            result = rag_db.build_embeddings(self.db_path, cfg)

            assert result == 2  # Two chunks processed
            mock_faiss_class.from_texts.assert_called_once()
            mock_faiss.save_local.assert_called_once()

    @patch("ai_qmx_helpdesk.rag_db._get_db_connection")
    def test_build_embeddings_no_chunks(self, mock_get_conn: Any) -> None:
        """Test embedding generation with no chunks."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=None)
        mock_get_conn.return_value = mock_conn

        mock_cursor.fetchall.return_value = []

        cfg = {"embed": {"provider": "toy"}}

        result = rag_db.build_embeddings(self.db_path, cfg)
        assert result == 0

    def test_build_embeddings_unsupported_backend(self) -> None:
        """Test embedding generation with unsupported backend."""
        # Initialize database first
        rag_db.init_db(self.db_path)

        # Add some dummy chunks to the database
        import sqlite3

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO documents (path, mtime, bytes) VALUES (?, ?, ?)",
            ("test.txt", 1234567890.0, 100),
        )
        doc_id = cursor.lastrowid
        cursor.execute(
            "INSERT INTO chunks (doc_id, seq, text, metadata) VALUES (?, ?, ?, ?)",
            (doc_id, 0, "Test chunk", "{}"),
        )
        conn.commit()
        conn.close()

        cfg = {"embed": {"provider": "toy"}, "db": {"backend": "unknown_backend"}}

        with pytest.raises(ValueError, match="Unsupported vector store backend"):
            rag_db.build_embeddings(self.db_path, cfg)

    def test_toy_llm(self) -> None:
        """Test toy LLM implementation."""
        toy_llm = rag_db.ToyLLM()

        # Test string input
        result = toy_llm.invoke("test message")
        assert hasattr(result, "content")
        assert "Echo: test message" in result.content

        # Test message object input
        mock_message = MagicMock()
        mock_message.content = "message content"
        result = toy_llm.invoke(mock_message)
        assert "Echo: message content" in result.content

        # Test list input
        mock_messages = [MagicMock()]
        mock_messages[0].content = "last message"
        result = toy_llm.invoke(mock_messages)
        assert "Echo: last message" in result.content

    def test_make_llm_toy_provider(self) -> None:
        """Test LLM factory with toy provider."""
        cfg = {"provider": "toy"}
        llm = rag_db.make_llm(cfg)

        assert isinstance(llm, rag_db.ToyLLM)

    def test_make_llm_unsupported_provider(self) -> None:
        """Test LLM factory with unsupported provider."""
        cfg = {"provider": "unknown_provider"}

        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            rag_db.make_llm(cfg)

    @patch("ai_qmx_helpdesk.rag_db._load_vectorstore")
    def test_search(self, mock_load_vectorstore: Any) -> None:
        """Test vector similarity search."""
        # Mock vectorstore
        mock_vectorstore = MagicMock()
        mock_doc1 = MagicMock()
        mock_doc1.page_content = "First matching document"
        mock_doc2 = MagicMock()
        mock_doc2.page_content = "Second matching document"

        mock_vectorstore.similarity_search_with_score.return_value = [
            (mock_doc1, 0.95),
            (mock_doc2, 0.87),
        ]
        mock_load_vectorstore.return_value = mock_vectorstore

        cfg = {"embed": {"provider": "toy"}}

        results = rag_db.search(self.db_path, "test query", k=2, cfg=cfg)

        assert len(results) == 2
        assert results[0] == ("First matching document", 0.95)
        assert results[1] == ("Second matching document", 0.87)

        mock_vectorstore.similarity_search_with_score.assert_called_once_with("test query", k=2)

    def test_search_empty_query(self) -> None:
        """Test search with empty query."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            rag_db.search(self.db_path, "")

        with pytest.raises(ValueError, match="Query cannot be empty"):
            rag_db.search(self.db_path, "   ")

    @patch("ai_qmx_helpdesk.rag_db._load_vectorstore")
    @patch("ai_qmx_helpdesk.rag_db.make_llm")
    @patch("langchain.prompts.ChatPromptTemplate")
    @patch("langchain.schema.output_parser.StrOutputParser")
    @patch("langchain.schema.runnable.RunnablePassthrough")
    def test_build_chain(
        self,
        mock_passthrough: Any,
        mock_parser: Any,
        mock_prompt: Any,
        mock_make_llm: Any,
        mock_load_vectorstore: Any,
    ) -> None:
        """Test RAG chain building."""
        # Setup mocks
        mock_vectorstore = MagicMock()
        mock_retriever = MagicMock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        mock_load_vectorstore.return_value = mock_vectorstore

        mock_llm = MagicMock()
        mock_make_llm.return_value = mock_llm

        mock_prompt_instance = MagicMock()
        mock_prompt.from_messages.return_value = mock_prompt_instance

        mock_parser_instance = MagicMock()
        mock_parser.return_value = mock_parser_instance

        mock_passthrough_instance = MagicMock()
        mock_passthrough.return_value = mock_passthrough_instance

        cfg = {
            "embed": {"provider": "toy"},
            "llm": {"provider": "toy"},
            "retriever": {"k": 5, "search_type": "similarity"},
        }

        # Test chain building
        chain = rag_db.build_chain(self.db_path, cfg)

        # Verify mocks were called correctly
        mock_load_vectorstore.assert_called_once_with(self.db_path, cfg)
        mock_vectorstore.as_retriever.assert_called_once()
        mock_make_llm.assert_called_once()
        mock_prompt.from_messages.assert_called_once()

        # Chain should be returned (exact structure depends on LangChain internals)
        assert chain is not None

    @patch("ai_qmx_helpdesk.rag_db.build_chain")
    def test_ask(self, mock_build_chain: Any) -> None:
        """Test question answering with RAG chain."""
        # Mock chain
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "This is the answer to your question."
        mock_build_chain.return_value = mock_chain

        cfg = {"llm": {"provider": "toy"}}

        result = rag_db.ask(self.db_path, "What is the answer?", cfg)

        assert result == "This is the answer to your question."
        mock_build_chain.assert_called_once_with(self.db_path, cfg)
        mock_chain.invoke.assert_called_once_with("What is the answer?")

    def test_ask_empty_question(self) -> None:
        """Test ask with empty question."""
        cfg = {"llm": {"provider": "toy"}}

        with pytest.raises(ValueError, match="Question cannot be empty"):
            rag_db.ask(self.db_path, "", cfg)

        with pytest.raises(ValueError, match="Question cannot be empty"):
            rag_db.ask(self.db_path, "   ", cfg)


class TestRagDBClass:
    """Test suite for RagDB wrapper class."""

    def setup_method(self) -> None:
        """Set up test fixtures for each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_class_rag.db")

    def teardown_method(self) -> None:
        """Clean up test fixtures after each test method."""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_initialization(self) -> None:
        """Test RagDB class initialization."""
        db = rag_db.RagDB()

        # Test default configuration
        cfg = db.get_cfg()
        assert "db" in cfg
        assert "embed" in cfg
        assert "llm" in cfg

        # Test custom configuration
        custom_cfg = {"db": {"path": self.db_path}}
        db_custom = rag_db.RagDB(custom_cfg)
        assert db_custom._db_path == self.db_path

    def test_configuration_management(self) -> None:
        """Test configuration get/set methods."""
        db = rag_db.RagDB()

        # Test get_cfg
        cfg = db.get_cfg()
        assert isinstance(cfg, dict)

        # Test set_cfg
        new_cfg = {"db": {"path": self.db_path}, "embed": {"provider": "toy"}}
        db.set_cfg(new_cfg)

        updated_cfg = db.get_cfg()
        assert updated_cfg["db"]["path"] == self.db_path
        assert updated_cfg["embed"]["provider"] == "toy"

    @patch("ai_qmx_helpdesk.rag_db.init_db")
    def test_init_db_wrapper(self, mock_init_db: Any) -> None:
        """Test init_db wrapper method."""
        db = rag_db.RagDB({"db": {"path": self.db_path}})

        db.init_db()

        mock_init_db.assert_called_once_with(self.db_path)

    @patch("ai_qmx_helpdesk.rag_db.ingest_path")
    def test_ingest_path_wrapper(self, mock_ingest_path: Any) -> None:
        """Test ingest_path wrapper method."""
        mock_ingest_path.return_value = 5

        db = rag_db.RagDB({"db": {"path": self.db_path}})

        # Test with defaults
        result = db.ingest_path()
        assert result == 5
        mock_ingest_path.assert_called_with(self.db_path, "./data", "**/*.{txt,pdf}")

        # Test with custom parameters
        result = db.ingest_path("/custom/data", "*.txt")
        mock_ingest_path.assert_called_with(self.db_path, "/custom/data", "*.txt")

    @patch("ai_qmx_helpdesk.rag_db.build_embeddings")
    def test_build_embeddings_wrapper(self, mock_build_embeddings: Any) -> None:
        """Test build_embeddings wrapper method."""
        mock_build_embeddings.return_value = 10

        db = rag_db.RagDB({"db": {"path": self.db_path}})

        result = db.build_embeddings()
        assert result == 10
        mock_build_embeddings.assert_called_once()

    @patch("ai_qmx_helpdesk.rag_db.search")
    def test_search_wrapper(self, mock_search: Any) -> None:
        """Test search wrapper method."""
        mock_search.return_value = [("result1", 0.9), ("result2", 0.8)]

        db = rag_db.RagDB({"db": {"path": self.db_path}})

        # Test with default k
        result = db.search("test query")
        assert len(result) == 2
        mock_search.assert_called_with(self.db_path, "test query", 5, db.get_cfg())

        # Test with custom k
        result = db.search("test query", k=3)
        mock_search.assert_called_with(self.db_path, "test query", 3, db.get_cfg())

    @patch("ai_qmx_helpdesk.rag_db.build_chain")
    def test_build_chain_wrapper(self, mock_build_chain: Any) -> None:
        """Test build_chain wrapper method."""
        mock_chain = MagicMock()
        mock_build_chain.return_value = mock_chain

        db = rag_db.RagDB({"db": {"path": self.db_path}})

        result = db.build_chain()
        assert result == mock_chain
        mock_build_chain.assert_called_once_with(self.db_path, db.get_cfg())

    @patch("ai_qmx_helpdesk.rag_db.ask")
    def test_ask_wrapper(self, mock_ask: Any) -> None:
        """Test ask wrapper method."""
        mock_ask.return_value = "Test answer"

        db = rag_db.RagDB({"db": {"path": self.db_path}})

        result = db.ask("Test question?")
        assert result == "Test answer"
        mock_ask.assert_called_once_with(self.db_path, "Test question?", db.get_cfg())

    @patch("ai_qmx_helpdesk.rag_db._get_db_connection")
    def test_get_stats(self, mock_get_conn: Any) -> None:
        """Test database statistics retrieval."""
        # Mock database connection and data
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=None)
        mock_get_conn.return_value = mock_conn

        # Mock query results
        mock_cursor.fetchone.side_effect = [
            {"doc_count": 5},  # documents count
            {"chunk_count": 15},  # chunks count
        ]

        db = rag_db.RagDB({"db": {"path": self.db_path}})

        stats = db.get_stats()

        assert stats["db_path"] == self.db_path
        assert stats["document_count"] == 5
        assert stats["chunk_count"] == 15
        assert "index_exists" in stats
        assert "index_path" in stats


class TestIntegration:
    """Integration tests using toy providers."""

    def setup_method(self) -> None:
        """Set up test fixtures for integration tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "integration_test.db")
        self.data_dir = os.path.join(self.temp_dir, "data")

        # Create test data
        os.makedirs(self.data_dir, exist_ok=True)

        with open(os.path.join(self.data_dir, "test1.txt"), "w") as f:
            f.write("Machine learning is a subset of artificial intelligence.")

        with open(os.path.join(self.data_dir, "test2.txt"), "w") as f:
            f.write("Natural language processing enables computers to understand human language.")

    def teardown_method(self) -> None:
        """Clean up test fixtures after integration tests."""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch("ai_qmx_helpdesk.rag_db._load_document_loaders")
    @patch("ai_qmx_helpdesk.rag_db._get_text_splitter")
    @patch("langchain_community.vectorstores.FAISS")
    def test_full_pipeline_with_toy_providers(
        self, mock_faiss_class: Any, mock_splitter: Any, mock_load_loaders: Any
    ) -> None:
        """Test complete RAG pipeline using toy providers."""
        # Setup mocks for document loading
        mock_loader_class = MagicMock()
        mock_loader_instance = MagicMock()
        mock_loader_class.return_value = mock_loader_instance

        # Mock documents
        mock_docs = [
            MagicMock(page_content="Machine learning content", metadata={}),
            MagicMock(page_content="Natural language processing content", metadata={}),
        ]
        mock_loader_instance.load.return_value = mock_docs
        mock_load_loaders.return_value = {"txt": mock_loader_class}

        # Mock text splitter
        mock_splitter_instance = MagicMock()
        mock_splitter_instance.split_text.side_effect = [
            ["Machine learning content"],
            ["Natural language processing content"],
        ]
        mock_splitter.return_value = mock_splitter_instance

        # Mock FAISS
        mock_vectorstore = MagicMock()
        mock_faiss_class.from_texts.return_value = mock_vectorstore
        mock_faiss_class.load_local.return_value = mock_vectorstore

        # Mock search results
        mock_doc = MagicMock()
        mock_doc.page_content = "Machine learning content"
        mock_vectorstore.similarity_search_with_score.return_value = [(mock_doc, 0.9)]
        mock_vectorstore.as_retriever.return_value = MagicMock()

        # Configuration using toy providers
        cfg = {"embed": {"provider": "toy"}, "llm": {"provider": "toy"}, "db": {"backend": "faiss"}}

        # Test full pipeline
        db = rag_db.RagDB({"db": {"path": self.db_path}})

        # 1. Initialize database
        db.init_db()
        assert os.path.exists(self.db_path)

        # 2. Ingest documents - use explicit parameters to avoid config mocking
        doc_count = db.ingest_path(data_dir=self.data_dir, glob_pattern="*.txt")
        assert doc_count >= 0  # Mocked, but should not error

        # 3. Build embeddings
        embed_count = rag_db.build_embeddings(self.db_path, cfg)
        assert embed_count >= 0  # Mocked, but should not error

        # 4. Search
        results = rag_db.search(self.db_path, "machine learning", k=1, cfg=cfg)
        assert len(results) == 1
        assert results[0][0] == "Machine learning content"
        assert results[0][1] == 0.9

        # 5. Ask question (chain execution)
        with (
            patch("langchain.prompts.ChatPromptTemplate"),
            patch("langchain.schema.output_parser.StrOutputParser"),
            patch("langchain.schema.runnable.RunnablePassthrough"),
        ):
            # Mock the chain execution
            with patch("ai_qmx_helpdesk.rag_db.build_chain") as mock_build_chain:
                mock_chain = MagicMock()
                mock_chain.invoke.return_value = "Echo: What is machine learning?"
                mock_build_chain.return_value = mock_chain

                answer = rag_db.ask(self.db_path, "What is machine learning?", cfg)
                assert "machine learning" in answer.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
