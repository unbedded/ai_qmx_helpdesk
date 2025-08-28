"""
rag_db.py - LangChain RAG Database Module (SQLite + FAISS/Chroma)
<DATE>: 2025-08-25

This module provides a comprehensive Retrieval-Augmented Generation (RAG) database
system built on LangChain with SQLite metadata storage and FAISS/Chroma vector stores.
Features provider-agnostic factories, LCEL chains, and comprehensive retrieval capabilities.

Example usage:
    import rag_db

    # Initialize database
    rag_db.init_db("my_rag.db")

    # Ingest documents
    doc_count = rag_db.ingest_path("my_rag.db", "./docs", "**/*.txt")

    # Build embeddings
    embed_count = rag_db.build_embeddings("my_rag.db", {"embed": {"provider": "toy"}})

    # Search documents
    results = rag_db.search("my_rag.db", "What is the main topic?", k=3)

    # Build and use chain
    chain = rag_db.build_chain("my_rag.db", {"llm": {"provider": "toy"}})
    answer = rag_db.ask("my_rag.db", "Explain the key concepts", {"llm": {"provider": "toy"}})

    # Or use class wrapper
    db = rag_db.RagDB({"db": {"path": "my_rag.db"}})
    results = db.search("query", k=5)
"""

import hashlib
import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config_manager import ConfigManager


# STEP_1: Define default configuration and constants
DEFAULT_CONFIG = {
    "db": {"path": "rag.db", "backend": "faiss", "persist_dir": "rag_index"},
    "ingest": {"data_dir": "./data", "glob": "**/*.{txt,pdf}", "pdf_ocr": False},
    "split": {"splitter": "recursive_character", "chunk_size": 1000, "chunk_overlap": 200},
    "embed": {
        "provider": "huggingface",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "batch_size": 64,
        "normalize": True,
    },
    "retriever": {"k": 5, "search_type": "similarity", "score_threshold": None},
    "llm": {"provider": "openai", "model": "gpt-4o-mini", "temperature": 0.2, "max_tokens": 512},
}

# SQL schemas
CREATE_DOCUMENTS_TABLE = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT UNIQUE NOT NULL,
    mtime REAL NOT NULL,
    bytes INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

CREATE_CHUNKS_TABLE = """
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id INTEGER NOT NULL,
    seq INTEGER NOT NULL,
    text TEXT NOT NULL,
    metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
)
"""

CREATE_EMBEDDINGS_TABLE = """
CREATE TABLE IF NOT EXISTS embeddings_info (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    dimensions INTEGER,
    total_vectors INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    config TEXT
)
"""

CREATE_INDEX_DOC_PATH = "CREATE INDEX IF NOT EXISTS idx_doc_path ON documents(path)"
CREATE_INDEX_CHUNK_DOC = "CREATE INDEX IF NOT EXISTS idx_chunk_doc ON chunks(doc_id, seq)"


def _record_embedding_info(
    db_path: str, embed_cfg: Dict[str, Any], backend: str, chunk_count: int, vectorstore: Any = None
) -> None:
    """
    Record embedding provider information in database.

    STEP_11: Embedding metadata tracking

    Args:
        db_path: Path to database file
        embed_cfg: Embedding configuration
        backend: Vector store backend (faiss/chroma)
        chunk_count: Number of vectors created
        vectorstore: Optional vector store instance for dimension detection
    """
    try:
        # Try to detect embedding dimensions
        dimensions = None
        if hasattr(vectorstore, "index") and hasattr(vectorstore.index, "d"):
            # FAISS index dimension
            dimensions = vectorstore.index.d
        elif embed_cfg.get("provider") == "toy":
            dimensions = 384  # ToyEmbeddings default

        with _get_db_connection(db_path) as conn:
            # Clear existing embedding info (only keep latest)
            conn.execute("DELETE FROM embeddings_info")

            # Insert new embedding info
            conn.execute(
                """
                INSERT INTO embeddings_info (provider, model, dimensions, total_vectors, config)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    embed_cfg.get("provider", "unknown"),
                    embed_cfg.get("model", "default"),
                    dimensions,
                    chunk_count,
                    json.dumps({"backend": backend, "embed_cfg": embed_cfg}),
                ),
            )
            conn.commit()
            _logger.info(
                "Recorded embedding info: %s/%s, %d vectors",
                embed_cfg.get("provider"),
                embed_cfg.get("model"),
                chunk_count,
            )
    except Exception as e:
        _logger.warning("Failed to record embedding info: %s", str(e))


def _setup_logging() -> logging.Logger:
    """
    Set up module logging configuration.

    STEP_2: Initialize logging as first step

    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.WARNING)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


# Module logger
_logger = _setup_logging()


def _get_db_connection(db_path: str) -> sqlite3.Connection:
    """
    Get SQLite database connection with proper configuration.

    STEP_3: Database connection setup with foreign keys enabled

    Args:
        db_path: Path to SQLite database file

    Returns:
        sqlite3.Connection: Configured database connection
    """
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: str) -> None:
    """
    Initialize RAG database with required tables and directories.

    STEP_4: Database initialization with schema creation

    Args:
        db_path: Path to database file

    Raises:
        sqlite3.Error: If database initialization fails
    """
    _logger.info("Initializing RAG database at: %s", db_path)

    try:
        # Ensure parent directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Create database and tables
        with _get_db_connection(db_path) as conn:
            conn.execute(CREATE_DOCUMENTS_TABLE)
            conn.execute(CREATE_CHUNKS_TABLE)
            conn.execute(CREATE_EMBEDDINGS_TABLE)
            conn.execute(CREATE_INDEX_DOC_PATH)
            conn.execute(CREATE_INDEX_CHUNK_DOC)
            conn.commit()

        # Create vector store directory
        # Use db_path stem (filename without extension) + '_index' as directory name
        db_stem = Path(db_path).stem  # e.g. 'rag' from 'rag.db'
        index_dir = Path(db_path).parent / f"{db_stem}_index"

        index_dir.mkdir(parents=True, exist_ok=True)

        _logger.info("RAG database initialized successfully")

    except Exception as e:
        _logger.exception("Failed to initialize database: %s", str(e))
        raise


def _load_document_loaders() -> Dict[str, Any]:
    """
    Dynamically import document loaders with fallback handling.

    STEP_5: Dynamic loader imports with graceful fallbacks

    Returns:
        Dict: Available loaders mapping
    """
    loaders = {}

    try:
        from langchain_community.document_loaders import TextLoader

        loaders["txt"] = TextLoader
        _logger.debug("TextLoader available")
    except ImportError:
        _logger.warning("TextLoader not available")

    try:
        from langchain_community.document_loaders import PyPDFLoader

        loaders["pdf"] = PyPDFLoader  # type: ignore[assignment]
        _logger.debug("PyPDFLoader available")
    except ImportError:
        _logger.warning("PyPDFLoader not available")

    return loaders


def _get_text_splitter(cfg: Dict[str, Any]) -> Any:
    """
    Get configured text splitter instance.

    STEP_6: Text splitter configuration and instantiation

    Args:
        cfg: Split configuration

    Returns:
        Text splitter instance
    """
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        return RecursiveCharacterTextSplitter(
            chunk_size=cfg.get("chunk_size", 1000),
            chunk_overlap=cfg.get("chunk_overlap", 200),
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
    except ImportError as e:
        _logger.error("Text splitter not available: %s", str(e))
        raise ImportError("langchain text_splitter required for document processing") from e


def ingest_path(db_path: str, data_dir: str, glob_pattern: str = "**/*.{txt,pdf}") -> int:
    """
    Ingest documents from directory path with idempotent processing.

    STEP_7: Document ingestion with change detection

    Args:
        db_path: Path to database file
        data_dir: Directory containing documents to ingest
        glob_pattern: File matching pattern

    Returns:
        int: Number of documents processed

    Raises:
        FileNotFoundError: If data directory doesn't exist
        ImportError: If required loaders unavailable
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    _logger.info("Starting ingestion from: %s", data_dir)

    # Load available document loaders
    loaders = _load_document_loaders()
    if not loaders:
        raise ImportError("No document loaders available")

    # Get text splitter
    split_cfg: Dict[str, Any] = DEFAULT_CONFIG["split"]  # type: ignore[assignment]
    splitter = _get_text_splitter(split_cfg)

    processed_count = 0

    try:
        with _get_db_connection(db_path) as conn:
            cursor = conn.cursor()

            # Find files matching pattern - handle brace expansion manually
            if glob_pattern == "**/*.{txt,pdf}":
                # Default pattern - expand manually since Python glob doesn't support brace expansion
                files = list(data_path.glob("**/*.txt")) + list(data_path.glob("**/*.pdf"))
            else:
                files = list(data_path.glob(glob_pattern))
            _logger.info("Found %d files matching pattern", len(files))

            for file_path in files:
                try:
                    # Check if file needs processing
                    file_stat = file_path.stat()
                    file_mtime = file_stat.st_mtime
                    file_size = file_stat.st_size

                    # Check existing record
                    cursor.execute(
                        "SELECT mtime, bytes FROM documents WHERE path = ?", (str(file_path),)
                    )
                    existing = cursor.fetchone()

                    if (
                        existing
                        and existing["mtime"] == file_mtime
                        and existing["bytes"] == file_size
                    ):
                        _logger.debug("Skipping unchanged file: %s", file_path)
                        continue

                    # Determine loader
                    file_ext = file_path.suffix.lower().lstrip(".")
                    if file_ext not in loaders:
                        _logger.warning("No loader for file type: %s", file_ext)
                        continue

                    # Load document
                    loader_class = loaders[file_ext]
                    loader = loader_class(str(file_path))
                    documents = loader.load()

                    if not documents:
                        _logger.warning("No content loaded from: %s", file_path)
                        continue

                    # Process document - combine all pages for PDFs
                    if len(documents) > 1:
                        # Multi-page document (PDF) - combine all pages
                        doc_content = "\n\n".join(doc.page_content for doc in documents)
                        doc_metadata = documents[0].metadata  # Use metadata from first page
                        doc_metadata.update(
                            {
                                "path": str(file_path),
                                "mtime": file_mtime,
                                "bytes": file_size,
                                "pages": len(documents),
                            }
                        )
                        _logger.info("Combined %d pages for: %s", len(documents), file_path.name)
                    else:
                        # Single page/document (TXT)
                        doc_content = documents[0].page_content
                        doc_metadata = documents[0].metadata
                        doc_metadata.update(
                            {"path": str(file_path), "mtime": file_mtime, "bytes": file_size}
                        )

                    # Delete existing record if updating
                    if existing:
                        cursor.execute("DELETE FROM documents WHERE path = ?", (str(file_path),))

                    # Insert document record
                    cursor.execute(
                        "INSERT INTO documents (path, mtime, bytes) VALUES (?, ?, ?)",
                        (str(file_path), file_mtime, file_size),
                    )
                    doc_id = cursor.lastrowid

                    # Split into chunks
                    chunks = splitter.split_text(doc_content)

                    # Insert chunks
                    for seq, chunk_text in enumerate(chunks):
                        chunk_metadata = doc_metadata.copy()
                        chunk_metadata.update({"doc_id": doc_id, "chunk_seq": seq})

                        cursor.execute(
                            "INSERT INTO chunks (doc_id, seq, text, metadata) VALUES (?, ?, ?, ?)",
                            (doc_id, seq, chunk_text, json.dumps(chunk_metadata)),
                        )

                    conn.commit()
                    processed_count += 1
                    _logger.info("Processed %s: %d chunks", file_path.name, len(chunks))

                except Exception as e:
                    _logger.error("Failed to process %s: %s", file_path, str(e))
                    conn.rollback()
                    continue

        _logger.info("Ingestion completed: %d documents processed", processed_count)
        return processed_count

    except Exception as e:
        _logger.exception("Ingestion failed: %s", str(e))
        raise


def make_embeddings(cfg: Dict[str, Any]) -> Any:
    """
    Factory function to create embeddings provider.

    STEP_8: Provider-agnostic embeddings factory

    Args:
        cfg: Embeddings configuration

    Returns:
        Embeddings instance

    Raises:
        ImportError: If provider dependencies unavailable
        ValueError: If provider not supported
    """
    provider = cfg.get("provider", "huggingface")

    if provider == "huggingface":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings

            model_name = cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2")
            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": cfg.get("normalize", True)},
            )
        except ImportError as e:
            raise ImportError(f"HuggingFace embeddings unavailable: {e}") from e

    elif provider == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings

            model = cfg.get("model", "text-embedding-3-small")
            return OpenAIEmbeddings(model=model)
        except ImportError as e:
            raise ImportError(f"OpenAI embeddings unavailable: {e}") from e

    elif provider == "toy":
        # Deterministic fallback for testing
        return ToyEmbeddings(cfg.get("model", "toy-model"))

    else:
        raise ValueError(f"Unsupported embeddings provider: {provider}")


class ToyEmbeddings:
    """
    Deterministic toy embeddings for testing without external dependencies.

    STEP_9: Toy embeddings implementation for offline testing
    """

    def __init__(self, model: str = "toy-model"):
        self.model = model
        self.dimension = 384  # Match common embedding dimensions

    def __call__(self, text: str) -> List[float]:
        """Make the class callable for FAISS compatibility."""
        return self.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """Embed single query text."""
        # Create deterministic embedding from text hash
        text_hash = hashlib.sha256(f"{self.model}:{text}".encode()).digest()

        # Convert to normalized float vector
        embedding: List[float] = []
        for i in range(0, len(text_hash), 2):
            if len(embedding) >= self.dimension:
                break
            if i + 1 < len(text_hash):
                val = (text_hash[i] * 256 + text_hash[i + 1]) / 65535.0 * 2.0 - 1.0
                embedding.append(val)

        # Pad to target dimension
        while len(embedding) < self.dimension:
            embedding.append(0.0)

        # Normalize
        magnitude = sum(x * x for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]

        return embedding


def build_embeddings(db_path: str, cfg: Dict[str, Any]) -> int:
    """
    Build embeddings for all chunks in database.

    STEP_10: Embedding generation and vector store building

    Args:
        db_path: Path to database file
        cfg: Full configuration dictionary

    Returns:
        int: Number of embeddings created

    Raises:
        ImportError: If required dependencies unavailable
    """
    _logger.info("Building embeddings for database: %s", db_path)

    embed_cfg = cfg.get("embed", DEFAULT_CONFIG["embed"])
    db_cfg = cfg.get("db", DEFAULT_CONFIG["db"])

    # Create embeddings instance
    embeddings = make_embeddings(embed_cfg)

    # Get chunks from database
    chunk_count = 0
    texts = []
    metadatas = []

    try:
        with _get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT text, metadata FROM chunks ORDER BY id")

            for row in cursor.fetchall():
                texts.append(row["text"])
                metadata = json.loads(row["metadata"]) if row["metadata"] else {}
                metadatas.append(metadata)
                chunk_count += 1

        if not texts:
            _logger.warning("No chunks found to embed")
            return 0

        _logger.info("Found %d chunks to embed", len(texts))

        # Create vector store
        backend = db_cfg.get("backend", "faiss").lower()

        if backend == "faiss":
            try:
                from langchain_community.vectorstores import FAISS

                vectorstore: Any = FAISS.from_texts(texts, embeddings, metadatas=metadatas)

                # Save to disk
                # Use db_path stem + '_index' as directory name
                db_stem = Path(db_path).stem
                persist_dir = Path(db_path).parent / f"{db_stem}_index"

                Path(persist_dir).mkdir(parents=True, exist_ok=True)
                vectorstore.save_local(str(persist_dir))

                _logger.info("FAISS index saved to: %s", persist_dir)

                # Record embedding provider info
                _record_embedding_info(db_path, embed_cfg, backend, chunk_count, vectorstore)

            except ImportError as e:
                raise ImportError(f"FAISS backend unavailable: {e}") from e

        elif backend == "chroma":
            try:
                from langchain_community.vectorstores import Chroma

                # Use db_path stem + '_index' as directory name
                db_stem = Path(db_path).stem
                persist_dir = Path(db_path).parent / f"{db_stem}_index"

                vectorstore = Chroma.from_texts(
                    texts, embeddings, metadatas=metadatas, persist_directory=str(persist_dir)
                )
                vectorstore.persist()

                _logger.info("Chroma index saved to: %s", persist_dir)

                # Record embedding provider info
                _record_embedding_info(db_path, embed_cfg, backend, chunk_count, vectorstore)

            except ImportError as e:
                raise ImportError(f"Chroma backend unavailable: {e}") from e

        else:
            raise ValueError(f"Unsupported vector store backend: {backend}")

        _logger.info("Successfully built embeddings for %d chunks", chunk_count)
        return chunk_count

    except Exception as e:
        _logger.exception("Failed to build embeddings: %s", str(e))
        raise


def _load_vectorstore(db_path: str, cfg: Dict[str, Any]) -> Any:
    """
    Load existing vector store from disk.

    STEP_11: Vector store loading with backend detection

    Args:
        db_path: Database path for directory inference
        cfg: Configuration dictionary

    Returns:
        Vector store instance
    """
    db_cfg = cfg.get("db", DEFAULT_CONFIG["db"])
    embed_cfg = cfg.get("embed", DEFAULT_CONFIG["embed"])
    backend = db_cfg.get("backend", "faiss").lower()

    # Create embeddings instance
    embeddings = make_embeddings(embed_cfg)

    # Determine persist directory
    # Use db_path stem + '_index' as directory name
    db_stem = Path(db_path).stem
    persist_dir = Path(db_path).parent / f"{db_stem}_index"

    if backend == "faiss":
        try:
            from langchain_community.vectorstores import FAISS

            return FAISS.load_local(
                str(persist_dir), embeddings, allow_dangerous_deserialization=True
            )
        except ImportError as e:
            raise ImportError(f"FAISS backend unavailable: {e}") from e
        except Exception as e:
            raise FileNotFoundError(
                f"FAISS index not found at {persist_dir}. Run build_embeddings first."
            ) from e

    elif backend == "chroma":
        try:
            from langchain_community.vectorstores import Chroma

            return Chroma(persist_directory=str(persist_dir), embedding_function=embeddings)
        except ImportError as e:
            raise ImportError(f"Chroma backend unavailable: {e}") from e

    else:
        raise ValueError(f"Unsupported vector store backend: {backend}")


def search(
    db_path: str, query: str, k: int = 5, cfg: Optional[Dict[str, Any]] = None
) -> List[Tuple[str, float]]:
    """
    Search for similar documents using vector similarity.

    STEP_12: Vector similarity search with scoring

    Args:
        db_path: Path to database file
        query: Search query text
        k: Number of results to return
        cfg: Optional configuration override

    Returns:
        List[Tuple[str, float]]: List of (text, score) tuples

    Raises:
        FileNotFoundError: If vector index not found
        ValueError: If query is empty
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    if cfg is None:
        cfg = DEFAULT_CONFIG.copy()

    _logger.info("Searching for: '%s' (k=%d)", query, k)

    try:
        # Load vector store
        vectorstore = _load_vectorstore(db_path, cfg)

        # Get retriever configuration
        retriever_cfg: Dict[str, Any] = cfg.get("retriever", DEFAULT_CONFIG["retriever"])  # type: ignore[assignment]
        search_type = retriever_cfg.get("search_type", "similarity")
        score_threshold = retriever_cfg.get("score_threshold")

        # Perform search
        if search_type == "similarity_score_threshold" and score_threshold is not None:
            docs_with_scores = vectorstore.similarity_search_with_score(
                query, k=k, score_threshold=score_threshold
            )
        else:
            docs_with_scores = vectorstore.similarity_search_with_score(query, k=k)

        # Format results
        results = [(doc.page_content, float(score)) for doc, score in docs_with_scores]

        _logger.info("Found %d results for query", len(results))
        return results

    except Exception as e:
        _logger.exception("Search failed: %s", str(e))
        raise


def make_llm(cfg: Dict[str, Any]) -> Any:
    """
    Factory function to create LLM provider.

    STEP_13: Provider-agnostic LLM factory

    Args:
        cfg: LLM configuration

    Returns:
        LLM instance

    Raises:
        ImportError: If provider dependencies unavailable
        ValueError: If provider not supported
    """
    provider = cfg.get("provider", "openai")

    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=cfg.get("model", "gpt-4o-mini"),
                temperature=cfg.get("temperature", 0.2),
                max_tokens=cfg.get("max_tokens", 512),  # type: ignore[call-arg]
            )
        except ImportError as e:
            raise ImportError(f"OpenAI LLM unavailable: {e}") from e

    elif provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(
                model_name=cfg.get("model", "claude-3-haiku-20240307"),  # type: ignore[call-arg]
                temperature=cfg.get("temperature", 0.2),
                max_tokens=cfg.get("max_tokens", 512),
            )
        except ImportError as e:
            raise ImportError(f"Anthropic LLM unavailable: {e}") from e

    elif provider == "toy":
        # Simple echo LLM for testing
        return ToyLLM()

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


class ToyLLM:
    """
    Toy LLM implementation for testing without external dependencies.

    STEP_14: Toy LLM for offline testing
    """

    def invoke(self, messages: Any) -> Any:
        """Simple invoke method that echoes input."""
        if isinstance(messages, str):
            content = f"Echo: {messages}"
        elif hasattr(messages, "content"):
            content = f"Echo: {messages.content}"
        elif isinstance(messages, list) and messages:
            last_msg = messages[-1]
            if hasattr(last_msg, "content"):
                content = f"Echo: {last_msg.content}"
            else:
                content = f"Echo: {str(last_msg)}"
        else:
            content = f"Echo: {str(messages)}"

        # Return minimal AIMessage-like object
        return type("AIMessage", (), {"content": content})()


def build_chain(db_path: str, cfg: Dict[str, Any]) -> Any:
    """
    Build LCEL RAG chain with retriever and LLM.

    STEP_15: LCEL chain construction with prompt template

    Args:
        db_path: Path to database file
        cfg: Configuration dictionary

    Returns:
        Runnable: LCEL chain instance

    Raises:
        ImportError: If LangChain components unavailable
    """
    _logger.info("Building RAG chain for database: %s", db_path)

    try:
        from langchain.prompts import ChatPromptTemplate
        from langchain.schema.output_parser import StrOutputParser
        from langchain.schema.runnable import RunnablePassthrough

        # Load vector store and create retriever
        vectorstore = _load_vectorstore(db_path, cfg)
        retriever_cfg = cfg.get("retriever", DEFAULT_CONFIG["retriever"])

        retriever = vectorstore.as_retriever(
            search_type=retriever_cfg.get("search_type", "similarity"),
            search_kwargs={"k": retriever_cfg.get("k", 5)},
        )

        # Create LLM
        llm_cfg = cfg.get("llm", DEFAULT_CONFIG["llm"])
        llm = make_llm(llm_cfg)

        # Define prompt template
        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant. Use the following context to answer the question. "
                    "If you don't know the answer based on the context, say so.",
                ),
                ("human", "Context:\n{context}\n\nQuestion: {question}"),
            ]
        )

        # Format documents helper
        def format_docs(docs: List[Any]) -> str:
            return "\n\n".join(doc.page_content for doc in docs)

        # Build LCEL chain
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt_template
            | llm
            | StrOutputParser()
        )

        _logger.info("RAG chain built successfully")
        return chain

    except ImportError as e:
        raise ImportError(f"LangChain components unavailable for chain building: {e}") from e
    except Exception as e:
        _logger.exception("Failed to build chain: %s", str(e))
        raise


def ask(db_path: str, question: str, cfg: Dict[str, Any]) -> str:
    """
    Ask question using RAG chain and return answer.

    STEP_16: Question answering with RAG chain execution

    Args:
        db_path: Path to database file
        question: Question to ask
        cfg: Configuration dictionary

    Returns:
        str: Generated answer

    Raises:
        ValueError: If question is empty
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")

    _logger.info("Processing question: %s", question)

    try:
        # Build and run chain
        chain = build_chain(db_path, cfg)
        result = chain.invoke(question)

        _logger.info("Generated answer successfully")
        return str(result)

    except Exception as e:
        _logger.exception("Failed to process question: %s", str(e))
        raise


class RagDB:
    """
    RAG Database wrapper class with configuration management.

    STEP_17: High-level wrapper class with ConfigManager integration

    This class provides a convenient wrapper around the RAG database functions
    with integrated configuration management and state tracking.
    """

    def __init__(self, cfg_dict: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize RAG database wrapper.

        Args:
            cfg_dict: Optional configuration dictionary
        """
        # Initialize logging first
        self._logger = logging.getLogger(__name__ + ".RagDB")
        self._logger.setLevel(logging.WARNING)

        if not self._logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)

        # Initialize configuration
        config = DEFAULT_CONFIG.copy()
        if cfg_dict:
            config.update(cfg_dict)

        self._config = ConfigManager(cfg_dict=config)
        # Access nested config values properly
        db_config = self._config.get_cfg().get("db", {})
        self._db_path = db_config.get("path", "rag.db")

        # Thread safety
        self._lock = threading.RLock()

        self._logger.info("RagDB initialized with database: %s", self._db_path)

    def get_cfg(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self._config.get_cfg()

    def set_cfg(self, cfg_dict: Dict[str, Any]) -> None:
        """Update configuration."""
        self._config.set_cfg(cfg_dict)
        # Update db_path if changed
        db_config = self._config.get_cfg().get("db", {})
        self._db_path = db_config.get("path", self._db_path)

    def init_db(self) -> None:
        """Initialize database."""
        with self._lock:
            init_db(self._db_path)

    def ingest_path(
        self, data_dir: Optional[str] = None, glob_pattern: Optional[str] = None
    ) -> int:
        """Ingest documents from path."""
        if data_dir is None:
            ingest_config = self._config.get_cfg().get("ingest", {})
            data_dir = ingest_config.get("data_dir", "./data")
        if glob_pattern is None:
            ingest_config = self._config.get_cfg().get("ingest", {})
            glob_pattern = ingest_config.get("glob", "**/*.{txt,pdf}")

        with self._lock:
            return ingest_path(self._db_path, data_dir, glob_pattern)

    def build_embeddings(self) -> int:
        """Build embeddings for all chunks."""
        with self._lock:
            return build_embeddings(self._db_path, self.get_cfg())

    def search(self, query: str, k: Optional[int] = None) -> List[Tuple[str, float]]:
        """Search for similar documents."""
        if k is None:
            retriever_config = self._config.get_cfg().get("retriever", {})
            k = retriever_config.get("k", 5)

        with self._lock:
            return search(self._db_path, query, k, self.get_cfg())

    def build_chain(self) -> Any:
        """Build RAG chain."""
        with self._lock:
            return build_chain(self._db_path, self.get_cfg())

    def ask(self, question: str) -> str:
        """Ask question and get answer."""
        with self._lock:
            return ask(self._db_path, question, self.get_cfg())

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            with _get_db_connection(self._db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("SELECT COUNT(*) as doc_count FROM documents")
                doc_count = cursor.fetchone()["doc_count"]

                cursor.execute("SELECT COUNT(*) as chunk_count FROM chunks")
                chunk_count = cursor.fetchone()["chunk_count"]

                # Check if vector index exists
                # Use db_path stem + '_index' as directory name
                db_stem = Path(self._db_path).stem
                persist_dir = Path(self._db_path).parent / f"{db_stem}_index"

                index_exists = Path(persist_dir).exists()

                return {
                    "db_path": self._db_path,
                    "document_count": doc_count,
                    "chunk_count": chunk_count,
                    "index_exists": index_exists,
                    "index_path": str(persist_dir),
                }

        except Exception as e:
            self._logger.error("Failed to get stats: %s", str(e))
            return {"error": str(e)}
