"""
ai_qmx_helpdesk package
Created: 2025-08-25

Package init with version constant and public exports.
"""

# Import RAG database functions for public API
from .rag_db import (
    init_db,
    ingest_path,
    build_embeddings,
    search,
    build_chain,
    ask,
    RagDB,
)

__all__: list[str] = [
    "init_db",
    "ingest_path",
    "build_embeddings",
    "search",
    "build_chain",
    "ask",
    "RagDB",
]
__version__ = "0.1.0"
