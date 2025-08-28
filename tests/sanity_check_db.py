#!/usr/bin/env python3
"""
Sanity check script to inspect RAG database contents
"""

import sqlite3
import json
from pathlib import Path


def check_database(db_path: str) -> None:
    """Check and display database contents."""

    if not Path(db_path).exists():
        print(f"❌ Database {db_path} does not exist!")
        return

    print(f"🔍 Inspecting database: {db_path}")
    print("=" * 60)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check documents table
    print("\n📄 DOCUMENTS TABLE:")
    cursor.execute("SELECT COUNT(*) FROM documents")
    doc_count = cursor.fetchone()[0]
    print(f"   Total documents: {doc_count}")

    if doc_count > 0:
        cursor.execute("SELECT id, path, bytes, created_at FROM documents")
        for row in cursor.fetchall():
            doc_id, source, file_size, created_at = row
            print(f"   📁 [{doc_id}] {source} ({file_size} bytes, {created_at})")

    # Check chunks table
    print("\n📝 CHUNKS TABLE:")
    cursor.execute("SELECT COUNT(*) FROM chunks")
    chunk_count = cursor.fetchone()[0]
    print(f"   Total chunks: {chunk_count}")

    if chunk_count > 0:
        print("\n   Sample chunks:")
        cursor.execute("""
            SELECT id, doc_id, seq, 
                   SUBSTR(text, 1, 80) as preview,
                   metadata 
            FROM chunks 
            ORDER BY doc_id, seq 
            LIMIT 5
        """)

        for row in cursor.fetchall():
            chunk_id, doc_id, chunk_seq, preview, metadata_json = row
            metadata = json.loads(metadata_json) if metadata_json else {}
            source = metadata.get("source", "unknown")

            print(f"   📄 Chunk {chunk_id} (doc={doc_id}, seq={chunk_seq})")
            print(f"      Source: {source}")
            print(f"      Text: {preview}...")
            print()

    # Check for corresponding vector index
    db_stem = Path(db_path).stem
    index_dir = Path(db_path).parent / f"{db_stem}_index"

    print(f"🔗 VECTOR INDEX: {index_dir}")
    if index_dir.exists():
        faiss_file = index_dir / "index.faiss"
        pkl_file = index_dir / "index.pkl"

        if faiss_file.exists():
            print(f"   ✅ FAISS index: {faiss_file} ({faiss_file.stat().st_size} bytes)")
        if pkl_file.exists():
            print(f"   ✅ Metadata: {pkl_file} ({pkl_file.stat().st_size} bytes)")

        if not (faiss_file.exists() and pkl_file.exists()):
            print("   ⚠️  Vector index incomplete - run 'ragdb embed' command")
    else:
        print("   ❌ Vector index directory not found - run 'ragdb embed' command")

    conn.close()
    print("\n✅ Database inspection complete!")


if __name__ == "__main__":
    import sys

    # Check command line args
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    else:
        # Check for common database names
        for db_name in ["rag.db", "my_docs.db"]:
            if Path(db_name).exists():
                db_path = db_name
                break
        else:
            print("Usage: python sanity_check_db.py [database.db]")
            print("Or run from directory containing rag.db or my_docs.db")
            sys.exit(1)

    check_database(db_path)
