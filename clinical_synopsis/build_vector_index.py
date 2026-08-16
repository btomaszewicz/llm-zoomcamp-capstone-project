import json
import os
import sqlite3
from pathlib import Path

import numpy as np

from embedder import Embedder


DEFAULT_DB_PATH = Path("data/retrieval/metadata.db")
DEFAULT_OUTPUT_DIR = Path("data/retrieval")

DB_PATH = Path(
    os.getenv(
        "CLINICAL_SYNOPSIS_METADATA_DB",
        str(DEFAULT_DB_PATH),
    )
)

OUTPUT_DIR = Path(
    os.getenv(
        "CLINICAL_SYNOPSIS_RETRIEVAL_OUTPUT_DIR",
        str(DEFAULT_OUTPUT_DIR),
    )
)

VECTORS_PATH = OUTPUT_DIR / "vector_index.npz"
METADATA_PATH = OUTPUT_DIR / "vector_index_metadata.json"


def load_chunk_documents(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT
            c.chunk_id,
            c.patient_id,
            c.document_id,
            d.doc_type,
            d.title,
            c.heading,
            c.chunk_text,
            c.chunk_index,
            c.is_oncology,
            c.date_start,
            c.date_end
        FROM chunks c
        JOIN documents d
          ON c.document_id = d.document_id
        ORDER BY c.patient_id, c.document_id, c.chunk_index
    """).fetchall()

    conn.close()

    documents = []
    for row in rows:
        documents.append(
            {
                "chunk_id": row["chunk_id"],
                "patient_id": row["patient_id"],
                "document_id": row["document_id"],
                "doc_type": row["doc_type"] or "",
                "title": row["title"] or "",
                "heading": row["heading"] or "",
                "chunk_text": row["chunk_text"] or "",
                "chunk_index": row["chunk_index"],
                "is_oncology": int(row["is_oncology"]),
                "date_start": row["date_start"] or "",
                "date_end": row["date_end"] or "",
            }
        )

    return documents


def build_embedding_text(doc):
    parts = [
        f"title: {doc['title']}" if doc["title"] else "",
        f"heading: {doc['heading']}" if doc["heading"] else "",
        f"doc_type: {doc['doc_type']}" if doc["doc_type"] else "",
        f"chunk_text: {doc['chunk_text']}" if doc["chunk_text"] else "",
    ]
    return "\n".join(part for part in parts if part).strip()


def batch_iterable(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DB_PATH.exists():
        raise FileNotFoundError(f"Metadata DB not found: {DB_PATH}")

    documents = load_chunk_documents(DB_PATH)
    if not documents:
        raise ValueError("No chunk documents found in metadata DB.")

    embedder = Embedder()

    texts = [build_embedding_text(doc) for doc in documents]
    chunk_ids = [doc["chunk_id"] for doc in documents]

    batch_size = 64
    vectors = []

    for batch in batch_iterable(texts, batch_size):
        batch_vectors = embedder.encode_batch(batch, normalize=True)
        vectors.append(batch_vectors)

    embeddings = np.vstack(vectors).astype(np.float32)

    np.savez_compressed(
        VECTORS_PATH,
        embeddings=embeddings,
        chunk_ids=np.array(chunk_ids, dtype=str),
    )

    metadata = {
        "embedding_model": "models/Xenova/all-MiniLM-L6-v2",
        "normalized": True,
        "shape": list(embeddings.shape),
        "documents": documents,
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Built vector index for {len(documents)} chunks")
    print(f"Saved vectors to {VECTORS_PATH}")
    print(f"Saved metadata to {METADATA_PATH}")


if __name__ == "__main__":
    main()
