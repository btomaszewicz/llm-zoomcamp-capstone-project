import json
import os
import pickle
import sqlite3
from pathlib import Path

from minsearch import Index


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

INDEX_PATH = OUTPUT_DIR / "minsearch_index.pkl"
DOCUMENTS_PATH = OUTPUT_DIR / "minsearch_documents.json"


def load_documents(db_path: Path) -> list[dict]:
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
                "id": row["chunk_id"],
                "chunk_id": row["chunk_id"],
                "patient_id": row["patient_id"],
                "document_id": row["document_id"],
                "doc_type": row["doc_type"],
                "title": row["title"] or "",
                "heading": row["heading"] or "",
                "chunk_text": row["chunk_text"] or "",
                "chunk_index": row["chunk_index"],
                "is_oncology": str(row["is_oncology"]),
                "date_start": row["date_start"] or "",
                "date_end": row["date_end"] or "",
            }
        )

    return documents


def build_index(documents: list[dict]) -> Index:
    index = Index(
        text_fields=["title", "heading", "chunk_text"],
        keyword_fields=["patient_id", "doc_type", "is_oncology"],
    )
    index.fit(documents)
    return index


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DB_PATH.exists():
        raise FileNotFoundError(f"Metadata DB not found: {DB_PATH}")

    documents = load_documents(DB_PATH)
    if not documents:
        raise ValueError("No chunk documents found in metadata DB.")

    index = build_index(documents)

    with open(INDEX_PATH, "wb") as f:
        pickle.dump(index, f)

    with open(DOCUMENTS_PATH, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2)

    print(f"Built minsearch index for {len(documents)} chunks")
    print(f"Saved index to {INDEX_PATH}")
    print(f"Saved documents to {DOCUMENTS_PATH}")


if __name__ == "__main__":
    main()
