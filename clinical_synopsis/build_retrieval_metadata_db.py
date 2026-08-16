import csv
import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
import os
from pathlib import Path


DEFAULT_DERIVED_ROOT = Path("data/derived/sample50")
DEFAULT_DB_PATH = Path("data/retrieval/metadata.db")

DERIVED_ROOT = Path(
    os.getenv(
        "CLINICAL_SYNOPSIS_DERIVED_ROOT",
        str(DEFAULT_DERIVED_ROOT),
    )
)

DB_PATH = Path(
    os.getenv(
        "CLINICAL_SYNOPSIS_METADATA_DB",
        str(DEFAULT_DB_PATH),
    )
)


ONCOLOGY_TERMS = [
    "cancer", "carcinoma", "tumor", "tumour", "neoplasm", "oncology",
    "malignant", "metast", "chemo", "chemotherapy", "radiation",
    "radiotherapy", "biopsy", "stage", "adenocarcinoma", "lymphoma",
    "leukemia", "melanoma", "sarcoma"
]

DATE_PATTERN = re.compile(r"\b(\d{4}-\d{2}-\d{2}(?:[T ][^;\s]+)?)\b")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def normalize_text(text) -> str:
    if text is None:
        return ""
    return str(text).strip()


def detect_doc_type(path: Path) -> str:
    stem = path.stem.lower()
    if stem == "patient_overview":
        return "patient_overview"
    if stem == "oncology_timeline":
        return "oncology_timeline"
    if stem == "oncology_timeline_events":
        return "oncology_timeline_events"
    if stem == "conditions":
        return "conditions"
    if stem == "observations":
        return "observations"
    if stem == "encounters":
        return "encounters"
    if stem == "medications":
        return "medications"
    if stem == "procedures":
        return "procedures"
    if stem == "diagnostic_reports":
        return "diagnostic_reports"
    return stem


def is_oncology_text(text: str) -> bool:
    t = normalize_text(text).lower()
    return any(term in t for term in ONCOLOGY_TERMS)


def extract_dates(text: str) -> tuple[str | None, str | None]:
    matches = DATE_PATTERN.findall(text or "")
    if not matches:
        return None, None
    dates = sorted(matches)
    return dates[0], dates[-1]


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def markdown_chunks(text: str, max_chars: int = 1400, overlap: int = 200) -> list[dict]:
    lines = text.splitlines()
    sections = []
    current_heading = "Document"
    current_lines = []

    for line in lines:
        if line.startswith("#"):
            if current_lines:
                sections.append((current_heading, "\n".join(current_lines).strip()))
                current_lines = []
            current_heading = line.lstrip("#").strip() or "Section"
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_heading, "\n".join(current_lines).strip()))

    chunks = []
    for heading, section_text in sections:
        if not section_text:
            continue
        start = 0
        while start < len(section_text):
            end = min(len(section_text), start + max_chars)
            chunk_text = section_text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "heading": heading,
                    "text": chunk_text,
                    "char_start": start,
                    "char_end": end,
                })
            if end >= len(section_text):
                break
            start = max(0, end - overlap)

    return chunks


def csv_row_chunks(path: Path) -> list[dict]:
    chunks = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            text_parts = []
            for key, value in row.items():
                val = normalize_text(value)
                if val:
                    text_parts.append(f"{key}: {val}")
            chunk_text = "; ".join(text_parts).strip()
            if not chunk_text:
                continue
            chunks.append({
                "heading": path.stem,
                "text": chunk_text,
                "char_start": 0,
                "char_end": len(chunk_text),
                "row": row,
                "row_index": idx,
            })
    return chunks


def extract_sources_from_csv_row(row: dict) -> list[tuple[str | None, str | None, str | None]]:
    resource_id = normalize_text(row.get("resource_id"))
    source_file = normalize_text(row.get("source_file"))
    resource_type = normalize_text(row.get("resource_type"))

    if not resource_type:
        if source_file and "Condition" in source_file:
            resource_type = "Condition"

    values = []
    if resource_id or resource_type or source_file:
        values.append((resource_id or None, resource_type or None, source_file or None))
    return values


def create_schema(conn: sqlite3.Connection):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS patients (
        patient_id TEXT PRIMARY KEY
    );

    CREATE TABLE IF NOT EXISTS documents (
        document_id TEXT PRIMARY KEY,
        patient_id TEXT NOT NULL,
        doc_type TEXT NOT NULL,
        file_path TEXT NOT NULL,
        source_file TEXT,
        title TEXT,
        format TEXT NOT NULL,
        created_at TEXT NOT NULL,
        is_oncology INTEGER NOT NULL DEFAULT 0,
        date_start TEXT,
        date_end TEXT,
        FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
    );

    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id TEXT PRIMARY KEY,
        document_id TEXT NOT NULL,
        patient_id TEXT NOT NULL,
        chunk_index INTEGER NOT NULL,
        chunk_text TEXT NOT NULL,
        char_start INTEGER NOT NULL,
        char_end INTEGER NOT NULL,
        token_estimate INTEGER,
        heading TEXT,
        is_oncology INTEGER NOT NULL DEFAULT 0,
        date_start TEXT,
        date_end TEXT,
        FOREIGN KEY (document_id) REFERENCES documents(document_id),
        FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
    );

    CREATE TABLE IF NOT EXISTS chunk_sources (
        chunk_id TEXT NOT NULL,
        resource_id TEXT,
        resource_type TEXT,
        source_file TEXT,
        PRIMARY KEY (chunk_id, resource_id, resource_type, source_file),
        FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id)
    );

    CREATE INDEX IF NOT EXISTS idx_documents_patient ON documents(patient_id);
    CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(doc_type);
    CREATE INDEX IF NOT EXISTS idx_chunks_patient ON chunks(patient_id);
    CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
    CREATE INDEX IF NOT EXISTS idx_chunks_oncology ON chunks(is_oncology);
    CREATE INDEX IF NOT EXISTS idx_chunk_sources_chunk ON chunk_sources(chunk_id);
    """)


def ingest_document(conn: sqlite3.Connection, patient_id: str, path: Path):
    text = path.read_text(encoding="utf-8") if path.suffix.lower() == ".md" else ""
    doc_type = detect_doc_type(path)
    fmt = path.suffix.lower().lstrip(".")
    document_id = sha1_text(f"{patient_id}|{path.as_posix()}")
    created_at = now_iso()

    if path.suffix.lower() == ".md":
        is_oncology = 1 if is_oncology_text(text) or "oncology" in doc_type else 0
        date_start, date_end = extract_dates(text)
        title = text.splitlines()[0].lstrip("# ").strip() if text.strip() else path.name
    else:
        is_oncology = 1 if "oncology" in doc_type else 0
        date_start, date_end = None, None
        title = path.name

    conn.execute("""
        INSERT OR REPLACE INTO documents
        (document_id, patient_id, doc_type, file_path, source_file, title, format, created_at, is_oncology, date_start, date_end)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        document_id, patient_id, doc_type, str(path), None, title, fmt, created_at,
        is_oncology, date_start, date_end
    ))

    if path.suffix.lower() == ".md":
        chunks = markdown_chunks(text)
        for idx, chunk in enumerate(chunks):
            chunk_text = chunk["text"]
            chunk_id = sha1_text(f"{document_id}|{idx}|{chunk_text}")
            c_date_start, c_date_end = extract_dates(chunk_text)
            c_oncology = 1 if is_oncology_text(chunk_text) or is_oncology else 0

            conn.execute("""
                INSERT OR REPLACE INTO chunks
                (chunk_id, document_id, patient_id, chunk_index, chunk_text, char_start, char_end,
                 token_estimate, heading, is_oncology, date_start, date_end)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                chunk_id, document_id, patient_id, idx, chunk_text,
                chunk["char_start"], chunk["char_end"],
                estimate_tokens(chunk_text), chunk["heading"], c_oncology,
                c_date_start, c_date_end
            ))

    elif path.suffix.lower() == ".csv":
        chunks = csv_row_chunks(path)
        for idx, chunk in enumerate(chunks):
            chunk_text = chunk["text"]
            row = chunk.get("row", {})
            chunk_id = sha1_text(f"{document_id}|{idx}|{chunk_text}")
            c_date_start, c_date_end = extract_dates(chunk_text)
            c_oncology = 1 if is_oncology_text(chunk_text) or is_oncology else 0

            conn.execute("""
                INSERT OR REPLACE INTO chunks
                (chunk_id, document_id, patient_id, chunk_index, chunk_text, char_start, char_end,
                 token_estimate, heading, is_oncology, date_start, date_end)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                chunk_id, document_id, patient_id, idx, chunk_text,
                chunk["char_start"], chunk["char_end"],
                estimate_tokens(chunk_text), chunk["heading"], c_oncology,
                c_date_start, c_date_end
            ))

            for resource_id, resource_type, source_file in extract_sources_from_csv_row(row):
                conn.execute("""
                    INSERT OR REPLACE INTO chunk_sources
                    (chunk_id, resource_id, resource_type, source_file)
                    VALUES (?, ?, ?, ?)
                """, (chunk_id, resource_id, resource_type, source_file))


def main():
    if not DERIVED_ROOT.exists():
        raise FileNotFoundError(
            f"Derived patient directory does not exist: {DERIVED_ROOT}"
        )

    if DB_PATH.resolve().parent == Path("data/retrieval").resolve():
        print(f"Building application metadata database: {DB_PATH}")
    else:
        print(f"Building sandbox metadata database: {DB_PATH}")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    create_schema(conn)

    patient_dirs = sorted([p for p in DERIVED_ROOT.iterdir() if p.is_dir()])
    if not patient_dirs:
        raise FileNotFoundError(f"No patient directories found in {DERIVED_ROOT}")

    doc_count = 0
    chunk_count = 0

    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        conn.execute("INSERT OR IGNORE INTO patients (patient_id) VALUES (?)", (patient_id,))

        for path in sorted(patient_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in {".md", ".csv"}:
                ingest_document(conn, patient_id, path)
                doc_count += 1

    conn.commit()

    chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    patient_count = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]

    summary = {
        "patients": patient_count,
        "documents": doc_count,
        "chunks": chunk_count,
        "db_path": str(DB_PATH),
    }
    print(json.dumps(summary, indent=2))

    conn.close()


if __name__ == "__main__":
    main()