import json
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np

try:
    from .embedder import Embedder
except ImportError:
    from embedder import Embedder


MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parent

INDEX_PATH = REPO_ROOT / "data" / "retrieval" / "minsearch_index.pkl"
VECTOR_INDEX_PATH = REPO_ROOT / "data" / "retrieval" / "vector_index.npz"
VECTOR_METADATA_PATH = REPO_ROOT / "data" / "retrieval" / "vector_index_metadata.json"

ENTRY_TEMPLATE = """
patient_id: {patient_id}
doc_type: {doc_type}
title: {title}
heading: {heading}
date_start: {date_start}
date_end: {date_end}
is_oncology: {is_oncology}
chunk_text: {chunk_text}
""".strip()


@lru_cache(maxsize=1) # this caches the loaded index to avoid reloading it multiple times
def load_index(index_path: Path = INDEX_PATH):
    with open(index_path, "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1) 
def load_vector_index():
    data = np.load(VECTOR_INDEX_PATH, allow_pickle=True)
    with open(VECTOR_METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    embeddings = data["embeddings"]
    chunk_ids = data["chunk_ids"].tolist()
    documents = metadata["documents"]

    docs_by_chunk_id = {doc["chunk_id"]: doc for doc in documents}
    ordered_docs = [docs_by_chunk_id[chunk_id] for chunk_id in chunk_ids]

    return embeddings, ordered_docs


@lru_cache(maxsize=1) 
def load_embedder():
    return Embedder()


def search(query, patient_id=None, doc_types=None, is_oncology=None, num_results=5):
    boost_dict = {
        "title": 1.0,
        "heading": 2.0,
        "chunk_text": 1.5,
    }

    filter_dict = {}

    if patient_id is not None:
        filter_dict["patient_id"] = patient_id

    if doc_types is not None:
        filter_dict["doc_type"] = doc_types

    if is_oncology is not None:
        filter_dict["is_oncology"] = str(int(bool(is_oncology)))

    return load_index().search(
        query=query,
        filter_dict=filter_dict,
        boost_dict=boost_dict,
        num_results=num_results,
    )


def semantic_search(query, patient_id=None, doc_types=None, is_oncology=None, num_results=5):
    query_vector = load_embedder().encode(query, normalize=True)
    embeddings, vector_documents = load_vector_index()
    scores = embeddings @ query_vector

    filtered = []
    for doc, score in zip(vector_documents, scores):
        if patient_id is not None and doc.get("patient_id") != patient_id:
            continue

        if doc_types is not None:
            allowed_doc_types = {doc_types} if isinstance(doc_types, str) else set(doc_types)
            if doc.get("doc_type") not in allowed_doc_types:
                continue

        if is_oncology is not None and int(doc.get("is_oncology", 0)) != int(bool(is_oncology)):
            continue

        doc_with_score = dict(doc)
        doc_with_score["semantic_score"] = float(score)
        filtered.append(doc_with_score)

    filtered = sorted(filtered, key=lambda x: x["semantic_score"], reverse=True)
    return filtered[:num_results]


def rrf(result_lists, k=60, num_results=5):
    scores = {}
    docs = {}

    for results in result_lists:
        for rank, doc in enumerate(results, start=1):
            key = doc["chunk_id"]
            scores[key] = scores.get(key, 0.0) + 1 / (k + rank)
            docs[key] = doc

    ranked_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    fused = []
    for key in ranked_keys[:num_results]:
        doc = dict(docs[key])
        doc["rrf_score"] = scores[key]
        fused.append(doc)

    return fused


def hybrid_search(query, patient_id=None, doc_types=None, is_oncology=None, num_results=5, rrf_k=60):
    lexical_results = search(
        query=query,
        patient_id=patient_id,
        doc_types=doc_types,
        is_oncology=is_oncology,
        num_results=10,
    )

    semantic_results = semantic_search(
        query=query,
        patient_id=patient_id,
        doc_types=doc_types,
        is_oncology=is_oncology,
        num_results=10,
    )

    return rrf([lexical_results, semantic_results], k=rrf_k, num_results=num_results)


def build_context(search_results):
    context = ""

    for doc in search_results:
        doc_copy = {
            "patient_id": doc.get("patient_id", ""),
            "doc_type": doc.get("doc_type", ""),
            "title": doc.get("title", ""),
            "heading": doc.get("heading", ""),
            "date_start": doc.get("date_start", ""),
            "date_end": doc.get("date_end", ""),
            "is_oncology": doc.get("is_oncology", ""),
            "chunk_text": doc.get("chunk_text", ""),
        }
        context = context + ENTRY_TEMPLATE.format(**doc_copy) + "\n\n"

    return context.strip()