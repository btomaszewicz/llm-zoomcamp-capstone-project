import json
import pickle
from pathlib import Path
from time import time

import numpy as np
from openai import OpenAI

from embedder import Embedder


client = OpenAI()

MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parent

INDEX_PATH = REPO_ROOT / "data" / "retrieval" / "minsearch_index.pkl"
VECTOR_INDEX_PATH = REPO_ROOT / "data" / "retrieval" / "vector_index.npz"
VECTOR_METADATA_PATH = REPO_ROOT / "data" / "retrieval" / "vector_index_metadata.json"


INSTRUCTIONS = """
Your task is to answer questions about a patient's clinical record
based only on the provided context.

Use the context to find relevant information and provide accurate answers.
If the answer is not found in the context, respond with "I don't know."

Do not make up facts that are not supported by the context.
When possible, mention the document type and date information that support the answer.
""".strip()


PROMPT_TEMPLATE = """
QUESTION: {question}

CONTEXT:
{context}
""".strip()


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


EVALUATION_PROMPT_TEMPLATE = """
You are an expert evaluator for a RAG system.

Your task is to evaluate the generated answer for:
1. Relevance to the user's question
2. Groundedness in the provided retrieved context

Classify relevance as one of:
- "NON_RELEVANT"
- "PARTLY_RELEVANT"
- "RELEVANT"

Classify groundedness as one of:
- "NOT_GROUNDED"
- "PARTLY_GROUNDED"
- "GROUNDED"

Question: {question}

Search type: {search_type}

Retrieved context:
{context}

Generated answer:
{answer}

Return parsable JSON only, without code fences, in exactly this format:

{{
  "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
  "Groundedness": "NOT_GROUNDED" | "PARTLY_GROUNDED" | "GROUNDED",
  "Explanation": "[Provide a brief explanation for your evaluation]"
}}
""".strip()


def load_index(index_path=INDEX_PATH):
    with open(index_path, "rb") as f:
        return pickle.load(f)


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


index = load_index()
vector_embeddings, vector_documents = load_vector_index()
embedder = Embedder()


def search(query, patient_id=None, doc_types=None, is_oncology=None, num_results=5):
    """Lexical search over the minsearch index."""
    boost_dict = {
        "title": 1.0,       # low; titles are often generic
        "heading": 2.0,     # strong; headings carry section semantics
        "chunk_text": 1.5,  # main content
    }

    filter_dict = {}

    if patient_id is not None:
        filter_dict["patient_id"] = patient_id

    if doc_types is not None:
        filter_dict["doc_type"] = doc_types

    if is_oncology is not None:
        # Adjust type (str/int) to match how is_oncology was indexed
        filter_dict["is_oncology"] = str(int(bool(is_oncology)))

    results = index.search(
        query=query,
        filter_dict=filter_dict,
        boost_dict=boost_dict,
        num_results=num_results,
    )

    return results


def semantic_search(query, patient_id=None, doc_types=None, is_oncology=None, num_results=5):
    """Semantic search over pre-computed embeddings."""
    query_vector = embedder.encode(query, normalize=True)
    scores = vector_embeddings @ query_vector  # dot product

    filtered = []
    for doc, score in zip(vector_documents, scores):
        if patient_id is not None and doc.get("patient_id") != patient_id:
            continue

        if doc_types is not None:
            if isinstance(doc_types, str):
                allowed_doc_types = {doc_types}
            else:
                allowed_doc_types = set(doc_types)

            if doc.get("doc_type") not in allowed_doc_types:
                continue

        if is_oncology is not None:
            if int(doc.get("is_oncology", 0)) != int(bool(is_oncology)):
                continue

        doc_with_score = dict(doc)
        doc_with_score["semantic_score"] = float(score)
        filtered.append(doc_with_score)

    filtered = sorted(filtered, key=lambda x: x["semantic_score"], reverse=True)
    return filtered[:num_results]


def rrf(result_lists, k=60, num_results=5):
    """Reciprocal Rank Fusion."""
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
    """Hybrid lexical + semantic search with RRF."""
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
    """Format retrieved chunks into a text context block."""
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


def build_prompt(query, search_results):
    context = build_context(search_results)
    prompt = PROMPT_TEMPLATE.format(question=query, context=context).strip()
    return prompt


def calculate_openai_cost(model, tokens):
    """
    Calculate OpenAI API cost in USD for a single call.

    Parameters
    ----------
    model : str
        Model name, e.g. "gpt-5.4-mini".
    tokens : dict
        Must contain 'input_tokens' and 'output_tokens' (ints).

    Returns
    -------
    dict with keys 'input_cost', 'output_cost', 'total_cost'.
    """

    pricing = {
        "gpt-5.4-mini": {
            "input_price_per_million": 0.75,   # USD per 1M input tokens
            "output_price_per_million": 4.50,  # USD per 1M output tokens
        },
        # Add other models here if needed.
    }

    info = pricing.get(model)
    if info is None:
        return {
            "input_cost": 0.0,
            "output_cost": 0.0,
            "total_cost": 0.0,
        }

    input_tokens = tokens.get("input_tokens", 0)
    output_tokens = tokens.get("output_tokens", 0)

    input_cost = (input_tokens / 1_000_000) * info["input_price_per_million"]
    output_cost = (output_tokens / 1_000_000) * info["output_price_per_million"]
    total_cost = input_cost + output_cost

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }


def llm(prompt, model="gpt-5.4-mini"):
    """Call the LLM to answer a question given a prompt."""
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "developer",
                "content": [
                    {
                        "type": "input_text",
                        "text": INSTRUCTIONS,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt,
                    }
                ],
            },
        ],
    )

    answer = response.output_text.strip()

    usage = getattr(response, "usage", None)
    if usage is None:
        token_stats = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
    else:
        input_tokens = getattr(usage, "input_tokens", 0)
        output_tokens = getattr(usage, "output_tokens", 0)
        total_tokens = getattr(usage, "total_tokens", input_tokens + output_tokens)

        token_stats = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }

    cost_info = calculate_openai_cost(model, token_stats)

    return {
        "answer": answer,
        "token_stats": token_stats,
        "cost": cost_info,
        "raw_response": response,
    }


def evaluate_relevance(question, answer, context, search_type, model="gpt-5.4-mini"):
    """LLM-as-a-judge for relevance and groundedness."""
    prompt = EVALUATION_PROMPT_TEMPLATE.format(
        question=question,
        answer=answer,
        context=context,
        search_type=search_type,
    )

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "developer",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Return valid JSON only. Do not include markdown or code fences.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt,
                    }
                ],
            },
        ],
    )

    evaluation_text = response.output_text.strip()

    usage = getattr(response, "usage", None)
    if usage is None:
        token_stats = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
    else:
        input_tokens = getattr(usage, "input_tokens", 0)
        output_tokens = getattr(usage, "output_tokens", 0)
        total_tokens = getattr(usage, "total_tokens", input_tokens + output_tokens)

        token_stats = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }

    cost_info = calculate_openai_cost(model, token_stats)

    try:
        evaluation = json.loads(evaluation_text)
    except json.JSONDecodeError:
        evaluation = {
            "Relevance": "UNKNOWN",
            "Groundedness": "UNKNOWN",
            "Explanation": "Failed to parse evaluation",
        }

    return {
        "evaluation": evaluation,
        "token_stats": token_stats,
        "cost": cost_info,
        "raw_text": evaluation_text,
    }


def rag(
    query,
    patient_id=None,
    doc_types=None,
    is_oncology=None,
    num_results=5,
    model="gpt-5.4-mini",
    search_type="lexical",
):
    """End-to-end RAG: retrieve → build prompt → answer → evaluate."""
    t0 = time()

    if search_type == "lexical":
        search_results = search(
            query=query,
            patient_id=patient_id,
            doc_types=doc_types,
            is_oncology=is_oncology,
            num_results=num_results,
        )
    elif search_type == "semantic":
        search_results = semantic_search(
            query=query,
            patient_id=patient_id,
            doc_types=doc_types,
            is_oncology=is_oncology,
            num_results=num_results,
        )
    elif search_type == "hybrid":
        search_results = hybrid_search(
            query=query,
            patient_id=patient_id,
            doc_types=doc_types,
            is_oncology=is_oncology,
            num_results=num_results,
        )
    else:
        raise ValueError("search_type must be one of: lexical, semantic, hybrid")

    context = build_context(search_results)
    prompt = build_prompt(query, search_results)

    llm_result = llm(prompt, model=model)
    answer = llm_result["answer"]
    token_stats = llm_result["token_stats"]
    answer_cost = llm_result["cost"]

    eval_result = evaluate_relevance(
        question=query,
        answer=answer,
        context=context,
        search_type=search_type,
        model=model,
    )
    evaluation = eval_result["evaluation"]
    eval_token_stats = eval_result["token_stats"]
    eval_cost = eval_result["cost"]

    took = time() - t0

    answer_data = {
        "answer": answer,
        "model_used": model,
        "search_type": search_type,
        "response_time": took,
        "relevance": evaluation.get("Relevance", "UNKNOWN"),
        "groundedness": evaluation.get("Groundedness", "UNKNOWN"),
        "evaluation_explanation": evaluation.get("Explanation", "Failed to parse evaluation"),
        "search_results": search_results,

        "prompt_tokens": token_stats["input_tokens"],
        "completion_tokens": token_stats["output_tokens"],
        "total_tokens": token_stats["total_tokens"],

        "answer_input_cost_usd": answer_cost["input_cost"],
        "answer_output_cost_usd": answer_cost["output_cost"],
        "answer_total_cost_usd": answer_cost["total_cost"],

        "eval_prompt_tokens": eval_token_stats["input_tokens"],
        "eval_completion_tokens": eval_token_stats["output_tokens"],
        "eval_total_tokens": eval_token_stats["total_tokens"],

        "eval_input_cost_usd": eval_cost["input_cost"],
        "eval_output_cost_usd": eval_cost["output_cost"],
        "eval_total_cost_usd": eval_cost["total_cost"],

        "overall_total_cost_usd": answer_cost["total_cost"] + eval_cost["total_cost"],
    }

    return answer_data