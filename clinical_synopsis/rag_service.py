import json
from datetime import date
import re
from typing import Any
from functools import lru_cache

from openai import OpenAI

try:
    from . import config
    from . import retrieval
except ImportError:  # Allows running the file in notebook/script contexts.
    import config
    import retrieval

client = OpenAI()

BASE_INSTRUCTIONS = config.BASE_INSTRUCTIONS
PROMPT_MODES = config.PROMPT_MODES
QUESTION_TYPES = config.QUESTION_TYPES
PROMPT_TEMPLATE = config.PROMPT_TEMPLATE
EVALUATION_PROMPT_TEMPLATE = config.EVALUATION_PROMPT_TEMPLATE


def _calculate_age(dob_str: str | None, as_of: date | None = None) -> int | None:
    if not dob_str:
        return None
    as_of = as_of or date.today()
    dob = date.fromisoformat(dob_str[:10])
    return as_of.year - dob.year - ((as_of.month, as_of.day) < (dob.month, dob.day))


def _extract_patient_identity(search_results: list[dict[str, Any]]) -> tuple[str | None, str | None, int | None, str | None]:
    patient_name = None
    patient_dob = None
    patient_gender = None

    for doc in search_results:
        title = doc.get("title", "")
        if title.startswith("Patient Overview:"):
            patient_name = title.split(":", 1)[1].strip()

        chunk_text = doc.get("chunk_text", "")
        dob_match = re.search(r"^\s*-\s*Birth date:\s*(\d{4}-\d{2}-\d{2})", chunk_text, re.M)
        gender_match = re.search(r"^\s*-\s*Gender:\s*([A-Za-z]+)", chunk_text, re.M)

        if dob_match:
            patient_dob = dob_match.group(1)
        if gender_match:
            patient_gender = gender_match.group(1)

    patient_age = _calculate_age(patient_dob)
    return patient_name, patient_dob, patient_age, patient_gender


def _search_patient_identity(search_type: str, patient_id: str, num_results: int = 10):
    identity_results = _search(
        search_type=search_type,
        query="patient overview",
        patient_id=patient_id,
        doc_types=["patient_overview"],
        num_results=num_results,
    )
    return _extract_patient_identity(identity_results)


@lru_cache(maxsize=1)
def get_patient_catalog() -> list[dict[str, Any]]:
    """
    Return one display-ready identity record per patient found in the index.

    Uses patient_overview chunks because they contain:
    - patient_id
    - title: "Patient Overview: <name>"
    - Identity chunk with birth date and gender
    """
    _, documents = retrieval.load_vector_index()

    docs_by_patient: dict[str, list[dict[str, Any]]] = {}

    for doc in documents:
        if doc.get("doc_type") != "patient_overview":
            continue

        patient_id = doc.get("patient_id")
        if not patient_id:
            continue

        docs_by_patient.setdefault(patient_id, []).append(doc)

    patients = []

    for patient_id, patient_docs in docs_by_patient.items():
        patient_name, patient_dob, patient_age, patient_gender = (
            _extract_patient_identity(patient_docs)
        )

        patient_name = patient_name or "Unnamed patient"

        label_parts = [patient_name]

        if patient_dob:
            label_parts.append(f"DOB {patient_dob}")

        if patient_gender:
            label_parts.append(patient_gender.title())

        # Keep the ID visible because synthetic names could potentially repeat.
        label_parts.append(f"ID {patient_id[:8]}…")

        patients.append({
            "patient_id": patient_id,
            "patient_name": patient_name,
            "patient_dob": patient_dob,
            "patient_age_years": patient_age,
            "patient_gender": patient_gender,
            "label": " — ".join(label_parts),
        })

    return sorted(
        patients,
        key=lambda patient: (
            patient["patient_name"].lower(),
            patient["patient_id"],
        ),
    )


def _search(
    search_type: str,
    query: str,
    patient_id: str,
    doc_types: list[str] | None,
    num_results: int,
) -> list[dict[str, Any]]:
    if search_type == "lexical":
        return retrieval.search(
            query=query,
            patient_id=patient_id,
            doc_types=doc_types,
            is_oncology=None,
            num_results=num_results,
        )

    if search_type == "semantic":
        return retrieval.semantic_search(
            query=query,
            patient_id=patient_id,
            doc_types=doc_types,
            is_oncology=None,
            num_results=num_results,
        )

    if search_type == "hybrid":
        return retrieval.hybrid_search(
            query=query,
            patient_id=patient_id,
            doc_types=doc_types,
            is_oncology=None,
            num_results=num_results,
        )

    raise ValueError("search_type must be one of: lexical, semantic, hybrid")


def _has_heading(docs: list[dict[str, Any]], heading: str) -> bool:
    return any(doc.get("heading") == heading for doc in docs)


def _dedupe_by_chunk_id(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    merged = []

    for doc in docs:
        chunk_id = doc.get("chunk_id")
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        merged.append(doc)

    return merged


# # this version sends BASE_INSTRUCTIONS which are already sent in the llm() function, so we don't need to include them again in the prompt
# def build_prompt_with_mode(question: str, context: str, prompt_mode: str) -> str:
#     extra = PROMPT_MODES.get(prompt_mode, "")
#     system_instructions = BASE_INSTRUCTIONS + "\n\n" + extra

#     prompt = f"""
# {system_instructions}

# CONTEXT:
# {context}

# QUESTION:
# {question}

# ANSWER:
# """.strip()

#     return prompt
def build_prompt_with_mode(question: str, context: str, prompt_mode: str) -> str:
    extra = PROMPT_MODES.get(prompt_mode, "")

    return f"""
{extra}

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
""".strip()


def build_prompt(query, search_results):
    context = retrieval.build_context(search_results)
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
                        "text": BASE_INSTRUCTIONS,
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
            "relevance_score": None,
            "groundedness_score": None,
            "overall_score": None,
            "relevance_label": "UNKNOWN",
            "groundedness_label": "UNKNOWN",
            "explanation": "Failed to parse evaluation JSON.",
        }

    relevance_score = evaluation.get("relevance_score")
    groundedness_score = evaluation.get("groundedness_score")

    if relevance_score is not None and groundedness_score is not None:
        overall_score = (float(relevance_score) + float(groundedness_score)) / 2
    else:
        overall_score = None

    return {
        "relevance_score": relevance_score,
        "groundedness_score": groundedness_score,
        "overall_score": overall_score,
        "relevance_label": evaluation.get("relevance_label", "UNKNOWN"),
        "groundedness_label": evaluation.get("groundedness_label", "UNKNOWN"),
        "explanation": evaluation.get("explanation", "No explanation returned."),
        "token_stats": token_stats,
        "cost": cost_info,
        "raw_text": evaluation_text,
    }


def rag_new(
    query: str,
    patient_id: str,
    question_type: str | None = None,
    num_results: int = 5,
    model: str = "gpt-5.4-mini",
    search_type: str = "hybrid",
) -> dict[str, Any]:
    cfg = QUESTION_TYPES.get(question_type, {})
    prompt_mode = cfg.get("prompt_mode", "summary")

    primary_doc_types = cfg.get("doc_types") or cfg.get("doc_types_primary")
    primary_headings = cfg.get("headings") or cfg.get("headings_primary")
    primary_num_results = cfg.get("primary_num_results", num_results)

    overview_results = _search(
        search_type=search_type,
        query=query,
        patient_id=patient_id,
        doc_types=primary_doc_types,
        num_results=primary_num_results,
    )

    if question_type == "patient_overview" and not _has_heading(overview_results, "Medications"):
        medication_results = _search(
            search_type=search_type,
            query="medications",
            patient_id=patient_id,
            doc_types=["patient_overview"],
            num_results=max(num_results, 10),
        )
        medication_results = [
            doc for doc in medication_results
            if doc.get("heading") == "Medications"
        ]
        overview_results = _dedupe_by_chunk_id(overview_results + medication_results)

    if primary_headings:
        overview_filtered = [
            doc for doc in overview_results
            if doc.get("heading") in primary_headings
        ]
        if not overview_filtered:
            overview_filtered = overview_results
    else:
        overview_filtered = overview_results

    conditions_results = []
    if question_type == "conditions":
        conditions_results = _search(
            search_type=search_type,
            query=query,
            patient_id=patient_id,
            doc_types=cfg.get("doc_types_conditions_supplement", ["conditions"]),
            num_results=cfg.get("conditions_num_results", 100),
        )
        conditions_results = _dedupe_by_chunk_id(conditions_results)

    oncology_results = []
    if question_type == "patient_overview":
        oncology_results = _search(
            search_type=search_type,
            query=(
                "Summarize the patient's oncology history, treatments, "
                "and documented response or progression."
            ),
            patient_id=patient_id,
            doc_types=cfg.get("doc_types_onco_fallback", ["oncology_timeline"]),
            num_results=20,
        )
        oncology_results = _dedupe_by_chunk_id(oncology_results)

    if question_type == "conditions":
        recent_context = retrieval.build_context(overview_filtered)
        longitudinal_context = retrieval.build_context(conditions_results)

        context = f"""
RECENT PATIENT OVERVIEW CONDITIONS:
{recent_context}

LONGITUDINAL CONDITIONS RECORDS:
{longitudinal_context}
""".strip()

    elif question_type == "patient_overview":
        overview_context = retrieval.build_context(overview_filtered)
        oncology_context = retrieval.build_context(oncology_results)

        context = f"""
PATIENT OVERVIEW CONTEXT:
{overview_context}

ONCOLOGY TIMELINE CONTEXT:
{oncology_context}
""".strip()

    else:
        all_results = _dedupe_by_chunk_id(overview_filtered + oncology_results)
        context = retrieval.build_context(all_results)

    prompt = build_prompt_with_mode(
        question=query,
        context=context,
        prompt_mode=prompt_mode,
    )

    llm_out = llm(prompt=prompt, model=model)
    answer = llm_out["answer"]
    token_stats = llm_out["token_stats"]
    cost_info = calculate_openai_cost(model, token_stats)

    patient_name, patient_dob, patient_age, patient_gender = _search_patient_identity(search_type, patient_id)

    return {
        "patient_id": patient_id,
        "patient_name": patient_name,
        "patient_dob": patient_dob,
        "patient_age_years": patient_age,
        "patient_gender": patient_gender,
        "answer": answer,
        "context": context,
        **token_stats,
        **cost_info,
    }