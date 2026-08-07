from __future__ import annotations # NOT NEEDED RIGHT?

from datetime import date
import re
from typing import Any


try:
    from . import rag as rag_module
except ImportError:  # Allows running the file in notebook/script contexts.
    import rag as rag_module


BASE_INSTRUCTIONS = rag_module.INSTRUCTIONS


QUESTION_TYPES = {
    "patient_overview": {
        "prompt_mode": "summary",
        "doc_types_primary": ["patient_overview"],
        "doc_types_med_fallback": ["medications"],
        "doc_types_onco_fallback": ["oncology_timeline"],
        "headings_primary": ["Recent Conditions", "Recent Results", "Procedures", "Medications"],
    },
    "conditions": {
        "prompt_mode": "extract_conditions",
        "doc_types_primary": ["patient_overview"],
        "headings_primary": ["Recent Conditions", "Recent Results"],
        "doc_types_conditions_supplement": ["conditions"],
        "primary_num_results": 20,
        "conditions_num_results": 100,
    },
    "medications": {
        "prompt_mode": "extract_medications",
        "doc_types": ["patient_overview"],
        "headings": ["Medications"],
    },
    "oncology_timeline": {
        "description": "Oncology history and major events",
        "doc_types": ["oncology_timeline", "oncology_timeline_events"],
        "prompt_mode": "summarize_oncology_timeline",
    },
}


PATIENT_OVERVIEW_EXTRA = """
For overview questions, use exactly these four sections and do not add others.

1. **Summary**
- Write exactly 3 sentences.
- Sentence 1: State the most important documented clinical conditions, including cancer history when documented in the oncology timeline.
- Sentence 2: State the current or recent clinically important status. Do not mention social, occupational, environmental, or administrative findings.
- Sentence 3: Give a one-sentence oncology synopsis using ONLY ONCOLOGY TIMELINE CONTEXT. Do not use medications as oncology evidence.

2. **Conditions**
- Include only documented clinical diagnoses and clinically meaningful comorbidities.
- Exclude social, occupational, environmental, administrative, and screening findings.
- Specifically exclude stress, employment status, not in labor force, and reports of violence in the environment.
- Do not reproduce every row from Recent Conditions.
- Preserve active/resolved status exactly.
- Format: **Condition** — status; date: YYYY-MM-DD.
- If none qualify, write: "No qualifying clinical conditions documented."

3. **Medications**
- Give a concise summary of medications explicitly documented as current or active.
- Do not list dose, route, strength, formulation, or duplicate ingredients.
- Group medications used for the same apparent purpose when documented together.
- For multiple active pain medicines, use one bullet named **Active analgesic regimen** and list only the medication names.
- Do not include completed, historical, inactive, or discontinued medications.
- If none are explicitly current or active, write exactly:
  "No current medication is documented in the provided medication snapshot."

4. **Oncology timeline**
- Use ONLY ONCOLOGY TIMELINE CONTEXT.
- Do not use patient_overview conditions, results, procedures, or medications as oncology evidence.
- Summarize diagnosis/staging, treatment episodes, and documented response or progression.
- Combine repeated treatment sessions and repeated identical response findings into date ranges.
- Do not infer remission, cure, recurrence, metastasis, or current cancer status.
- If no oncology timeline context is supplied, write:
  "No oncology timeline information documented."

Rules:
- Use only facts in the supplied context.
- Do not infer missing facts.
- Do not omit any of the four sections.
"""


CONDITIONS_EXTRA = """
For questions about diagnosed conditions:

- Use the supplied context only.
- Preserve each condition or finding's documented status and date.
- Do not infer diagnoses, status, dates, recurrence, remission, or causality.
- Separate diagnoses/disorders from findings according to the wording in the context.
- Entries labelled "(finding)" belong in Findings and social/functional history.
- Entries labelled "(disorder)" and documented diagnoses belong in Diagnoses and disorders.
- Within each section, list entries documented as active first, followed by resolved entries.
- Do not reorder entries further; retain their order from the supplied context within each status group.

Format your answer using exactly these two sections:

**Diagnoses and disorders**
- List documented diagnoses and disorders.
- Each bullet: **Name** — status; date: YYYY-MM-DD.
- If there are no documented diagnoses or disorders in the supplied context, write:
  "No diagnoses or disorders documented."

**Findings and social/functional history**
- List documented findings, including social, occupational, environmental, behavioral,
  and functional findings.
- Each bullet: **Name** — status; date: YYYY-MM-DD.
- If there are no documented findings in the supplied context, write:
  "No findings or social/functional history documented."
"""


MEDICATIONS_EXTRA = """
For questions about medications:

- Treat this as a medication-history extraction and prioritization task based only on the context.
- Use the Medications section from patient_overview.md as the primary source.
- Preserve each medication's documented status exactly. Do NOT describe a medication as current,
  active, ongoing, or discontinued unless that status is explicitly documented.
- If all listed medications are historical or marked completed, explicitly state:
  "No current medication is documented in the provided medication snapshot."
- Prioritize clinically significant therapies over routine, duplicate, short-term, or remote medications.
- In an oncology patient, prioritize documented antineoplastic or endocrine cancer therapies.
- Group all documented historical hormonal contraceptive therapies into exactly one bullet named
  "**Historical hormonal contraception**." This includes oral contraceptives, transdermal contraceptive
  patches, and contraceptive implants when they appear in the context.
- Do NOT create separate bullets for individual historical contraceptive products after grouping them.
- Exclude one-off symptomatic or short-course medications (for example, cold/flu, cough, pain, or
  sleep products) when other clinically significant medication history is present.
- Do NOT list duplicate historical entries for the same medication or medication class. Retain only
  the most recent documented date within a grouped bullet.
- List antineoplastic, endocrine cancer therapy, or other disease-modifying therapy as separate
  medication bullets, even if marked completed.
- Do NOT infer indication, current use, dose, regimen, treatment response, or clinical importance
  beyond what is documented.
- If a date is missing, write "date: not documented"; do not invent one.

Format:
- Begin with one status sentence:
  - If no medication is explicitly current/active: "No current medication is documented in the provided medication snapshot."
  - Otherwise: "Current/recent medications documented in the provided snapshot:"
- Then use one bullet per medication or clinically coherent medication group.
- Each bullet: **Medication or group** — documented status; date or date range in YYYY-MM-DD format; brief factual description only when supported by the context.
- Convert documented timestamps to their calendar date only. Do not include a time, time zone, or infer a date that is not documented.
"""


ONCOLOGY_TIMELINE_EXTRA = """
For questions about oncology history:

- Treat this as an extraction and summarization task based on the context.
- Focus on the patient's main *oncology-related* events (e.g., diagnoses, staging, treatments, progression or response),
  not unrelated conditions or encounters.
- Use oncology-related sections (e.g., oncology_timeline chunks and relevant parts of patient_overview.md) as the primary source.
- List events in strict chronological order by their documented date (earliest first).
- For each event, preserve its type and status exactly as documented (diagnosis, treatment start, progression, response, etc.).
- Do NOT infer oncology events that are not mentioned.
- If a date is missing, say "date: not documented" instead of inventing one.

Format:
- Combine repeated records of the same treatment into one treatment episode with a date range.
- Combine repeated identical response/progression records into one date-range bullet.
- Use 3–6 clinically meaningful bullets, in chronological order.
- Each bullet: **YYYY-MM-DD** or **YYYY-MM-DD to YYYY-MM-DD** — event type; concise factual description.
"""


PROMPT_MODES = {
    "summary": PATIENT_OVERVIEW_EXTRA,
    "extract_conditions": CONDITIONS_EXTRA,
    "extract_medications": MEDICATIONS_EXTRA,
    "summarize_oncology_timeline": ONCOLOGY_TIMELINE_EXTRA,
}


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


def _search(
    search_type: str,
    query: str,
    patient_id: str,
    doc_types: list[str] | None,
    num_results: int,
) -> list[dict[str, Any]]:
    if search_type == "lexical":
        return rag_module.search(
            query=query,
            patient_id=patient_id,
            doc_types=doc_types,
            is_oncology=None,
            num_results=num_results,
        )

    if search_type == "semantic":
        return rag_module.semantic_search(
            query=query,
            patient_id=patient_id,
            doc_types=doc_types,
            is_oncology=None,
            num_results=num_results,
        )

    if search_type == "hybrid":
        return rag_module.hybrid_search(
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


def build_prompt_with_mode(question: str, context: str, prompt_mode: str) -> str:
    extra = PROMPT_MODES.get(prompt_mode, "")
    system_instructions = BASE_INSTRUCTIONS + "\n\n" + extra

    prompt = f"""
{system_instructions}

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
""".strip()

    return prompt


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
        recent_context = rag_module.build_context(overview_filtered)
        longitudinal_context = rag_module.build_context(conditions_results)

        context = f"""
RECENT PATIENT OVERVIEW CONDITIONS:
{recent_context}

LONGITUDINAL CONDITIONS RECORDS:
{longitudinal_context}
""".strip()

    elif question_type == "patient_overview":
        overview_context = rag_module.build_context(overview_filtered)
        oncology_context = rag_module.build_context(oncology_results)

        context = f"""
PATIENT OVERVIEW CONTEXT:
{overview_context}

ONCOLOGY TIMELINE CONTEXT:
{oncology_context}
""".strip()

    else:
        all_results = _dedupe_by_chunk_id(overview_filtered + oncology_results)
        context = rag_module.build_context(all_results)

    prompt = build_prompt_with_mode(
        question=query,
        context=context,
        prompt_mode=prompt_mode,
    )

    llm_out = rag_module.llm(prompt=prompt, model=model)
    answer = llm_out["answer"]
    token_stats = llm_out["token_stats"]
    cost_info = rag_module.calculate_openai_cost(model, token_stats)

    patient_name, patient_dob, patient_age, patient_gender = _extract_patient_identity(overview_results)

    return {
        "patient_id": patient_id,
        "patient_name": patient_name,
        "patient_dob": patient_dob,
        "patient_age_years": patient_age,
        "patient_gender": patient_gender,
        "answer": answer,
        **token_stats,
        **cost_info,
    }