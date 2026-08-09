
PROMPT_TEMPLATE = """
QUESTION: {question}

CONTEXT:
{context}
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


BASE_INSTRUCTIONS = """
Your task is to answer questions about a patient's clinical record
based only on the provided context.

Use the context to find relevant information and provide accurate answers.
If the answer is not found in the context, respond with "I don't know."

Do not make up facts that are not supported by the context.
When possible, mention the document type and date information that support the answer.
""".strip()


PATIENT_OVERVIEW_EXTRA = """
For overview questions, use exactly these four sections and do not add others.

1. **Summary:**
- Write exactly 3 sentences.
- Sentence 1: State the most important documented clinical conditions, including cancer history when documented in the oncology timeline.
- Sentence 2: State the current or recent clinically important status. Do not mention social, occupational, environmental, or administrative findings.
- Sentence 3: Give a one-sentence oncology synopsis using ONLY ONCOLOGY TIMELINE CONTEXT. Do not use medications as oncology evidence.

2. **Active Conditions:**
- Include only documented clinical diagnoses and clinically meaningful comorbidities that are listed as active.
- Exclude social, occupational, environmental, administrative, and screening findings.
- Specifically exclude stress, employment status, not in labor force, and reports of violence in the environment.
- Do not reproduce every row from Recent Conditions.
- Preserve active/resolved status exactly.
- Format: **Condition** — status; date: YYYY-MM-DD.
- If none qualify, write: "No qualifying clinical conditions documented."

3. **Medications:**
- Give a concise summary of medications explicitly documented as current or active.
- Do not list dose, route, strength, formulation, or duplicate ingredients.
- Group medications used for the same apparent purpose when documented together.
- For multiple active pain medicines, use one bullet named **Active analgesic regimen** and list only the medication names.
- Do not include completed, historical, inactive, or discontinued medications.
- If none are explicitly current or active, write exactly:
  "No current medication is documented in the provided medication snapshot."

4. **Oncology timeline:**
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
""".strip()


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
""".strip()


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
""".strip()


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
""".strip()


PROMPT_MODES = {
    "summary": PATIENT_OVERVIEW_EXTRA,
    "extract_conditions": CONDITIONS_EXTRA,
    "extract_medications": MEDICATIONS_EXTRA,
    "summarize_oncology_timeline": ONCOLOGY_TIMELINE_EXTRA,
}


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


__all__ = [
    "BASE_INSTRUCTIONS",
    "QUESTION_TYPES",
    "PROMPT_MODES",
    "PATIENT_OVERVIEW_EXTRA",
    "CONDITIONS_EXTRA",
    "MEDICATIONS_EXTRA",
    "ONCOLOGY_TIMELINE_EXTRA",
]