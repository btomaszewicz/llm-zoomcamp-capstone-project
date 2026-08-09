import os

import streamlit as st

from clinical_synopsis.question_router import route_question
from clinical_synopsis.rag_service import rag_new


QUESTION_TYPE_LABELS = {
    "patient_overview": "Patient overview",
    "conditions": "Conditions",
    "medications": "Medications",
    "oncology_timeline": "Oncology timeline",
}

QUESTION_TYPE_OPTIONS = list(QUESTION_TYPE_LABELS)


st.set_page_config(
    page_title="Clinical Synopsis",
    page_icon="🩺",
    layout="centered",
)

st.title("Clinical Synopsis")
st.caption("Record-grounded clinical summaries. Verify important information against the source record.")

if not os.environ.get("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is not configured for this Streamlit process.")
    st.stop()

with st.sidebar:
    st.header("Patient")
    patient_id = st.text_input(
        "Patient ID",
        placeholder="e.g. f203e11d-5573-1624-69b8-af8436987b3e",
    ).strip()

    st.header("Answer settings")
    search_type = st.selectbox(
        "Retrieval method",
        options=["hybrid", "semantic", "lexical"],
        index=0,
        help="Hybrid is the default. Use semantic or lexical only when comparing retrieval behavior.",
    )
    model = st.selectbox("Model", options=["gpt-5.4-mini"], index=0)

question = st.text_area(
    "Question",
    placeholder="For example: Summarize this patient's oncology history.",
    height=110,
)

route = route_question(question) if question.strip() else None

if route and route.question_type:
    suggested_type = route.question_type
    suggested_index = QUESTION_TYPE_OPTIONS.index(suggested_type)
    st.info(
        f"Suggested question type: **{QUESTION_TYPE_LABELS[suggested_type]}** "
        f"(routing confidence: {route.confidence:.0%})."
    )
else:
    suggested_index = 0
    if question.strip():
        st.warning("I could not confidently classify this question. Please choose the question type.")

question_type = st.selectbox(
    "Question type",
    options=QUESTION_TYPE_OPTIONS,
    index=suggested_index,
    format_func=lambda value: QUESTION_TYPE_LABELS[value],
    help="You can override the suggested question type at any time.",
)

ask = st.button("Generate synopsis", type="primary", use_container_width=True)

if ask:
    if not patient_id:
        st.error("Enter a patient ID.")
        st.stop()
    if not question.strip():
        st.error("Enter a question.")
        st.stop()

    try:
        with st.spinner("Retrieving the patient record and generating an answer..."):
            result = rag_new(
                query=question.strip(),
                patient_id=patient_id,
                question_type=question_type,
                search_type=search_type,
                model=model,
            )
    except Exception as exc:
        st.error("The synopsis could not be generated.")
        st.exception(exc)
        st.stop()

    patient_name = result.get("patient_name") or "Patient"
    dob = result.get("patient_dob") or "Not documented"
    age = result.get("patient_age_years")
    gender = result.get("patient_gender") or "Not documented"

    st.subheader(patient_name)
    demographics = [f"DOB: {dob}", f"Gender: {gender}"]
    if age is not None:
        demographics.insert(1, f"Age: {age}")
    st.caption(" · ".join(demographics))

    st.markdown(result["answer"])

    with st.expander("Answer details"):
        st.write(f"Question type: {QUESTION_TYPE_LABELS[question_type]}")
        st.write(f"Retrieval method: {search_type}")
        st.write(f"Model: {model}")
        st.write(f"Input tokens: {result.get('input_tokens', 0):,}")
        st.write(f"Output tokens: {result.get('output_tokens', 0):,}")
        st.write(f"Estimated cost: ${result.get('total_cost', 0.0):.6f}")

    if route:
        with st.expander("Routing details"):
            st.write(f"Suggested type: {route.question_type or 'Clarification required'}")
            st.write(f"Routing confidence: {route.confidence:.0%}")
            st.json(route.scores)
