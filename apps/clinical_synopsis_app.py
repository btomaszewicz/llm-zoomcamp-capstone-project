import os
import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clinical_synopsis.question_router import route_question
from clinical_synopsis.rag_service import rag_new
from clinical_synopsis.feedback import save_feedback


QUESTION_TYPE_LABELS = {
    "patient_overview": "Patient overview",
    "conditions": "Conditions",
    "medications": "Medications",
    "oncology_timeline": "Oncology timeline",
}

QUESTION_TYPE_OPTIONS = list(QUESTION_TYPE_LABELS)

DEFAULT_DERIVED_ROOT = REPO_ROOT / "data" / "derived" / "sample50"

DERIVED_ROOT = Path(
    os.getenv(
        "CLINICAL_SYNOPSIS_DERIVED_ROOT",
        str(DEFAULT_DERIVED_ROOT),
    )
)


def render_source_documents(patient_id: str) -> None:
    patient_dir = DERIVED_ROOT / patient_id
    overview_path = patient_dir / "patient_overview.md"

    # Show the readable overview first
    if overview_path.exists():
        with st.expander("View patient overview", expanded=False):
            st.markdown(overview_path.read_text(encoding="utf-8"))
    else:
        st.info("No patient overview is available for this patient.")

    # Keep detailed structured source files as downloads
    source_files = [
        ("Patient overview", "patient_overview.md", "text/markdown"),
        ("Conditions", "conditions.csv", "text/csv"),
        ("Medications", "medications.csv", "text/csv"),
        ("Oncology timeline", "oncology_timeline.md", "text/markdown"),
        ("Oncology timeline events", "oncology_timeline_events.csv", "text/csv"),
        ("Procedures", "procedures.csv", "text/csv"),
        ("Diagnostic reports", "diagnostic_reports.csv", "text/csv"),
        ("Encounters", "encounters.csv", "text/csv"),
    ]

    available_sources = [
        (label, patient_dir / filename, mime_type)
        for label, filename, mime_type in source_files
        if (patient_dir / filename).exists()
    ]

    if not available_sources:
        st.info("No source documents are available for this patient.")
        return

    st.caption("Download detailed source records:")

    columns = st.columns(2)

    for index, (label, file_path, mime_type) in enumerate(available_sources):
        with columns[index % 2]:
            st.download_button(
                label=f"Download {label}",
                data=file_path.read_bytes(),
                file_name=file_path.name,
                mime=mime_type,
                use_container_width=True,
                key=f"{patient_id}_{file_path.name}",
            )

    overview_path = patient_dir / "patient_overview.md"

    # # if overview_path.exists():
    # #     with st.expander("View patient overview"):
    # #         st.markdown(overview_path.read_text(encoding="utf-8"))
    # if overview_path.exists():
    #     with st.expander("View patient overview"):
    #         try:
    #             st.markdown(overview_path.read_text(encoding="utf-8"))
    #         except OSError:
    #             st.warning("The patient overview could not be read.")



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

    st.divider()
    st.subheader("Source documents")
    # st.caption(
    #     "Download the underlying patient summary and structured records "
    #     "to review the evidence behind this answer."
    # )

    render_source_documents(patient_id)

    # user feedback
    st.divider()
    st.subheader("Answer feedback")

    feedback_score = st.radio(
        "Was this answer useful and accurate?",
        options=[5, 4, 3, 2, 1],
        horizontal=True,
        format_func=lambda score: {
            5: "5 — Excellent",
            4: "4 — Good",
            3: "3 — Usable with changes",
            2: "2 — Poor",
            1: "1 — Unusable",
        }[score],
    )

    accuracy_issue = st.checkbox(
        "Potential clinical accuracy issue",
        help="Select this if the answer appears inaccurate, unsupported, or potentially misleading.",
    )

    issue_type = st.selectbox(
        "Issue type",
        options=[
            "No issue",
            "Incorrect fact",
            "Missing important information",
            "Too much or irrelevant information",
            "Wrong question type",
            "Unclear format or wording",
            "Other",
        ],
    )

    feedback_comment = st.text_area(
        "Optional comment",
        placeholder="For example: Breast-cancer treatment history was omitted.",
        height=90,
    )

    if st.button("Submit feedback"):
        feedback_event = {
            "patient_id": patient_id,
            "question": question.strip(),
            "question_type": question_type,
            "search_type": search_type,
            "model": model,
            "answer": result["answer"],
            "routing_suggestion": route.question_type if route else None,
            "routing_confidence": route.confidence if route else None,
            "feedback_score": feedback_score,
            "accuracy_issue": accuracy_issue,
            "issue_type": issue_type,
            "comment": feedback_comment.strip(),
            "input_tokens": result.get("input_tokens"),
            "output_tokens": result.get("output_tokens"),
            "total_cost": result.get("total_cost"),
        }

        save_feedback(feedback_event)
        st.success("Feedback saved. Thank you.")

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
