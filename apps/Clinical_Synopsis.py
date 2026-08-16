import os
import sys
from pathlib import Path
import pandas as pd
import time

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clinical_synopsis.question_router import route_question
from clinical_synopsis.rag_service import get_patient_catalog, rag_new, evaluate_relevance
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
    oncology_timeline_path = patient_dir / "oncology_timeline.md"

    # 1. Readable patient overview
    if overview_path.exists():
        with st.expander("View patient overview", expanded=False):
            overview_text = overview_path.read_text(encoding="utf-8")

            st.text_area(
                "Patient overview record",
                value=overview_text,
                height=420,
                disabled=True,
                label_visibility="collapsed",
                key=f"{patient_id}_patient_overview",
            )

    # 2. Readable oncology timeline, only when available
    if oncology_timeline_path.exists():
        with st.expander("View oncology timeline", expanded=False):
            timeline_text = oncology_timeline_path.read_text(encoding="utf-8")

            st.text_area(
                "Oncology timeline record",
                value=timeline_text,
                height=320,
                disabled=True,
                label_visibility="collapsed",
                key=f"{patient_id}_oncology_timeline",
            )

    # 3. Interactive previews of structured source records
    csv_files = [
        ("Conditions", "conditions.csv"),
        ("Medications", "medications.csv"),
        ("Oncology timeline events", "oncology_timeline_events.csv"),
        ("Procedures", "procedures.csv"),
        ("Diagnostic reports", "diagnostic_reports.csv"),
        ("Encounters", "encounters.csv"),
    ]

    available_csvs = [
        (label, patient_dir / filename)
        for label, filename in csv_files
        if (patient_dir / filename).exists()
    ]

    if not overview_path.exists() and not available_csvs:
        st.info("No source documents are available for this patient.")
        return

    if available_csvs:
        st.caption("View detailed structured source records:")

    for label, file_path in available_csvs:
        with st.expander(f"View {label}", expanded=False):
            try:
                df = pd.read_csv(file_path)

                st.caption(f"{len(df):,} records")

                st.dataframe(
                    df,
                    hide_index=True,
                    use_container_width=True,
                    height=320,
                )
            except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
                st.warning(f"Could not display {file_path.name}: {exc}")



st.set_page_config(
    page_title="Clinical Synopsis",
    page_icon="🩺",
    layout="centered",
)

if "response" not in st.session_state:
    st.session_state.response = None

if "feedback_message" not in st.session_state:
    st.session_state.feedback_message = None

st.title("Clinical Synopsis")
st.caption("Generate clinical summaries with supporting evidence from available patient records.")

if not os.environ.get("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is not configured for this Streamlit process.")
    st.stop()

with st.sidebar:
    # st.header("Patient")
    # patient_id = st.text_input(
    #     "Patient ID",
    #     placeholder="e.g. f203e11d-5573-1624-69b8-af8436987b3e",
    # ).strip()
    st.header("Synthetic patient")
    patients = get_patient_catalog()
    patient_options = [""] + [
        patient["patient_id"]
        for patient in patients
    ]
    patient_labels = {
        patient["patient_id"]: patient["label"]
        for patient in patients
    }
    selected_patient_id = st.selectbox(
        "Select a synthetic patient",
        options=patient_options,
        format_func=lambda value: (
            "Select a synthetic patient..."
            if not value
            else patient_labels[value]
        ),
    )
    patient_id = selected_patient_id

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
    placeholder="For example: Provide an overview of this patient's medical background and current status.",
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
        st.error("Select a synthetic patient.")
        st.stop()

    if not question.strip():
        st.error("Enter a question.")
        st.stop()

    try:
        with st.spinner("Retrieving the patient record and generating an answer..."):
            started_at = time.perf_counter()

            result = rag_new(
                query=question.strip(),
                patient_id=patient_id,
                question_type=question_type,
                search_type=search_type,
                model=model,
            )

            judge_result = evaluate_relevance(
                question=question.strip(),
                answer=result["answer"],
                context=result["context"],
                search_type=search_type,
                model=model,
            )
            
            latency_seconds = time.perf_counter() - started_at
    except Exception as exc:
        st.error("The synopsis could not be generated.")
        st.exception(exc)
        st.stop()

    # Store a snapshot. It survives future widget-triggered reruns.
    st.session_state.response = {
        "patient_id": patient_id,
        "question": question.strip(),
        "question_type": question_type,
        "search_type": search_type,
        "model": model,
        "answer": result["answer"],
        "patient_name": result.get("patient_name") or "Patient",
        "patient_dob": result.get("patient_dob") or "Not documented",
        "patient_age_years": result.get("patient_age_years"),
        "patient_gender": result.get("patient_gender") or "Not documented",
        "routing_suggestion": route.question_type if route else None,
        "routing_confidence": route.confidence if route else None,
        "routing_scores": route.scores if route else None,
        "input_tokens": result.get("input_tokens"),
        "output_tokens": result.get("output_tokens"),
        "total_cost": result.get("total_cost"),
        "latency_seconds": latency_seconds,
        "judge_relevance_score": judge_result["relevance_score"],
        "judge_groundedness_score": judge_result["groundedness_score"],
        "judge_overall_score": judge_result["overall_score"],
        "judge_relevance_label": judge_result["relevance_label"],
        "judge_groundedness_label": judge_result["groundedness_label"],
        "judge_explanation": judge_result["explanation"],
        "judge_input_tokens": judge_result["token_stats"]["input_tokens"],
        "judge_output_tokens": judge_result["token_stats"]["output_tokens"],
        "judge_total_cost": judge_result["cost"]["total_cost"],
    }

    # Do not show a “feedback saved” message for a newly generated answer.
    st.session_state.feedback_message = None


# Note: this section is deliberately OUTSIDE `if ask:`.
# It remains visible when the user clicks any feedback widget.
response = st.session_state.response

if response is not None:
    patient_id = response["patient_id"]

    st.subheader(response["patient_name"])

    demographics = [
        f"DOB: {response['patient_dob']}",
        f"Gender: {response['patient_gender']}",
    ]

    if response["patient_age_years"] is not None:
        demographics.insert(1, f"Age: {response['patient_age_years']}")

    st.caption(" · ".join(demographics))

    st.markdown(response["answer"])

    st.divider()
    st.subheader("Available electronic health records")
    st.caption(f"Patient: {response['patient_name']}")
    render_source_documents(patient_id)

    st.divider()
    st.subheader("Answer feedback")

    if st.session_state.feedback_message:
        st.success(st.session_state.feedback_message)

    # The form prevents a rerun each time a rating, checkbox, or comment changes.
    with st.form(
        key=f"feedback_form_{patient_id}_{hash(response['question'])}",
        clear_on_submit=False,
    ):
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
            help=(
                "Select this if the answer appears inaccurate, unsupported, "
                "or potentially misleading."
            ),
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
            placeholder=(
                "For example: Breast-cancer treatment history was omitted."
            ),
            height=90,
        )

        submit_feedback = st.form_submit_button(
            "Submit feedback",
            type="primary",
        )

    if submit_feedback:
        feedback_event = {
            "patient_id": response["patient_id"],
            "question": response["question"],
            "question_type": response["question_type"],
            "search_type": response["search_type"],
            "model": response["model"],
            "answer": response["answer"],
            "routing_suggestion": response["routing_suggestion"],
            "routing_confidence": response["routing_confidence"],
            "feedback_score": feedback_score,
            "accuracy_issue": int(accuracy_issue),
            "issue_type": issue_type,
            "comment": feedback_comment.strip() or None,
            "input_tokens": response["input_tokens"],
            "output_tokens": response["output_tokens"],
            "total_cost": response["total_cost"],
            "latency_seconds": response["latency_seconds"],
            "judge_relevance_score": response["judge_relevance_score"],
            "judge_groundedness_score": response["judge_groundedness_score"],
            "judge_overall_score": response["judge_overall_score"],
            "judge_relevance_label": response["judge_relevance_label"],
            "judge_groundedness_label": response["judge_groundedness_label"],
            "judge_explanation": response["judge_explanation"],
            "judge_input_tokens": response["judge_input_tokens"],
            "judge_output_tokens": response["judge_output_tokens"],
            "judge_total_cost": response["judge_total_cost"],
        }

        try:
            feedback_id = save_feedback(feedback_event)
            st.session_state.feedback_message = (
                f"Feedback saved successfully — record #{feedback_id}."
            )
            st.success(st.session_state.feedback_message)

        except Exception as exc:
            st.error("Feedback could not be saved.")
            st.exception(exc)

    with st.expander("Answer details"):
        st.write(
            f"Question type: "
            f"{QUESTION_TYPE_LABELS[response['question_type']]}"
        )
        st.write(f"Retrieval method: {response['search_type']}")
        st.write(f"Model: {response['model']}")
        st.write(
            f"Input tokens: {response['input_tokens'] or 0:,}"
        )
        st.write(
            f"Output tokens: {response['output_tokens'] or 0:,}"
        )
        st.write(
            f"Estimated cost: ${response['total_cost'] or 0.0:.6f}"
        )

    if response["routing_suggestion"] or response["routing_scores"]:
        with st.expander("Routing details"):
            st.write(
                "Suggested type: "
                f"{response['routing_suggestion'] or 'Clarification required'}"
            )

            if response["routing_confidence"] is not None:
                st.write(
                    "Routing confidence: "
                    f"{response['routing_confidence']:.0%}"
                )

            if response["routing_scores"] is not None:
                st.json(response["routing_scores"])