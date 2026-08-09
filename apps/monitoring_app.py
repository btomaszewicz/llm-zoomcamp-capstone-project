import os
import sys
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_FEEDBACK_DB = REPO_ROOT / "data" / "monitoring" / "clinical_synopsis_feedback.db"
DEFAULT_DERIVED_ROOT = REPO_ROOT / "data" / "derived" / "sample50"

FEEDBACK_DB = Path(
    os.getenv("CLINICAL_SYNOPSIS_FEEDBACK_DB", str(DEFAULT_FEEDBACK_DB))
)
DERIVED_ROOT = Path(
    os.getenv("CLINICAL_SYNOPSIS_DERIVED_ROOT", str(DEFAULT_DERIVED_ROOT))
)

EXPECTED_COLUMNS = [
    "id",
    "created_at",
    "patient_id",
    "question",
    "question_type",
    "search_type",
    "model",
    "answer",
    "routing_suggestion",
    "routing_confidence",
    "feedback_score",
    "accuracy_issue",
    "issue_type",
    "comment",
    "input_tokens",
    "output_tokens",
    "total_cost",
]


st.set_page_config(
    page_title="Clinical Synopsis Monitoring",
    page_icon="📊",
    layout="wide",
)


@st.cache_data(ttl=15)
def load_feedback(db_path: str) -> pd.DataFrame:
    path = Path(db_path)

    if not path.exists():
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    try:
        with sqlite3.connect(path) as conn:
            tables = pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='feedback'",
                conn,
            )

            if tables.empty:
                return pd.DataFrame(columns=EXPECTED_COLUMNS)

            feedback = pd.read_sql_query(
                "SELECT * FROM feedback ORDER BY created_at DESC",
                conn,
            )
    except sqlite3.Error as exc:
        raise RuntimeError(f"Could not read feedback database: {exc}") from exc

    for column in EXPECTED_COLUMNS:
        if column not in feedback.columns:
            feedback[column] = None

    feedback["created_at"] = pd.to_datetime(
        feedback["created_at"], errors="coerce", utc=True
    )
    feedback["feedback_score"] = pd.to_numeric(
        feedback["feedback_score"], errors="coerce")
    feedback["total_cost"] = pd.to_numeric(
        feedback["total_cost"], errors="coerce")
    feedback["accuracy_issue"] = pd.to_numeric(
        feedback["accuracy_issue"], errors="coerce").fillna(0).astype(int)

    return feedback[EXPECTED_COLUMNS]


def metric_value(value, fmt="{:.2f}") -> str:
    if pd.isna(value):
        return "—"
    return fmt.format(value)


def render_source_preview(patient_id: str, feedback_id: int) -> None:
    patient_dir = DERIVED_ROOT / patient_id
    overview_path = patient_dir / "patient_overview.md"

    if not overview_path.exists():
        st.info("No patient overview source file was found for this feedback record.")
        return

    with st.expander("View patient overview source", expanded=False):
        st.text_area(
            "Patient overview source",
            value=overview_path.read_text(encoding="utf-8"),
            height=420,
            disabled=True,
            label_visibility="collapsed",
            key=f"monitor_overview_{feedback_id}",
        )


st.title("Clinical Synopsis Monitoring")
st.caption(
    "Review clinician feedback, answer quality signals, routing behavior, and cost. "
    "This interface contains patient-level data and should be access-controlled in deployment."
)

if st.button("Refresh data"):
    load_feedback.clear()

try:
    feedback = load_feedback(str(FEEDBACK_DB))
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

if feedback.empty:
    st.info(
        "No feedback records have been saved yet. Submit feedback from the Clinical Synopsis app "
        "to populate this dashboard."
    )
    st.code(
        "from clinical_synopsis.feedback import save_feedback\n"
        "save_feedback(feedback_event)",
        language="python",
    )
    st.stop()

with st.sidebar:
    st.header("Filters")

    question_types = sorted(feedback["question_type"].dropna().unique())
    selected_question_types = st.multiselect(
        "Question types",
        options=question_types,
        default=question_types,
    )

    search_types = sorted(feedback["search_type"].dropna().unique())
    selected_search_types = st.multiselect(
        "Retrieval methods",
        options=search_types,
        default=search_types,
    )

    models = sorted(feedback["model"].dropna().unique())
    selected_models = st.multiselect(
        "Models",
        options=models,
        default=models,
    )

    rating_options = sorted(
        feedback["feedback_score"].dropna().astype(int).unique(), reverse=True
    )
    selected_ratings = st.multiselect(
        "Feedback scores",
        options=rating_options,
        default=rating_options,
    )

    accuracy_filter = st.selectbox(
        "Clinical accuracy flag",
        options=["All", "Flagged", "Not flagged"],
    )

    issue_types = sorted(feedback["issue_type"].dropna().unique())
    selected_issue_types = st.multiselect(
        "Issue categories",
        options=issue_types,
        default=issue_types,
    )

    patient_search = st.text_input(
        "Patient ID contains",
        placeholder="Optional patient ID fragment",
    ).strip()

filtered = feedback.copy()

if selected_question_types:
    filtered = filtered[filtered["question_type"].isin(selected_question_types)]
else:
    filtered = filtered.iloc[0:0]

if selected_search_types:
    filtered = filtered[filtered["search_type"].isin(selected_search_types)]
else:
    filtered = filtered.iloc[0:0]

if selected_models:
    filtered = filtered[filtered["model"].isin(selected_models)]
else:
    filtered = filtered.iloc[0:0]

if selected_ratings:
    filtered = filtered[
        filtered["feedback_score"].isin(selected_ratings)
    ]
else:
    filtered = filtered.iloc[0:0]

if selected_issue_types:
    filtered = filtered[filtered["issue_type"].isin(selected_issue_types)]
else:
    filtered = filtered.iloc[0:0]

if accuracy_filter == "Flagged":
    filtered = filtered[filtered["accuracy_issue"] == 1]
elif accuracy_filter == "Not flagged":
    filtered = filtered[filtered["accuracy_issue"] == 0]

if patient_search:
    filtered = filtered[
        filtered["patient_id"].fillna("").str.contains(
            patient_search,
            case=False,
            na=False,
        )
    ]

filtered = filtered.sort_values("created_at", ascending=False)
rated = filtered.dropna(subset=["feedback_score"])

average_score = rated["feedback_score"].mean()
low_score_count = int((rated["feedback_score"] <= 2).sum())
low_score_rate = (
    100 * low_score_count / len(rated)
    if len(rated)
    else float("nan")
)
accuracy_flag_count = int((filtered["accuracy_issue"] == 1).sum())
accuracy_flag_rate = (
    100 * accuracy_flag_count / len(filtered)
    if len(filtered)
    else float("nan")
)
total_cost = filtered["total_cost"].sum(min_count=1)

st.subheader("Quality summary")
metric_columns = st.columns(5)
metric_columns[0].metric("Feedback records", f"{len(filtered):,}")
metric_columns[1].metric("Average score", metric_value(average_score))
metric_columns[2].metric("Scores ≤ 2", metric_value(low_score_rate, "{:.1f}%"))
metric_columns[3].metric(
    "Accuracy flags",
    metric_value(accuracy_flag_rate, "{:.1f}%"),
)
metric_columns[4].metric("Estimated cost", metric_value(total_cost, "${:.4f}"))

chart_columns = st.columns(2)

with chart_columns[0]:
    st.subheader("Rating distribution")
    rating_counts = (
        rated["feedback_score"]
        .astype(int)
        .value_counts()
        .reindex([5, 4, 3, 2, 1], fill_value=0)
        .rename_axis("score")
        .to_frame("responses")
    )
    st.bar_chart(rating_counts)

with chart_columns[1]:
    st.subheader("Issue categories")
    issues = filtered[filtered["issue_type"].notna()].copy()
    issues = issues[issues["issue_type"] != "No issue"]

    if issues.empty:
        st.caption("No issue categories selected or recorded.")
    else:
        issue_counts = issues["issue_type"].value_counts().to_frame("responses")
        st.bar_chart(issue_counts)

st.divider()
st.subheader("Feedback records")

TABLE_COLUMNS = [
    "id",
    "created_at",
    "patient_id",
    "question_type",
    "search_type",
    "model",
    "feedback_score",
    "accuracy_issue",
    "issue_type",
    "total_cost",
    "question",
    "comment",
]

st.dataframe(
    filtered[TABLE_COLUMNS],
    hide_index=True,
    use_container_width=True,
    height=360,
    column_config={
        "created_at": st.column_config.DatetimeColumn("Created at", format="YYYY-MM-DD HH:mm"),
        "feedback_score": st.column_config.NumberColumn("Score", format="%d / 5"),
        "accuracy_issue": st.column_config.CheckboxColumn("Accuracy flag"),
        "total_cost": st.column_config.NumberColumn("Cost", format="$%.6f"),
        "question": st.column_config.TextColumn("Question", width="large"),
        "comment": st.column_config.TextColumn("Comment", width="large"),
    },
)

csv_export = filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download filtered feedback as CSV",
    data=csv_export,
    file_name="clinical_synopsis_feedback.csv",
    mime="text/csv",
)

if filtered.empty:
    st.info("No feedback records match the selected filters.")
    st.stop()

st.divider()
st.subheader("Inspect an answer")

record_ids = filtered["id"].tolist()
record_labels = {
    int(row.id): (
        f"#{int(row.id)} — "
        f"{row.created_at.strftime('%Y-%m-%d %H:%M') if pd.notna(row.created_at) else 'unknown time'} — "
        f"{row.question_type or 'unknown type'} — "
        f"score {int(row.feedback_score) if pd.notna(row.feedback_score) else 'not rated'}"
    )
    for row in filtered.itertuples()
}

selected_record_id = st.selectbox(
    "Feedback record",
    options=record_ids,
    format_func=lambda value: record_labels[int(value)],
)

record = filtered[filtered["id"] == selected_record_id].iloc[0]

left, right = st.columns([2, 1])

with left:
    st.markdown("#### Question")
    st.write(record["question"])

    st.markdown("#### Generated answer")
    st.markdown(record["answer"])

    if pd.notna(record["comment"]) and str(record["comment"]).strip():
        st.markdown("#### Clinician comment")
        st.info(record["comment"])

with right:
    st.markdown("#### Review metadata")
    st.write(f"**Score:** {record['feedback_score'] if pd.notna(record['feedback_score']) else 'Not rated'}")
    st.write(f"**Accuracy flagged:** {'Yes' if record['accuracy_issue'] else 'No'}")
    st.write(f"**Issue type:** {record['issue_type'] or 'Not selected'}")
    st.write(f"**Question type:** {record['question_type']}")
    st.write(f"**Router suggestion:** {record['routing_suggestion'] or 'None'}")
    st.write(f"**Router confidence:** {record['routing_confidence'] if pd.notna(record['routing_confidence']) else 'Not recorded'}")
    st.write(f"**Retrieval:** {record['search_type']}")
    st.write(f"**Model:** {record['model']}")
    st.write(f"**Input tokens:** {record['input_tokens'] if pd.notna(record['input_tokens']) else 'Not recorded'}")
    st.write(f"**Output tokens:** {record['output_tokens'] if pd.notna(record['output_tokens']) else 'Not recorded'}")
    st.write(f"**Estimated cost:** ${record['total_cost']:.6f}" if pd.notna(record['total_cost']) else "**Estimated cost:** Not recorded")

render_source_preview(str(record["patient_id"]), int(record["id"]))
