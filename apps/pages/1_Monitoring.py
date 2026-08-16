import os
import sys
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_FEEDBACK_DB = (
    REPO_ROOT / "data" / "monitoring" / "clinical_synopsis_feedback.db"
)
DEFAULT_DERIVED_ROOT = REPO_ROOT / "data" / "derived" / "sample50"

FEEDBACK_DB = Path(os.getenv("CLINICAL_SYNOPSIS_FEEDBACK_DB", str(DEFAULT_FEEDBACK_DB)))
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
    "latency_seconds",
    "judge_relevance_score",
    "judge_groundedness_score",
    "judge_overall_score",
    "judge_relevance_label",
    "judge_groundedness_label",
    "judge_explanation",
    "judge_input_tokens",
    "judge_output_tokens",
    "judge_total_cost",
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
        feedback["feedback_score"], errors="coerce"
    )
    feedback["total_cost"] = pd.to_numeric(feedback["total_cost"], errors="coerce")
    feedback["latency_seconds"] = pd.to_numeric(
        feedback["latency_seconds"], errors="coerce"
    )
    feedback["accuracy_issue"] = (
        pd.to_numeric(feedback["accuracy_issue"], errors="coerce").fillna(0).astype(int)
    )

    for column in [
        "judge_relevance_score",
        "judge_groundedness_score",
        "judge_overall_score",
        "judge_input_tokens",
        "judge_output_tokens",
        "judge_total_cost",
    ]:
        feedback[column] = pd.to_numeric(
            feedback[column],
            errors="coerce",
        )

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
    filtered = filtered[filtered["feedback_score"].isin(selected_ratings)]
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
        filtered["patient_id"]
        .fillna("")
        .str.contains(
            patient_search,
            case=False,
            na=False,
        )
    ]

filtered = filtered.sort_values("created_at", ascending=False)
if filtered.empty:
    st.info("No feedback records match the selected filters.")
    st.stop()

rated = filtered.dropna(subset=["feedback_score"])

average_score = rated["feedback_score"].mean()
low_score_count = int((rated["feedback_score"] <= 2).sum())
low_score_rate = 100 * low_score_count / len(rated) if len(rated) else float("nan")
accuracy_flag_count = int((filtered["accuracy_issue"] == 1).sum())
accuracy_flag_rate = (
    100 * accuracy_flag_count / len(filtered) if len(filtered) else float("nan")
)
total_cost = filtered["total_cost"].sum(min_count=1)

filtered["combined_cost"] = filtered["total_cost"].fillna(0) + filtered[
    "judge_total_cost"
].fillna(0)

st.subheader("Quality summary")
metric_columns = st.columns(4)
metric_columns[0].metric("Feedback records", f"{len(filtered):,}")
metric_columns[1].metric("Average score", metric_value(average_score))
metric_columns[2].metric("Scores ≤ 2", metric_value(low_score_rate, "{:.1f}%"))
metric_columns[3].metric(
    "Accuracy flags",
    metric_value(accuracy_flag_rate, "{:.1f}%"),
)
# metric_columns[4].metric("Estimated cost", metric_value(total_cost, "${:.4f}"))

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
st.subheader("Usage and cost")

usage_columns = st.columns(4)

# usage_columns[0].metric(
#     "Total queries",          #we're saving feedback, not queries, which can be run without feedback
#     f"{len(filtered):,}",
# )
usage_columns[0].metric(
    "Feedback submissions",
    f"{len(filtered):,}",
)

usage_columns[1].metric(
    "Total input tokens",
    f"{filtered['input_tokens'].fillna(0).sum():,.0f}",
)

usage_columns[2].metric(
    "Total output tokens",
    f"{filtered['output_tokens'].fillna(0).sum():,.0f}",
)

usage_columns[3].metric(
    "Total estimated cost",
    f"${filtered['total_cost'].fillna(0).sum():.4f}",
)

st.metric(
    "Generation + judge cost",
    f"${filtered['combined_cost'].sum():.4f}",
)

chart_left, chart_right = st.columns(2)

with chart_left:
    st.subheader("Feedback volume by day")

    daily_queries = (
        filtered.dropna(subset=["created_at"])
        .assign(date=lambda df: df["created_at"].dt.date)
        .groupby("date")
        .size()
        .rename("queries")
    )

    if daily_queries.empty:
        st.caption("No dated query records are available.")
    else:
        st.line_chart(daily_queries)

with chart_right:
    st.subheader("Estimated cost by question type")

    cost_by_question_type = (
        filtered.groupby("question_type")["total_cost"]
        .sum()
        .sort_values(ascending=False)
        .rename("estimated_cost")
    )

    if cost_by_question_type.empty:
        st.caption("No cost data is available.")
    else:
        st.bar_chart(cost_by_question_type)

st.subheader("Token usage by retrieval method")

tokens_by_search_type = (
    filtered.groupby("search_type")[["input_tokens", "output_tokens"]].sum().fillna(0)
)

if tokens_by_search_type.empty:
    st.caption("No token-usage data is available.")
else:
    st.bar_chart(tokens_by_search_type)

st.divider()
st.subheader("Answer-generation latency")

latency_by_type = (
    filtered.dropna(subset=["latency_seconds"])
    .groupby("question_type")["latency_seconds"]
    .mean()
    .sort_values(ascending=False)
    .rename("average_seconds")
)

if latency_by_type.empty:
    st.caption("Latency will appear after new answer requests are recorded.")
else:
    st.bar_chart(latency_by_type)


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
    "latency_seconds",
    "question",
    "comment",
]

st.dataframe(
    filtered[TABLE_COLUMNS],
    hide_index=True,
    use_container_width=True,
    height=360,
    column_config={
        "created_at": st.column_config.DatetimeColumn(
            "Created at", format="YYYY-MM-DD HH:mm"
        ),
        "feedback_score": st.column_config.NumberColumn("Score", format="%d / 5"),
        "accuracy_issue": st.column_config.CheckboxColumn("Accuracy flag"),
        "total_cost": st.column_config.NumberColumn("Cost", format="$%.6f"),
        "latency_seconds": st.column_config.NumberColumn("Latency", format="%.2f s"),
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

st.divider()
st.subheader("Automated evaluation")
st.caption(
    "LLM-as-a-judge results for clinician-reviewed answers. "
    "Use clinician feedback as the primary quality signal."
)

judged = filtered.dropna(subset=["judge_overall_score"]).copy()

if judged.empty:
    st.info(
        "No judge results have been saved yet. "
        "They will appear after new evaluated answers receive feedback."
    )
else:
    evaluation_metrics = st.columns(4)

    evaluation_metrics[0].metric(
        "Judged answers",
        f"{len(judged):,}",
    )

    evaluation_metrics[1].metric(
        "Average overall score",
        f"{judged['judge_overall_score'].mean():.2f} / 2",
    )

    evaluation_metrics[2].metric(
        "Average relevance",
        f"{judged['judge_relevance_score'].mean():.2f} / 2",
    )

    evaluation_metrics[3].metric(
        "Average groundedness",
        f"{judged['judge_groundedness_score'].mean():.2f} / 2",
    )

    judge_chart_left, judge_chart_right = st.columns(2)

    with judge_chart_left:
        st.markdown("#### Judge-score distribution")

        score_distribution = (
            judged["judge_overall_score"]
            .value_counts()
            .sort_index()
            .rename_axis("overall_score")
            .to_frame("answers")
        )

        st.bar_chart(score_distribution)

    with judge_chart_right:
        st.markdown("#### Groundedness outcomes")

        groundedness_distribution = (
            judged["judge_groundedness_label"]
            .fillna("UNKNOWN")
            .value_counts()
            .to_frame("answers")
        )

        st.bar_chart(groundedness_distribution)


comparison = judged.dropna(subset=["feedback_score", "judge_overall_score"]).copy()

comparison["clinician_normalized"] = (comparison["feedback_score"] - 1) / 4

comparison["judge_normalized"] = comparison["judge_overall_score"] / 2

if comparison.empty:
    st.caption(
        "A comparison will appear once answers have both clinician feedback "
        "and a stored judge score."
    )
else:
    st.markdown("#### Judge versus clinician feedback")

    mean_absolute_gap = (
        (comparison["clinician_normalized"] - comparison["judge_normalized"])
        .abs()
        .mean()
    )

    agreement_rate = (
        (comparison["clinician_normalized"] >= 0.75)
        == (comparison["judge_normalized"] >= 0.75)
    ).mean() * 100

    compare_metrics = st.columns(3)

    compare_metrics[0].metric(
        "Comparable answers",
        f"{len(comparison):,}",
    )

    compare_metrics[1].metric(
        "Mean score gap",
        f"{mean_absolute_gap:.2f}",
        help="0 means perfect agreement after both scores are normalized to 0–1.",
    )

    compare_metrics[2].metric(
        "High/low agreement",
        f"{agreement_rate:.0f}%",
        help=(
            "Agreement on whether both the clinician and judge consider "
            "the answer high quality."
        ),
    )

    st.markdown("#### Disagreement cases")

    disagreement_cases = comparison.assign(
        score_gap=(
            comparison["clinician_normalized"] - comparison["judge_normalized"]
        ).abs()
    ).sort_values("score_gap", ascending=False)

    st.dataframe(
        disagreement_cases[
            [
                "id",
                "created_at",
                "patient_id",
                "question_type",
                "feedback_score",
                "judge_overall_score",
                "judge_relevance_label",
                "judge_groundedness_label",
                "score_gap",
                "question",
                "judge_explanation",
                "comment",
            ]
        ].head(20),
        hide_index=True,
        use_container_width=True,
        column_config={
            "score_gap": st.column_config.NumberColumn(
                "Normalized gap",
                format="%.2f",
            ),
            "question": st.column_config.TextColumn(
                "Question",
                width="large",
            ),
            "judge_explanation": st.column_config.TextColumn(
                "Judge explanation",
                width="large",
            ),
            "comment": st.column_config.TextColumn(
                "Clinician comment",
                width="large",
            ),
        },
    )


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
    st.write(
        f"**Score:** {record['feedback_score'] if pd.notna(record['feedback_score']) else 'Not rated'}"
    )
    st.write(f"**Accuracy flagged:** {'Yes' if record['accuracy_issue'] else 'No'}")
    st.write(f"**Issue type:** {record['issue_type'] or 'Not selected'}")
    st.write(f"**Question type:** {record['question_type']}")
    st.write(f"**Router suggestion:** {record['routing_suggestion'] or 'None'}")
    st.write(
        f"**Router confidence:** {record['routing_confidence'] if pd.notna(record['routing_confidence']) else 'Not recorded'}"
    )
    st.write(f"**Retrieval:** {record['search_type']}")
    st.write(f"**Model:** {record['model']}")
    st.write(
        f"**Latency:** {record['latency_seconds']:.2f} seconds"
        if pd.notna(record["latency_seconds"])
        else "**Latency:** Not recorded"
    )
    st.write(
        f"**Input tokens:** {record['input_tokens'] if pd.notna(record['input_tokens']) else 'Not recorded'}"
    )
    st.write(
        f"**Output tokens:** {record['output_tokens'] if pd.notna(record['output_tokens']) else 'Not recorded'}"
    )
    st.write(
        f"**Estimated cost:** ${record['total_cost']:.6f}"
        if pd.notna(record["total_cost"])
        else "**Estimated cost:** Not recorded"
    )

render_source_preview(str(record["patient_id"]), int(record["id"]))
