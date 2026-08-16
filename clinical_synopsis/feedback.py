import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


DB_PATH = Path("data/monitoring/clinical_synopsis_feedback.db")


def save_feedback(event: dict) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                patient_id TEXT,
                question TEXT,
                question_type TEXT,
                search_type TEXT,
                model TEXT,
                answer TEXT,
                routing_suggestion TEXT,
                routing_confidence REAL,
                feedback_score INTEGER,
                accuracy_issue INTEGER,
                issue_type TEXT,
                comment TEXT,
                input_tokens INTEGER,
                output_tokens INTEGER,
                total_cost REAL,
                latency_seconds REAL,
                judge_relevance_score REAL,
                judge_groundedness_score REAL,
                judge_overall_score REAL,
                judge_relevance_label TEXT,
                judge_groundedness_label TEXT,
                judge_explanation TEXT,
                judge_input_tokens INTEGER,
                judge_output_tokens INTEGER,
                judge_total_cost REAL
            )
        """)

        new_columns = {
            "judge_relevance_score": "REAL",
            "judge_groundedness_score": "REAL",
            "judge_overall_score": "REAL",
            "judge_relevance_label": "TEXT",
            "judge_groundedness_label": "TEXT",
            "judge_explanation": "TEXT",
            "judge_input_tokens": "INTEGER",
            "judge_output_tokens": "INTEGER",
            "judge_total_cost": "REAL",
        }

        for column_name, column_type in new_columns.items():
            try:
                conn.execute(
                    f"ALTER TABLE feedback ADD COLUMN {column_name} {column_type}"
                )
            except sqlite3.OperationalError:
                pass

        try:
            conn.execute(
                "ALTER TABLE feedback ADD COLUMN latency_seconds REAL"
            )
        except sqlite3.OperationalError:
            pass

        conn.execute("""
            INSERT INTO feedback (
                created_at, patient_id, question, question_type,
                search_type, model, answer,
                routing_suggestion, routing_confidence,
                feedback_score, accuracy_issue, issue_type, comment,
                input_tokens, output_tokens, total_cost, latency_seconds,
                judge_relevance_score, judge_groundedness_score, judge_overall_score,
                judge_relevance_label, judge_groundedness_label, judge_explanation,
                judge_input_tokens, judge_output_tokens, judge_total_cost
            )
            VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, (
            datetime.now(timezone.utc).isoformat(),
            event.get("patient_id"),
            event.get("question"),
            event.get("question_type"),
            event.get("search_type"),
            event.get("model"),
            event.get("answer"),
            event.get("routing_suggestion"),
            event.get("routing_confidence"),
            event.get("feedback_score"),
            int(bool(event.get("accuracy_issue"))),
            event.get("issue_type"),
            event.get("comment"),
            event.get("input_tokens"),
            event.get("output_tokens"),
            event.get("total_cost"),
            event.get("latency_seconds"),
            event.get("judge_relevance_score"),
            event.get("judge_groundedness_score"),
            event.get("judge_overall_score"),
            event.get("judge_relevance_label"),
            event.get("judge_groundedness_label"),
            event.get("judge_explanation"),
            event.get("judge_input_tokens"),
            event.get("judge_output_tokens"),
            event.get("judge_total_cost")
        ))