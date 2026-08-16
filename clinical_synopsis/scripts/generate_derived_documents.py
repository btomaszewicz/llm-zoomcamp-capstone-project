import json
import re
from pathlib import Path

import pandas as pd


INPUT_ROOT = Path("data/interim/sample50")
OUTPUT_ROOT = Path("data/derived/sample50")


ONCOLOGY_TERMS = [
    "cancer",
    "carcinoma",
    "tumor",
    "tumour",
    "neoplasm",
    "oncology",
    "malignant",
    "metast",
    "chemo",
    "chemotherapy",
    "radiation",
    "radiotherapy",
    "biopsy",
    "stage",
    "adenocarcinoma",
    "lymphoma",
    "leukemia",
    "melanoma",
    "sarcoma",
]


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def normalize_text(value) -> str:
    if pd.isna(value) or value is None:
        return ""
    return str(value).strip()


def has_rows(df: pd.DataFrame) -> bool:
    return df is not None and not df.empty


def text_matches_oncology(text: str) -> bool:
    t = normalize_text(text).lower()
    return any(term in t for term in ONCOLOGY_TERMS)


def first_nonempty(row: pd.Series, columns: list[str]) -> str:
    for col in columns:
        if col in row and normalize_text(row[col]):
            return normalize_text(row[col])
    return ""


def patient_display_name(patient_df: pd.DataFrame, patient_id: str) -> str:
    if has_rows(patient_df):
        for col in ["full_name", "given_name", "family_name"]:
            if col in patient_df.columns:
                val = normalize_text(patient_df.iloc[0].get(col))
                if val:
                    if col == "full_name":
                        return val
        given = normalize_text(patient_df.iloc[0].get("given_name"))
        family = normalize_text(patient_df.iloc[0].get("family_name"))
        combined = " ".join(x for x in [given, family] if x)
        if combined:
            return combined
    return patient_id


def sort_by_date(df: pd.DataFrame, date_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    chosen_col = None
    for col in date_cols:
        if col in df.columns:
            df["_sort_date"] = pd.to_datetime(df[col], errors="coerce", utc=True)
            chosen_col = col
            break
    if chosen_col is None:
        df["_sort_date"] = pd.NaT
    return df.sort_values("_sort_date", ascending=False, na_position="last").drop(
        columns=["_sort_date"]
    )


def write_csv_if_needed(df: pd.DataFrame, path: Path, always_write: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if always_write or has_rows(df):
        df.to_csv(path, index=False)


def summarize_conditions(df: pd.DataFrame, limit: int = 10) -> list[str]:
    if not has_rows(df):
        return []
    df2 = sort_by_date(df, ["recorded_date", "onset_datetime"])
    lines = []
    for _, row in df2.head(limit).iterrows():
        label = first_nonempty(row, ["display", "text", "code"])
        when = first_nonempty(row, ["recorded_date", "onset_datetime"])
        status = first_nonempty(row, ["clinical_status", "verification_status"])
        parts = [label]
        if when:
            parts.append(f"date: {when}")
        if status:
            parts.append(f"status: {status}")
        if label:
            lines.append("- " + "; ".join(parts))
    return lines


def summarize_meds(df: pd.DataFrame, limit: int = 10) -> list[str]:
    if not has_rows(df):
        return []
    df2 = sort_by_date(
        df, ["authored_on", "effective_datetime", "effective_period_start"]
    )
    lines = []
    for _, row in df2.head(limit).iterrows():
        label = first_nonempty(row, ["display", "text", "code"])
        when = first_nonempty(
            row, ["authored_on", "effective_datetime", "effective_period_start"]
        )
        status = first_nonempty(row, ["status", "intent"])
        if label:
            parts = [label]
            if when:
                parts.append(f"date: {when}")
            if status:
                parts.append(f"status: {status}")
            lines.append("- " + "; ".join(parts))
    return lines


def summarize_observations(df: pd.DataFrame, limit: int = 12) -> list[str]:
    if not has_rows(df):
        return []
    df2 = sort_by_date(df, ["effective_datetime", "issued"])
    lines = []
    for _, row in df2.head(limit).iterrows():
        label = first_nonempty(row, ["display", "text", "code"])
        val_num = normalize_text(row.get("value_numeric"))
        val_unit = normalize_text(row.get("value_unit"))
        val_str = first_nonempty(row, ["value_string", "value_display", "value_code"])
        when = first_nonempty(row, ["effective_datetime", "issued"])
        value_text = ""
        if val_num:
            value_text = f"{val_num} {val_unit}".strip()
        elif val_str:
            value_text = val_str
        if label:
            parts = [label]
            if value_text:
                parts.append(f"value: {value_text}")
            if when:
                parts.append(f"date: {when}")
            lines.append("- " + "; ".join(parts))
    return lines


def summarize_encounters(df: pd.DataFrame, limit: int = 8) -> list[str]:
    if not has_rows(df):
        return []
    df2 = sort_by_date(df, ["period_start", "period_end"])
    lines = []
    for _, row in df2.head(limit).iterrows():
        label = first_nonempty(
            row, ["encounter_type_display", "encounter_type_code", "encounter_class"]
        )
        start = first_nonempty(row, ["period_start"])
        end = first_nonempty(row, ["period_end"])
        provider = first_nonempty(row, ["service_provider"])
        parts = [label or "Encounter"]
        if start:
            parts.append(f"start: {start}")
        if end:
            parts.append(f"end: {end}")
        if provider:
            parts.append(f"provider: {provider}")
        lines.append("- " + "; ".join(parts))
    return lines


def summarize_procedures(df: pd.DataFrame, limit: int = 10) -> list[str]:
    if not has_rows(df):
        return []
    df2 = sort_by_date(df, ["performed_datetime", "performed_period_start"])
    lines = []
    for _, row in df2.head(limit).iterrows():
        label = first_nonempty(row, ["display", "text", "code"])
        when = first_nonempty(row, ["performed_datetime", "performed_period_start"])
        if label:
            parts = [label]
            if when:
                parts.append(f"date: {when}")
            lines.append("- " + "; ".join(parts))
    return lines


def summarize_reports(df: pd.DataFrame, limit: int = 10) -> list[str]:
    if not has_rows(df):
        return []
    df2 = sort_by_date(df, ["effective_datetime", "issued"])
    lines = []
    for _, row in df2.head(limit).iterrows():
        label = first_nonempty(row, ["display", "text", "code"])
        when = first_nonempty(row, ["effective_datetime", "issued"])
        if label:
            parts = [label]
            if when:
                parts.append(f"date: {when}")
            lines.append("- " + "; ".join(parts))
    return lines


def build_overview_md(
    patient_id: str,
    patient_df: pd.DataFrame,
    encounters_df: pd.DataFrame,
    conditions_df: pd.DataFrame,
    observations_df: pd.DataFrame,
    meds_df: pd.DataFrame,
    procedures_df: pd.DataFrame,
    reports_df: pd.DataFrame,
    metadata: dict,
) -> str:
    name = patient_display_name(patient_df, patient_id)
    demo = patient_df.iloc[0].to_dict() if has_rows(patient_df) else {}

    lines = []
    lines.append(f"# Patient Overview: {name}")
    lines.append("")
    lines.append("## Identity")
    lines.append(f"- Patient ID: {patient_id}")
    if normalize_text(demo.get("gender")):
        lines.append(f"- Gender: {normalize_text(demo.get('gender'))}")
    if normalize_text(demo.get("birth_date")):
        lines.append(f"- Birth date: {normalize_text(demo.get('birth_date'))}")
    lines.append("")

    lines.append("## Record Snapshot")
    lines.append(f"- Source bundle: {metadata.get('source_file', '')}")
    lines.append(f"- Bundle entries: {metadata.get('n_entries', 0)}")
    lines.append(f"- Encounters: {len(encounters_df)}")
    lines.append(f"- Conditions: {len(conditions_df)}")
    lines.append(f"- Observations: {len(observations_df)}")
    lines.append(f"- Medications: {len(meds_df)}")
    lines.append(f"- Procedures: {len(procedures_df)}")
    lines.append(f"- Diagnostic reports: {len(reports_df)}")
    lines.append("")

    lines.append("## Recent Conditions")
    condition_lines = summarize_conditions(conditions_df)
    lines.extend(
        condition_lines if condition_lines else ["- No condition records found."]
    )
    lines.append("")

    lines.append("## Recent Results")
    obs_lines = summarize_observations(observations_df)
    lines.extend(obs_lines if obs_lines else ["- No observation records found."])
    lines.append("")

    lines.append("## Recent Encounters")
    enc_lines = summarize_encounters(encounters_df)
    lines.extend(enc_lines if enc_lines else ["- No encounter records found."])
    lines.append("")

    if has_rows(meds_df):
        lines.append("## Medications")
        lines.extend(
            summarize_meds(meds_df) or ["- Medication data present but not summarized."]
        )
        lines.append("")

    if has_rows(procedures_df):
        lines.append("## Procedures")
        lines.extend(
            summarize_procedures(procedures_df)
            or ["- Procedure data present but not summarized."]
        )
        lines.append("")

    if has_rows(reports_df):
        lines.append("## Diagnostic Reports")
        lines.extend(
            summarize_reports(reports_df)
            or ["- Diagnostic report data present but not summarized."]
        )
        lines.append("")

    lines.append("## Provenance")
    lines.append(
        "- This document is derived from normalized CSV tables in the same patient folder."
    )
    lines.append(
        "- Use the `source_file` and `resource_id` columns in CSV files to trace facts back to the original FHIR bundle."
    )

    return "\n".join(lines).strip() + "\n"


def collect_oncology_events(
    conditions_df: pd.DataFrame,
    observations_df: pd.DataFrame,
    meds_df: pd.DataFrame,
    procedures_df: pd.DataFrame,
    reports_df: pd.DataFrame,
) -> pd.DataFrame:
    events = []

    for _, row in conditions_df.iterrows() if has_rows(conditions_df) else []:
        text = " ".join(
            [
                first_nonempty(row, ["display", "text", "code"]),
                first_nonempty(row, ["clinical_status"]),
            ]
        )
        if text_matches_oncology(text):
            events.append(
                {
                    "event_type": "Condition",
                    "date": first_nonempty(row, ["recorded_date", "onset_datetime"]),
                    "label": first_nonempty(row, ["display", "text", "code"]),
                    "status": first_nonempty(
                        row, ["clinical_status", "verification_status"]
                    ),
                    "resource_id": first_nonempty(row, ["resource_id"]),
                    "source_file": first_nonempty(row, ["source_file"]),
                }
            )

    for _, row in observations_df.iterrows() if has_rows(observations_df) else []:
        text = " ".join(
            [
                first_nonempty(row, ["display", "text", "code"]),
                first_nonempty(row, ["value_display", "value_string", "value_code"]),
            ]
        )
        if text_matches_oncology(text):
            value_text = first_nonempty(
                row, ["value_display", "value_string", "value_code"]
            )
            if normalize_text(row.get("value_numeric")):
                value_text = f"{normalize_text(row.get('value_numeric'))} {normalize_text(row.get('value_unit'))}".strip()
            events.append(
                {
                    "event_type": "Observation",
                    "date": first_nonempty(row, ["effective_datetime", "issued"]),
                    "label": first_nonempty(row, ["display", "text", "code"]),
                    "status": value_text,
                    "resource_id": first_nonempty(row, ["resource_id"]),
                    "source_file": first_nonempty(row, ["source_file"]),
                }
            )

    for _, row in meds_df.iterrows() if has_rows(meds_df) else []:
        text = first_nonempty(row, ["display", "text", "code"])
        if text_matches_oncology(text):
            events.append(
                {
                    "event_type": "Medication",
                    "date": first_nonempty(
                        row,
                        ["authored_on", "effective_datetime", "effective_period_start"],
                    ),
                    "label": text,
                    "status": first_nonempty(row, ["status", "intent"]),
                    "resource_id": first_nonempty(row, ["resource_id"]),
                    "source_file": first_nonempty(row, ["source_file"]),
                }
            )

    for _, row in procedures_df.iterrows() if has_rows(procedures_df) else []:
        text = first_nonempty(row, ["display", "text", "code"])
        if text_matches_oncology(text):
            events.append(
                {
                    "event_type": "Procedure",
                    "date": first_nonempty(
                        row, ["performed_datetime", "performed_period_start"]
                    ),
                    "label": text,
                    "status": first_nonempty(row, ["status"]),
                    "resource_id": first_nonempty(row, ["resource_id"]),
                    "source_file": first_nonempty(row, ["source_file"]),
                }
            )

    for _, row in reports_df.iterrows() if has_rows(reports_df) else []:
        text = first_nonempty(row, ["display", "text", "code"])
        if text_matches_oncology(text):
            events.append(
                {
                    "event_type": "DiagnosticReport",
                    "date": first_nonempty(row, ["effective_datetime", "issued"]),
                    "label": text,
                    "status": first_nonempty(row, ["status"]),
                    "resource_id": first_nonempty(row, ["resource_id"]),
                    "source_file": first_nonempty(row, ["source_file"]),
                }
            )

    events_df = pd.DataFrame(events)
    if not events_df.empty:
        events_df["_sort_date"] = pd.to_datetime(
            events_df["date"], errors="coerce", utc=True
        )
        events_df = events_df.sort_values(
            "_sort_date", ascending=True, na_position="last"
        ).drop(columns=["_sort_date"])
    return events_df


def build_oncology_timeline_md(
    patient_id: str, patient_name: str, events_df: pd.DataFrame
) -> str:
    lines = []
    lines.append(f"# Oncology Timeline: {patient_name}")
    lines.append("")
    lines.append(f"- Patient ID: {patient_id}")
    lines.append(f"- Oncology-related dated events: {len(events_df)}")
    lines.append("")
    lines.append("## Timeline")
    for _, row in events_df.iterrows():
        date = first_nonempty(row, ["date"]) or "undated"
        event_type = first_nonempty(row, ["event_type"])
        label = first_nonempty(row, ["label"])
        status = first_nonempty(row, ["status"])
        resource_id = first_nonempty(row, ["resource_id"])
        if status:
            lines.append(
                f"- {date} — {event_type}: {label}; detail: {status}; resource_id: {resource_id}"
            )
        else:
            lines.append(
                f"- {date} — {event_type}: {label}; resource_id: {resource_id}"
            )
    lines.append("")
    lines.append("## Provenance")
    lines.append(
        "- Each timeline event was selected from normalized source tables by keyword matching."
    )
    lines.append(
        "- Confirm clinical relevance against the original CSV rows before downstream use."
    )
    return "\n".join(lines).strip() + "\n"


def combine_medications(
    med_req_df: pd.DataFrame,
    med_admin_df: pd.DataFrame,
) -> pd.DataFrame:
    frames = []

    if has_rows(med_req_df):
        df = med_req_df.copy()
        df["medication_event_type"] = "MedicationRequest"
        df["event_date"] = df.get("authored_on")
        frames.append(df)

    if has_rows(med_admin_df):
        df = med_admin_df.copy()
        df["medication_event_type"] = "MedicationAdministration"
        df["event_date"] = df.get("effective_datetime")
        if "effective_period_start" in df.columns:
            df["event_date"] = df["event_date"].fillna(df["effective_period_start"])
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["_sort_date"] = pd.to_datetime(
        combined["event_date"], errors="coerce", utc=True
    )
    combined = combined.sort_values(
        "_sort_date", ascending=False, na_position="last"
    ).drop(columns=["_sort_date"])
    return combined


def generate_for_patient(patient_dir: Path) -> dict:
    patient_id = patient_dir.name
    out_dir = OUTPUT_ROOT / patient_id
    out_dir.mkdir(parents=True, exist_ok=True)

    patient_df = safe_read_csv(patient_dir / "patient.csv")
    encounters_df = safe_read_csv(patient_dir / "encounters.csv")
    conditions_df = safe_read_csv(patient_dir / "conditions.csv")
    observations_df = safe_read_csv(patient_dir / "observations.csv")
    med_req_df = safe_read_csv(patient_dir / "medication_requests.csv")
    med_admin_df = safe_read_csv(patient_dir / "medication_administrations.csv")
    procedures_df = safe_read_csv(patient_dir / "procedures.csv")
    reports_df = safe_read_csv(patient_dir / "diagnostic_reports.csv")

    metadata_path = patient_dir / "bundle_metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )

    meds_df = combine_medications(med_req_df, med_admin_df)

    write_csv_if_needed(encounters_df, out_dir / "encounters.csv", always_write=True)
    write_csv_if_needed(conditions_df, out_dir / "conditions.csv", always_write=True)
    write_csv_if_needed(
        observations_df, out_dir / "observations.csv", always_write=True
    )

    if has_rows(meds_df):
        write_csv_if_needed(meds_df, out_dir / "medications.csv")

    if has_rows(procedures_df):
        write_csv_if_needed(procedures_df, out_dir / "procedures.csv")

    if has_rows(reports_df):
        write_csv_if_needed(reports_df, out_dir / "diagnostic_reports.csv")

    overview_md = build_overview_md(
        patient_id=patient_id,
        patient_df=patient_df,
        encounters_df=encounters_df,
        conditions_df=conditions_df,
        observations_df=observations_df,
        meds_df=meds_df,
        procedures_df=procedures_df,
        reports_df=reports_df,
        metadata=metadata,
    )
    (out_dir / "patient_overview.md").write_text(overview_md, encoding="utf-8")

    patient_name = patient_display_name(patient_df, patient_id)
    oncology_events_df = collect_oncology_events(
        conditions_df=conditions_df,
        observations_df=observations_df,
        meds_df=meds_df,
        procedures_df=procedures_df,
        reports_df=reports_df,
    )
    oncology_events_dated = (
        oncology_events_df[
            oncology_events_df["date"].fillna("").astype(str).str.len() > 0
        ]
        if not oncology_events_df.empty
        else oncology_events_df
    )

    created_timeline = False
    if len(oncology_events_dated) >= 3:
        timeline_md = build_oncology_timeline_md(
            patient_id, patient_name, oncology_events_dated
        )
        (out_dir / "oncology_timeline.md").write_text(timeline_md, encoding="utf-8")
        oncology_events_dated.to_csv(
            out_dir / "oncology_timeline_events.csv", index=False
        )
        created_timeline = True

    manifest = {
        "patient_id": patient_id,
        "files_created": sorted([p.name for p in out_dir.glob("*") if p.is_file()]),
        "row_counts": {
            "encounters": int(len(encounters_df)),
            "conditions": int(len(conditions_df)),
            "observations": int(len(observations_df)),
            "medications": int(len(meds_df)),
            "procedures": int(len(procedures_df)),
            "diagnostic_reports": int(len(reports_df)),
            "oncology_events": int(len(oncology_events_dated)),
        },
        "created_oncology_timeline": created_timeline,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    patient_dirs = sorted([p for p in INPUT_ROOT.iterdir() if p.is_dir()])
    if not patient_dirs:
        raise FileNotFoundError(f"No patient folders found in {INPUT_ROOT}")

    manifests = []
    for patient_dir in patient_dirs:
        manifests.append(generate_for_patient(patient_dir))

    summary_df = pd.DataFrame(manifests)
    summary_df.to_csv(OUTPUT_ROOT / "derived_generation_summary.csv", index=False)

    totals = {
        "patients_processed": len(manifests),
        "patient_overviews": len(manifests),
        "oncology_timelines": sum(
            1 for m in manifests if m["created_oncology_timeline"]
        ),
    }
    (OUTPUT_ROOT / "run_summary.json").write_text(
        json.dumps(totals, indent=2), encoding="utf-8"
    )

    print(f"Processed {len(manifests)} patients into {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
