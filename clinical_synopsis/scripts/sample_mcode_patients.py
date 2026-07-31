import json
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


DATA_DIR = Path("data/raw/longitudinalMCODEBreast")
OUTPUT_DIR = Path("data/processed")
SEED = 42
N_SAMPLE = 50


RESOURCE_TYPES_DATE_FIELDS = {
    "Encounter": [("period", "start"), ("period", "end")],
    "Observation": [("effectiveDateTime",), ("issued",)],
    "Condition": [("onsetDateTime",), ("abatementDateTime",), ("recordedDate",)],
    "Procedure": [("performedDateTime",), ("performedPeriod", "start"), ("performedPeriod", "end")],
    "MedicationRequest": [("authoredOn",)],
    "MedicationAdministration": [("effectiveDateTime",), ("effectivePeriod", "start"), ("effectivePeriod", "end")],
    "DiagnosticReport": [("effectiveDateTime",), ("issued",)],
}


@dataclass
class PatientStats:
    filename: str
    patient_id: str | None
    patient_name: str | None
    n_resources: int
    n_encounters: int
    n_observations: int
    n_conditions: int
    n_procedures: int
    n_medication_requests: int
    n_medication_administrations: int
    n_diagnostic_reports: int
    first_date: str | None
    last_date: str | None
    followup_days: int
    complexity_score: int
    complexity_bucket: str | None = None


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def get_nested(d: dict, path: tuple[str, ...]):
    cur = d
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
        if cur is None:
            return None
    return cur


def extract_dates(resource: dict) -> list[datetime]:
    resource_type = resource.get("resourceType")
    date_paths = RESOURCE_TYPES_DATE_FIELDS.get(resource_type, [])
    dates = []

    for path in date_paths:
        value = get_nested(resource, path)
        dt = parse_datetime(value) if isinstance(value, str) else None
        if dt is not None:
            dates.append(dt)

    return dates


def extract_patient_info(entries: list[dict]) -> tuple[str | None, str | None]:
    for entry in entries:
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "Patient":
            patient_id = resource.get("id")
            names = resource.get("name", [])
            if names:
                first = " ".join(names[0].get("given", []))
                last = names[0].get("family", "")
                patient_name = f"{first} {last}".strip() or None
            else:
                patient_name = None
            return patient_id, patient_name
    return None, None

# filter for bundles that include only patient info, not practitioner or hospital info from mCode
def load_bundle(file_path: Path) -> dict:
    return json.loads(file_path.read_text(encoding="utf-8"))


def is_patient_bundle(file_path: Path) -> bool:
    try:
        bundle = load_bundle(file_path)
        entries = bundle.get("entry", [])
        patient_id, _ = extract_patient_info(entries)
        return patient_id is not None
    except Exception:
        return False


def compute_patient_stats(file_path: Path) -> PatientStats:
    bundle = load_bundle(file_path)
    entries = bundle.get("entry", [])

    patient_id, patient_name = extract_patient_info(entries)

    counts = {
        "Encounter": 0,
        "Observation": 0,
        "Condition": 0,
        "Procedure": 0,
        "MedicationRequest": 0,
        "MedicationAdministration": 0,
        "DiagnosticReport": 0,
    }

    all_dates = []

    for entry in entries:
        resource = entry.get("resource", {})
        resource_type = resource.get("resourceType")

        if resource_type in counts:
            counts[resource_type] += 1

        all_dates.extend(extract_dates(resource))

    all_dates = sorted(set(all_dates))
    first_date = all_dates[0] if all_dates else None
    last_date = all_dates[-1] if all_dates else None
    followup_days = (last_date - first_date).days if first_date and last_date else 0

    complexity_score = (
        counts["Encounter"] * 3
        + counts["Observation"] * 1
        + counts["Condition"] * 2
        + counts["Procedure"] * 2
        + counts["MedicationRequest"] * 2
        + counts["MedicationAdministration"] * 2
        + counts["DiagnosticReport"] * 2
        + min(followup_days // 180, 20)
    )

    return PatientStats(
        filename=str(file_path),
        patient_id=patient_id,
        patient_name=patient_name,
        n_resources=len(entries),
        n_encounters=counts["Encounter"],
        n_observations=counts["Observation"],
        n_conditions=counts["Condition"],
        n_procedures=counts["Procedure"],
        n_medication_requests=counts["MedicationRequest"],
        n_medication_administrations=counts["MedicationAdministration"],
        n_diagnostic_reports=counts["DiagnosticReport"],
        first_date=first_date.isoformat() if first_date else None,
        last_date=last_date.isoformat() if last_date else None,
        followup_days=followup_days,
        complexity_score=complexity_score,
    )


def assign_complexity_buckets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["complexity_bucket"] = pd.qcut(
        df["complexity_score"],
        q=3,
        labels=["low", "medium", "high"],
        duplicates="drop",
    )

    return df


def stratified_sample(
    df: pd.DataFrame,
    n_total: int = 50,
    seed: int = 42,
    bucket_col: str = "complexity_bucket",
) -> pd.DataFrame:
    rng = random.Random(seed)
    df = df.copy()

    bucket_counts = df[bucket_col].value_counts().to_dict()
    buckets = sorted(df[bucket_col].dropna().unique())

    target = {}
    remaining = n_total

    for i, bucket in enumerate(buckets):
        if i == len(buckets) - 1:
            target[bucket] = remaining
        else:
            share = bucket_counts[bucket] / len(df)
            n_bucket = round(n_total * share)
            target[bucket] = n_bucket
            remaining -= n_bucket

    sampled_parts = []

    for bucket in buckets:
        bucket_df = df[df[bucket_col] == bucket].copy()
        n_take = min(target[bucket], len(bucket_df))
        indices = list(bucket_df.index)
        rng.shuffle(indices)
        sampled_parts.append(bucket_df.loc[indices[:n_take]])

    sampled = pd.concat(sampled_parts).copy()

    if len(sampled) < n_total:
        remaining_df = df.loc[~df.index.isin(sampled.index)].copy()
        remaining_indices = list(remaining_df.index)
        rng.shuffle(remaining_indices)
        extra = remaining_df.loc[remaining_indices[: n_total - len(sampled)]]
        sampled = pd.concat([sampled, extra])

    return sampled.sort_values(["complexity_bucket", "complexity_score", "filename"]).reset_index(drop=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_files = sorted(DATA_DIR.glob("*.json"))
    if not all_files:
        raise FileNotFoundError(f"No JSON files found in {DATA_DIR}")

    files = [fp for fp in all_files if is_patient_bundle(fp)]
    skipped_files = [fp for fp in all_files if fp not in files]

    if not files:
        raise FileNotFoundError(f"No patient JSON bundles found in {DATA_DIR}")

    stats = [compute_patient_stats(fp) for fp in files]
    df = pd.DataFrame([asdict(s) for s in stats])

    df = assign_complexity_buckets(df)

    summary_path = OUTPUT_DIR / "mcode_breast_patient_stats.csv"
    df.to_csv(summary_path, index=False)

    sample_df = stratified_sample(df, n_total=N_SAMPLE, seed=SEED)
    sample_df["sample_seed"] = SEED

    manifest_path = OUTPUT_DIR / "mcode_breast_sample_50_manifest.csv"
    sample_df.to_csv(manifest_path, index=False)

    print(f"Found {len(all_files)} JSON files total")
    print(f"Kept {len(files)} patient bundles")
    print(f"Skipped {len(skipped_files)} non-patient bundles")
    print()

    if skipped_files:
        print("Example skipped files:")
        for fp in skipped_files[:10]:
            print(" -", fp.name)
        print()

    print(f"Wrote patient stats to: {summary_path}")
    print(f"Wrote 50-patient manifest to: {manifest_path}")
    print()
    print("Sample bucket counts:")
    print(sample_df["complexity_bucket"].value_counts(dropna=False).to_string())
    print()
    print("Example sampled files:")
    print(sample_df[["filename", "complexity_bucket", "complexity_score", "followup_days"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()