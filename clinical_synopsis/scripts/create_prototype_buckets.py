import shutil
from pathlib import Path

import pandas as pd


MANIFEST_PATH = Path("data/processed/mcode_breast_sample_50_manifest.csv")
OUTPUT_ROOT = Path("data/prototype")
ALL_DIR = OUTPUT_ROOT / "sample50"
BUCKETS = ["low", "medium", "high"]


def ensure_directories() -> None:
    ALL_DIR.mkdir(parents=True, exist_ok=True)
    for bucket in BUCKETS:
        (OUTPUT_ROOT / bucket).mkdir(parents=True, exist_ok=True)


def validate_manifest(df: pd.DataFrame) -> None:
    required_columns = {"filename", "complexity_bucket"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")


def copy_selected_files(df: pd.DataFrame) -> tuple[int, int]:
    copied = 0
    missing = 0

    for _, row in df.iterrows():
        src = Path(row["filename"])
        bucket = str(row["complexity_bucket"]).strip().lower()

        if not src.exists():
            print(f"[WARN] Missing source file: {src}")
            missing += 1
            continue

        shutil.copy2(src, ALL_DIR / src.name)

        if bucket in BUCKETS:
            shutil.copy2(src, (OUTPUT_ROOT / bucket) / src.name)
        else:
            print(f"[WARN] Unknown bucket '{bucket}' for file: {src.name}")

        copied += 1

    return copied, missing


def print_summary(df: pd.DataFrame) -> None:
    print("\nBucket counts in manifest:")
    print(df["complexity_bucket"].value_counts(dropna=False).to_string())

    print("\nOutput folders:")
    print(f"- {ALL_DIR}")
    for bucket in BUCKETS:
        print(f"- {OUTPUT_ROOT / bucket}")


def main() -> None:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    df = pd.read_csv(MANIFEST_PATH)
    validate_manifest(df)
    ensure_directories()

    copied, missing = copy_selected_files(df)

    print(f"Copied {copied} files.")
    if missing:
        print(f"Skipped {missing} missing files.")

    print_summary(df)


if __name__ == "__main__":
    main()
