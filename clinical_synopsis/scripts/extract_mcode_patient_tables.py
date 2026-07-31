import json
from pathlib import Path
from datetime import datetime

import pandas as pd


INPUT_DIR = Path("data/prototype/sample50")
OUTPUT_DIR = Path("data/interim/sample50")


def parse_datetime(value):
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def get_nested(obj, *path):
    cur = obj
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
        if cur is None:
            return None
    return cur


def coding_to_text(code_obj):
    if not isinstance(code_obj, dict):
        return None, None, None
    coding = code_obj.get("coding", [])
    if coding and isinstance(coding, list):
        first = coding[0]
        return first.get("system"), first.get("code"), first.get("display")
    return None, None, code_obj.get("text")


def extract_patient(resource, source_file):
    names = resource.get("name", [])
    given = None
    family = None
    full_name = None

    if names:
        given = " ".join(names[0].get("given", [])) or None
        family = names[0].get("family")
        full_name = " ".join([x for x in [given, family] if x]) or None

    birth_date = resource.get("birthDate")
    gender = resource.get("gender")
    deceased = resource.get("deceasedBoolean", False)

    return {
        "patient_id": resource.get("id"),
        "source_file": str(source_file),
        "full_name": full_name,
        "given_name": given,
        "family_name": family,
        "gender": gender,
        "birth_date": birth_date,
        "deceased": deceased,
    }


def extract_encounter(resource, patient_id, source_file):
    encounter_class = get_nested(resource, "class", "code")
    encounter_type_system, encounter_type_code, encounter_type_display = coding_to_text(resource.get("type", [{}])[0] if resource.get("type") else {})
    subject_ref = get_nested(resource, "subject", "reference")
    period_start = get_nested(resource, "period", "start")
    period_end = get_nested(resource, "period", "end")
    service_provider = get_nested(resource, "serviceProvider", "display")

    return {
        "patient_id": patient_id,
        "source_file": str(source_file),
        "resource_id": resource.get("id"),
        "subject_reference": subject_ref,
        "encounter_class": encounter_class,
        "encounter_type_system": encounter_type_system,
        "encounter_type_code": encounter_type_code,
        "encounter_type_display": encounter_type_display,
        "status": resource.get("status"),
        "period_start": period_start,
        "period_end": period_end,
        "service_provider": service_provider,
    }


def extract_condition(resource, patient_id, source_file):
    system, code, display = coding_to_text(resource.get("code"))
    clinical_status = get_nested(resource, "clinicalStatus", "coding")
    verification_status = get_nested(resource, "verificationStatus", "coding")
    encounter_ref = get_nested(resource, "encounter", "reference")

    return {
        "patient_id": patient_id,
        "source_file": str(source_file),
        "resource_id": resource.get("id"),
        "encounter_reference": encounter_ref,
        "code_system": system,
        "code": code,
        "display": display,
        "text": get_nested(resource, "code", "text"),
        "clinical_status": clinical_status[0].get("code") if clinical_status else None,
        "verification_status": verification_status[0].get("code") if verification_status else None,
        "onset_datetime": resource.get("onsetDateTime"),
        "abatement_datetime": resource.get("abatementDateTime"),
        "recorded_date": resource.get("recordedDate"),
    }


def extract_observation(resource, patient_id, source_file):
    system, code, display = coding_to_text(resource.get("code"))
    encounter_ref = get_nested(resource, "encounter", "reference")
    category = resource.get("category", [])
    category_code = None
    if category and isinstance(category, list):
        coding = category[0].get("coding", [])
        if coding:
            category_code = coding[0].get("code")

    value_quantity = resource.get("valueQuantity", {})
    value = value_quantity.get("value")
    unit = value_quantity.get("unit")

    value_string = resource.get("valueString")
    value_codeable = resource.get("valueCodeableConcept", {})
    _, value_code, value_display = coding_to_text(value_codeable)

    return {
        "patient_id": patient_id,
        "source_file": str(source_file),
        "resource_id": resource.get("id"),
        "encounter_reference": encounter_ref,
        "category_code": category_code,
        "code_system": system,
        "code": code,
        "display": display,
        "text": get_nested(resource, "code", "text"),
        "status": resource.get("status"),
        "effective_datetime": resource.get("effectiveDateTime"),
        "issued": resource.get("issued"),
        "value_numeric": value,
        "value_unit": unit,
        "value_string": value_string,
        "value_code": value_code,
        "value_display": value_display,
    }


def extract_medication_request(resource, patient_id, source_file):
    med = resource.get("medicationCodeableConcept", {})
    system, code, display = coding_to_text(med)
    encounter_ref = get_nested(resource, "encounter", "reference")

    return {
        "patient_id": patient_id,
        "source_file": str(source_file),
        "resource_id": resource.get("id"),
        "encounter_reference": encounter_ref,
        "code_system": system,
        "code": code,
        "display": display,
        "text": med.get("text"),
        "status": resource.get("status"),
        "intent": resource.get("intent"),
        "authored_on": resource.get("authoredOn"),
    }


def extract_medication_administration(resource, patient_id, source_file):
    med = resource.get("medicationCodeableConcept", {})
    system, code, display = coding_to_text(med)
    context_ref = get_nested(resource, "context", "reference")

    effective_datetime = resource.get("effectiveDateTime")
    effective_start = get_nested(resource, "effectivePeriod", "start")
    effective_end = get_nested(resource, "effectivePeriod", "end")

    return {
        "patient_id": patient_id,
        "source_file": str(source_file),
        "resource_id": resource.get("id"),
        "context_reference": context_ref,
        "code_system": system,
        "code": code,
        "display": display,
        "text": med.get("text"),
        "status": resource.get("status"),
        "effective_datetime": effective_datetime,
        "effective_period_start": effective_start,
        "effective_period_end": effective_end,
    }


def extract_procedure(resource, patient_id, source_file):
    system, code, display = coding_to_text(resource.get("code"))
    encounter_ref = get_nested(resource, "encounter", "reference")

    return {
        "patient_id": patient_id,
        "source_file": str(source_file),
        "resource_id": resource.get("id"),
        "encounter_reference": encounter_ref,
        "code_system": system,
        "code": code,
        "display": display,
        "text": get_nested(resource, "code", "text"),
        "status": resource.get("status"),
        "performed_datetime": resource.get("performedDateTime"),
        "performed_period_start": get_nested(resource, "performedPeriod", "start"),
        "performed_period_end": get_nested(resource, "performedPeriod", "end"),
    }


def extract_diagnostic_report(resource, patient_id, source_file):
    system, code, display = coding_to_text(resource.get("code"))
    encounter_ref = get_nested(resource, "encounter", "reference")

    return {
        "patient_id": patient_id,
        "source_file": str(source_file),
        "resource_id": resource.get("id"),
        "encounter_reference": encounter_ref,
        "code_system": system,
        "code": code,
        "display": display,
        "text": get_nested(resource, "code", "text"),
        "status": resource.get("status"),
        "effective_datetime": resource.get("effectiveDateTime"),
        "issued": resource.get("issued"),
    }


def write_table(rows, output_path):
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)


def process_bundle(file_path: Path):
    bundle = json.loads(file_path.read_text(encoding="utf-8"))
    entries = bundle.get("entry", [])

    patient_rows = []
    encounter_rows = []
    condition_rows = []
    observation_rows = []
    medication_request_rows = []
    medication_admin_rows = []
    procedure_rows = []
    diagnostic_report_rows = []

    patient_id = None

    for entry in entries:
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "Patient":
            patient = extract_patient(resource, file_path)
            patient_id = patient["patient_id"]
            patient_rows.append(patient)
            break

    if patient_id is None:
        patient_id = file_path.stem

    for entry in entries:
        resource = entry.get("resource", {})
        rtype = resource.get("resourceType")

        if rtype == "Encounter":
            encounter_rows.append(extract_encounter(resource, patient_id, file_path))
        elif rtype == "Condition":
            condition_rows.append(extract_condition(resource, patient_id, file_path))
        elif rtype == "Observation":
            observation_rows.append(extract_observation(resource, patient_id, file_path))
        elif rtype == "MedicationRequest":
            medication_request_rows.append(extract_medication_request(resource, patient_id, file_path))
        elif rtype == "MedicationAdministration":
            medication_admin_rows.append(extract_medication_administration(resource, patient_id, file_path))
        elif rtype == "Procedure":
            procedure_rows.append(extract_procedure(resource, patient_id, file_path))
        elif rtype == "DiagnosticReport":
            diagnostic_report_rows.append(extract_diagnostic_report(resource, patient_id, file_path))

    patient_dir = OUTPUT_DIR / patient_id
    patient_dir.mkdir(parents=True, exist_ok=True)

    write_table(patient_rows, patient_dir / "patient.csv")
    write_table(encounter_rows, patient_dir / "encounters.csv")
    write_table(condition_rows, patient_dir / "conditions.csv")
    write_table(observation_rows, patient_dir / "observations.csv")
    write_table(medication_request_rows, patient_dir / "medication_requests.csv")
    write_table(medication_admin_rows, patient_dir / "medication_administrations.csv")
    write_table(procedure_rows, patient_dir / "procedures.csv")
    write_table(diagnostic_report_rows, patient_dir / "diagnostic_reports.csv")

    metadata = {
        "patient_id": patient_id,
        "source_file": str(file_path),
        "bundle_resource_type": bundle.get("resourceType"),
        "bundle_type": bundle.get("type"),
        "n_entries": len(entries),
        "n_encounters": len(encounter_rows),
        "n_conditions": len(condition_rows),
        "n_observations": len(observation_rows),
        "n_medication_requests": len(medication_request_rows),
        "n_medication_administrations": len(medication_admin_rows),
        "n_procedures": len(procedure_rows),
        "n_diagnostic_reports": len(diagnostic_report_rows),
    }

    with open(patient_dir / "bundle_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(INPUT_DIR.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files found in {INPUT_DIR}")

    for file_path in files:
        process_bundle(file_path)

    print(f"Processed {len(files)} patient files into {OUTPUT_DIR}")


if __name__ == "__main__":
    main()