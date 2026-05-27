from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
META = ROOT / "data" / "metadata"
META.mkdir(parents=True, exist_ok=True)

SIGUNGU_LIST = [
    "천안시", "공주시", "보령시", "아산시", "서산시", "논산시", "계룡시", "당진시",
    "금산군", "부여군", "서천군", "청양군", "홍성군", "예산군", "태안군"
]

FILES = {
    "01_reservoir_sigungu_daily": {
        "path": "01_reservoir_sigungu_daily.csv",
        "required_cols": ["date", "sigungu", "avg_reservoir_rate", "reservoir_risk_score"],
        "score_cols": ["reservoir_risk_score"],
        "expected_sigungu": "allow_missing_gyeryong",
    },
    "01_reservoir_facility_summary": {
        "path": "01_reservoir_facility_summary.csv",
        "required_cols": ["sigungu"],
        "score_cols": [],
        "expected_sigungu": "allow_missing",
    },
    "04_weather_drought_latest": {
        "path": "04_weather_drought_latest_by_sigungu.csv",
        "required_cols": ["sigungu", "weather_drought_risk_score", "weather_drought_level"],
        "score_cols": ["weather_drought_risk_score"],
        "expected_sigungu": "strict_15",
    },
    "03_crop_vulnerability": {
        "path": "03_crop_vulnerability_by_sigungu.csv",
        "required_cols": ["sigungu", "crop_vulnerability_index", "crop_vulnerability_level"],
        "score_cols": ["crop_vulnerability_index"],
        "expected_sigungu": "strict_15",
    },
    "05_agri_impact": {
        "path": "05_agri_impact_by_sigungu.csv",
        "required_cols": ["sigungu", "agri_impact_index", "agri_impact_level"],
        "score_cols": ["agri_impact_index"],
        "expected_sigungu": "strict_15",
    },
    "02_well_latest": {
        "path": "02_well_by_sigungu.csv",
        "required_cols": [
            "sigungu", "well_count", "groundwater_well_count",
            "drilling_developed_well_count", "well_support_score", "well_shortage_score"
        ],
        "score_cols": ["well_support_score", "well_shortage_score"],
        "expected_sigungu": "strict_15",
    },
    "02_well_yearly": {
        "path": "02_well_yearly_by_sigungu.csv",
        "required_cols": ["sigungu", "source_zip", "source_type", "year", "well_count"],
        "score_cols": ["well_support_score", "well_shortage_score"],
        "expected_sigungu": "yearly",
    },
    "02_well_trend": {
        "path": "02_well_trend_by_sigungu.csv",
        "required_cols": ["sigungu"],
        "score_cols": ["well_growth_index"],
        "expected_sigungu": "strict_15",
    },
}

def status(level, name, message):
    return {"level": level, "file_key": name, "message": message}

def check_score_range(df, col):
    if col not in df.columns:
        return [f"missing score column: {col}"]
    s = pd.to_numeric(df[col], errors="coerce")
    problems = []
    if s.notna().sum() == 0:
        problems.append(f"{col}: all NaN")
    if (s.dropna() < 0).any() or (s.dropna() > 100).any():
        problems.append(f"{col}: out of 0~100 range")
    return problems

def check_sigungu(df, mode):
    problems = []
    warnings = []

    if "sigungu" not in df.columns:
        return ["missing sigungu column"], warnings

    actual = sorted(df["sigungu"].dropna().unique().tolist())
    expected = sorted(SIGUNGU_LIST)

    extra = sorted(set(actual) - set(expected))
    missing = sorted(set(expected) - set(actual))

    if extra:
        problems.append(f"extra sigungu values: {extra}")

    if mode == "strict_15":
        if len(actual) != 15:
            problems.append(f"sigungu count should be 15, actual={len(actual)}, missing={missing}")
    elif mode == "allow_missing_gyeryong":
        missing_except_gyeryong = [x for x in missing if x != "계룡시"]
        if missing_except_gyeryong:
            problems.append(f"unexpected missing sigungu: {missing_except_gyeryong}")
        if "계룡시" in missing:
            warnings.append("계룡시 missing: acceptable if no agricultural reservoir data")
    elif mode == "allow_missing":
        if missing:
            warnings.append(f"missing sigungu allowed: {missing}")
    elif mode == "yearly":
        if len(actual) < 10:
            problems.append(f"yearly sigungu coverage too low: {len(actual)}")

    return problems, warnings

def audit_file(key, cfg):
    records = []
    path = PROCESSED / cfg["path"]

    if not path.exists():
        records.append(status("ERROR", key, f"missing file: {path}"))
        return records, None

    try:
        df = pd.read_csv(path)
    except Exception as e:
        records.append(status("ERROR", key, f"cannot read csv: {e}"))
        return records, None

    records.append(status("INFO", key, f"rows={len(df)}, cols={len(df.columns)}"))

    if len(df) == 0:
        records.append(status("ERROR", key, "empty dataframe"))

    for col in cfg["required_cols"]:
        if col not in df.columns:
            records.append(status("ERROR", key, f"missing required column: {col}"))

    sigungu_errors, sigungu_warnings = check_sigungu(df, cfg["expected_sigungu"])
    for msg in sigungu_errors:
        records.append(status("ERROR", key, msg))
    for msg in sigungu_warnings:
        records.append(status("WARN", key, msg))

    if "sigungu" in df.columns and cfg["expected_sigungu"] != "yearly":
        dup = df["sigungu"].duplicated().sum()
        if dup > 0:
            # reservoir daily는 날짜별 반복이라 중복 허용
            if key != "01_reservoir_sigungu_daily":
                records.append(status("ERROR", key, f"duplicated sigungu rows: {dup}"))

    for col in cfg["score_cols"]:
        for msg in check_score_range(df, col):
            records.append(status("ERROR", key, msg))

    return records, df

def audit_well_yearly(df):
    records = []

    if df is None or df.empty:
        return records

    if "source_zip" not in df.columns:
        return records

    summary = (
        df.groupby(["source_type", "year", "source_zip"], dropna=False)
          .agg(
              total_well_count=("well_count", "sum"),
              total_groundwater=("groundwater_well_count", "sum"),
              total_drilling=("drilling_developed_well_count", "sum"),
          )
          .reset_index()
    )

    out_path = META / "well_yearly_source_summary.csv"
    summary.to_csv(out_path, index=False, encoding="utf-8-sig")

    for _, r in summary.iterrows():
        source_zip = str(r["source_zip"])
        source_type = str(r["source_type"])
        total_groundwater = float(r["total_groundwater"])

        if source_type == "groundwater_well":
            if "20231231" in source_zip and total_groundwater < 10000:
                records.append(status("WARN", "02_well_yearly", f"2023 groundwater likely under-counted: {source_zip}, count={total_groundwater}"))
            if "gdb_well_info_new_2024" in source_zip and total_groundwater < 10000:
                records.append(status("WARN", "02_well_yearly", f"2024 gdb groundwater likely under-counted: {source_zip}, count={total_groundwater}"))
            if "20241231" in source_zip and total_groundwater < 50000:
                records.append(status("ERROR", "02_well_yearly", f"latest 20241231 groundwater too low: {source_zip}, count={total_groundwater}"))

    return records

def main():
    all_records = []
    loaded = {}

    for key, cfg in FILES.items():
        records, df = audit_file(key, cfg)
        all_records.extend(records)
        loaded[key] = df

    all_records.extend(audit_well_yearly(loaded.get("02_well_yearly")))

    audit = pd.DataFrame(all_records)
    audit_path = META / "processed_audit_report.csv"
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")

    print()
    print("[AUDIT SUMMARY]")
    print(audit.groupby(["level"]).size())

    print()
    print("[ERRORS / WARNINGS]")
    ew = audit[audit["level"].isin(["ERROR", "WARN"])]
    if len(ew) == 0:
        print("No ERROR/WARN found.")
    else:
        print(ew.to_string(index=False))

    print()
    print(f"[Saved] {audit_path}")
    print(f"[Saved] {META / 'well_yearly_source_summary.csv'}")

if __name__ == "__main__":
    main()
