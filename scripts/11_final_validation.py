from pathlib import Path
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "README.md",
    "app.py",

    "data/processed/aquaguard_sigungu_features.csv",
    "data/processed/aquaguard_priority_top15.csv",
    "data/processed/alternative_source_candidates.csv",

    "reports/tables/alternative_source_top5_by_sigungu.csv",
    "reports/tables/top_priority_summary.csv",
    "reports/tables/main_driver_summary.csv",

    "reports/figures/01_final_risk_ranking.png",
    "reports/figures/02_risk_components_stacked.png",
    "reports/figures/03_reservoir_vs_alternative_shortage_scatter.png",
    "reports/figures/04_top5_priority_table.png",
    "reports/figures/05_alternative_source_top1_by_risk_area.png",

    "data/metadata/final_feature_formula.md",
    "data/metadata/alternative_source_recommendation_method.md",
    "data/metadata/processed_audit_report.csv",
    "data/metadata/visual_generation_summary.md",

    "docs/DEMO_SCENARIO.md",
    "docs/SUBMISSION_CHECKLIST.md",
]

EXPECTED_SIGUNGU_COUNT = 15
EXPECTED_CANDIDATE_ROWS = 75


def check_file_exists():
    rows = []
    for rel in REQUIRED_FILES:
        path = ROOT / rel
        rows.append({
            "check": "file_exists",
            "target": rel,
            "status": "PASS" if path.exists() else "FAIL",
            "detail": "" if path.exists() else "missing file",
        })
    return rows


def check_features():
    rows = []
    path = ROOT / "data/processed/aquaguard_sigungu_features.csv"

    if not path.exists():
        return [{
            "check": "features",
            "target": str(path),
            "status": "FAIL",
            "detail": "feature file missing",
        }]

    df = pd.read_csv(path)

    required_cols = [
        "sigungu",
        "final_priority_rank",
        "final_water_risk_score",
        "final_water_risk_level",
        "main_risk_driver",
        "rain_shortage_score",
        "reservoir_risk_score",
        "groundwater_dependency_score",
        "crop_water_demand_score",
        "alternative_source_access_shortage_score",
    ]

    for col in required_cols:
        rows.append({
            "check": "feature_required_column",
            "target": col,
            "status": "PASS" if col in df.columns else "FAIL",
            "detail": "",
        })

    rows.append({
        "check": "feature_row_count",
        "target": "aquaguard_sigungu_features.csv",
        "status": "PASS" if len(df) == EXPECTED_SIGUNGU_COUNT else "FAIL",
        "detail": f"rows={len(df)}",
    })

    rows.append({
        "check": "feature_sigungu_unique",
        "target": "sigungu",
        "status": "PASS" if df["sigungu"].nunique() == EXPECTED_SIGUNGU_COUNT else "FAIL",
        "detail": f"unique={df['sigungu'].nunique()}",
    })

    score = pd.to_numeric(df["final_water_risk_score"], errors="coerce")

    rows.append({
        "check": "final_score_not_null",
        "target": "final_water_risk_score",
        "status": "PASS" if score.isna().sum() == 0 else "FAIL",
        "detail": f"missing={score.isna().sum()}",
    })

    rows.append({
        "check": "final_score_range",
        "target": "final_water_risk_score",
        "status": "PASS" if ((score >= 0) & (score <= 100)).all() else "FAIL",
        "detail": f"min={score.min()}, max={score.max()}",
    })

    return rows


def check_candidates():
    rows = []
    path = ROOT / "reports/tables/alternative_source_top5_by_sigungu.csv"

    if not path.exists():
        return [{
            "check": "candidates",
            "target": str(path),
            "status": "FAIL",
            "detail": "candidate file missing",
        }]

    df = pd.read_csv(path)

    rows.append({
        "check": "candidate_row_count",
        "target": "alternative_source_top5_by_sigungu.csv",
        "status": "PASS" if len(df) == EXPECTED_CANDIDATE_ROWS else "FAIL",
        "detail": f"rows={len(df)}",
    })

    rows.append({
        "check": "candidate_target_count",
        "target": "target_sigungu",
        "status": "PASS" if df["target_sigungu"].nunique() == EXPECTED_SIGUNGU_COUNT else "FAIL",
        "detail": f"targets={df['target_sigungu'].nunique()}",
    })

    by_target = df.groupby("target_sigungu")["candidate_reservoir_name"].nunique()
    min_unique = int(by_target.min())

    rows.append({
        "check": "candidate_top5_unique_per_target",
        "target": "candidate_reservoir_name",
        "status": "PASS" if min_unique >= 5 else "FAIL",
        "detail": f"min_unique_candidates={min_unique}",
    })

    dup = df.duplicated(["target_sigungu", "candidate_reservoir_name", "candidate_sigungu"]).sum()

    rows.append({
        "check": "candidate_duplicate_count",
        "target": "target_sigungu + candidate_reservoir_name + candidate_sigungu",
        "status": "PASS" if dup == 0 else "FAIL",
        "detail": f"duplicate_count={dup}",
    })

    return rows


def check_docs_text():
    rows = []

    readme = ROOT / "README.md"
    if readme.exists():
        text = readme.read_text(encoding="utf-8", errors="ignore")
        keywords = [
            "25:20",
            "대체 수원 후보",
            "streamlit run app.py",
            "관로, 수리권, 수질",
        ]

        for kw in keywords:
            rows.append({
                "check": "readme_keyword",
                "target": kw,
                "status": "PASS" if kw in text else "WARN",
                "detail": "",
            })

    app = ROOT / "app.py"
    if app.exists():
        text = app.read_text(encoding="utf-8", errors="ignore")
        keywords = [
            "st.set_page_config",
            "HTML 행정 리포트 다운로드",
            "alternative_source",
            "scatter_mapbox",
        ]

        for kw in keywords:
            rows.append({
                "check": "app_keyword",
                "target": kw,
                "status": "PASS" if kw in text else "FAIL",
                "detail": "",
            })

    return rows


def main():
    records = []
    records.extend(check_file_exists())
    records.extend(check_features())
    records.extend(check_candidates())
    records.extend(check_docs_text())

    report = pd.DataFrame(records)
    out_path = ROOT / "data/metadata/final_validation_report.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("[FINAL VALIDATION SUMMARY]")
    print(report.groupby("status").size())

    problems = report[report["status"].isin(["FAIL", "WARN"])]

    print()
    print("[FAIL/WARN]")
    if problems.empty:
        print("No FAIL/WARN found.")
    else:
        print(problems.to_string(index=False))

    print()
    print(f"[Saved] {out_path}")

    if (report["status"] == "FAIL").any():
        sys.exit(1)


if __name__ == "__main__":
    main()
