from pathlib import Path
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "README.md",
    "app.py",
    "pages/01_Reservoir_Watchlist.py",

    "data/processed/aquaguard_sigungu_features.csv",
    "data/processed/aquaguard_priority_top15.csv",
    "data/processed/alternative_source_candidates.csv",
    "data/processed/reservoir_facility_status_for_dashboard.csv",

    "reports/tables/alternative_source_top5_by_sigungu.csv",
    "reports/tables/top_priority_summary.csv",
    "reports/tables/main_driver_summary.csv",
    "reports/tables/reservoir_watchlist.csv",
    "reports/tables/reservoir_facility_status_by_sigungu.csv",

    "reports/figures/01_final_risk_ranking.png",
    "reports/figures/02_risk_components_stacked.png",
    "reports/figures/03_reservoir_vs_alternative_shortage_scatter.png",
    "reports/figures/04_top5_priority_table.png",
    "reports/figures/05_alternative_source_top1_by_risk_area.png",

    "data/metadata/final_feature_formula.md",
    "data/metadata/alternative_source_recommendation_method.md",
    "data/metadata/reservoir_watchlist_method.md",
    "data/metadata/processed_audit_report.csv",
    "data/metadata/visual_generation_summary.md",

    "docs/DEMO_SCENARIO.md",
    "docs/SUBMISSION_CHECKLIST.md",
]

EXPECTED_SIGUNGU_COUNT = 15
EXPECTED_CANDIDATE_ROWS = 75


def row(check, target, status, detail=""):
    return {
        "check": check,
        "target": target,
        "status": status,
        "detail": detail,
    }


def main():
    records = []

    for rel in REQUIRED_FILES:
        path = ROOT / rel
        records.append(row(
            "file_exists",
            rel,
            "PASS" if path.exists() else "FAIL",
            "" if path.exists() else "missing file",
        ))

    feature_path = ROOT / "data/processed/aquaguard_sigungu_features.csv"
    if feature_path.exists():
        df = pd.read_csv(feature_path)
        score = pd.to_numeric(df["final_water_risk_score"], errors="coerce")

        records.append(row("feature_rows", "aquaguard_sigungu_features.csv", "PASS" if len(df) == 15 else "FAIL", f"rows={len(df)}"))
        records.append(row("feature_sigungu_unique", "sigungu", "PASS" if df["sigungu"].nunique() == 15 else "FAIL", f"unique={df['sigungu'].nunique()}"))
        records.append(row("final_score_missing", "final_water_risk_score", "PASS" if score.isna().sum() == 0 else "FAIL", f"missing={score.isna().sum()}"))
        records.append(row("final_score_range", "final_water_risk_score", "PASS" if ((score >= 0) & (score <= 100)).all() else "FAIL", f"min={score.min()}, max={score.max()}"))

        required_cols = [
            "rain_shortage_score",
            "reservoir_risk_score",
            "groundwater_dependency_score",
            "crop_water_demand_score",
            "alternative_source_access_shortage_score",
        ]
        for c in required_cols:
            records.append(row("feature_required_col", c, "PASS" if c in df.columns else "FAIL"))

    cand_path = ROOT / "reports/tables/alternative_source_top5_by_sigungu.csv"
    if cand_path.exists():
        cdf = pd.read_csv(cand_path)
        dup = cdf.duplicated(["target_sigungu", "candidate_reservoir_name", "candidate_sigungu"]).sum()
        min_unique = cdf.groupby("target_sigungu")["candidate_reservoir_name"].nunique().min()

        records.append(row("candidate_rows", "alternative_source_top5_by_sigungu.csv", "PASS" if len(cdf) == 75 else "FAIL", f"rows={len(cdf)}"))
        records.append(row("candidate_target_count", "target_sigungu", "PASS" if cdf["target_sigungu"].nunique() == 15 else "FAIL", f"targets={cdf['target_sigungu'].nunique()}"))
        records.append(row("candidate_duplicate_count", "candidate duplicate", "PASS" if dup == 0 else "FAIL", f"duplicate_count={dup}"))
        records.append(row("candidate_top5_unique", "candidate_reservoir_name", "PASS" if min_unique >= 5 else "FAIL", f"min_unique={min_unique}"))

    watch_path = ROOT / "reports/tables/reservoir_watchlist.csv"
    if watch_path.exists():
        wdf = pd.read_csv(watch_path)
        records.append(row("watchlist_rows", "reservoir_watchlist.csv", "PASS" if len(wdf) == 15 else "FAIL", f"rows={len(wdf)}"))
        records.append(row("watchlist_sigungu_unique", "sigungu", "PASS" if wdf["sigungu"].nunique() == 15 else "FAIL", f"unique={wdf['sigungu'].nunique()}"))

    facility_path = ROOT / "data/processed/reservoir_facility_status_for_dashboard.csv"
    if facility_path.exists():
        fdf = pd.read_csv(facility_path)
        records.append(row("facility_status_non_empty", "reservoir_facility_status_for_dashboard.csv", "PASS" if len(fdf) > 0 else "FAIL", f"rows={len(fdf)}"))

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
