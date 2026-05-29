from pathlib import Path
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "README.md",
    "app.py",
    "pages/01_Reservoir_Watchlist.py",
    "pages/02_Deep_AI_Insights.py",
    "pages/03_Live_Data_Update.py",

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

    # Live data outputs
    "data/processed/latest_weather_30d_by_sigungu.csv",
    "data/processed/latest_kma_aws_weather_30d_by_sigungu.csv",
    "data/processed/latest_adms_soil_moisture_by_sigungu.csv",
    "data/processed/latest_adms_reservoir_support_by_sigungu.csv",
    "data/processed/latest_live_sigungu_features.csv",
    "reports/tables/latest_live_data_status.csv",
    "reports/tables/latest_live_risk_summary.csv",
    "reports/tables/latest_oldam_status_summary.csv",
    "reports/tables/latest_kma_aws_weather_status.csv",
    "reports/tables/latest_adms_soil_moisture_status.csv",
    "reports/tables/latest_adms_reservoir_support_status.csv",
    "reports/tables/latest_reservoir_source_crosscheck.csv",
]

EXPECTED_SIGUNGU_COUNT = 15
EXPECTED_CANDIDATE_ROWS = 75
REQUIRED_REPORT_FIGURES = [
    "reports/figures/01_final_risk_ranking.png",
    "reports/figures/02_risk_components_stacked.png",
    "reports/figures/03_reservoir_vs_alternative_shortage_scatter.png",
    "reports/figures/04_top5_priority_table.png",
    "reports/figures/05_alternative_source_top1_by_risk_area.png",
]


def row(check, target, status, detail=""):
    return {
        "check": check,
        "target": target,
        "status": status,
        "detail": detail,
    }


def safe_read_csv(rel_path):
    path = ROOT / rel_path
    if not path.exists():
        return None
    return pd.read_csv(path)


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

    for rel in REQUIRED_REPORT_FIGURES:
        path = ROOT / rel
        is_non_empty = path.exists() and path.stat().st_size > 0
        detail = f"bytes={path.stat().st_size}" if path.exists() else "missing file"
        records.append(row(
            "figure_non_empty",
            rel,
            "PASS" if is_non_empty else "FAIL",
            detail,
        ))

    # Static / PDF 기준 feature 검증
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

    # 대체수원 추천 검증
    cand_path = ROOT / "reports/tables/alternative_source_top5_by_sigungu.csv"
    if cand_path.exists():
        cdf = pd.read_csv(cand_path)
        dup = cdf.duplicated(["target_sigungu", "candidate_reservoir_name", "candidate_sigungu"]).sum()
        min_unique = cdf.groupby("target_sigungu")["candidate_reservoir_name"].nunique().min()

        records.append(row("candidate_rows", "alternative_source_top5_by_sigungu.csv", "PASS" if len(cdf) == 75 else "FAIL", f"rows={len(cdf)}"))
        records.append(row("candidate_target_count", "target_sigungu", "PASS" if cdf["target_sigungu"].nunique() == 15 else "FAIL", f"targets={cdf['target_sigungu'].nunique()}"))
        records.append(row("candidate_duplicate_count", "candidate duplicate", "PASS" if dup == 0 else "FAIL", f"duplicate_count={dup}"))
        records.append(row("candidate_top5_unique", "candidate_reservoir_name", "PASS" if min_unique >= 5 else "FAIL", f"min_unique={min_unique}"))

    # 저수지 watchlist 검증
    watch_path = ROOT / "reports/tables/reservoir_watchlist.csv"
    if watch_path.exists():
        wdf = pd.read_csv(watch_path)
        records.append(row("watchlist_rows", "reservoir_watchlist.csv", "PASS" if len(wdf) == 15 else "FAIL", f"rows={len(wdf)}"))
        records.append(row("watchlist_sigungu_unique", "sigungu", "PASS" if wdf["sigungu"].nunique() == 15 else "FAIL", f"unique={wdf['sigungu'].nunique()}"))

    facility_path = ROOT / "data/processed/reservoir_facility_status_for_dashboard.csv"
    if facility_path.exists():
        fdf = pd.read_csv(facility_path)
        records.append(row("facility_status_non_empty", "reservoir_facility_status_for_dashboard.csv", "PASS" if len(fdf) > 0 else "FAIL", f"rows={len(fdf)}"))

        dup_cols = [c for c in ["sigungu", "facility_name", "address"] if c in fdf.columns]
        if dup_cols:
            dup_count = int(fdf.duplicated(dup_cols).sum())
            records.append(row("facility_duplicate_count", "reservoir_facility_status_for_dashboard.csv", "PASS" if dup_count == 0 else "FAIL", f"duplicate_count={dup_count}"))

        priority_cols = [
            "facility_priority_rank",
            "facility_scale_score",
            "inspection_priority_score",
            "facility_priority_reason",
            "facility_priority_level",
        ]
        for c in priority_cols:
            records.append(row("facility_priority_col", c, "PASS" if c in fdf.columns else "FAIL"))

        if {"sigungu", "facility_priority_rank"}.issubset(fdf.columns):
            rank_missing = int(pd.to_numeric(fdf["facility_priority_rank"], errors="coerce").isna().sum())
            records.append(row("facility_priority_rank_missing", "facility_priority_rank", "PASS" if rank_missing == 0 else "FAIL", f"missing={rank_missing}"))

        if {"sigungu", "inspection_priority_score"}.issubset(fdf.columns):
            score = pd.to_numeric(fdf["inspection_priority_score"], errors="coerce")
            records.append(row("facility_priority_score_missing", "inspection_priority_score", "PASS" if score.isna().sum() == 0 else "FAIL", f"missing={score.isna().sum()}"))
            records.append(row("facility_priority_score_range", "inspection_priority_score", "PASS" if ((score >= 0) & (score <= 100)).all() else "FAIL", f"min={score.min()}, max={score.max()}"))
            dangjin = fdf[fdf["sigungu"] == "당진시"].copy()
            if len(dangjin) > 1:
                spread = pd.to_numeric(dangjin["inspection_priority_score"], errors="coerce").max() - pd.to_numeric(dangjin["inspection_priority_score"], errors="coerce").min()
                records.append(row("facility_priority_spread", "당진시", "PASS" if spread > 0 else "FAIL", f"spread={spread:.3f}"))

    # Live 데이터 검증
    live_status = safe_read_csv("reports/tables/latest_live_data_status.csv")
    if live_status is not None and len(live_status):
        s = live_status.iloc[-1]

        records.append(row("live_feature_rows", "latest_live_data_status.csv", "PASS" if int(s.get("live_feature_rows", 0)) == 15 else "FAIL", f"live_feature_rows={s.get('live_feature_rows')}"))
        records.append(row("live_weather_coverage", "live_weather_kma_count", "PASS" if int(s.get("live_weather_kma_count", 0)) == 15 else "FAIL", f"live_weather_kma_count={s.get('live_weather_kma_count')}"))
        records.append(row("live_soil_coverage", "live_soil_adms_count", "PASS" if int(s.get("live_soil_adms_count", 0)) == 15 else "FAIL", f"live_soil_adms_count={s.get('live_soil_adms_count')}"))
        records.append(row("live_status_success", "latest_live_data_status.csv", "PASS" if str(s.get("status", "")) == "SUCCESS" else "FAIL", f"status={s.get('status')}"))

    live_feature = safe_read_csv("data/processed/latest_live_sigungu_features.csv")
    if live_feature is not None:
        records.append(row("live_feature_sigungu_unique", "latest_live_sigungu_features.csv", "PASS" if live_feature["sigungu"].nunique() == 15 else "FAIL", f"unique={live_feature['sigungu'].nunique()}"))

        live_score = pd.to_numeric(live_feature["final_live_water_risk_score"], errors="coerce")
        records.append(row("live_score_missing", "final_live_water_risk_score", "PASS" if live_score.isna().sum() == 0 else "FAIL", f"missing={live_score.isna().sum()}"))
        records.append(row("live_score_range", "final_live_water_risk_score", "PASS" if ((live_score >= 0) & (live_score <= 100)).all() else "FAIL", f"min={live_score.min()}, max={live_score.max()}"))

        for c in ["live_weather_source", "live_reservoir_source", "live_soil_source"]:
            records.append(row("live_required_col", c, "PASS" if c in live_feature.columns else "FAIL"))

    oldam_status = safe_read_csv("reports/tables/latest_oldam_status_summary.csv")
    if oldam_status is not None and len(oldam_status):
        s = oldam_status.iloc[-1]
        records.append(row("oldam_status_success", "latest_oldam_status_summary.csv", "PASS" if str(s.get("status", "")) == "SUCCESS" else "FAIL", f"status={s.get('status')}"))
        records.append(row("oldam_matched_rows", "matched_rows", "PASS" if int(s.get("matched_rows", 0)) > 0 else "FAIL", f"matched_rows={s.get('matched_rows')}"))
        records.append(row("oldam_sigungu_count", "sigungu_count", "PASS" if int(s.get("sigungu_count", 0)) >= 5 else "FAIL", f"sigungu_count={s.get('sigungu_count')}"))

    aws_status = safe_read_csv("reports/tables/latest_kma_aws_weather_status.csv")
    if aws_status is not None and len(aws_status):
        s = aws_status.iloc[-1]
        records.append(row("aws_status_success", "latest_kma_aws_weather_status.csv", "PASS" if str(s.get("status", "")) == "SUCCESS" else "FAIL", f"status={s.get('status')}"))
        records.append(row("aws_sigungu_count", "sigungu_count", "PASS" if int(s.get("sigungu_count", 0)) == 15 else "FAIL", f"sigungu_count={s.get('sigungu_count')}"))

    soil_status = safe_read_csv("reports/tables/latest_adms_soil_moisture_status.csv")
    if soil_status is not None and len(soil_status):
        s = soil_status.iloc[-1]
        records.append(row("soil_status_success", "latest_adms_soil_moisture_status.csv", "PASS" if str(s.get("status", "")) == "SUCCESS" else "FAIL", f"status={s.get('status')}"))
        records.append(row("soil_sigungu_count", "sigungu_count", "PASS" if int(s.get("sigungu_count", 0)) == 15 else "FAIL", f"sigungu_count={s.get('sigungu_count')}"))

    adms_rvow_status = safe_read_csv("reports/tables/latest_adms_reservoir_support_status.csv")
    if adms_rvow_status is not None and len(adms_rvow_status):
        s = adms_rvow_status.iloc[-1]
        records.append(row("adms_rvow_status_success", "latest_adms_reservoir_support_status.csv", "PASS" if str(s.get("status", "")) == "SUCCESS" else "FAIL", f"status={s.get('status')}"))
        records.append(row("adms_rvow_success_rows", "success_rows", "PASS" if int(s.get("success_rows", 0)) == 15 else "FAIL", f"success_rows={s.get('success_rows')}"))

    crosscheck = safe_read_csv("reports/tables/latest_reservoir_source_crosscheck.csv")
    if crosscheck is not None:
        records.append(row("crosscheck_rows", "latest_reservoir_source_crosscheck.csv", "PASS" if len(crosscheck) == 15 else "FAIL", f"rows={len(crosscheck)}"))
        if "crosscheck_status" in crosscheck.columns:
            both_count = int((crosscheck["crosscheck_status"] == "ADMS_AND_OLDAM").sum())
            records.append(row("crosscheck_oldam_overlap", "ADMS_AND_OLDAM", "PASS" if both_count >= 5 else "WARN", f"ADMS_AND_OLDAM={both_count}"))

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
