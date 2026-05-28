from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORT_TABLES = ROOT / "reports" / "tables"
META = ROOT / "data" / "metadata"

REPORT_TABLES.mkdir(parents=True, exist_ok=True)
META.mkdir(parents=True, exist_ok=True)

FEATURE_PATH = PROCESSED / "aquaguard_sigungu_features.csv"
FACILITY_PATH = PROCESSED / "01_reservoir_facility_clean.csv"


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def minmax_0_100(s):
    s = to_num(s)
    if s.notna().sum() == 0:
        return pd.Series(50.0, index=s.index)
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(50.0, index=s.index)
    return ((s - mn) / (mx - mn) * 100).clip(0, 100)


def watch_level(row):
    min_rate = row.get("min_reservoir_rate", np.nan)
    avg_rate = row.get("avg_reservoir_rate", np.nan)
    low30 = row.get("low_reservoir_count_30", 0)
    low40 = row.get("low_reservoir_count_40", 0)
    risk = row.get("reservoir_risk_score", np.nan)

    min_rate = np.nan if pd.isna(min_rate) else float(min_rate)
    avg_rate = np.nan if pd.isna(avg_rate) else float(avg_rate)
    low30 = 0 if pd.isna(low30) else float(low30)
    low40 = 0 if pd.isna(low40) else float(low40)
    risk = 0 if pd.isna(risk) else float(risk)

    if (not pd.isna(min_rate) and min_rate <= 30) or low30 >= 1 or risk >= 80:
        return "심각후보"
    if (not pd.isna(min_rate) and min_rate <= 40) or low40 >= 1 or risk >= 60:
        return "경계후보"
    if (not pd.isna(avg_rate) and avg_rate < 70) or risk >= 40:
        return "주의후보"
    return "정상"


def watch_reason(row):
    reasons = []

    if pd.notna(row.get("min_reservoir_rate")) and float(row["min_reservoir_rate"]) <= 30:
        reasons.append("최저 저수율 30% 이하")
    elif pd.notna(row.get("min_reservoir_rate")) and float(row["min_reservoir_rate"]) <= 40:
        reasons.append("최저 저수율 40% 이하")

    if row.get("low_reservoir_count_30", 0) >= 1:
        reasons.append("30% 이하 저수지 존재")
    elif row.get("low_reservoir_count_40", 0) >= 1:
        reasons.append("40% 이하 저수지 존재")

    if pd.notna(row.get("avg_reservoir_rate")) and float(row["avg_reservoir_rate"]) < 70:
        reasons.append("시군 평균 저수율 70% 미만")

    if pd.notna(row.get("reservoir_risk_score")) and float(row["reservoir_risk_score"]) >= 40:
        reasons.append("저수율 위험도 주의 이상")

    if not reasons:
        reasons.append("정상 모니터링")

    return " / ".join(reasons)


def build_watchlist(features):
    cols = [
        "sigungu",
        "reservoir_latest_date",
        "avg_reservoir_rate",
        "min_reservoir_rate",
        "max_reservoir_rate",
        "reservoir_count",
        "low_reservoir_count_40",
        "low_reservoir_count_30",
        "total_effective_capacity",
        "reservoir_risk_score",
        "final_water_risk_score",
        "final_water_risk_level",
        "main_risk_driver",
    ]

    cols = [c for c in cols if c in features.columns]
    out = features[cols].copy()

    numeric_cols = [
        "avg_reservoir_rate",
        "min_reservoir_rate",
        "max_reservoir_rate",
        "reservoir_count",
        "low_reservoir_count_40",
        "low_reservoir_count_30",
        "total_effective_capacity",
        "reservoir_risk_score",
        "final_water_risk_score",
    ]

    for c in numeric_cols:
        if c in out.columns:
            out[c] = to_num(out[c])

    out["watch_level"] = out.apply(watch_level, axis=1)
    out["watch_reason"] = out.apply(watch_reason, axis=1)

    level_order = {
        "심각후보": 4,
        "경계후보": 3,
        "주의후보": 2,
        "정상": 1,
    }

    out["watch_priority_score"] = (
        out["watch_level"].map(level_order).fillna(0) * 25
        + out.get("reservoir_risk_score", 0).fillna(0) * 0.5
        + out.get("final_water_risk_score", 0).fillna(0) * 0.2
    ).clip(0, 100)

    out = out.sort_values(
        ["watch_priority_score", "reservoir_risk_score", "final_water_risk_score"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    out["watch_rank"] = np.arange(1, len(out) + 1)

    front = ["watch_rank", "sigungu", "watch_level", "watch_reason", "watch_priority_score"]
    other = [c for c in out.columns if c not in front]
    return out[front + other]


def build_facility_status(features, facilities):
    sigungu_cols = [
        "sigungu",
        "reservoir_latest_date",
        "avg_reservoir_rate",
        "min_reservoir_rate",
        "reservoir_risk_score",
        "final_water_risk_score",
        "final_water_risk_level",
    ]
    sigungu_cols = [c for c in sigungu_cols if c in features.columns]

    sigungu_info = features[sigungu_cols].copy()
    sigungu_info = sigungu_info.drop_duplicates(["sigungu"], keep="first")

    df = facilities.copy()

    # 숫자 컬럼 정리
    for c in ["benefit_area", "effective_capacity", "total_capacity", "basin_area", "full_water_area"]:
        if c in df.columns:
            df[c] = to_num(df[c]).fillna(0)

    # 문자열 컬럼 정리
    for c in ["sigungu", "facility_name", "address", "source_file"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str).str.strip()

    # 핵심: 동일 시군 + 동일 저수지명 + 동일 주소 중복 제거
    # 원천 데이터에 같은 시설이 여러 번 반복되어 대시보드 표가 중복되는 문제를 방지한다.
    dedup_cols = [c for c in ["sigungu", "facility_name", "address"] if c in df.columns]

    if dedup_cols:
        before_rows = len(df)

        sort_cols = [c for c in ["benefit_area", "effective_capacity", "total_capacity"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(
                sort_cols,
                ascending=[False] * len(sort_cols),
                na_position="last",
            )

        df = df.drop_duplicates(subset=dedup_cols, keep="first").copy()
        after_rows = len(df)
        print(f"[DEDUP facility_status] removed={before_rows - after_rows}, rows={after_rows}")
    else:
        print("[WARN] facility dedup columns not found. Skip dedup.")

    df = df.merge(
        sigungu_info.rename(columns={
            "avg_reservoir_rate": "sigungu_avg_reservoir_rate",
            "min_reservoir_rate": "sigungu_min_reservoir_rate",
            "reservoir_risk_score": "sigungu_reservoir_risk_score",
            "final_water_risk_score": "sigungu_final_water_risk_score",
            "final_water_risk_level": "sigungu_final_water_risk_level",
        }),
        on="sigungu",
        how="left"
    )

    df["benefit_area_index"] = minmax_0_100(df.get("benefit_area", 0))
    df["effective_capacity_index"] = minmax_0_100(df.get("effective_capacity", 0))

    df["inspection_priority_score"] = (
        df["sigungu_reservoir_risk_score"].fillna(50) * 0.50
        + df["benefit_area_index"] * 0.25
        + df["effective_capacity_index"] * 0.25
    ).clip(0, 100)

    df["reservoir_status_note"] = df.apply(
        lambda r: "시군 저수율 위험 높음"
        if pd.notna(r.get("sigungu_reservoir_risk_score")) and r["sigungu_reservoir_risk_score"] >= 40
        else "일반 모니터링",
        axis=1
    )

    keep = [
        "sigungu",
        "facility_name",
        "address",
        "benefit_area",
        "effective_capacity",
        "total_capacity",
        "basin_area",
        "full_water_area",
        "reservoir_latest_date",
        "sigungu_avg_reservoir_rate",
        "sigungu_min_reservoir_rate",
        "sigungu_reservoir_risk_score",
        "sigungu_final_water_risk_score",
        "sigungu_final_water_risk_level",
        "inspection_priority_score",
        "reservoir_status_note",
        "source_file",
    ]
    keep = [c for c in keep if c in df.columns]

    out = df[keep].copy()

    # 출력 직전 2차 중복 방어
    final_dedup_cols = [c for c in ["sigungu", "facility_name", "address"] if c in out.columns]
    if final_dedup_cols:
        before_rows = len(out)
        out = out.drop_duplicates(subset=final_dedup_cols, keep="first").copy()
        print(f"[DEDUP final_output] removed={before_rows - len(out)}, rows={len(out)}")

    out = out.sort_values(
        ["sigungu", "inspection_priority_score", "benefit_area", "effective_capacity"],
        ascending=[True, False, False, False]
    ).reset_index(drop=True)

    return out


def write_method_doc():
    text = """# AquaGuard AI 저수지 현황 및 이상탐지 Watchlist

## 목적

제출 제안서의 MVP 범위에 포함된 저수지 현황 테이블과 저수율 이상탐지 대시보드 기능을 구현하기 위한 보조 산출물이다.

## 산출물

- reports/tables/reservoir_watchlist.csv
- reports/tables/reservoir_facility_status_by_sigungu.csv
- data/processed/reservoir_facility_status_for_dashboard.csv

## Watchlist 판정 기준

- 심각후보: 최저 저수율 30% 이하, 30% 이하 저수지 존재, 또는 저수율 위험도 80 이상
- 경계후보: 최저 저수율 40% 이하, 40% 이하 저수지 존재, 또는 저수율 위험도 60 이상
- 주의후보: 평균 저수율 70% 미만 또는 저수율 위험도 40 이상
- 정상: 위 조건에 해당하지 않는 경우

## 해석 주의

MVP 단계에서는 공개데이터 기반 규칙형 Watchlist로 구현한다.
이는 실제 현장 이상 여부를 확정하는 기능이 아니라, 행정 담당자가 우선 점검할 후보를 좁히는 참고 지표이다.
향후 장기 시계열이 충분히 확보되면 Isolation Forest, RandomForest, LightGBM 기반 이상탐지로 고도화할 수 있다.
"""
    path = META / "reservoir_watchlist_method.md"
    path.write_text(text, encoding="utf-8")
    return path


def main():
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(f"Missing feature file: {FEATURE_PATH}")
    if not FACILITY_PATH.exists():
        raise FileNotFoundError(f"Missing facility file: {FACILITY_PATH}")

    features = pd.read_csv(FEATURE_PATH)
    facilities = pd.read_csv(FACILITY_PATH)

    watchlist = build_watchlist(features)
    facility_status = build_facility_status(features, facilities)

    watchlist_path = REPORT_TABLES / "reservoir_watchlist.csv"
    facility_table_path = REPORT_TABLES / "reservoir_facility_status_by_sigungu.csv"
    dashboard_path = PROCESSED / "reservoir_facility_status_for_dashboard.csv"
    method_path = write_method_doc()

    watchlist.to_csv(watchlist_path, index=False, encoding="utf-8-sig")
    facility_status.to_csv(facility_table_path, index=False, encoding="utf-8-sig")
    facility_status.to_csv(dashboard_path, index=False, encoding="utf-8-sig")

    print("[Saved]")
    print(f"- {watchlist_path} rows={len(watchlist)}")
    print(f"- {facility_table_path} rows={len(facility_status)}")
    print(f"- {dashboard_path} rows={len(facility_status)}")
    print(f"- {method_path}")

    print()
    print("[Watchlist Preview]")
    print(watchlist[[
        "watch_rank",
        "sigungu",
        "watch_level",
        "watch_reason",
        "avg_reservoir_rate",
        "min_reservoir_rate",
        "reservoir_risk_score",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
