from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORT_TABLES = ROOT / "reports" / "tables"
META = ROOT / "data" / "metadata"

REPORT_TABLES.mkdir(parents=True, exist_ok=True)
META.mkdir(parents=True, exist_ok=True)

SIGUNGU_LIST = [
    "천안시", "공주시", "보령시", "아산시", "서산시", "논산시", "계룡시", "당진시",
    "금산군", "부여군", "서천군", "청양군", "홍성군", "예산군", "태안군"
]

# 제출 PDF 기준 산식
PDF_WEIGHTS = {
    "rain_shortage_score": 0.25,
    "reservoir_risk_score": 0.25,
    "groundwater_dependency_score": 0.20,
    "crop_water_demand_score": 0.20,
    "alternative_source_access_shortage_score": 0.10,
}

DRIVER_LABELS = {
    "rain_shortage_score": "강우 부족도",
    "reservoir_risk_score": "저수율 위험도",
    "groundwater_dependency_score": "관정 의존도",
    "crop_water_demand_score": "작물 물수요 지수",
    "alternative_source_access_shortage_score": "대체 수원 접근성 부족도",
}


def read_csv(filename):
    path = PROCESSED / filename
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


def minmax_0_100(s):
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(np.nan, index=s.index)
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(50.0, index=s.index)
    return ((s - mn) / (mx - mn) * 100).clip(0, 100)


def ensure_score(df, col, fill_value=50):
    if col not in df.columns:
        df[col] = np.nan
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[f"{col}_missing"] = df[col].isna().astype(int)
    df[f"{col}_for_score"] = df[col].fillna(fill_value).clip(0, 100)
    return df


def risk_level(score):
    if pd.isna(score):
        return "판정불가"
    if score >= 80:
        return "심각"
    if score >= 60:
        return "경계"
    if score >= 40:
        return "주의"
    return "낮음"


def recommended_action(level):
    if level == "심각":
        return "즉시 현장점검 및 대체 수원 확보 검토"
    if level == "경계":
        return "우선 점검 대상 지정 및 대체 수원 후보 검토"
    if level == "주의":
        return "정기 점검 강화 및 농업용수 수요 변화 관찰"
    if level == "낮음":
        return "일반 모니터링 유지"
    return "데이터 보완 후 재판정"


def load_reservoir_latest():
    df = read_csv("01_reservoir_sigungu_daily.csv")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["sigungu", "date"])
    latest = df.groupby("sigungu", as_index=False).tail(1).copy()

    keep = [
        "sigungu", "date", "avg_reservoir_rate", "min_reservoir_rate",
        "max_reservoir_rate", "reservoir_count", "low_reservoir_count_40",
        "low_reservoir_count_30", "total_effective_capacity",
        "reservoir_risk_score", "risk_level_by_reservoir"
    ]
    keep = [c for c in keep if c in latest.columns]
    latest = latest[keep].rename(columns={"date": "reservoir_latest_date"})

    base = pd.DataFrame({"sigungu": SIGUNGU_LIST})
    out = base.merge(latest, on="sigungu", how="left")
    out["reservoir_data_missing"] = out["reservoir_risk_score"].isna().astype(int)

    for col in ["reservoir_count", "low_reservoir_count_40", "low_reservoir_count_30", "total_effective_capacity"]:
        if col not in out.columns:
            out[col] = 0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    if "risk_level_by_reservoir" not in out.columns:
        out["risk_level_by_reservoir"] = np.nan

    out.loc[out["reservoir_data_missing"] == 1, "risk_level_by_reservoir"] = "데이터없음"
    return out


def load_weather():
    df = read_csv("04_weather_drought_latest_by_sigungu.csv")

    keep = [
        "sigungu", "date", "drought_stage_score", "avg_reservoir_rate_from_drought",
        "avg_normal_rate", "avg_normal_ratio", "normal_ratio_deficit_score",
        "avg_rainfall", "reservoir_rate_risk_from_drought",
        "weather_drought_risk_score", "weather_drought_level"
    ]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()

    if "date" in out.columns:
        out = out.rename(columns={"date": "weather_latest_date"})

    return out


def load_crop():
    df = read_csv("03_crop_vulnerability_by_sigungu.csv")

    keep = [
        "sigungu", "total_households", "rice_households",
        "crop_household_demand_index", "farm_household_scale_index",
        "rice_area", "rice_yield_10a", "rice_production",
        "rice_area_index", "rice_production_index", "rice_productivity_index",
        "crop_vulnerability_index", "crop_vulnerability_level"
    ]
    keep = [c for c in keep if c in df.columns]
    return df[keep].copy()


def load_agri():
    df = read_csv("05_agri_impact_by_sigungu.csv")

    keep = [
        "sigungu", "farm_households_agri_stats", "farm_population",
        "farm_population_index", "farm_households_agri_stats_index",
        "total_benefit_area", "facility_effective_capacity",
        "benefit_area_index", "reservoir_capacity_index",
        "agri_impact_index", "agri_impact_level"
    ]
    keep = [c for c in keep if c in df.columns]
    return df[keep].copy()


def load_well():
    df = read_csv("02_well_by_sigungu.csv")

    keep = [
        "sigungu", "well_count", "groundwater_well_count",
        "drilling_developed_well_count", "total_pump_capacity",
        "avg_pump_capacity", "avg_well_depth", "well_count_index",
        "pump_capacity_index", "well_support_score",
        "well_shortage_score", "well_support_level"
    ]
    keep = [c for c in keep if c in df.columns]
    return df[keep].copy()


def build_features():
    base = pd.DataFrame({"sigungu": SIGUNGU_LIST})

    parts = [
        load_reservoir_latest(),
        load_weather(),
        load_crop(),
        load_agri(),
        load_well(),
    ]

    df = base.copy()
    for p in parts:
        df = df.merge(p, on="sigungu", how="left")

    # PDF 지표 ① 강우 부족도
    # 1순위: 평년 대비 부족도(normal_ratio_deficit_score)
    # 없으면 weather_drought_risk_score로 대체
    if "normal_ratio_deficit_score" in df.columns:
        df["rain_shortage_score"] = pd.to_numeric(df["normal_ratio_deficit_score"], errors="coerce")
    else:
        df["rain_shortage_score"] = np.nan

    if "weather_drought_risk_score" in df.columns:
        df["rain_shortage_score"] = df["rain_shortage_score"].fillna(
            pd.to_numeric(df["weather_drought_risk_score"], errors="coerce")
        )

    # PDF 지표 ② 저수율 위험도
    # reservoir_risk_score 그대로 사용

    # PDF 지표 ③ 관정 의존도
    # 현재 확보된 관정 수와 양수능력을 반영한 well_support_score를 관정 의존 가능성 지표로 사용
    df["groundwater_dependency_score"] = pd.to_numeric(
        df.get("well_support_score", np.nan),
        errors="coerce"
    )

    # PDF 지표 ④ 작물 물수요 지수
    # 작물·논벼·농가 구조를 반영한 crop_vulnerability_index를 MVP의 작물 물수요 지수로 사용
    df["crop_water_demand_score"] = pd.to_numeric(
        df.get("crop_vulnerability_index", np.nan),
        errors="coerce"
    )

    # PDF 지표 ⑤ 대체 수원 접근성 부족도
    # MVP 1차에서는 관정 기반 대체 수원 부족도를 사용.
    # 이후 대체 수원 후보 추천 단계에서 거리·저수율·수혜면적 기반으로 고도화한다.
    df["alternative_source_access_shortage_score"] = pd.to_numeric(
        df.get("well_shortage_score", np.nan),
        errors="coerce"
    )

    # 산식용 점수 보정
    df = ensure_score(df, "rain_shortage_score", fill_value=50)
    df = ensure_score(df, "reservoir_risk_score", fill_value=50)
    df = ensure_score(df, "groundwater_dependency_score", fill_value=50)
    df = ensure_score(df, "crop_water_demand_score", fill_value=50)
    df = ensure_score(df, "alternative_source_access_shortage_score", fill_value=50)

    # 보조 지표도 0~100 보정
    df = ensure_score(df, "agri_impact_index", fill_value=50)
    df = ensure_score(df, "well_shortage_score", fill_value=50)
    df = ensure_score(df, "well_support_score", fill_value=50)

    df["final_water_risk_score"] = 0.0
    for col, w in PDF_WEIGHTS.items():
        df["final_water_risk_score"] += df[f"{col}_for_score"] * w

    df["final_water_risk_score"] = df["final_water_risk_score"].clip(0, 100)
    df["final_water_risk_level"] = df["final_water_risk_score"].apply(risk_level)

    driver_score_cols = [f"{c}_for_score" for c in PDF_WEIGHTS.keys()]
    driver_idx = df[driver_score_cols].idxmax(axis=1)
    reverse_map = {f"{k}_for_score": v for k, v in DRIVER_LABELS.items()}
    df["main_risk_driver"] = driver_idx.map(reverse_map)

    df["recommended_action"] = df["final_water_risk_level"].apply(recommended_action)

    # 설명용 보조 축
    df["water_supply_pressure_score"] = (
        df["rain_shortage_score_for_score"] * 0.50
        + df["reservoir_risk_score_for_score"] * 0.50
    ).clip(0, 100)

    df["agricultural_demand_score"] = df["crop_water_demand_score_for_score"]

    df["alternative_water_risk_score"] = (
        df["groundwater_dependency_score_for_score"] * 0.65
        + df["alternative_source_access_shortage_score_for_score"] * 0.35
    ).clip(0, 100)

    # 농축어업 통계는 PDF에서 보조 판단 자료 성격이 강하므로 동점 보정에 활용
    df = df.sort_values(
        ["final_water_risk_score", "agri_impact_index_for_score"],
        ascending=[False, False]
    ).reset_index(drop=True)

    df["final_priority_rank"] = np.arange(1, len(df) + 1)

    front_cols = [
        "final_priority_rank",
        "sigungu",
        "final_water_risk_score",
        "final_water_risk_level",
        "main_risk_driver",
        "recommended_action",

        "rain_shortage_score",
        "reservoir_risk_score",
        "groundwater_dependency_score",
        "crop_water_demand_score",
        "alternative_source_access_shortage_score",

        "water_supply_pressure_score",
        "agricultural_demand_score",
        "alternative_water_risk_score",

        "agri_impact_index",
        "well_support_score",
        "well_shortage_score",
        "reservoir_data_missing",
    ]

    front_cols = [c for c in front_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in front_cols]
    return df[front_cols + other_cols]


def write_formula_doc():
    text = """# AquaGuard AI 최종 위험도 산식

## 제출 제안서 기준 산식 반영

제출 PDF의 기본 위험지수 구조에 맞춰 최종 MVP 산식을 아래와 같이 적용한다.

final_water_risk_score =
0.25 * rain_shortage_score
+ 0.25 * reservoir_risk_score
+ 0.20 * groundwater_dependency_score
+ 0.20 * crop_water_demand_score
+ 0.10 * alternative_source_access_shortage_score

## 지표 매핑

| PDF 지표 | 구현 컬럼 | 설명 |
|---|---|---|
| 강우 부족도 | rain_shortage_score | 평년 대비 강우/저수율 부족 지표 기반 |
| 저수율 위험도 | reservoir_risk_score | 농업용저수지 수위조회 기반 |
| 관정 의존도 | groundwater_dependency_score | 관정 수·양수능력 기반 well_support_score 활용 |
| 작물 물수요 지수 | crop_water_demand_score | 재배작물·논벼·농가 구조 기반 crop_vulnerability_index 활용 |
| 대체 수원 접근성 부족도 | alternative_source_access_shortage_score | MVP 1차에서는 well_shortage_score 활용 |

## 처리 원칙

- 제출 PDF와의 정합성을 위해 최종 위험도 산식은 PDF의 25:25:20:20:10 구조를 따른다.
- 농축어업 통계 기반 agri_impact_index는 최종 점수 직접 가중치가 아니라 피해 규모 보조 지표 및 동점 보정 지표로 사용한다.
- 02_well_trend_by_sigungu.csv는 일부 중간 연도 과소집계 경고가 있어 최종 산식에서 제외한다.
- 계룡시처럼 농업용저수지 데이터가 없는 지역은 reservoir_risk_score 원본은 결측으로 유지하고, 최종 산식에서는 중립값 50으로 보정한다.
- 대체 수원 접근성 부족도는 다음 단계에서 거리·저수율·수혜면적 기반 후보 추천 알고리즘으로 고도화한다.
"""
    path = META / "final_feature_formula.md"
    path.write_text(text, encoding="utf-8")


def main():
    df = build_features()

    feature_path = PROCESSED / "aquaguard_sigungu_features.csv"
    priority_path = PROCESSED / "aquaguard_priority_top15.csv"
    report_table_path = REPORT_TABLES / "aquaguard_priority_table.csv"

    df.to_csv(feature_path, index=False, encoding="utf-8-sig")
    df.to_csv(priority_path, index=False, encoding="utf-8-sig")

    report_cols = [
        "final_priority_rank",
        "sigungu",
        "final_water_risk_score",
        "final_water_risk_level",
        "main_risk_driver",
        "recommended_action",
        "rain_shortage_score",
        "reservoir_risk_score",
        "groundwater_dependency_score",
        "crop_water_demand_score",
        "alternative_source_access_shortage_score",
        "agri_impact_index",
    ]
    report_cols = [c for c in report_cols if c in df.columns]
    df[report_cols].to_csv(report_table_path, index=False, encoding="utf-8-sig")

    write_formula_doc()

    print("[Saved]")
    print(f"- {feature_path} rows={len(df)} cols={len(df.columns)}")
    print(f"- {priority_path} rows={len(df)}")
    print(f"- {report_table_path} rows={len(df)}")
    print(f"- {META / 'final_feature_formula.md'}")

    print()
    print("[Top priority]")
    print(df[[
        "final_priority_rank",
        "sigungu",
        "final_water_risk_score",
        "final_water_risk_level",
        "main_risk_driver",
        "rain_shortage_score",
        "reservoir_risk_score",
        "groundwater_dependency_score",
        "crop_water_demand_score",
        "alternative_source_access_shortage_score",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
