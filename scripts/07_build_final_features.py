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

WEIGHTS = {
    "reservoir_risk_score": 0.35,
    "weather_drought_risk_score": 0.25,
    "crop_vulnerability_index": 0.15,
    "agri_impact_index": 0.15,
    "well_shortage_score": 0.10,
}

DRIVER_LABELS = {
    "reservoir_risk_score": "저수율 위험",
    "weather_drought_risk_score": "기상·가뭄 위험",
    "crop_vulnerability_index": "작물 취약성",
    "agri_impact_index": "농업 영향 규모",
    "well_shortage_score": "대체 수원 부족",
}


def read_required_csv(filename):
    path = PROCESSED / filename
    if not path.exists():
        raise FileNotFoundError(f"Required processed file not found: {path}")
    return pd.read_csv(path)


def ensure_score_0_100(df, col, fill_value=50):
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
        return "우선 점검 대상 지정 및 저수율·가뭄상황 집중 모니터링"
    if level == "주의":
        return "정기 점검 강화 및 농업용수 수요 변화 관찰"
    if level == "낮음":
        return "일반 모니터링 유지"
    return "데이터 보완 후 재판정"


def load_reservoir_latest():
    df = read_required_csv("01_reservoir_sigungu_daily.csv")

    if "date" not in df.columns:
        raise KeyError("01_reservoir_sigungu_daily.csv must contain date column")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["sigungu", "date"])

    latest = df.groupby("sigungu", as_index=False).tail(1).copy()

    keep_cols = [
        "sigungu",
        "date",
        "avg_reservoir_rate",
        "min_reservoir_rate",
        "max_reservoir_rate",
        "reservoir_count",
        "low_reservoir_count_40",
        "low_reservoir_count_30",
        "total_effective_capacity",
        "reservoir_risk_score",
        "risk_level_by_reservoir",
    ]

    keep_cols = [c for c in keep_cols if c in latest.columns]
    latest = latest[keep_cols].copy()

    latest = latest.rename(columns={"date": "reservoir_latest_date"})

    base = pd.DataFrame({"sigungu": SIGUNGU_LIST})
    out = base.merge(latest, on="sigungu", how="left")

    out["reservoir_data_missing"] = out["reservoir_risk_score"].isna().astype(int)

    # 계룡시처럼 농업용 저수지 데이터가 없는 경우 극단값이 아니라 중립값 50으로 보정
    # 단, 원본 값은 NaN으로 남기고 *_for_score에서만 보정한다.
    for col in ["reservoir_count", "low_reservoir_count_40", "low_reservoir_count_30", "total_effective_capacity"]:
        if col not in out.columns:
            out[col] = 0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    if "risk_level_by_reservoir" not in out.columns:
        out["risk_level_by_reservoir"] = np.nan

    out.loc[out["reservoir_data_missing"] == 1, "risk_level_by_reservoir"] = "데이터없음"

    return out


def load_weather_latest():
    df = read_required_csv("04_weather_drought_latest_by_sigungu.csv")

    keep_cols = [
        "sigungu",
        "date",
        "drought_stage_score",
        "avg_reservoir_rate_from_drought",
        "avg_normal_rate",
        "avg_normal_ratio",
        "normal_ratio_deficit_score",
        "avg_rainfall",
        "reservoir_rate_risk_from_drought",
        "weather_drought_risk_score",
        "weather_drought_level",
    ]

    keep_cols = [c for c in keep_cols if c in df.columns]
    out = df[keep_cols].copy()

    if "date" in out.columns:
        out = out.rename(columns={"date": "weather_latest_date"})

    return out


def load_crop():
    df = read_required_csv("03_crop_vulnerability_by_sigungu.csv")

    keep_cols = [
        "sigungu",
        "total_households",
        "rice_households",
        "crop_household_demand_index",
        "farm_household_scale_index",
        "rice_area",
        "rice_yield_10a",
        "rice_production",
        "rice_area_index",
        "rice_production_index",
        "rice_productivity_index",
        "crop_vulnerability_index",
        "crop_vulnerability_level",
    ]

    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols].copy()


def load_agri_impact():
    df = read_required_csv("05_agri_impact_by_sigungu.csv")

    keep_cols = [
        "sigungu",
        "farm_households_agri_stats",
        "farm_population",
        "farm_population_index",
        "farm_households_agri_stats_index",
        "total_benefit_area",
        "facility_effective_capacity",
        "benefit_area_index",
        "reservoir_capacity_index",
        "agri_impact_index",
        "agri_impact_level",
    ]

    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols].copy()


def load_well():
    df = read_required_csv("02_well_by_sigungu.csv")

    keep_cols = [
        "sigungu",
        "well_count",
        "groundwater_well_count",
        "drilling_developed_well_count",
        "total_pump_capacity",
        "avg_pump_capacity",
        "avg_well_depth",
        "well_count_index",
        "pump_capacity_index",
        "well_support_score",
        "well_shortage_score",
        "well_support_level",
    ]

    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols].copy()


def build_features():
    base = pd.DataFrame({"sigungu": SIGUNGU_LIST})

    reservoir = load_reservoir_latest()
    weather = load_weather_latest()
    crop = load_crop()
    agri = load_agri_impact()
    well = load_well()

    df = base.copy()

    for part in [reservoir, weather, crop, agri, well]:
        df = df.merge(part, on="sigungu", how="left")

    # 점수형 컬럼 보정
    # reservoir는 계룡시 등 데이터 없는 곳이 있어 중립값 50으로 산식에 반영
    df = ensure_score_0_100(df, "reservoir_risk_score", fill_value=50)
    df = ensure_score_0_100(df, "weather_drought_risk_score", fill_value=50)
    df = ensure_score_0_100(df, "crop_vulnerability_index", fill_value=50)
    df = ensure_score_0_100(df, "agri_impact_index", fill_value=50)
    df = ensure_score_0_100(df, "well_shortage_score", fill_value=50)
    df = ensure_score_0_100(df, "well_support_score", fill_value=50)

    df["final_water_risk_score"] = 0.0

    for col, weight in WEIGHTS.items():
        df["final_water_risk_score"] += df[f"{col}_for_score"] * weight

    df["final_water_risk_score"] = df["final_water_risk_score"].clip(0, 100)
    df["final_water_risk_level"] = df["final_water_risk_score"].apply(risk_level)

    # 세부 축 점수
    df["water_supply_pressure_score"] = (
        df["reservoir_risk_score_for_score"] * 0.60
        + df["weather_drought_risk_score_for_score"] * 0.40
    ).clip(0, 100)

    df["agricultural_damage_potential_score"] = (
        df["crop_vulnerability_index_for_score"] * 0.50
        + df["agri_impact_index_for_score"] * 0.50
    ).clip(0, 100)

    df["alternative_water_shortage_score"] = df["well_shortage_score_for_score"]

    driver_cols = list(WEIGHTS.keys())
    driver_score_cols = [f"{c}_for_score" for c in driver_cols]

    driver_idx = df[driver_score_cols].idxmax(axis=1)
    reverse_map = {f"{k}_for_score": DRIVER_LABELS[k] for k in driver_cols}
    df["main_risk_driver"] = driver_idx.map(reverse_map)

    df["recommended_action"] = df["final_water_risk_level"].apply(recommended_action)

    df = df.sort_values("final_water_risk_score", ascending=False).reset_index(drop=True)
    df["final_priority_rank"] = np.arange(1, len(df) + 1)

    # 보고서용 핵심 컬럼 앞으로 배치
    front_cols = [
        "final_priority_rank",
        "sigungu",
        "final_water_risk_score",
        "final_water_risk_level",
        "main_risk_driver",
        "recommended_action",
        "water_supply_pressure_score",
        "agricultural_damage_potential_score",
        "alternative_water_shortage_score",
        "reservoir_risk_score",
        "weather_drought_risk_score",
        "crop_vulnerability_index",
        "agri_impact_index",
        "well_support_score",
        "well_shortage_score",
        "reservoir_data_missing",
    ]

    existing_front = [c for c in front_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in existing_front]
    df = df[existing_front + other_cols]

    return df


def write_formula_doc():
    text = """# AquaGuard AI 최종 위험도 산식

## 최종 점수

final_water_risk_score =
0.35 * reservoir_risk_score
+ 0.25 * weather_drought_risk_score
+ 0.15 * crop_vulnerability_index
+ 0.15 * agri_impact_index
+ 0.10 * well_shortage_score

## 해석

- reservoir_risk_score: 저수율 부족 위험
- weather_drought_risk_score: 기상·가뭄 위험
- crop_vulnerability_index: 작물 구조상 물 부족 취약성
- agri_impact_index: 농가·농가인구·수혜면적 기반 영향 규모
- well_shortage_score: 관정 기반 대체 수원 부족도

## 처리 원칙

- 02_well_trend_by_sigungu.csv는 일부 중간 연도 과소집계 경고가 있어 최종 산식에서 제외한다.
- 계룡시처럼 농업용 저수지 데이터가 없는 지역은 reservoir_risk_score 원본은 결측으로 유지하고, 최종 산식에서는 중립값 50으로 보정한다.
- 최종 우선순위는 final_water_risk_score 내림차순으로 산정한다.
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
        "reservoir_risk_score",
        "weather_drought_risk_score",
        "crop_vulnerability_index",
        "agri_impact_index",
        "well_shortage_score",
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
        "well_shortage_score",
        "recommended_action",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
