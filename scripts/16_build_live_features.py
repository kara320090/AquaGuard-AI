from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

PROCESSED = ROOT / "data" / "processed"
REPORT_TABLES = ROOT / "reports" / "tables"
META = ROOT / "data" / "metadata"

BASE_PATH = PROCESSED / "aquaguard_sigungu_features.csv"
OLDAM_PATH = REPORT_TABLES / "latest_live_reservoir_by_sigungu.csv"
KMA_PATH = PROCESSED / "latest_weather_30d_by_sigungu.csv"
SOIL_PATH = PROCESSED / "latest_adms_soil_moisture_by_sigungu.csv"

OUT_FEATURE_PATH = PROCESSED / "latest_live_sigungu_features.csv"
OUT_SUMMARY_PATH = REPORT_TABLES / "latest_live_risk_summary.csv"
OUT_STATUS_PATH = REPORT_TABLES / "latest_live_data_status.csv"
OUT_METHOD_PATH = META / "live_feature_method.md"

# 실시간 데이터 3축 반영:
# 기상청 AWS 강수량 20%, 올담 저수율 20%, ADMS 토양수분 20%
WEIGHTS = {
    "rain": 0.20,
    "reservoir": 0.20,
    "soil": 0.20,
    "groundwater": 0.15,
    "crop": 0.15,
    "alternative": 0.10,
}


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def risk_level(score):
    if pd.isna(score):
        return "미산정"
    if score >= 80:
        return "심각"
    if score >= 60:
        return "경계"
    if score >= 40:
        return "주의"
    return "낮음"


def main_driver(row):
    candidates = {
        "강우 부족도": row.get("live_rain_shortage_score_for_score", np.nan),
        "저수율 위험도": row.get("live_reservoir_risk_score_for_score", np.nan),
        "토양수분 가뭄도": row.get("live_soil_moisture_drought_score_for_score", np.nan),
        "관정 의존도": row.get("groundwater_dependency_score_for_score", np.nan),
        "작물 물수요": row.get("crop_water_demand_score_for_score", np.nan),
        "대체 수원 접근성 부족도": row.get("alternative_source_access_shortage_score_for_score", np.nan),
    }
    clean = {k: v for k, v in candidates.items() if pd.notna(v)}
    if not clean:
        return "데이터 부족"
    return max(clean, key=clean.get)


def recommended_action(level):
    if level == "심각":
        return "즉시 현장 점검, 용수 공급 대책 검토, 대체 수원 후보 우선 검토"
    if level == "경계":
        return "우선 점검 대상 지정, 저수율·강우·토양수분 집중 모니터링"
    if level == "주의":
        return "정기 점검 강화 및 농업용수 수요 변화 관찰"
    return "일반 모니터링 유지"


def load_base():
    if not BASE_PATH.exists():
        raise FileNotFoundError(f"Missing base feature file: {BASE_PATH}")

    df = pd.read_csv(BASE_PATH)

    required = [
        "sigungu",
        "final_water_risk_score",
        "rain_shortage_score",
        "reservoir_risk_score",
        "groundwater_dependency_score",
        "crop_water_demand_score",
        "alternative_source_access_shortage_score",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required base columns: {missing}")

    return df


def load_oldam():
    if not OLDAM_PATH.exists():
        print(f"[WARN] OLDAM live reservoir file missing: {OLDAM_PATH}")
        return pd.DataFrame()

    df = pd.read_csv(OLDAM_PATH)
    keep = [
        "date",
        "sigungu",
        "today_reservoir_count",
        "today_avg_reservoir_rate",
        "today_min_reservoir_rate",
        "today_max_reservoir_rate",
        "today_low_reservoir_count_40",
        "today_low_reservoir_count_30",
        "today_over_100_count",
        "today_reservoir_risk_score",
    ]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    for c in df.columns:
        if c not in ["date", "sigungu"]:
            df[c] = to_num(df[c])

    return df


def load_kma():
    if not KMA_PATH.exists():
        print(f"[WARN] KMA weather file missing: {KMA_PATH}")
        return pd.DataFrame()

    df = pd.read_csv(KMA_PATH)
    keep = [
        "sigungu",
        "weather_start_date",
        "weather_end_date",
        "station_count",
        "rainfall_30d",
        "rainfall_7d",
        "rain_days_30d",
        "rain_days_7d",
        "avg_temperature_30d",
        "avg_humidity_30d",
        "avg_wind_speed_30d",
        "latest_rain_shortage_score",
        "weather_data_status",
    ]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    for c in df.columns:
        if c not in ["sigungu", "weather_start_date", "weather_end_date", "weather_data_status"]:
            df[c] = to_num(df[c])

    return df


def load_soil():
    if not SOIL_PATH.exists():
        print(f"[WARN] ADMS soil moisture file missing: {SOIL_PATH}")
        return pd.DataFrame()

    df = pd.read_csv(SOIL_PATH)
    keep = [
        "sigungu",
        "soil_data_date",
        "soil_row_count",
        "soil_moisture_avg",
        "soil_moisture_min",
        "soil_moisture_max",
        "soil_dry_count",
        "soil_severe_count",
        "soil_moisture_drought_score",
        "soil_data_status",
    ]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    for c in df.columns:
        if c not in ["sigungu", "soil_data_date", "soil_data_status"]:
            df[c] = to_num(df[c])

    return df


def build_live_features(base, oldam, kma, soil):
    df = base.copy()

    # 1. 올담 저수율 병합
    if not oldam.empty:
        df = df.merge(oldam, on="sigungu", how="left", suffixes=("", "_oldam"))
    else:
        for c in [
            "date",
            "today_reservoir_count",
            "today_avg_reservoir_rate",
            "today_min_reservoir_rate",
            "today_max_reservoir_rate",
            "today_low_reservoir_count_40",
            "today_low_reservoir_count_30",
            "today_over_100_count",
            "today_reservoir_risk_score",
        ]:
            df[c] = np.nan

    # 2. 기상청 AWS/ASOS 강수량 병합
    if not kma.empty:
        df = df.merge(kma, on="sigungu", how="left", suffixes=("", "_kma"))
    else:
        for c in [
            "weather_start_date",
            "weather_end_date",
            "station_count",
            "rainfall_30d",
            "rainfall_7d",
            "rain_days_30d",
            "rain_days_7d",
            "avg_temperature_30d",
            "avg_humidity_30d",
            "avg_wind_speed_30d",
            "latest_rain_shortage_score",
            "weather_data_status",
        ]:
            df[c] = np.nan

    # 3. ADMS 토양수분 병합
    if not soil.empty:
        df = df.merge(soil, on="sigungu", how="left", suffixes=("", "_soil"))
    else:
        for c in [
            "soil_data_date",
            "soil_row_count",
            "soil_moisture_avg",
            "soil_moisture_min",
            "soil_moisture_max",
            "soil_dry_count",
            "soil_severe_count",
            "soil_moisture_drought_score",
            "soil_data_status",
        ]:
            df[c] = np.nan

    # 최신 저수율 위험도: 올담 있으면 사용, 없으면 기존 baseline
    df["live_reservoir_risk_score"] = df["today_reservoir_risk_score"].combine_first(df["reservoir_risk_score"])
    df["live_reservoir_source"] = np.where(
        df["today_reservoir_risk_score"].notna(),
        "OLDAM_TODAY",
        "BASELINE_2025",
    )

    # 최신 강우 부족도: AWS/ASOS 있으면 사용, 없으면 기존 baseline
    df["live_rain_shortage_score"] = df["latest_rain_shortage_score"].combine_first(df["rain_shortage_score"])

    if "weather_data_status" in df.columns:
        weather_status = df["weather_data_status"].fillna("KMA_WEATHER_30D").astype(str)
    else:
        weather_status = pd.Series("KMA_WEATHER_30D", index=df.index)

    df["live_weather_source"] = np.where(
        df["latest_rain_shortage_score"].notna(),
        weather_status,
        "BASELINE_WEATHER",
    )

    # 토양수분 가뭄도: ADMS 있으면 사용, 없으면 50점 중립값
    df["live_soil_moisture_drought_score"] = df["soil_moisture_drought_score"]
    df["live_soil_source"] = np.where(
        df["soil_moisture_drought_score"].notna(),
        "ADMS_SOIL_AUTO",
        "NO_SOIL_DATA",
    )

    score_cols = [
        "live_rain_shortage_score",
        "live_reservoir_risk_score",
        "live_soil_moisture_drought_score",
        "groundwater_dependency_score",
        "crop_water_demand_score",
        "alternative_source_access_shortage_score",
    ]

    for c in score_cols:
        df[f"{c}_for_score"] = to_num(df[c]).fillna(50).clip(0, 100)

    df["final_live_water_risk_score"] = (
        WEIGHTS["rain"] * df["live_rain_shortage_score_for_score"]
        + WEIGHTS["reservoir"] * df["live_reservoir_risk_score_for_score"]
        + WEIGHTS["soil"] * df["live_soil_moisture_drought_score_for_score"]
        + WEIGHTS["groundwater"] * df["groundwater_dependency_score_for_score"]
        + WEIGHTS["crop"] * df["crop_water_demand_score_for_score"]
        + WEIGHTS["alternative"] * df["alternative_source_access_shortage_score_for_score"]
    ).clip(0, 100)

    df["final_live_water_risk_level"] = df["final_live_water_risk_score"].apply(risk_level)
    df["live_main_risk_driver"] = df.apply(main_driver, axis=1)
    df["live_recommended_action"] = df["final_live_water_risk_level"].apply(recommended_action)

    df["live_score_delta_from_baseline"] = (
        df["final_live_water_risk_score"] - to_num(df["final_water_risk_score"])
    )

    df = df.sort_values("final_live_water_risk_score", ascending=False).reset_index(drop=True)
    df["final_live_priority_rank"] = np.arange(1, len(df) + 1)

    front = [
        "final_live_priority_rank",
        "sigungu",
        "final_live_water_risk_score",
        "final_live_water_risk_level",
        "live_score_delta_from_baseline",
        "live_main_risk_driver",
        "live_recommended_action",
        "live_rain_shortage_score",
        "live_reservoir_risk_score",
        "live_soil_moisture_drought_score",
        "groundwater_dependency_score",
        "crop_water_demand_score",
        "alternative_source_access_shortage_score",
        "live_weather_source",
        "live_reservoir_source",
        "live_soil_source",
    ]

    other = [c for c in df.columns if c not in front]
    return df[front + other]


def build_summary(live):
    keep = [
        "final_live_priority_rank",
        "sigungu",
        "final_live_water_risk_score",
        "final_live_water_risk_level",
        "live_score_delta_from_baseline",
        "live_main_risk_driver",
        "live_weather_source",
        "live_reservoir_source",
        "live_soil_source",
        "today_avg_reservoir_rate",
        "today_min_reservoir_rate",
        "today_reservoir_count",
        "rainfall_30d",
        "rainfall_7d",
        "latest_rain_shortage_score",
        "soil_data_date",
        "soil_moisture_avg",
        "soil_moisture_min",
        "soil_moisture_drought_score",
        "final_water_risk_score",
        "final_water_risk_level",
    ]
    keep = [c for c in keep if c in live.columns]
    return live[keep].copy()


def build_status(live, oldam, kma, soil):
    status = pd.DataFrame([{
        "live_feature_rows": len(live),
        "oldam_rows": len(oldam),
        "oldam_sigungu_count": oldam["sigungu"].nunique() if not oldam.empty and "sigungu" in oldam.columns else 0,
        "kma_rows": len(kma),
        "kma_sigungu_count": kma["sigungu"].nunique() if not kma.empty and "sigungu" in kma.columns else 0,
        "soil_rows": len(soil),
        "soil_sigungu_count": soil["sigungu"].nunique() if not soil.empty and "sigungu" in soil.columns else 0,
        "live_reservoir_oldam_count": int((live["live_reservoir_source"] == "OLDAM_TODAY").sum()),
        "live_weather_kma_count": int((live["live_weather_source"] != "BASELINE_WEATHER").sum()),
        "live_soil_adms_count": int((live["live_soil_source"] == "ADMS_SOIL_AUTO").sum()),
        "top_live_sigungu": live.iloc[0]["sigungu"] if len(live) else "",
        "top_live_score": live.iloc[0]["final_live_water_risk_score"] if len(live) else np.nan,
        "status": "SUCCESS",
    }])
    return status


def write_method_doc():
    text = """# AquaGuard AI Live Feature Method

## 목적

올담 최신 저수지 수위 데이터, 기상청 AWS/ASOS 최근 30일 강수량, ADMS 토양수분현황을 결합해 현재 기준 농업용수 위험도를 갱신한다.

## Live 위험도 산식

final_live_water_risk_score =
0.20 * live_rain_shortage_score
+ 0.20 * live_reservoir_risk_score
+ 0.20 * live_soil_moisture_drought_score
+ 0.15 * groundwater_dependency_score
+ 0.15 * crop_water_demand_score
+ 0.10 * alternative_source_access_shortage_score

## 데이터 소스 우선순위

- 강우 부족도: 기상청 AWS/ASOS 최근 30일 강수량
- 저수율 위험도: 올담 최신 저수율 데이터
- 토양수분 가뭄도: ADMS 밭토양수분현황
- 관정, 작물, 대체 수원 접근성: 정적·반정적 공공데이터 기반 기존 feature

## 해석 주의

ADMS 토양유효수분 값이 충분한 상태이면 soil_moisture_drought_score는 낮게 계산된다.
이는 위험을 과장하지 않고 실제 농지 건조 상태를 보정하기 위한 것이다.
"""
    OUT_METHOD_PATH.write_text(text, encoding="utf-8")


def main():
    print("[AquaGuard AI] Build latest live features")

    base = load_base()
    oldam = load_oldam()
    kma = load_kma()
    soil = load_soil()

    live = build_live_features(base, oldam, kma, soil)
    summary = build_summary(live)
    status = build_status(live, oldam, kma, soil)

    live.to_csv(OUT_FEATURE_PATH, index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    status.to_csv(OUT_STATUS_PATH, index=False, encoding="utf-8-sig")
    write_method_doc()

    print()
    print("[Saved]")
    print(f"- {OUT_FEATURE_PATH} rows={len(live)}")
    print(f"- {OUT_SUMMARY_PATH} rows={len(summary)}")
    print(f"- {OUT_STATUS_PATH}")
    print(f"- {OUT_METHOD_PATH}")

    print()
    print("[Live Status]")
    print(status.to_string(index=False))

    print()
    print("[Live Risk Summary]")
    print(summary.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
