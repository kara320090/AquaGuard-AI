from pathlib import Path
import re
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "05_agri_stats"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SIGUNGU_LIST = [
    "천안시", "공주시", "보령시", "아산시", "서산시", "논산시", "계룡시", "당진시",
    "금산군", "부여군", "서천군", "청양군", "홍성군", "예산군", "태안군"
]


def clean_number(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        x = (
            x.replace(",", "")
             .replace("%", "")
             .replace(" ", "")
             .replace("가구", "")
             .replace("명", "")
             .strip()
        )
        if x in ["", "-", "X", "nan", "None", "NULL", "…"]:
            return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan


def normalize_sigungu(x):
    if pd.isna(x):
        return np.nan

    s = str(x).strip()

    if s in ["", "계", "합계", "총계", "전국", "충청남도", "충남", "소계"]:
        return np.nan

    if "천안" in s:
        return "천안시"

    for sg in SIGUNGU_LIST:
        if sg in s:
            return sg

    return np.nan


def minmax_0_100(s):
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(np.nan, index=s.index)
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(50.0, index=s.index)
    return ((s - mn) / (mx - mn) * 100).clip(0, 100)


def find_file(keyword, suffix=None):
    files = []
    for p in RAW_DIR.rglob("*"):
        if not p.is_file():
            continue
        if keyword in p.name:
            if suffix is None or p.suffix.lower() == suffix:
                files.append(p)
    return files


def load_farm_population():
    candidates = []
    for p in RAW_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in [".xlsx", ".xls", ".csv"]:
            if any(k in p.name for k in ["농가", "농가인구", "시군별"]):
                candidates.append(p)

    if not candidates:
        print("[WARN] 05 농가/농가인구 파일을 찾지 못했습니다.")
        return pd.DataFrame({"sigungu": SIGUNGU_LIST})

    path = candidates[0]
    print(f"[LOAD] farm population: {path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, encoding="cp949")
    else:
        df = pd.read_excel(path)

    df.columns = [str(c).strip() for c in df.columns]

    print("[INFO] columns:", df.columns.tolist())

    # 예상 구조:
    # 시점, 시도별(1), 시도별(2), 농가, 농가인구, 남, 여
    sig_col = None
    for c in ["시도별(2)", "시군명", "시군", "행정구역별", "지역"]:
        if c in df.columns:
            sig_col = c
            break

    if sig_col is None:
        raise KeyError(f"시군 컬럼을 찾지 못했습니다. 현재 컬럼: {df.columns.tolist()}")

    year_col = None
    for c in ["시점", "연도", "기준연도", "년도"]:
        if c in df.columns:
            year_col = c
            break

    farm_col = None
    for c in df.columns:
        if str(c).strip() == "농가" or "농가 " in str(c):
            farm_col = c
            break

    pop_col = None
    for c in df.columns:
        if "농가인구" in str(c):
            pop_col = c
            break

    male_col = None
    female_col = None
    for c in df.columns:
        if str(c).strip() in ["남", "남자"]:
            male_col = c
        if str(c).strip() in ["여", "여자"]:
            female_col = c

    out = pd.DataFrame()
    out["sigungu"] = df[sig_col].apply(normalize_sigungu)
    out["year"] = df[year_col].apply(clean_number) if year_col else np.nan
    out["farm_households_agri_stats"] = df[farm_col].apply(clean_number) if farm_col else np.nan
    out["farm_population"] = df[pop_col].apply(clean_number) if pop_col else np.nan
    out["farm_population_male"] = df[male_col].apply(clean_number) if male_col else np.nan
    out["farm_population_female"] = df[female_col].apply(clean_number) if female_col else np.nan
    out["source_file"] = path.name

    out = out[out["sigungu"].notna()].copy()

    if out.empty:
        print("[WARN] farm population parsed empty.")
        return pd.DataFrame({"sigungu": SIGUNGU_LIST})

    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    latest_year = out.groupby("sigungu")["year"].max().reset_index()
    latest = out.merge(latest_year, on=["sigungu", "year"], how="inner")

    latest = latest.groupby(["sigungu", "year"], as_index=False).agg(
        farm_households_agri_stats=("farm_households_agri_stats", "sum"),
        farm_population=("farm_population", "sum"),
        farm_population_male=("farm_population_male", "sum"),
        farm_population_female=("farm_population_female", "sum"),
    )

    latest["farm_population_index"] = minmax_0_100(latest["farm_population"])
    latest["farm_households_agri_stats_index"] = minmax_0_100(latest["farm_households_agri_stats"])

    return latest.sort_values("sigungu").reset_index(drop=True)


def load_reservoir_facility_summary():
    path = PROCESSED_DIR / "01_reservoir_facility_summary.csv"
    if not path.exists():
        print("[WARN] reservoir facility summary not found.")
        return pd.DataFrame({"sigungu": SIGUNGU_LIST})

    df = pd.read_csv(path)

    # 계룡시처럼 시설 데이터가 없는 곳도 최종에서는 남겨야 함.
    base = pd.DataFrame({"sigungu": SIGUNGU_LIST})
    df = base.merge(df, on="sigungu", how="left")

    for col in [
        "facility_reservoir_count",
        "total_benefit_area",
        "facility_total_capacity",
        "facility_effective_capacity",
        "facility_basin_area",
        "facility_full_water_area",
    ]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["benefit_area_index"] = minmax_0_100(df["total_benefit_area"])
    df["reservoir_capacity_index"] = minmax_0_100(df["facility_effective_capacity"])
    df["reservoir_facility_count_index"] = minmax_0_100(df["facility_reservoir_count"])

    return df


def load_crop_vulnerability():
    path = PROCESSED_DIR / "03_crop_vulnerability_by_sigungu.csv"
    if not path.exists():
        raise FileNotFoundError("03_crop_vulnerability_by_sigungu.csv가 없습니다. ③ 전처리를 먼저 실행하세요.")

    df = pd.read_csv(path)

    base = pd.DataFrame({"sigungu": SIGUNGU_LIST})
    df = base.merge(df, on="sigungu", how="left")

    for col in [
        "crop_vulnerability_index",
        "farm_household_scale_index",
        "rice_area_index",
        "rice_production_index",
        "rice_productivity_index",
    ]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def build_agri_impact(farm_pop, reservoir_summary, crop):
    base = pd.DataFrame({"sigungu": SIGUNGU_LIST})

    df = base.merge(farm_pop, on="sigungu", how="left")
    df = df.merge(reservoir_summary, on="sigungu", how="left", suffixes=("", "_reservoir"))
    df = df.merge(crop, on="sigungu", how="left", suffixes=("", "_crop"))

    component_weights = {
        "farm_population_index": 0.25,
        "farm_households_agri_stats_index": 0.20,
        "benefit_area_index": 0.20,
        "reservoir_capacity_index": 0.10,
        "rice_production_index": 0.15,
        "rice_area_index": 0.10,
    }

    score = pd.Series(0.0, index=df.index)
    weight_sum = pd.Series(0.0, index=df.index)

    for col, w in component_weights.items():
        if col in df.columns:
            valid = df[col].notna()
            score.loc[valid] += df.loc[valid, col] * w
            weight_sum.loc[valid] += w

    df["agri_impact_index"] = np.where(weight_sum > 0, score / weight_sum, np.nan)

    df["agri_impact_level"] = pd.cut(
        df["agri_impact_index"],
        bins=[-1, 39, 59, 79, 100],
        labels=["낮음", "주의", "경계", "심각"]
    ).astype(str)

    return df.sort_values("agri_impact_index", ascending=False).reset_index(drop=True)


def main():
    print("[AquaGuard AI] Step 05 - Agri stats / agri impact preprocessing")

    farm_pop = load_farm_population()
    reservoir_summary = load_reservoir_facility_summary()
    crop = load_crop_vulnerability()

    agri_impact = build_agri_impact(farm_pop, reservoir_summary, crop)

    farm_pop_path = PROCESSED_DIR / "05_farm_population_by_sigungu.csv"
    agri_impact_path = PROCESSED_DIR / "05_agri_impact_by_sigungu.csv"

    farm_pop.to_csv(farm_pop_path, index=False, encoding="utf-8-sig")
    agri_impact.to_csv(agri_impact_path, index=False, encoding="utf-8-sig")

    print()
    print("[Saved]")
    print(f"- {farm_pop_path} rows={len(farm_pop)}")
    print(f"- {agri_impact_path} rows={len(agri_impact)}")

    print()
    print("[Preview: agri impact]")
    print(agri_impact[[
        "sigungu",
        "farm_population",
        "total_benefit_area",
        "facility_effective_capacity",
        "rice_production_index",
        "agri_impact_index",
        "agri_impact_level"
    ]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
