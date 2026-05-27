from pathlib import Path
import re
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "03_crop"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
             .replace("ha", "")
             .replace("톤", "")
             .replace("kg", "")
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

    if s in ["", "계", "합계", "총계", "전국", "충청남도", "충남"]:
        return np.nan

    # 천안시 동남구/서북구는 최종 행정단위에서 천안시로 통합
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


def load_crop_households():
    # 우선 CSV를 기준으로 처리한다. 인벤토리상 이 파일의 컬럼 구조가 가장 명확함.
    candidates = find_file("충청남도_재배작물별 농가현황", ".csv")
    if not candidates:
        raise FileNotFoundError("충청남도_재배작물별 농가현황.csv 파일을 찾지 못했습니다.")

    path = candidates[0]
    print(f"[LOAD] crop households: {path}")

    df = pd.read_csv(path, encoding="cp949")
    df.columns = [str(c).strip() for c in df.columns]

    sig_col = "행정구역별"
    if sig_col not in df.columns:
        raise KeyError(f"'{sig_col}' 컬럼이 없습니다. 현재 컬럼: {df.columns.tolist()}")

    out = pd.DataFrame()
    out["sigungu"] = df[sig_col].apply(normalize_sigungu)
    out["year"] = 2025

    col_map = {
        "rice_households": "논벼_농가 (가구)",
        "food_crop_households": "식량작물_농가 (가구)",
        "vegetable_households": "채소·산나물_농가 (가구)",
        "special_crop_households": "특용작물·버섯_농가 (가구)",
        "fruit_households": "과수_농가 (가구)",
        "medicinal_households": "약용작물_농가 (가구)",
        "flower_households": "화초·관상작물_농가 (가구)",
        "other_households": "기타작물_농가 (가구)",
        "livestock_households": "축산_농가 (가구)",
    }

    for new_col, src_col in col_map.items():
        if src_col in df.columns:
            out[new_col] = df[src_col].apply(clean_number)
        else:
            out[new_col] = np.nan
            print(f"[WARN] missing column: {src_col}")

    out = out[out["sigungu"].notna()].copy()

    component_cols = list(col_map.keys())
    out["total_households"] = out[component_cols].sum(axis=1, skipna=True)

    # 천안시 동남구/서북구 통합
    num_cols = [c for c in out.columns if c not in ["sigungu"]]
    out = out.groupby("sigungu", as_index=False)[num_cols].sum()

    # 비율
    for c in component_cols:
        ratio_col = c.replace("_households", "_ratio")
        out[ratio_col] = out[c] / out["total_households"].replace(0, np.nan)

    out["crop_household_demand_index_raw"] = (
        out["rice_ratio"].fillna(0) * 1.00
        + out["food_crop_ratio"].fillna(0) * 0.70
        + out["vegetable_ratio"].fillna(0) * 0.50
        + out["special_crop_ratio"].fillna(0) * 0.40
    )

    out["crop_household_demand_index"] = minmax_0_100(out["crop_household_demand_index_raw"])
    out["farm_household_scale_index"] = minmax_0_100(out["total_households"])

    out["source_file"] = path.name

    return out.sort_values("sigungu").reset_index(drop=True)


def load_rice_productivity():
    candidates = find_file("시군별_논벼_생산량", ".xlsx")
    if not candidates:
        print("[WARN] KOSIS 시군별 논벼 생산량 파일 없음")
        return pd.DataFrame(columns=["sigungu"])

    path = candidates[0]
    print(f"[LOAD] rice productivity: {path}")

    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]

    if "행정구역별(2)" not in df.columns:
        raise KeyError(f"KOSIS 파일에 '행정구역별(2)' 컬럼이 없습니다. 현재 컬럼: {df.columns.tolist()}")

    # KOSIS wide format:
    # 행정구역별(1), 행정구역별(2), 2018, 2018.1, 2018.2, ...
    # 일반적으로 연도 3개 묶음 = 재배면적, 10a당 생산량, 생산량
    rows = []

    for _, r in df.iterrows():
        sigungu = normalize_sigungu(r.get("행정구역별(2)"))
        if pd.isna(sigungu):
            continue

        for year in range(2018, 2026):
            c_area = str(year)
            c_yield = f"{year}.1"
            c_prod = f"{year}.2"

            if c_area not in df.columns:
                continue

            rows.append({
                "sigungu": sigungu,
                "year": year,
                "rice_area": clean_number(r.get(c_area)),
                "rice_yield_10a": clean_number(r.get(c_yield)),
                "rice_production": clean_number(r.get(c_prod)),
                "source_file": path.name,
            })

    out = pd.DataFrame(rows)

    if out.empty:
        print("[WARN] rice productivity parsed empty")
        return pd.DataFrame(columns=["sigungu"])

    # 천안시 통합 및 최신연도 선택
    out = out.groupby(["sigungu", "year"], as_index=False).agg(
        rice_area=("rice_area", "sum"),
        rice_yield_10a=("rice_yield_10a", "mean"),
        rice_production=("rice_production", "sum")
    )

    latest_year = out.groupby("sigungu")["year"].max().reset_index()
    latest = out.merge(latest_year, on=["sigungu", "year"], how="inner")

    latest["rice_area_index"] = minmax_0_100(latest["rice_area"])
    latest["rice_production_index"] = minmax_0_100(latest["rice_production"])
    latest["rice_productivity_index"] = minmax_0_100(latest["rice_yield_10a"])

    return latest.sort_values("sigungu").reset_index(drop=True)


def build_crop_vulnerability(households, rice):
    merged = households.merge(rice, on="sigungu", how="outer", suffixes=("", "_rice"))

    component_weights = {
        "crop_household_demand_index": 0.40,
        "farm_household_scale_index": 0.20,
        "rice_area_index": 0.25,
        "rice_production_index": 0.15,
    }

    score = pd.Series(0.0, index=merged.index)
    weight_sum = pd.Series(0.0, index=merged.index)

    for col, w in component_weights.items():
        if col in merged.columns:
            valid = merged[col].notna()
            score.loc[valid] += merged.loc[valid, col] * w
            weight_sum.loc[valid] += w

    merged["crop_vulnerability_index"] = np.where(weight_sum > 0, score / weight_sum, np.nan)

    merged["crop_vulnerability_level"] = pd.cut(
        merged["crop_vulnerability_index"],
        bins=[-1, 39, 59, 79, 100],
        labels=["낮음", "주의", "경계", "심각"]
    ).astype(str)

    return merged.sort_values("crop_vulnerability_index", ascending=False).reset_index(drop=True)


def main():
    print("[AquaGuard AI] Step 04 - Crop preprocessing fixed")
    print(f"RAW_DIR: {RAW_DIR}")

    households = load_crop_households()
    rice = load_rice_productivity()
    vulnerability = build_crop_vulnerability(households, rice)

    households_path = OUT_DIR / "03_crop_households_by_sigungu.csv"
    rice_path = OUT_DIR / "03_rice_productivity_by_sigungu.csv"
    vulnerability_path = OUT_DIR / "03_crop_vulnerability_by_sigungu.csv"

    households.to_csv(households_path, index=False, encoding="utf-8-sig")
    rice.to_csv(rice_path, index=False, encoding="utf-8-sig")
    vulnerability.to_csv(vulnerability_path, index=False, encoding="utf-8-sig")

    print()
    print("[Saved]")
    print(f"- {households_path} rows={len(households)}")
    print(f"- {rice_path} rows={len(rice)}")
    print(f"- {vulnerability_path} rows={len(vulnerability)}")

    print()
    print("[Preview: households]")
    print(households[["sigungu", "total_households", "rice_households", "crop_household_demand_index"]].head(20).to_string(index=False))

    print()
    print("[Preview: rice]")
    if not rice.empty:
        print(rice[["sigungu", "year", "rice_area", "rice_yield_10a", "rice_production"]].head(20).to_string(index=False))
    else:
        print(rice)

    print()
    print("[Preview: vulnerability]")
    print(vulnerability[["sigungu", "crop_vulnerability_index", "crop_vulnerability_level"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
