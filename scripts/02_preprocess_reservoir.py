from pathlib import Path
import re
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "01_reservoir"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ENCODINGS = ["utf-8-sig", "cp949", "euc-kr", "utf-8"]

CHUNGNAM_SIGUNGU = [
    "천안시", "공주시", "보령시", "아산시", "서산시", "논산시", "계룡시", "당진시",
    "금산군", "부여군", "서천군", "청양군", "홍성군", "예산군", "태안군"
]

DATE_COL_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def read_csv_safely(path: Path) -> pd.DataFrame:
    last_error = None
    for enc in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_error = e
    raise RuntimeError(f"Failed to read {path}: {last_error}")


def clean_number(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        x = x.replace(",", "").replace("%", "").strip()
        if x in ["", "-", "nan", "None"]:
            return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan


def normalize_sigungu(text):
    if pd.isna(text):
        return np.nan
    text = str(text)

    # 천안시 동남구/서북구는 신청서류·대시보드 기준에서 천안시로 통합
    if "천안" in text:
        return "천안시"

    for sg in CHUNGNAM_SIGUNGU:
        if sg in text:
            return sg

    return np.nan


def is_daily_rate_file(df: pd.DataFrame) -> bool:
    cols = list(map(str, df.columns))
    date_cols = [c for c in cols if DATE_COL_PATTERN.match(c)]
    return ("저수지명" in cols) and ("위치" in cols) and len(date_cols) >= 30


def is_facility_file(df: pd.DataFrame) -> bool:
    cols = list(map(str, df.columns))
    required = {"시설명", "소재지"}
    capacity_like = any(c in cols for c in ["수혜면적", "총저수량", "유효저수량"])
    return required.issubset(set(cols)) and capacity_like


def load_daily_rate_files():
    rows = []
    source_files = []

    for path in RAW_DIR.rglob("*.csv"):
        df = read_csv_safely(path)

        if not is_daily_rate_file(df):
            continue

        source_files.append(path.name)

        date_cols = [c for c in df.columns if DATE_COL_PATTERN.match(str(c))]

        id_cols = [c for c in ["저수지명", "위치", "유효저수량(천m3)", "유효저수량"] if c in df.columns]

        long_df = df[id_cols + date_cols].melt(
            id_vars=id_cols,
            value_vars=date_cols,
            var_name="date",
            value_name="storage_rate"
        )

        long_df["source_file"] = path.name
        rows.append(long_df)

    if not rows:
        raise FileNotFoundError("No daily reservoir rate files found under data/raw/01_reservoir")

    out = pd.concat(rows, ignore_index=True)

    if "유효저수량(천m3)" in out.columns:
        out["effective_capacity"] = out["유효저수량(천m3)"].apply(clean_number)
    elif "유효저수량" in out.columns:
        out["effective_capacity"] = out["유효저수량"].apply(clean_number)
    else:
        out["effective_capacity"] = np.nan

    out["storage_rate"] = out["storage_rate"].apply(clean_number)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["sigungu"] = out["위치"].apply(normalize_sigungu)

    out = out.dropna(subset=["date", "sigungu", "storage_rate"])

    out = out.rename(columns={
        "저수지명": "reservoir_name",
        "위치": "location"
    })

    out = out[
        [
            "date",
            "sigungu",
            "reservoir_name",
            "location",
            "effective_capacity",
            "storage_rate",
            "source_file"
        ]
    ]

    print(f"[OK] daily rate files: {len(source_files)}")
    for f in source_files:
        print(f"  - {f}")

    return out


def load_facility_files():
    rows = []
    source_files = []

    for path in RAW_DIR.rglob("*.csv"):
        df = read_csv_safely(path)

        if not is_facility_file(df):
            continue

        source_files.append(path.name)

        keep_cols = [c for c in [
            "표준코드", "본부", "지사", "시설명", "소재지", "수혜면적",
            "총저수량", "유효저수량", "유역면적", "만수면적", "착공년도", "준공년도"
        ] if c in df.columns]

        part = df[keep_cols].copy()
        part["source_file"] = path.name
        rows.append(part)

    if not rows:
        print("[WARN] No facility files found.")
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)

    out["sigungu"] = out["소재지"].apply(normalize_sigungu)
    out = out.dropna(subset=["sigungu"])

    rename_map = {
        "시설명": "facility_name",
        "소재지": "address",
        "수혜면적": "benefit_area",
        "총저수량": "total_capacity",
        "유효저수량": "effective_capacity",
        "유역면적": "basin_area",
        "만수면적": "full_water_area",
        "착공년도": "start_year",
        "준공년도": "completion_year",
    }

    out = out.rename(columns=rename_map)

    for col in [
        "benefit_area", "total_capacity", "effective_capacity",
        "basin_area", "full_water_area"
    ]:
        if col in out.columns:
            out[col] = out[col].apply(clean_number)

    print(f"[OK] facility files: {len(source_files)}")
    for f in source_files:
        print(f"  - {f}")

    return out


def build_sigungu_daily_features(daily: pd.DataFrame):
    df = daily.copy()

    grouped = df.groupby(["date", "sigungu"], as_index=False).agg(
        avg_reservoir_rate=("storage_rate", "mean"),
        min_reservoir_rate=("storage_rate", "min"),
        max_reservoir_rate=("storage_rate", "max"),
        reservoir_count=("reservoir_name", "nunique"),
        low_reservoir_count_40=("storage_rate", lambda x: int((x < 40).sum())),
        low_reservoir_count_30=("storage_rate", lambda x: int((x < 30).sum())),
        total_effective_capacity=("effective_capacity", "sum"),
    )

    grouped["reservoir_risk_score"] = 100 - grouped["avg_reservoir_rate"]
    grouped["reservoir_risk_score"] = grouped["reservoir_risk_score"].clip(0, 100)

    grouped["risk_level_by_reservoir"] = pd.cut(
        grouped["reservoir_risk_score"],
        bins=[-1, 39, 59, 79, 100],
        labels=["낮음", "주의", "경계", "심각"]
    ).astype(str)

    return grouped


def build_facility_summary(facility: pd.DataFrame):
    if facility.empty:
        return facility

    agg_dict = {
        "facility_name": "nunique",
    }

    optional_sum_cols = [
        "benefit_area", "total_capacity", "effective_capacity",
        "basin_area", "full_water_area"
    ]

    for col in optional_sum_cols:
        if col in facility.columns:
            agg_dict[col] = "sum"

    summary = facility.groupby("sigungu", as_index=False).agg(agg_dict)

    summary = summary.rename(columns={
        "facility_name": "facility_reservoir_count",
        "benefit_area": "total_benefit_area",
        "total_capacity": "facility_total_capacity",
        "effective_capacity": "facility_effective_capacity",
        "basin_area": "facility_basin_area",
        "full_water_area": "facility_full_water_area",
    })

    return summary


def main():
    print("[AquaGuard AI] Step 02 - Reservoir preprocessing")
    print(f"RAW_DIR: {RAW_DIR}")

    daily = load_daily_rate_files()
    facility = load_facility_files()

    sigungu_daily = build_sigungu_daily_features(daily)
    facility_summary = build_facility_summary(facility)

    daily_path = OUT_DIR / "01_reservoir_daily_long.csv"
    sigungu_daily_path = OUT_DIR / "01_reservoir_sigungu_daily.csv"
    facility_path = OUT_DIR / "01_reservoir_facility_clean.csv"
    facility_summary_path = OUT_DIR / "01_reservoir_facility_summary.csv"

    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    sigungu_daily.to_csv(sigungu_daily_path, index=False, encoding="utf-8-sig")

    if not facility.empty:
        facility.to_csv(facility_path, index=False, encoding="utf-8-sig")
        facility_summary.to_csv(facility_summary_path, index=False, encoding="utf-8-sig")

    print()
    print("[Saved]")
    print(f"- {daily_path}")
    print(f"- {sigungu_daily_path}")
    if not facility.empty:
        print(f"- {facility_path}")
        print(f"- {facility_summary_path}")

    print()
    print("[Preview: sigungu daily]")
    print(sigungu_daily.head(10))


if __name__ == "__main__":
    main()
