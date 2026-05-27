from pathlib import Path
from datetime import datetime
import re
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "04_weather_drought"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ENCODINGS = ["utf-8-sig", "cp949", "euc-kr", "utf-8"]

CHUNGNAM_SIGUNGU = [
    "천안시", "공주시", "보령시", "아산시", "서산시", "논산시", "계룡시", "당진시",
    "금산군", "부여군", "서천군", "청양군", "홍성군", "예산군", "태안군"
]

DROUGHT_STAGE_SCORE = {
    "정상": 0,
    "관심": 25,
    "약한가뭄": 25,
    "주의": 50,
    "보통가뭄": 50,
    "경계": 75,
    "심한가뭄": 75,
    "심각": 100,
    "극심한가뭄": 100,
    "0": 0,
    "1": 25,
    "2": 50,
    "3": 75,
    "4": 100,
}


def clean_number(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        x = (
            x.replace(",", "")
             .replace("%", "")
             .replace("㎜", "")
             .replace("mm", "")
             .strip()
        )
        if x in ["", "-", "nan", "None", "NULL"]:
            return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan


def normalize_sigungu(text):
    if pd.isna(text):
        return np.nan
    text = str(text)

    if "천안" in text:
        return "천안시"

    for sg in CHUNGNAM_SIGUNGU:
        if sg in text:
            return sg

    return np.nan


def normalize_sido(text):
    if pd.isna(text):
        return ""
    text = str(text)
    if "충남" in text or "충청남도" in text:
        return "충남"
    return text


def infer_date_column(df):
    candidates = [
        "기준일자", "일자", "날짜", "관측일자", "측정일자",
        "base_date", "date", "ymd"
    ]

    for c in candidates:
        if c in df.columns:
            return c

    for c in df.columns:
        name = str(c)
        if "일자" in name or "날짜" in name or "기준" in name:
            return c

    return None


def infer_sigungu_column(df):
    candidates = [
        "시군명", "시군", "시군구", "지역", "지역명", "관측지역",
        "sigungu", "sgg"
    ]

    for c in candidates:
        if c in df.columns:
            return c

    for c in df.columns:
        name = str(c)
        if "시군" in name or "시군구" in name or "지역" in name:
            return c

    return None


def infer_sido_column(df):
    candidates = ["시도명", "시도", "도명", "sido"]

    for c in candidates:
        if c in df.columns:
            return c

    for c in df.columns:
        name = str(c)
        if "시도" in name:
            return c

    return None


def find_col_contains(df, keywords):
    for c in df.columns:
        name = str(c)
        if any(k in name for k in keywords):
            return c
    return None


def read_csv_safely(path: Path):
    last_error = None
    for enc in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_error = e
    raise RuntimeError(f"CSV read failed: {path} / {last_error}")


def read_excel_safely(path: Path):
    try:
        xls = pd.ExcelFile(path)
        frames = []
        for sheet in xls.sheet_names:
            try:
                df = pd.read_excel(path, sheet_name=sheet)
                if len(df) > 0:
                    df["__sheet_name__"] = sheet
                    frames.append(df)
            except Exception:
                continue
        if frames:
            return pd.concat(frames, ignore_index=True)
    except Exception:
        pass

    # Some .xls files are HTML tables
    try:
        tables = pd.read_html(path)
        if tables:
            return pd.concat(tables, ignore_index=True)
    except Exception as e:
        raise RuntimeError(f"Excel/HTML read failed: {path} / {e}")

    raise RuntimeError(f"No readable table found: {path}")


def read_table(path: Path):
    ext = path.suffix.lower()

    if ext == ".csv":
        return read_csv_safely(path)

    if ext in [".xlsx", ".xls"]:
        return read_excel_safely(path)

    return None


def normalize_stage_value(x):
    if pd.isna(x):
        return np.nan

    s = str(x).strip().replace(" ", "")

    for key, score in DROUGHT_STAGE_SCORE.items():
        if key in s:
            return score

    num = clean_number(s)
    if pd.notna(num):
        if num <= 4:
            return DROUGHT_STAGE_SCORE.get(str(int(num)), np.nan)
        return np.clip(num, 0, 100)

    return np.nan


def parse_date_series(s):
    out = pd.to_datetime(s, errors="coerce")

    # Handle yyyymmdd numeric/string
    mask = out.isna()
    if mask.any():
        alt = s.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
        out2 = pd.to_datetime(alt, format="%Y%m%d", errors="coerce")
        out = out.where(~mask, out2)

    return out


def extract_weather_drought_records(path: Path, df: pd.DataFrame):
    original_cols = list(df.columns)
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    date_col = infer_date_column(df)
    sigungu_col = infer_sigungu_column(df)
    sido_col = infer_sido_column(df)

    # If no sigungu column, skip because we need city/county level
    if sigungu_col is None:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["sigungu"] = df[sigungu_col].apply(normalize_sigungu)

    if date_col is not None:
        out["date"] = parse_date_series(df[date_col])
    else:
        out["date"] = pd.NaT

    if sido_col is not None:
        out["sido"] = df[sido_col].apply(normalize_sido)
    else:
        out["sido"] = ""

    # Keep only Chungnam rows
    out = out[out["sigungu"].notna()].copy()

    if len(out) == 0:
        return pd.DataFrame()

    # Numeric indicators
    reservoir_rate_col = find_col_contains(df, ["저수율"])
    normal_rate_col = find_col_contains(df, ["평년"])
    normal_ratio_col = find_col_contains(df, ["평년대비", "대비"])
    rainfall_col = find_col_contains(df, ["강우량", "강수량", "누적강우"])
    drought_stage_col = find_col_contains(df, ["가뭄단계", "단계", "가뭄현황", "가뭄전망"])

    out["reservoir_rate_from_drought"] = (
        df[reservoir_rate_col].apply(clean_number) if reservoir_rate_col else np.nan
    )
    out["normal_rate"] = (
        df[normal_rate_col].apply(clean_number) if normal_rate_col else np.nan
    )
    out["normal_ratio"] = (
        df[normal_ratio_col].apply(clean_number) if normal_ratio_col else np.nan
    )
    out["rainfall"] = (
        df[rainfall_col].apply(clean_number) if rainfall_col else np.nan
    )
    out["drought_stage_score"] = (
        df[drought_stage_col].apply(normalize_stage_value) if drought_stage_col else np.nan
    )

    # Risk components
    out["reservoir_rate_risk"] = 100 - out["reservoir_rate_from_drought"]
    out["reservoir_rate_risk"] = out["reservoir_rate_risk"].clip(0, 100)

    # If normal_ratio is 100 = normal, less than 100 = deficit
    out["normal_ratio_deficit_score"] = 100 - out["normal_ratio"]
    out["normal_ratio_deficit_score"] = out["normal_ratio_deficit_score"].clip(0, 100)

    # Rainfall risk is hard without normal baseline; preserve rainfall only.
    out["source_file"] = path.name
    out["source_path"] = str(path.relative_to(ROOT))

    return out


def load_all_weather_drought():
    records = []

    files = [
        p for p in RAW_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in [".csv", ".xlsx", ".xls"]
    ]

    for path in files:
        try:
            df = read_table(path)
            if df is None or len(df) == 0:
                continue

            part = extract_weather_drought_records(path, df)
            if len(part) > 0:
                records.append(part)
                print(f"[OK] parsed: {path.name} -> rows={len(part)}")
            else:
                print(f"[SKIP] no Chungnam sigungu rows: {path.name}")

        except Exception as e:
            print(f"[ERROR] {path.name}: {e}")

    if not records:
        raise RuntimeError("No weather/drought records parsed.")

    return pd.concat(records, ignore_index=True)


def build_sigungu_daily(df):
    valid = df.copy()

    # If date is missing, assign NaT and still keep source-level records
    grouped_cols = ["date", "sigungu"]

    agg = valid.groupby(grouped_cols, dropna=False, as_index=False).agg(
        drought_stage_score=("drought_stage_score", "max"),
        avg_reservoir_rate_from_drought=("reservoir_rate_from_drought", "mean"),
        avg_normal_rate=("normal_rate", "mean"),
        avg_normal_ratio=("normal_ratio", "mean"),
        normal_ratio_deficit_score=("normal_ratio_deficit_score", "mean"),
        avg_rainfall=("rainfall", "mean"),
        record_count=("sigungu", "size"),
    )

    # Composite drought/weather risk
    risk_cols = [
        "drought_stage_score",
        "normal_ratio_deficit_score",
    ]

    # If stage and normal deficit are missing but reservoir rate exists, use water rate risk
    if "avg_reservoir_rate_from_drought" in agg.columns:
        agg["reservoir_rate_risk_from_drought"] = 100 - agg["avg_reservoir_rate_from_drought"]
        agg["reservoir_rate_risk_from_drought"] = agg["reservoir_rate_risk_from_drought"].clip(0, 100)
        risk_cols.append("reservoir_rate_risk_from_drought")

    agg["weather_drought_risk_score"] = agg[risk_cols].mean(axis=1, skipna=True)
    agg["weather_drought_risk_score"] = agg["weather_drought_risk_score"].fillna(0).clip(0, 100)

    agg["weather_drought_level"] = pd.cut(
        agg["weather_drought_risk_score"],
        bins=[-1, 39, 59, 79, 100],
        labels=["낮음", "주의", "경계", "심각"]
    ).astype(str)

    return agg


def build_latest_sigungu_features(sigungu_daily):
    df = sigungu_daily.copy()

    dated = df[df["date"].notna()].copy()

    if len(dated) == 0:
        latest = df.groupby("sigungu", as_index=False).tail(1)
        return latest

    latest_dates = dated.groupby("sigungu")["date"].max().reset_index()
    latest = dated.merge(latest_dates, on=["sigungu", "date"], how="inner")

    # If duplicate latest records exist, average them
    latest = latest.groupby(["date", "sigungu"], as_index=False).agg(
        drought_stage_score=("drought_stage_score", "max"),
        avg_reservoir_rate_from_drought=("avg_reservoir_rate_from_drought", "mean"),
        avg_normal_rate=("avg_normal_rate", "mean"),
        avg_normal_ratio=("avg_normal_ratio", "mean"),
        normal_ratio_deficit_score=("normal_ratio_deficit_score", "mean"),
        avg_rainfall=("avg_rainfall", "mean"),
        reservoir_rate_risk_from_drought=("reservoir_rate_risk_from_drought", "mean"),
        weather_drought_risk_score=("weather_drought_risk_score", "mean"),
        record_count=("record_count", "sum"),
    )

    latest["weather_drought_level"] = pd.cut(
        latest["weather_drought_risk_score"],
        bins=[-1, 39, 59, 79, 100],
        labels=["낮음", "주의", "경계", "심각"]
    ).astype(str)

    return latest


def main():
    print("[AquaGuard AI] Step 03 - Weather/Drought preprocessing")
    print(f"RAW_DIR: {RAW_DIR}")

    raw = load_all_weather_drought()
    sigungu_daily = build_sigungu_daily(raw)
    latest = build_latest_sigungu_features(sigungu_daily)

    raw_path = OUT_DIR / "04_weather_drought_raw_combined.csv"
    daily_path = OUT_DIR / "04_weather_drought_sigungu_daily.csv"
    latest_path = OUT_DIR / "04_weather_drought_latest_by_sigungu.csv"

    raw.to_csv(raw_path, index=False, encoding="utf-8-sig")
    sigungu_daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    latest.to_csv(latest_path, index=False, encoding="utf-8-sig")

    print()
    print("[Saved]")
    print(f"- {raw_path}")
    print(f"- {daily_path}")
    print(f"- {latest_path}")

    print()
    print("[Preview: latest by sigungu]")
    print(latest.sort_values("weather_drought_risk_score", ascending=False).head(15))


if __name__ == "__main__":
    main()
