from pathlib import Path
from datetime import datetime, timezone, timedelta
import re
import sys
import requests
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

OLDAM_URL = "https://alldam.chungnam.go.kr/bigdata/collect/csvDownLoad.do?apiIdx=2869"

RAW_ROOT = ROOT / "data" / "raw" / "live_snapshots"
PROCESSED = ROOT / "data" / "processed"
REPORT_TABLES = ROOT / "reports" / "tables"
META = ROOT / "data" / "metadata"

FACILITY_PATH = PROCESSED / "01_reservoir_facility_clean.csv"

KST = timezone(timedelta(hours=9))
TODAY = datetime.now(KST).strftime("%Y%m%d")

SNAPSHOT_DIR = RAW_ROOT / TODAY
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)
REPORT_TABLES.mkdir(parents=True, exist_ok=True)
META.mkdir(parents=True, exist_ok=True)

RAW_CSV_PATH = SNAPSHOT_DIR / "oldam_reservoir_today_raw.csv"
STD_PATH = PROCESSED / "latest_oldam_reservoir_today.csv"
SIGUNGU_PATH = REPORT_TABLES / "latest_live_reservoir_by_sigungu.csv"
STATUS_PATH = REPORT_TABLES / "latest_oldam_status_summary.csv"
LOG_PATH = META / "live_data_collection_log.csv"


def normalize_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()




CHUNGNAM_SIGUNGU = [
    "천안시", "공주시", "보령시", "아산시", "서산시",
    "논산시", "계룡시", "당진시", "금산군", "부여군",
    "서천군", "청양군", "홍성군", "예산군", "태안군",
]


def extract_sigungu_from_location(location):
    text = normalize_text(location)
    for sigungu in CHUNGNAM_SIGUNGU:
        if sigungu in text:
            return sigungu
    return ""


def make_key(x):
    x = normalize_text(x)
    x = re.sub(r"\s+", "", x)
    x = re.sub(r"[^0-9A-Za-z가-힣()]", "", x)
    return x


def find_col(cols, include_keywords, exclude_keywords=None):
    exclude_keywords = exclude_keywords or []

    for col in cols:
        c = str(col).strip()
        if all(k in c for k in include_keywords) and not any(e in c for e in exclude_keywords):
            return col

    return None


def download_oldam_csv():
    headers = {
        "User-Agent": "Mozilla/5.0 AquaGuard-AI/1.0",
        "Accept": "text/csv,application/csv,application/octet-stream,*/*",
        "Referer": "https://alldam.chungnam.go.kr/bigdata/collect/view.chungnam?menuCd=DOM_000000201001001000&apiIdx=2869",
    }

    r = requests.get(OLDAM_URL, headers=headers, timeout=60)
    r.raise_for_status()

    content = r.content

    if content[:50].lstrip().startswith(b"<"):
        raise RuntimeError(
            "Downloaded content looks like HTML, not CSV. "
            "The endpoint may require session/cookie or block direct download."
        )

    RAW_CSV_PATH.write_bytes(content)

    return RAW_CSV_PATH, len(content)


def read_csv_with_encoding(path):
    encodings = ["utf-8-sig", "cp949", "euc-kr", "utf-8"]

    last_error = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc, dtype=str)
            return df, enc
        except Exception as e:
            last_error = e

    raise RuntimeError(f"CSV read failed. last_error={last_error}")


def standardize_oldam_columns(raw):
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    cols = df.columns.tolist()

    name_col = (
        find_col(cols, ["저수지", "이름"])
        or find_col(cols, ["저수지", "명"])
        or find_col(cols, ["시설", "명"])
    )

    rate_col = find_col(cols, ["저수율"])
    date_col = (
        find_col(cols, ["측정", "날짜"])
        or find_col(cols, ["기준", "일자"])
        or find_col(cols, ["날짜"])
        or find_col(cols, ["일자"])
    )

    location_col = (
        find_col(cols, ["저수지", "위치"])
        or find_col(cols, ["위치"])
        or find_col(cols, ["주소"])
        or find_col(cols, ["소재"])
    )

    code_col = (
        find_col(cols, ["저수지", "코드"])
        or find_col(cols, ["코드"])
        or find_col(cols, ["시설", "번호"])
    )

    water_level_col = find_col(cols, ["수위"], exclude_keywords=["저수율"])

    required = {
        "facility_name": name_col,
        "reservoir_rate": rate_col,
        "date": date_col,
    }

    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise KeyError(
            f"Required columns not found: {missing}\n"
            f"Detected columns={cols}"
        )

    out = pd.DataFrame()

    out["facility_name"] = df[name_col].map(normalize_text)
    out["reservoir_rate_raw"] = df[rate_col].map(normalize_text)
    out["reservoir_rate"] = pd.to_numeric(
        out["reservoir_rate_raw"].str.replace(",", "", regex=False),
        errors="coerce",
    )

    out["date_raw"] = df[date_col].map(normalize_text)
    out["date"] = pd.to_datetime(out["date_raw"], errors="coerce")

    # 20260522 같은 숫자형 날짜 보정
    mask = out["date"].isna() & out["date_raw"].str.match(r"^\d{8}$", na=False)
    out.loc[mask, "date"] = pd.to_datetime(out.loc[mask, "date_raw"], format="%Y%m%d", errors="coerce")

    if location_col is not None:
        out["location_raw"] = df[location_col].map(normalize_text)
    else:
        out["location_raw"] = ""

    if code_col is not None:
        out["facility_code"] = df[code_col].map(normalize_text)
    else:
        out["facility_code"] = ""

    if water_level_col is not None:
        out["water_level_raw"] = df[water_level_col].map(normalize_text)
        out["water_level"] = pd.to_numeric(
            out["water_level_raw"].str.replace(",", "", regex=False),
            errors="coerce",
        )
    else:
        out["water_level_raw"] = ""
        out["water_level"] = np.nan

    out["facility_key"] = out["facility_name"].map(make_key)

    out["reservoir_rate_for_score"] = out["reservoir_rate"].clip(lower=0, upper=100)
    out["reservoir_rate_over_100"] = (out["reservoir_rate"] > 100).astype(int)
    out["reservoir_rate_missing"] = out["reservoir_rate"].isna().astype(int)

    out = out[out["facility_name"].ne("")].copy()

    return out


def attach_facility_info(today):
    if not FACILITY_PATH.exists():
        today["sigungu"] = ""
        today["address"] = ""
        today["benefit_area"] = np.nan
        today["effective_capacity"] = np.nan
        today["facility_match_status"] = "facility_file_missing"
        return today

    fac = pd.read_csv(FACILITY_PATH, dtype=str)
    fac.columns = [str(c).strip() for c in fac.columns]

    if "facility_name" not in fac.columns:
        raise KeyError(f"facility_name not found in {FACILITY_PATH}. columns={fac.columns.tolist()}")

    fac["facility_key"] = fac["facility_name"].map(make_key)

    keep_cols = [
        "facility_key",
        "sigungu",
        "address",
        "benefit_area",
        "effective_capacity",
        "total_capacity",
        "basin_area",
        "full_water_area",
    ]
    keep_cols = [c for c in keep_cols if c in fac.columns]

    fac_small = fac[keep_cols].drop_duplicates("facility_key", keep="first").copy()

    merged = today.merge(fac_small, on="facility_key", how="left")

    if "sigungu" not in merged.columns:
        merged["sigungu"] = ""

    merged["sigungu"] = merged["sigungu"].fillna("").astype(str)

    merged["sigungu_from_location"] = merged["location_raw"].apply(extract_sigungu_from_location)

    name_matched = merged["sigungu"].astype(str).str.len() > 0
    location_matched = (~name_matched) & merged["sigungu_from_location"].astype(str).str.len().gt(0)

    merged.loc[location_matched, "sigungu"] = merged.loc[location_matched, "sigungu_from_location"]

    merged["facility_match_status"] = np.select(
        [
            name_matched,
            location_matched,
        ],
        [
            "facility_name_matched",
            "location_matched",
        ],
        default="unmatched",
    )

    return merged


def build_sigungu_summary(today):
    work = today.copy()

    # 시군 매칭이 안 된 행은 별도 표시는 하되 시군 집계에서는 제외
    matched = work[work["sigungu"].astype(str).str.len() > 0].copy()

    if matched.empty:
        return pd.DataFrame(columns=[
            "date", "sigungu", "today_reservoir_count",
            "today_avg_reservoir_rate", "today_min_reservoir_rate",
            "today_max_reservoir_rate", "today_low_reservoir_count_40",
            "today_low_reservoir_count_30", "today_over_100_count",
            "today_reservoir_risk_score"
        ])

    g = matched.groupby("sigungu", dropna=False)

    out = g.agg(
        date=("date", "max"),
        today_reservoir_count=("facility_name", "count"),
        today_avg_reservoir_rate=("reservoir_rate_for_score", "mean"),
        today_min_reservoir_rate=("reservoir_rate_for_score", "min"),
        today_max_reservoir_rate=("reservoir_rate_for_score", "max"),
        today_low_reservoir_count_40=("reservoir_rate_for_score", lambda s: int((s <= 40).sum())),
        today_low_reservoir_count_30=("reservoir_rate_for_score", lambda s: int((s <= 30).sum())),
        today_over_100_count=("reservoir_rate_over_100", "sum"),
    ).reset_index()

    out["today_reservoir_risk_score"] = (
        (100 - out["today_avg_reservoir_rate"]) * 0.70
        + (out["today_low_reservoir_count_40"] / out["today_reservoir_count"].clip(lower=1)) * 20
        + (out["today_low_reservoir_count_30"] / out["today_reservoir_count"].clip(lower=1)) * 30
    ).clip(0, 100)

    out = out.sort_values("today_reservoir_risk_score", ascending=False).reset_index(drop=True)
    out["today_reservoir_risk_rank"] = np.arange(1, len(out) + 1)

    front = ["today_reservoir_risk_rank", "date", "sigungu"]
    other = [c for c in out.columns if c not in front]
    return out[front + other]


def build_status(raw, today, sigungu_summary, raw_bytes, encoding):
    latest_date = today["date"].max() if "date" in today.columns and len(today) else pd.NaT

    status = {
        "collection_date_kst": TODAY,
        "source_url": OLDAM_URL,
        "raw_csv_path": str(RAW_CSV_PATH.relative_to(ROOT)),
        "raw_bytes": raw_bytes,
        "csv_encoding": encoding,
        "raw_rows": len(raw),
        "standardized_rows": len(today),
        "matched_rows": int((today["facility_match_status"] == "matched").sum()) if "facility_match_status" in today.columns else 0,
        "unmatched_rows": int((today["facility_match_status"] == "unmatched").sum()) if "facility_match_status" in today.columns else len(today),
        "latest_measurement_date": "" if pd.isna(latest_date) else str(pd.Timestamp(latest_date).date()),
        "sigungu_count": sigungu_summary["sigungu"].nunique() if not sigungu_summary.empty else 0,
        "avg_reservoir_rate_today": float(today["reservoir_rate_for_score"].mean()) if len(today) else np.nan,
        "low_40_count_today": int((today["reservoir_rate_for_score"] <= 40).sum()) if len(today) else 0,
        "low_30_count_today": int((today["reservoir_rate_for_score"] <= 30).sum()) if len(today) else 0,
        "over_100_count_today": int((today["reservoir_rate"] > 100).sum()) if len(today) else 0,
        "status": "SUCCESS",
    }

    return pd.DataFrame([status])


def append_log(status_df):
    row = status_df.copy()
    row["logged_at_kst"] = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")

    if LOG_PATH.exists():
        old = pd.read_csv(LOG_PATH)
        out = pd.concat([old, row], ignore_index=True)
    else:
        out = row

    out.to_csv(LOG_PATH, index=False, encoding="utf-8-sig")


def main():
    print("[AquaGuard AI] Fetch OLDAM reservoir today CSV")
    print(f"URL: {OLDAM_URL}")

    raw_path, raw_bytes = download_oldam_csv()
    print(f"[OK] downloaded: {raw_path} bytes={raw_bytes}")

    raw, enc = read_csv_with_encoding(raw_path)
    print(f"[OK] read csv encoding={enc}, rows={len(raw)}, cols={len(raw.columns)}")
    print("[COLUMNS]")
    print(raw.columns.tolist())

    today = standardize_oldam_columns(raw)
    today = attach_facility_info(today)

    sigungu_summary = build_sigungu_summary(today)
    status = build_status(raw, today, sigungu_summary, raw_bytes, enc)

    today.to_csv(STD_PATH, index=False, encoding="utf-8-sig")
    sigungu_summary.to_csv(SIGUNGU_PATH, index=False, encoding="utf-8-sig")
    status.to_csv(STATUS_PATH, index=False, encoding="utf-8-sig")
    append_log(status)

    print()
    print("[Saved]")
    print(f"- {STD_PATH} rows={len(today)}")
    print(f"- {SIGUNGU_PATH} rows={len(sigungu_summary)}")
    print(f"- {STATUS_PATH}")
    print(f"- {LOG_PATH}")

    print()
    print("[Status]")
    print(status.to_string(index=False))

    print()
    print("[Sigungu Summary Preview]")
    if sigungu_summary.empty:
        print("No matched sigungu summary. Check facility name matching.")
    else:
        show_cols = [
            "today_reservoir_risk_rank",
            "sigungu",
            "today_reservoir_count",
            "today_avg_reservoir_rate",
            "today_min_reservoir_rate",
            "today_low_reservoir_count_40",
            "today_low_reservoir_count_30",
            "today_reservoir_risk_score",
        ]
        print(sigungu_summary[show_cols].to_string(index=False))

    unmatched = today[today["facility_match_status"] == "unmatched"]
    if len(unmatched):
        print()
        print(f"[WARN] unmatched reservoirs: {len(unmatched)}")
        print(unmatched[["facility_name", "reservoir_rate", "date_raw", "location_raw"]].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
