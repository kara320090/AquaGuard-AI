from pathlib import Path
from datetime import datetime, timezone, timedelta
from io import BytesIO, StringIO
import re
import requests
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

RAW_ROOT = ROOT / "data" / "raw" / "live_snapshots"
PROCESSED = ROOT / "data" / "processed"
REPORT_TABLES = ROOT / "reports" / "tables"
META = ROOT / "data" / "metadata"

KST = timezone(timedelta(hours=9))
TODAY_DT = datetime.now(KST)
TODAY = TODAY_DT.strftime("%Y%m%d")

SNAPSHOT_DIR = RAW_ROOT / TODAY
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)
REPORT_TABLES.mkdir(parents=True, exist_ok=True)
META.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://adms.ekr.or.kr"
MAIN_URL = BASE_URL + "/page/farmFarmWaterValueMain.do"
EXCEL_URL = BASE_URL + "/droughtInfo/retrieveFarmWaterValueListExcel.do"
LIST_URL = BASE_URL + "/droughtInfo/retrieveFarmWaterValueList.do"

RAW_PATH = SNAPSHOT_DIR / "adms_soil_moisture_today_raw.bin"
RAW_HTML_PATH = SNAPSHOT_DIR / "adms_soil_moisture_today_raw.html"

STD_PATH = PROCESSED / "latest_adms_soil_moisture.csv"
SIGUNGU_PATH = PROCESSED / "latest_adms_soil_moisture_by_sigungu.csv"
STATUS_PATH = REPORT_TABLES / "latest_adms_soil_moisture_status.csv"
LOG_PATH = META / "adms_soil_moisture_collection_log.csv"

CHUNGNAM_SIGUNGU = [
    "천안시", "공주시", "보령시", "아산시", "서산시",
    "논산시", "계룡시", "당진시", "금산군", "부여군",
    "서천군", "청양군", "홍성군", "예산군", "태안군",
]


def normalize_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def clean_col(c):
    c = normalize_text(c)
    c = re.sub(r"\s+", "", c)
    c = c.replace("\n", "").replace("\r", "")
    return c


def find_col(cols, include_keywords, exclude_keywords=None):
    exclude_keywords = exclude_keywords or []
    for col in cols:
        c = clean_col(col)
        if all(k in c for k in include_keywords) and not any(e in c for e in exclude_keywords):
            return col
    return None


def extract_default_date(html):
    # <input ... id="dateFrom" ... value="2026-05-26" />
    m = re.search(r'id=["\']dateFrom["\'][^>]*value=["\']([^"\']+)["\']', html)
    if m:
        return m.group(1).strip()

    m = re.search(r'name=["\']dateFrom["\'][^>]*value=["\']([^"\']+)["\']', html)
    if m:
        return m.group(1).strip()

    # fallback: 오늘 날짜
    return datetime.now(KST).strftime("%Y-%m-%d")


def decode_bytes(content):
    for enc in ["utf-8", "cp949", "euc-kr"]:
        try:
            return content.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="replace"), "utf-8-replace"


def request_adms():
    session = requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0 AquaGuard-AI/1.0",
        "Referer": MAIN_URL,
        "Origin": BASE_URL,
    }

    main = session.get(MAIN_URL, headers=headers, timeout=60)
    main.raise_for_status()
    main_text, main_enc = decode_bytes(main.content)
    default_date = extract_default_date(main_text)

    payload = {
        "dateFrom": default_date,
        "sidoNm": "",
        "sigunNm": "",
        "pageIndex": "1",
    }

    r = session.post(EXCEL_URL, data=payload, headers=headers, timeout=90)
    r.raise_for_status()

    RAW_PATH.write_bytes(r.content)

    return {
        "content": r.content,
        "content_type": r.headers.get("Content-Type", ""),
        "content_disposition": r.headers.get("Content-Disposition", ""),
        "bytes": len(r.content),
        "query_date": default_date,
        "main_encoding": main_enc,
    }


def try_parse_excel_or_html(content):
    # 1) xlsx zip signature
    if content[:2] == b"PK":
        return pd.read_excel(BytesIO(content), dtype=str), "xlsx"

    # 2) old xls signature
    if content[:8].startswith(b"\xd0\xcf\x11\xe0"):
        return pd.read_excel(BytesIO(content), dtype=str), "xls"

    # 3) text/html or csv
    text, enc = decode_bytes(content)
    RAW_HTML_PATH.write_text(text, encoding="utf-8")

    if "<html" in text.lower() or "<table" in text.lower():
        tables = pd.read_html(StringIO(text))
        if not tables:
            raise RuntimeError("HTML response found, but no tables parsed.")
        # 가장 행이 많은 테이블 선택
        df = max(tables, key=len)
        return df.astype(str), f"html_table_{enc}"

    # 4) CSV/TSV fallback
    try:
        df = pd.read_csv(StringIO(text), dtype=str)
        if df.shape[1] >= 2:
            return df, f"csv_{enc}"
    except Exception:
        pass

    try:
        df = pd.read_csv(StringIO(text), sep=r"\s+", engine="python", dtype=str)
        if df.shape[1] >= 2:
            return df, f"whitespace_{enc}"
    except Exception:
        pass

    raise RuntimeError(
        "Could not parse ADMS soil moisture response. "
        f"head={text[:300]}"
    )


def extract_sigungu_from_row(row):
    joined = " ".join(normalize_text(v) for v in row.values)
    for s in CHUNGNAM_SIGUNGU:
        if s in joined:
            return s
    return ""


def stage_to_score(x):
    t = normalize_text(x)

    if not t:
        return np.nan

    # ADMS/가뭄 계열 표현을 넓게 수용
    mapping = [
        ("심각", 100),
        ("위험", 90),
        ("경계", 75),
        ("주의", 55),
        ("관심", 35),
        ("약한", 30),
        ("건조", 60),
        ("매우건조", 85),
        ("부족", 70),
        ("정상", 0),
        ("양호", 0),
        ("충분", 0),
    ]

    for k, v in mapping:
        if k in t:
            return v

    return np.nan


def standardize(df, query_date):
    raw = df.copy()
    raw.columns = [normalize_text(c) for c in raw.columns]

    # MultiIndex/중복 컬럼 대비
    raw.columns = [clean_col(c) if clean_col(c) else f"col_{i}" for i, c in enumerate(raw.columns)]

    # 완전히 빈 행 제거
    raw = raw.dropna(how="all").copy()

    cols = raw.columns.tolist()

    date_col = (
        find_col(cols, ["기준", "일"])
        or find_col(cols, ["조사", "일"])
        or find_col(cols, ["관측", "일"])
        or find_col(cols, ["날짜"])
        or find_col(cols, ["일자"])
    )

    sido_col = (
        find_col(cols, ["시도"])
        or find_col(cols, ["도"])
    )

    sigungu_col = (
        find_col(cols, ["시군"])
        or find_col(cols, ["시군구"])
        or find_col(cols, ["시", "군"])
    )

    eupmyeon_col = (
        find_col(cols, ["읍면"])
        or find_col(cols, ["읍", "면"])
        or find_col(cols, ["동"])
    )

    moisture_col = (
        find_col(cols, ["유효", "수분"])
        or find_col(cols, ["토양", "수분"])
        or find_col(cols, ["수분", "율"])
        or find_col(cols, ["수분"])
    )

    stage_col = (
        find_col(cols, ["가뭄", "단계"])
        or find_col(cols, ["단계"])
        or find_col(cols, ["상태"])
        or find_col(cols, ["등급"])
        or find_col(cols, ["판정"])
    )

    out = pd.DataFrame()
    out["source_query_date"] = query_date

    if date_col:
        out["date_raw"] = raw[date_col].map(normalize_text)
    else:
        out["date_raw"] = query_date

    out["date"] = pd.to_datetime(out["date_raw"], errors="coerce")
    out["date"] = out["date"].fillna(pd.to_datetime(query_date, errors="coerce"))

    if sido_col:
        out["sido"] = raw[sido_col].map(normalize_text)
    else:
        out["sido"] = ""

    if sigungu_col:
        out["sigungu"] = raw[sigungu_col].map(normalize_text)
    else:
        out["sigungu"] = raw.apply(extract_sigungu_from_row, axis=1)

    if eupmyeon_col:
        out["eupmyeon"] = raw[eupmyeon_col].map(normalize_text)
    else:
        out["eupmyeon"] = ""

    if moisture_col:
        out["soil_moisture_raw"] = raw[moisture_col].map(normalize_text)
        out["soil_moisture"] = pd.to_numeric(
            out["soil_moisture_raw"].str.replace("%", "", regex=False).str.replace(",", "", regex=False),
            errors="coerce",
        )
    else:
        out["soil_moisture_raw"] = ""
        out["soil_moisture"] = np.nan

    if stage_col:
        out["soil_status"] = raw[stage_col].map(normalize_text)
        out["soil_stage_score"] = out["soil_status"].map(stage_to_score)
    else:
        out["soil_status"] = ""
        out["soil_stage_score"] = np.nan

    # 시군 값이 비어있으면 전체 행 문자열에서 재추출
    missing_sigungu = out["sigungu"].astype(str).str.len() == 0
    if missing_sigungu.any():
        out.loc[missing_sigungu, "sigungu"] = raw[missing_sigungu].apply(extract_sigungu_from_row, axis=1)

    # 충남 필터: 시도에 충남/충청남도 포함 또는 15개 시군명 포함
    joined_rows = raw.apply(lambda r: " ".join(normalize_text(v) for v in r.values), axis=1)
    is_chungnam = (
        out["sido"].astype(str).str.contains("충남|충청남도", regex=True, na=False)
        | out["sigungu"].isin(CHUNGNAM_SIGUNGU)
        | joined_rows.str.contains("|".join(CHUNGNAM_SIGUNGU), regex=True, na=False)
    )

    out = out[is_chungnam].copy()

    # 빈 DataFrame에서 scalar 컬럼을 먼저 만든 경우 NaN으로 남을 수 있어 필터 후 재보정
    out["source_query_date"] = query_date

    # 토양수분 수치만 있고 단계가 없으면 낮은 수분일수록 위험
    if out["soil_stage_score"].notna().sum() == 0:
        if out["soil_moisture"].notna().sum() > 0:
            out["soil_stage_score"] = (100 - out["soil_moisture"]).clip(0, 100)
        else:
            out["soil_stage_score"] = 50

    out["soil_dry_flag"] = (out["soil_stage_score"] >= 55).astype(int)
    out["soil_severe_flag"] = (out["soil_stage_score"] >= 75).astype(int)

    out["adms_parse_columns"] = " | ".join(cols)

    return out.reset_index(drop=True), {
        "date_col": date_col or "",
        "sido_col": sido_col or "",
        "sigungu_col": sigungu_col or "",
        "eupmyeon_col": eupmyeon_col or "",
        "moisture_col": moisture_col or "",
        "stage_col": stage_col or "",
        "raw_columns": cols,
    }


def build_sigungu_summary(std):
    if std.empty:
        return pd.DataFrame(columns=[
            "soil_moisture_rank", "sigungu", "soil_data_date",
            "soil_row_count", "soil_moisture_avg", "soil_moisture_min",
            "soil_dry_count", "soil_severe_count", "soil_moisture_drought_score",
            "soil_data_status"
        ])

    g = std.groupby("sigungu", dropna=False)

    out = g.agg(
        soil_data_date=("date", "max"),
        soil_row_count=("sigungu", "count"),
        soil_moisture_avg=("soil_moisture", "mean"),
        soil_moisture_min=("soil_moisture", "min"),
        soil_moisture_max=("soil_moisture", "max"),
        soil_dry_count=("soil_dry_flag", "sum"),
        soil_severe_count=("soil_severe_flag", "sum"),
        soil_moisture_drought_score=("soil_stage_score", "mean"),
    ).reset_index()

    out["soil_moisture_drought_score"] = out["soil_moisture_drought_score"].clip(0, 100)

    # ADMS 표에는 별도 날짜 컬럼이 없을 수 있으므로 source_query_date로 기준일 보정
    if "source_query_date" in std.columns and len(std) > 0:
        source_dates = std["source_query_date"].dropna().astype(str)
        fallback_date = source_dates.iloc[0] if len(source_dates) else ""
        if fallback_date:
            out["soil_data_date"] = out["soil_data_date"].fillna(fallback_date)

    out["soil_data_status"] = "ADMS_SOIL_AUTO"

    out = out.sort_values("soil_moisture_drought_score", ascending=False).reset_index(drop=True)
    out["soil_moisture_rank"] = np.arange(1, len(out) + 1)

    front = ["soil_moisture_rank", "sigungu"]
    other = [c for c in out.columns if c not in front]
    return out[front + other]


def append_log(status):
    row = status.copy()
    row["logged_at_kst"] = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")

    if LOG_PATH.exists():
        old = pd.read_csv(LOG_PATH)
        out = pd.concat([old, row], ignore_index=True)
    else:
        out = row

    out.to_csv(LOG_PATH, index=False, encoding="utf-8-sig")


def main():
    print("[AquaGuard AI] Fetch ADMS soil moisture")
    print(f"main: {MAIN_URL}")
    print(f"excel: {EXCEL_URL}")

    meta = request_adms()
    print(f"[OK] downloaded bytes={meta['bytes']} content-type={meta['content_type']} date={meta['query_date']}")
    print(f"[HEAD] {meta['content'][:80]!r}")

    raw_df, parse_type = try_parse_excel_or_html(meta["content"])
    print(f"[OK] parsed type={parse_type}, rows={len(raw_df)}, cols={len(raw_df.columns)}")
    print("[COLUMNS]")
    print(raw_df.columns.tolist())

    std, colmap = standardize(raw_df, meta["query_date"])
    summary = build_sigungu_summary(std)

    std.to_csv(STD_PATH, index=False, encoding="utf-8-sig")
    summary.to_csv(SIGUNGU_PATH, index=False, encoding="utf-8-sig")

    status = pd.DataFrame([{
        "collection_date_kst": TODAY,
        "source": "ADMS_SOIL_MOISTURE",
        "main_url": MAIN_URL,
        "excel_url": EXCEL_URL,
        "query_date": meta["query_date"],
        "raw_path": str(RAW_PATH.relative_to(ROOT)),
        "raw_bytes": meta["bytes"],
        "content_type": meta["content_type"],
        "content_disposition": meta["content_disposition"],
        "parse_type": parse_type,
        "raw_rows": len(raw_df),
        "standardized_chungnam_rows": len(std),
        "sigungu_count": summary["sigungu"].nunique() if not summary.empty else 0,
        "date_col": colmap["date_col"],
        "sido_col": colmap["sido_col"],
        "sigungu_col": colmap["sigungu_col"],
        "moisture_col": colmap["moisture_col"],
        "stage_col": colmap["stage_col"],
        "status": "SUCCESS",
    }])

    status.to_csv(STATUS_PATH, index=False, encoding="utf-8-sig")
    append_log(status)

    print()
    print("[Saved]")
    print(f"- {RAW_PATH}")
    print(f"- {STD_PATH} rows={len(std)}")
    print(f"- {SIGUNGU_PATH} rows={len(summary)}")
    print(f"- {STATUS_PATH}")
    print(f"- {LOG_PATH}")

    print()
    print("[Status]")
    print(status.to_string(index=False))

    print()
    print("[Soil Summary]")
    if summary.empty:
        print("No Chungnam soil moisture rows parsed. Check ADMS response columns.")
        print("[Raw Preview]")
        print(raw_df.head(20).to_string(index=False))
    else:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
