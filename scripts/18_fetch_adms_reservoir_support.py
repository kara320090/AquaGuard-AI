from pathlib import Path
from datetime import datetime, timezone, timedelta
import re
import json
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
MAIN_URL = BASE_URL + "/page/myareainfo.do"
SIGUN_URL = BASE_URL + "/common/retrieveSiGunCd.do"
RVOW_SIGUN_URL = BASE_URL + "/droughtInfo/retrieveDroughtMapPaddyClickMyareaSigunTotal.do"

RAW_JSON_PATH = SNAPSHOT_DIR / "adms_reservoir_myarea_sigungu_raw.json"
OUT_PATH = PROCESSED / "latest_adms_reservoir_support_by_sigungu.csv"
CROSSCHECK_PATH = REPORT_TABLES / "latest_reservoir_source_crosscheck.csv"
STATUS_PATH = REPORT_TABLES / "latest_adms_reservoir_support_status.csv"
LOG_PATH = META / "adms_reservoir_support_collection_log.csv"

OLDAM_SIGUNGU_PATH = REPORT_TABLES / "latest_live_reservoir_by_sigungu.csv"

CHUNGNAM_SIDO_CD = "44"


def normalize_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def to_num(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    s = s.replace("%", "").replace(",", "")
    s = re.sub(r"[^0-9.\-]", "", s)
    return pd.to_numeric(s, errors="coerce")


def decode_response(resp):
    for enc in ["utf-8", "cp949", "euc-kr"]:
        try:
            return resp.content.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return resp.text, "response_text"


def extract_hidden_value(html, name_or_id, default=""):
    patterns = [
        rf'id=["\']{re.escape(name_or_id)}["\'][^>]*value=["\']([^"\']*)["\']',
        rf'name=["\']{re.escape(name_or_id)}["\'][^>]*value=["\']([^"\']*)["\']',
    ]
    for pat in patterns:
        m = re.search(pat, html)
        if m:
            return m.group(1).strip()
    return default


def request_json(session, method, url, data=None, headers=None):
    headers = headers or {}
    if method.upper() == "POST":
        r = session.post(url, data=data, headers=headers, timeout=60)
    else:
        r = session.get(url, params=data, headers=headers, timeout=60)

    r.raise_for_status()

    try:
        return r.json(), r
    except Exception:
        text, enc = decode_response(r)
        raise RuntimeError(
            f"JSON parse failed. url={url}, encoding={enc}, head={text[:500]}"
        )


def fetch_sigungu_codes(session, headers):
    payload = {"choiceSiDoCd": CHUNGNAM_SIDO_CD}
    data, resp = request_json(session, "POST", SIGUN_URL, data=payload, headers=headers)

    result = data.get("result", [])
    rows = []

    for item in result:
        code = normalize_text(item.get("code_id", ""))
        name = normalize_text(item.get("code_name", ""))
        if code and name:
            rows.append({"sigun_cd": code, "sigungu": name})

    if not rows:
        raise RuntimeError(f"No Chungnam sigungu codes returned. raw={data}")

    return pd.DataFrame(rows), data


def fetch_sigungu_reservoir(session, headers, sigun_cd, sigungu, yearmonthday, monthday, term):
    payload = {
        "yearMonthDay": yearmonthday,
        "monthDay": monthday,
        "term": term,
        "sigunCd": sigun_cd,
    }

    data, resp = request_json(session, "POST", RVOW_SIGUN_URL, data=payload, headers=headers)

    result = data.get("result", [])
    if not result:
        return {
            "sigungu": sigungu,
            "sigun_cd": sigun_cd,
            "adms_query_date": yearmonthday,
            "adms_monthday": monthday,
            "adms_term": term,
            "adms_rvow": np.nan,
            "adms_normal_rvow": np.nan,
            "adms_normal_ratio": np.nan,
            "adms_sido_name": "",
            "adms_sigun_name": sigungu,
            "adms_sido_cd": CHUNGNAM_SIDO_CD,
            "adms_lat": np.nan,
            "adms_lon": np.nan,
            "adms_status": "NO_RESULT",
            "raw_result_count": 0,
        }, data

    item = result[0]

    return {
        "sigungu": normalize_text(item.get("sigunName", "")) or sigungu,
        "sigun_cd": normalize_text(item.get("sigunCd", "")) or sigun_cd,
        "adms_query_date": yearmonthday,
        "adms_monthday": monthday,
        "adms_term": term,
        "adms_rvow": to_num(item.get("nRvow", np.nan)),
        "adms_normal_rvow": to_num(item.get("nAvow", np.nan)),
        "adms_normal_ratio": to_num(item.get("nRatio", np.nan)),
        "adms_sido_name": normalize_text(item.get("sidoName", "")),
        "adms_sigun_name": normalize_text(item.get("sigunName", "")),
        "adms_sido_cd": normalize_text(item.get("sidoCd", CHUNGNAM_SIDO_CD)),
        "adms_lat": to_num(item.get("lat", np.nan)),
        "adms_lon": to_num(item.get("lon", np.nan)),
        "adms_status": "SUCCESS",
        "raw_result_count": len(result),
    }, data


def build_crosscheck(adms):
    if not OLDAM_SIGUNGU_PATH.exists():
        out = adms.copy()
        out["oldam_available"] = 0
        out["oldam_avg_reservoir_rate"] = np.nan
        out["rvow_diff_oldam_minus_adms"] = np.nan
        out["crosscheck_status"] = "NO_OLDAM_FILE"
        return out

    oldam = pd.read_csv(OLDAM_SIGUNGU_PATH)

    keep = [
        "sigungu",
        "today_reservoir_count",
        "today_avg_reservoir_rate",
        "today_min_reservoir_rate",
        "today_reservoir_risk_score",
    ]
    keep = [c for c in keep if c in oldam.columns]
    oldam = oldam[keep].copy()

    oldam = oldam.rename(columns={
        "today_reservoir_count": "oldam_reservoir_count",
        "today_avg_reservoir_rate": "oldam_avg_reservoir_rate",
        "today_min_reservoir_rate": "oldam_min_reservoir_rate",
        "today_reservoir_risk_score": "oldam_reservoir_risk_score",
    })

    out = adms.merge(oldam, on="sigungu", how="left")

    out["oldam_available"] = out["oldam_avg_reservoir_rate"].notna().astype(int)
    out["rvow_diff_oldam_minus_adms"] = (
        out["oldam_avg_reservoir_rate"] - out["adms_rvow"]
    )

    out["crosscheck_status"] = np.where(
        out["oldam_available"].eq(1),
        "ADMS_AND_OLDAM",
        "ADMS_ONLY",
    )

    return out


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
    print("[AquaGuard AI] Fetch ADMS reservoir support data")
    print(f"main: {MAIN_URL}")

    session = requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0 AquaGuard-AI/1.0",
        "Referer": MAIN_URL,
        "Origin": BASE_URL,
        "X-Requested-With": "XMLHttpRequest",
    }

    main = session.get(MAIN_URL, headers=headers, timeout=60)
    main.raise_for_status()
    html, enc = decode_response(main)

    max_date = extract_hidden_value(html, "maxDate", TODAY)
    term = extract_hidden_value(html, "term", "0")

    yearmonthday = re.sub(r"[^0-9]", "", max_date)
    if len(yearmonthday) != 8:
        yearmonthday = TODAY

    monthday = yearmonthday[4:8]

    print(f"[OK] page loaded encoding={enc}, yearmonthday={yearmonthday}, monthday={monthday}, term={term}")

    sigungu_df, sigungu_raw = fetch_sigungu_codes(session, headers)
    print(f"[OK] sigungu codes: {len(sigungu_df)}")

    rows = []
    raw_payload = {
        "main_url": MAIN_URL,
        "sigungu_raw": sigungu_raw,
        "sigungu_results": {},
    }

    for _, r in sigungu_df.iterrows():
        sigun_cd = str(r["sigun_cd"])
        sigungu = str(r["sigungu"])

        print(f"  - fetch {sigungu} ({sigun_cd})")
        row, raw = fetch_sigungu_reservoir(
            session=session,
            headers=headers,
            sigun_cd=sigun_cd,
            sigungu=sigungu,
            yearmonthday=yearmonthday,
            monthday=monthday,
            term=term,
        )
        rows.append(row)
        raw_payload["sigungu_results"][sigungu] = raw

    RAW_JSON_PATH.write_text(
        json.dumps(raw_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    adms = pd.DataFrame(rows)

    for c in ["adms_rvow", "adms_normal_rvow", "adms_normal_ratio", "adms_lat", "adms_lon"]:
        if c in adms.columns:
            adms[c] = pd.to_numeric(adms[c], errors="coerce")

    # 보조 위험도: 현재 ADMS 저수율이 낮을수록 위험. 산식에는 바로 반영하지 않고 검증용으로 보관.
    adms["adms_reservoir_support_risk_score"] = (100 - adms["adms_rvow"]).clip(0, 100)
    adms["adms_reservoir_support_status"] = "ADMS_RVOW_SUPPORT"

    adms = adms.sort_values("adms_reservoir_support_risk_score", ascending=False).reset_index(drop=True)
    adms["adms_support_rank"] = np.arange(1, len(adms) + 1)

    front = ["adms_support_rank", "sigungu", "adms_query_date"]
    other = [c for c in adms.columns if c not in front]
    adms = adms[front + other]

    cross = build_crosscheck(adms)

    adms.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    cross.to_csv(CROSSCHECK_PATH, index=False, encoding="utf-8-sig")

    status = pd.DataFrame([{
        "collection_date_kst": TODAY,
        "source": "ADMS_MYAREA_RESERVOIR_SUPPORT",
        "main_url": MAIN_URL,
        "sigungu_url": SIGUN_URL,
        "rvow_url": RVOW_SIGUN_URL,
        "query_date": yearmonthday,
        "monthday": monthday,
        "term": term,
        "sigungu_code_count": len(sigungu_df),
        "result_rows": len(adms),
        "success_rows": int((adms["adms_status"] == "SUCCESS").sum()),
        "crosscheck_rows": len(cross),
        "oldam_crosscheck_rows": int((cross.get("oldam_available", pd.Series(dtype=int)) == 1).sum()),
        "raw_json_path": str(RAW_JSON_PATH.relative_to(ROOT)),
        "status": "SUCCESS",
    }])

    status.to_csv(STATUS_PATH, index=False, encoding="utf-8-sig")
    append_log(status)

    print()
    print("[Saved]")
    print(f"- {RAW_JSON_PATH}")
    print(f"- {OUT_PATH} rows={len(adms)}")
    print(f"- {CROSSCHECK_PATH} rows={len(cross)}")
    print(f"- {STATUS_PATH}")
    print(f"- {LOG_PATH}")

    print()
    print("[Status]")
    print(status.to_string(index=False))

    print()
    print("[ADMS Reservoir Support]")
    print(adms.to_string(index=False))

    print()
    print("[Crosscheck]")
    show_cols = [
        "sigungu",
        "adms_rvow",
        "adms_normal_rvow",
        "adms_normal_ratio",
        "oldam_avg_reservoir_rate",
        "rvow_diff_oldam_minus_adms",
        "crosscheck_status",
    ]
    show_cols = [c for c in show_cols if c in cross.columns]
    print(cross[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
