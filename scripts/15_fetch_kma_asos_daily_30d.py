from pathlib import Path
from datetime import datetime, timezone, timedelta
from io import StringIO
import os
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
START_DT = TODAY_DT - timedelta(days=30)

TM1 = START_DT.strftime("%Y%m%d")
TM2 = TODAY_DT.strftime("%Y%m%d")

SNAPSHOT_DIR = RAW_ROOT / TODAY
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)
REPORT_TABLES.mkdir(parents=True, exist_ok=True)
META.mkdir(parents=True, exist_ok=True)

RAW_PATH = SNAPSHOT_DIR / "kma_asos_daily_30d_raw.txt"
STD_PATH = PROCESSED / "latest_kma_asos_daily_30d.csv"
SIGUNGU_PATH = PROCESSED / "latest_weather_30d_by_sigungu.csv"
STATUS_PATH = REPORT_TABLES / "latest_kma_weather_status.csv"
LOG_PATH = META / "kma_weather_collection_log.csv"

KMA_URL = "https://apihub.kma.go.kr/api/typ01/url/kma_sfcdd3.php"

# 1차 MVP용 충남 ASOS 대표 관측소
# ASOS는 15개 시군 전체를 촘촘히 커버하지 않으므로, 이후 AWS로 보완한다.
CHUNGNAM_ASOS_STATIONS = {
    129: {"station_name": "서산", "sigungu": "서산시"},
    177: {"station_name": "홍성", "sigungu": "홍성군"},
    232: {"station_name": "천안", "sigungu": "천안시"},
    235: {"station_name": "보령", "sigungu": "보령시"},
    236: {"station_name": "부여", "sigungu": "부여군"},
    238: {"station_name": "금산", "sigungu": "금산군"},
}

KMA_DAILY_COLUMNS = [
    "TM", "STN",
    "WS_AVG", "WR_DAY", "WD_MAX", "WS_MAX", "WS_MAX_TM",
    "WD_INS", "WS_INS", "WS_INS_TM",
    "TA_AVG", "TA_MAX", "TA_MAX_TM", "TA_MIN", "TA_MIN_TM",
    "TD_AVG", "TS_AVG", "TG_MIN",
    "HM_AVG", "HM_MIN", "HM_MIN_TM",
    "PV_AVG", "EV_S", "EV_L", "FG_DUR",
    "PA_AVG", "PS_AVG", "PS_MAX", "PS_MAX_TM", "PS_MIN", "PS_MIN_TM",
    "CA_TOT", "SS_DAY", "SS_DUR", "SS_CMB", "SI_DAY",
    "SI_60M_MAX", "SI_60M_MAX_TM",
    "RN_DAY", "RN_D99", "RN_DUR",
    "RN_60M_MAX", "RN_60M_MAX_TM",
    "RN_10M_MAX", "RN_10M_MAX_TM",
    "RN_POW_MAX", "RN_POW_MAX_TM",
    "SD_NEW", "SD_NEW_TM", "SD_MAX", "SD_MAX_TM",
    "TE_05", "TE_10", "TE_15", "TE_30", "TE_50",
]


def get_api_key():
    key = os.getenv("KMA_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "KMA_API_KEY 환경변수가 없습니다. PowerShell에서 먼저 실행하세요: "
            "$env:KMA_API_KEY='발급받은키'"
        )
    return key


def download_kma():
    params = {
        "tm1": TM1,
        "tm2": TM2,
        "stn": "0",
        "help": "0",
        "authKey": get_api_key(),
    }

    r = requests.get(KMA_URL, params=params, timeout=90)
    r.raise_for_status()

    RAW_PATH.write_bytes(r.content)

    text = None
    for enc in ["utf-8", "cp949", "euc-kr"]:
        try:
            text = r.content.decode(enc)
            break
        except UnicodeDecodeError:
            continue

    if text is None:
        text = r.text

    if "ERROR" in text.upper() or "인증" in text[:500]:
        print(text[:1000])
        raise RuntimeError("KMA API response looks like an error. Check auth key or parameters.")

    return text, len(r.content), r.url


def parse_kma_text(text):
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            continue
        if s.startswith("7777") or s.endswith("7777"):
            continue
        lines.append(s)

    if not lines:
        raise RuntimeError("No data lines parsed from KMA response.")

    df = pd.read_csv(
        StringIO("\n".join(lines)),
        sep=r"\s+",
        header=None,
        engine="python",
    )

    if df.shape[1] > len(KMA_DAILY_COLUMNS):
        df = df.iloc[:, :len(KMA_DAILY_COLUMNS)].copy()

    df.columns = KMA_DAILY_COLUMNS[:df.shape[1]]

    required = ["TM", "STN", "RN_DAY"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Required columns missing after parse: {missing}, columns={df.columns.tolist()}")

    df["date"] = pd.to_datetime(df["TM"].astype(str), errors="coerce")
    df["station_id"] = pd.to_numeric(df["STN"], errors="coerce").astype("Int64")

    for c in ["RN_DAY", "TA_AVG", "HM_AVG", "WS_AVG"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df.loc[df[c] <= -90, c] = np.nan

    # 강수량 결측은 우선 0으로 처리. 이후 품질검증에서 결측률 확인.
    df["rainfall"] = df["RN_DAY"].fillna(0).clip(lower=0)

    if "TA_AVG" in df.columns:
        df["avg_temperature"] = df["TA_AVG"]
    else:
        df["avg_temperature"] = np.nan

    if "HM_AVG" in df.columns:
        df["avg_humidity"] = df["HM_AVG"]
    else:
        df["avg_humidity"] = np.nan

    if "WS_AVG" in df.columns:
        df["avg_wind_speed"] = df["WS_AVG"]
    else:
        df["avg_wind_speed"] = np.nan

    return df


def filter_chungnam_asos(df):
    out = df[df["station_id"].isin(CHUNGNAM_ASOS_STATIONS.keys())].copy()

    out["station_name"] = out["station_id"].map(
        {k: v["station_name"] for k, v in CHUNGNAM_ASOS_STATIONS.items()}
    )
    out["sigungu"] = out["station_id"].map(
        {k: v["sigungu"] for k, v in CHUNGNAM_ASOS_STATIONS.items()}
    )

    keep = [
        "date",
        "station_id",
        "station_name",
        "sigungu",
        "rainfall",
        "avg_temperature",
        "avg_humidity",
        "avg_wind_speed",
    ]

    out = out[keep].sort_values(["sigungu", "date"]).reset_index(drop=True)

    return out


def minmax_inverse_score(s):
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(50.0, index=s.index)

    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(50.0, index=s.index)

    # 강수량이 적을수록 위험도 높음
    return ((mx - s) / (mx - mn) * 100).clip(0, 100)


def build_sigungu_summary(df):
    if df.empty:
        return pd.DataFrame()

    end_date = df["date"].max()
    start_7d = end_date - pd.Timedelta(days=6)

    g30 = df.groupby("sigungu").agg(
        weather_start_date=("date", "min"),
        weather_end_date=("date", "max"),
        station_count=("station_id", "nunique"),
        rainfall_30d=("rainfall", "sum"),
        rain_days_30d=("rainfall", lambda s: int((s > 0).sum())),
        avg_temperature_30d=("avg_temperature", "mean"),
        avg_humidity_30d=("avg_humidity", "mean"),
        avg_wind_speed_30d=("avg_wind_speed", "mean"),
    ).reset_index()

    g7 = df[df["date"] >= start_7d].groupby("sigungu").agg(
        rainfall_7d=("rainfall", "sum"),
        rain_days_7d=("rainfall", lambda s: int((s > 0).sum())),
    ).reset_index()

    out = g30.merge(g7, on="sigungu", how="left")
    out["rainfall_7d"] = out["rainfall_7d"].fillna(0)
    out["rain_days_7d"] = out["rain_days_7d"].fillna(0).astype(int)

    out["latest_rain_shortage_score"] = minmax_inverse_score(out["rainfall_30d"])

    out["weather_data_status"] = "ASOS_AUTO"
    out = out.sort_values("latest_rain_shortage_score", ascending=False).reset_index(drop=True)
    out["rain_shortage_rank"] = np.arange(1, len(out) + 1)

    front = ["rain_shortage_rank", "sigungu"]
    other = [c for c in out.columns if c not in front]
    return out[front + other]


def write_status(raw_rows, filtered_rows, sigungu_count, raw_bytes, request_url):
    status = pd.DataFrame([{
        "collection_date_kst": TODAY,
        "tm1": TM1,
        "tm2": TM2,
        "source": "KMA_APIHUB_ASOS_DAILY_PERIOD",
        "request_url_without_key": request_url.split("authKey=")[0] + "authKey=***",
        "raw_path": str(RAW_PATH.relative_to(ROOT)),
        "raw_bytes": raw_bytes,
        "raw_rows": raw_rows,
        "filtered_chungnam_asos_rows": filtered_rows,
        "sigungu_count": sigungu_count,
        "status": "SUCCESS",
    }])

    status.to_csv(STATUS_PATH, index=False, encoding="utf-8-sig")

    if LOG_PATH.exists():
        old = pd.read_csv(LOG_PATH)
        out = pd.concat([old, status], ignore_index=True)
    else:
        out = status

    out.to_csv(LOG_PATH, index=False, encoding="utf-8-sig")

    return status


def main():
    print("[AquaGuard AI] Fetch KMA ASOS daily 30d")
    print(f"period: {TM1} ~ {TM2}")

    text, raw_bytes, request_url = download_kma()
    parsed = parse_kma_text(text)
    filtered = filter_chungnam_asos(parsed)
    summary = build_sigungu_summary(filtered)

    filtered.to_csv(STD_PATH, index=False, encoding="utf-8-sig")
    summary.to_csv(SIGUNGU_PATH, index=False, encoding="utf-8-sig")
    status = write_status(len(parsed), len(filtered), summary["sigungu"].nunique() if not summary.empty else 0, raw_bytes, request_url)

    print()
    print("[Saved]")
    print(f"- {RAW_PATH}")
    print(f"- {STD_PATH} rows={len(filtered)}")
    print(f"- {SIGUNGU_PATH} rows={len(summary)}")
    print(f"- {STATUS_PATH}")
    print(f"- {LOG_PATH}")

    print()
    print("[Status]")
    print(status.to_string(index=False))

    print()
    print("[Weather Summary]")
    if summary.empty:
        print("No Chungnam ASOS data found.")
    else:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
