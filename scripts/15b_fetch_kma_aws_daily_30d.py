from pathlib import Path
from datetime import datetime, timezone, timedelta
from io import StringIO
import os
import math
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

KMA_AWS_URL = "https://apihub.kma.go.kr/api/typ01/url/sfc_aws_day.php"

RAW_PATH = SNAPSHOT_DIR / "kma_aws_daily_rainfall_30d_raw.txt"
AWS_DAILY_PATH = PROCESSED / "latest_kma_aws_daily_30d.csv"
AWS_SIGUNGU_PATH = PROCESSED / "latest_kma_aws_weather_30d_by_sigungu.csv"
ASOS_BACKUP_PATH = PROCESSED / "latest_kma_asos_weather_30d_by_sigungu.csv"
COMBINED_WEATHER_PATH = PROCESSED / "latest_weather_30d_by_sigungu.csv"
STATUS_PATH = REPORT_TABLES / "latest_kma_aws_weather_status.csv"
LOG_PATH = META / "kma_aws_weather_collection_log.csv"

# 충남 15개 시군 대표 좌표. AWS 관측소를 가장 가까운 시군으로 배정하기 위한 MVP용 매핑.
CHUNGNAM_CENTROIDS = {
    "천안시": (36.8151, 127.1139),
    "공주시": (36.4465, 127.1190),
    "보령시": (36.3334, 126.6128),
    "아산시": (36.7898, 127.0026),
    "서산시": (36.7848, 126.4503),
    "논산시": (36.1871, 127.0987),
    "계룡시": (36.2746, 127.2486),
    "당진시": (36.8939, 126.6283),
    "금산군": (36.1088, 127.4882),
    "부여군": (36.2754, 126.9098),
    "서천군": (36.0803, 126.6919),
    "청양군": (36.4592, 126.8023),
    "홍성군": (36.6013, 126.6608),
    "예산군": (36.6826, 126.8487),
    "태안군": (36.7456, 126.2979),
}

# 충남 주변 bounding box. 너무 엄격하면 경계 관측소가 빠지고, 너무 넓으면 타시도 관측소가 섞인다.
LAT_MIN, LAT_MAX = 35.90, 37.20
LON_MIN, LON_MAX = 125.80, 127.80


def get_api_key():
    key = os.getenv("KMA_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "KMA_API_KEY 환경변수가 없습니다. PowerShell에서 먼저 실행하세요:\n"
            "$env:KMA_API_KEY='발급받은키'"
        )
    return key


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def nearest_sigungu(lat, lon):
    if pd.isna(lat) or pd.isna(lon):
        return "", np.nan

    best_name = ""
    best_dist = float("inf")

    for name, (clat, clon) in CHUNGNAM_CENTROIDS.items():
        d = haversine_km(float(lat), float(lon), clat, clon)
        if d < best_dist:
            best_name = name
            best_dist = d

    return best_name, best_dist


def download_aws():
    params = {
        "tm1": TM1,
        "tm2": TM2,
        "obs": "rn_day",
        "stn": "0",
        "disp": "0",
        "help": "0",
        "authKey": get_api_key(),
    }

    r = requests.get(KMA_AWS_URL, params=params, timeout=120)
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

    if "ERROR" in text.upper() or "인증" in text[:1000]:
        print(text[:1500])
        raise RuntimeError("KMA AWS API response looks like an error. Check KMA_API_KEY or parameters.")

    return text, len(r.content), r.url


def parse_aws_text(text):
    """
    KMA AWS sfc_aws_day.php 응답은 help/지점명/공백 형태에 따라
    행별 필드 수가 달라질 수 있다.
    따라서 pandas read_csv로 한 번에 읽지 않고,
    각 행에서 날짜, 지점번호, 숫자값(LON, LAT, HT, VAL)을 직접 추출한다.
    """
    records = []
    bad_lines = []

    for line_no, line in enumerate(text.splitlines(), start=1):
        s = line.strip()

        if not s:
            continue
        if s.startswith("#"):
            continue
        if s.startswith("7777") or s.endswith("7777"):
            continue

        parts = s.split()

        if len(parts) < 2:
            bad_lines.append((line_no, s, "too_few_parts"))
            continue

        tm = parts[0].strip()
        if not (len(tm) == 8 and tm.isdigit()):
            bad_lines.append((line_no, s, "not_data_line"))
            continue

        try:
            stn = int(float(parts[1]))
        except Exception:
            bad_lines.append((line_no, s, "bad_station"))
            continue

        nums = []
        for p in parts[2:]:
            try:
                nums.append(float(p))
            except Exception:
                # 지점명 같은 문자열은 건너뜀
                continue

        if len(nums) < 4:
            bad_lines.append((line_no, s, "not_enough_numeric_values"))
            continue

        lon, lat, ht, val = nums[0], nums[1], nums[2], nums[3]

        records.append({
            "TM": tm,
            "STN": stn,
            "LON": lon,
            "LAT": lat,
            "HT": ht,
            "VAL": val,
        })

    if not records:
        sample = "\n".join([f"{n}: {line} [{reason}]" for n, line, reason in bad_lines[:20]])
        raise RuntimeError(f"No valid AWS data rows parsed.\nSample bad lines:\n{sample}")

    df = pd.DataFrame(records)

    df["date"] = pd.to_datetime(df["TM"].astype(str), format="%Y%m%d", errors="coerce")
    df["station_id"] = pd.to_numeric(df["STN"], errors="coerce").astype("Int64")
    df["lon"] = pd.to_numeric(df["LON"], errors="coerce")
    df["lat"] = pd.to_numeric(df["LAT"], errors="coerce")
    df["height"] = pd.to_numeric(df["HT"], errors="coerce")
    df["rainfall"] = pd.to_numeric(df["VAL"], errors="coerce")

    df.loc[df["rainfall"] <= -90, "rainfall"] = np.nan
    df["rainfall"] = df["rainfall"].fillna(0).clip(lower=0)

    df = df[df["date"].notna()].copy()

    if bad_lines:
        print(f"[WARN] skipped non-data/bad AWS lines: {len(bad_lines)}")
        for n, line, reason in bad_lines[:10]:
            print(f"  line={n} reason={reason} raw={line[:120]}")

    return df


def filter_assign_chungnam(df):
    bbox = df[
        df["lat"].between(LAT_MIN, LAT_MAX)
        & df["lon"].between(LON_MIN, LON_MAX)
    ].copy()

    assigned = []
    distances = []

    for lat, lon in zip(bbox["lat"], bbox["lon"]):
        name, dist = nearest_sigungu(lat, lon)
        assigned.append(name)
        distances.append(dist)

    bbox["sigungu"] = assigned
    bbox["nearest_sigungu_distance_km"] = distances

    # 너무 먼 관측소는 제외. 35km 이상이면 충남 외곽 관측소가 잘못 들어왔을 가능성이 있음.
    bbox = bbox[bbox["nearest_sigungu_distance_km"] <= 35].copy()

    keep = [
        "date",
        "station_id",
        "lat",
        "lon",
        "height",
        "sigungu",
        "nearest_sigungu_distance_km",
        "rainfall",
    ]

    bbox = bbox[keep].sort_values(["sigungu", "station_id", "date"]).reset_index(drop=True)
    return bbox


def minmax_inverse_score(s):
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(50.0, index=s.index)

    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(50.0, index=s.index)

    return ((mx - s) / (mx - mn) * 100).clip(0, 100)


def build_sigungu_summary(daily):
    if daily.empty:
        return pd.DataFrame()

    end_date = daily["date"].max()
    start_7d = end_date - pd.Timedelta(days=6)

    station_30d = daily.groupby(["sigungu", "station_id"], dropna=False).agg(
        station_lat=("lat", "first"),
        station_lon=("lon", "first"),
        weather_start_date=("date", "min"),
        weather_end_date=("date", "max"),
        rainfall_30d=("rainfall", "sum"),
        rain_days_30d=("rainfall", lambda s: int((s > 0).sum())),
        nearest_sigungu_distance_km=("nearest_sigungu_distance_km", "first"),
    ).reset_index()

    station_7d = daily[daily["date"] >= start_7d].groupby(["sigungu", "station_id"], dropna=False).agg(
        rainfall_7d=("rainfall", "sum"),
        rain_days_7d=("rainfall", lambda s: int((s > 0).sum())),
    ).reset_index()

    station = station_30d.merge(station_7d, on=["sigungu", "station_id"], how="left")
    station["rainfall_7d"] = station["rainfall_7d"].fillna(0)
    station["rain_days_7d"] = station["rain_days_7d"].fillna(0).astype(int)

    # 시군 내 관측소가 여러 개면 합계가 아니라 평균을 사용한다.
    # 이유: 관측소 수가 많은 시군이 강수량 과대평가되는 것을 방지.
    sigungu = station.groupby("sigungu").agg(
        weather_start_date=("weather_start_date", "min"),
        weather_end_date=("weather_end_date", "max"),
        station_count=("station_id", "nunique"),
        rainfall_30d=("rainfall_30d", "mean"),
        rainfall_7d=("rainfall_7d", "mean"),
        rain_days_30d=("rain_days_30d", "mean"),
        rain_days_7d=("rain_days_7d", "mean"),
        mean_station_distance_km=("nearest_sigungu_distance_km", "mean"),
    ).reset_index()

    sigungu["rain_days_30d"] = sigungu["rain_days_30d"].round(1)
    sigungu["rain_days_7d"] = sigungu["rain_days_7d"].round(1)

    sigungu["avg_temperature_30d"] = np.nan
    sigungu["avg_humidity_30d"] = np.nan
    sigungu["avg_wind_speed_30d"] = np.nan

    sigungu["latest_rain_shortage_score"] = minmax_inverse_score(sigungu["rainfall_30d"])
    sigungu["weather_data_status"] = "AWS_AUTO"

    sigungu = sigungu.sort_values("latest_rain_shortage_score", ascending=False).reset_index(drop=True)
    sigungu["rain_shortage_rank"] = np.arange(1, len(sigungu) + 1)

    front = ["rain_shortage_rank", "sigungu"]
    other = [c for c in sigungu.columns if c not in front]
    return sigungu[front + other], station


def preserve_asos_summary():
    if not COMBINED_WEATHER_PATH.exists():
        return pd.DataFrame()

    try:
        old = pd.read_csv(COMBINED_WEATHER_PATH)
    except Exception:
        return pd.DataFrame()

    if "weather_data_status" in old.columns and old["weather_data_status"].astype(str).str.contains("ASOS").any():
        old.to_csv(ASOS_BACKUP_PATH, index=False, encoding="utf-8-sig")
        return old

    if ASOS_BACKUP_PATH.exists():
        return pd.read_csv(ASOS_BACKUP_PATH)

    return pd.DataFrame()


def combine_weather(asos, aws):
    if aws.empty and asos.empty:
        return pd.DataFrame()

    if asos.empty:
        return aws.copy()

    if aws.empty:
        return asos.copy()

    # AWS가 있는 시군은 AWS를 우선 사용하고, 없는 시군만 ASOS fallback.
    aws_sigungu = set(aws["sigungu"].astype(str))
    asos_only = asos[~asos["sigungu"].astype(str).isin(aws_sigungu)].copy()

    combined = pd.concat([aws, asos_only], ignore_index=True, sort=False)
    combined = combined.sort_values("latest_rain_shortage_score", ascending=False).reset_index(drop=True)
    combined["rain_shortage_rank"] = np.arange(1, len(combined) + 1)

    return combined


def write_status(raw_rows, chungnam_rows, station_count, sigungu_count, raw_bytes, request_url):
    status = pd.DataFrame([{
        "collection_date_kst": TODAY,
        "tm1": TM1,
        "tm2": TM2,
        "source": "KMA_APIHUB_AWS_DAILY_RAINFALL_PERIOD",
        "request_url_without_key": request_url.split("authKey=")[0] + "authKey=***",
        "raw_path": str(RAW_PATH.relative_to(ROOT)),
        "raw_bytes": raw_bytes,
        "raw_rows": raw_rows,
        "filtered_chungnam_aws_rows": chungnam_rows,
        "station_count": station_count,
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
    print("[AquaGuard AI] Fetch KMA AWS daily rainfall 30d")
    print(f"period: {TM1} ~ {TM2}")

    text, raw_bytes, request_url = download_aws()
    parsed = parse_aws_text(text)
    daily = filter_assign_chungnam(parsed)
    aws_summary, station_summary = build_sigungu_summary(daily)

    asos_summary = preserve_asos_summary()
    combined = combine_weather(asos_summary, aws_summary)

    daily.to_csv(AWS_DAILY_PATH, index=False, encoding="utf-8-sig")
    aws_summary.to_csv(AWS_SIGUNGU_PATH, index=False, encoding="utf-8-sig")
    combined.to_csv(COMBINED_WEATHER_PATH, index=False, encoding="utf-8-sig")

    status = write_status(
        raw_rows=len(parsed),
        chungnam_rows=len(daily),
        station_count=daily["station_id"].nunique() if len(daily) else 0,
        sigungu_count=aws_summary["sigungu"].nunique() if not aws_summary.empty else 0,
        raw_bytes=raw_bytes,
        request_url=request_url,
    )

    print()
    print("[Saved]")
    print(f"- {RAW_PATH}")
    print(f"- {AWS_DAILY_PATH} rows={len(daily)}")
    print(f"- {AWS_SIGUNGU_PATH} rows={len(aws_summary)}")
    print(f"- {COMBINED_WEATHER_PATH} rows={len(combined)}")
    print(f"- {STATUS_PATH}")
    print(f"- {LOG_PATH}")

    print()
    print("[Status]")
    print(status.to_string(index=False))

    print()
    print("[AWS Weather Summary]")
    if aws_summary.empty:
        print("No Chungnam AWS data found.")
    else:
        print(aws_summary.to_string(index=False))

    print()
    print("[Combined Weather Summary]")
    if combined.empty:
        print("No combined weather data.")
    else:
        print(combined.to_string(index=False))


if __name__ == "__main__":
    main()
