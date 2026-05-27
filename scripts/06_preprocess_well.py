from pathlib import Path
import zipfile
import re
import pandas as pd
import numpy as np
from dbfread import DBF

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "02_well"
OUT_DIR = ROOT / "data" / "processed"
INTERIM_DIR = ROOT / "data" / "interim" / "well_yearly"
EXTRACT_DIR = ROOT / "data" / "interim" / "well_extract"

OUT_DIR.mkdir(parents=True, exist_ok=True)
INTERIM_DIR.mkdir(parents=True, exist_ok=True)
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

SIGUNGU_LIST = [
    "천안시", "공주시", "보령시", "아산시", "서산시", "논산시", "계룡시", "당진시",
    "금산군", "부여군", "서천군", "청양군", "홍성군", "예산군", "태안군"
]

PNU_SIGUNGU_MAP = {
    "44131": "천안시",
    "44133": "천안시",
    "44150": "공주시",
    "44180": "보령시",
    "44200": "아산시",
    "44210": "서산시",
    "44230": "논산시",
    "44250": "계룡시",
    "44270": "당진시",
    "44710": "금산군",
    "44760": "부여군",
    "44770": "서천군",
    "44790": "청양군",
    "44800": "홍성군",
    "44810": "예산군",
    "44825": "태안군",
}

ENCODINGS = ["cp949", "euc-kr", "utf-8", "utf-8-sig"]


def clean_number(x):
    if x is None or pd.isna(x):
        return np.nan
    if isinstance(x, str):
        x = (
            x.replace(",", "")
             .replace("%", "")
             .replace(" ", "")
             .replace("㎥", "")
             .replace("m3", "")
             .replace("톤", "")
             .replace("일", "")
             .strip()
        )
        if x in ["", "-", "X", "nan", "None", "NULL", "…"]:
            return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan


def normalize_sigungu(x):
    if x is None or pd.isna(x):
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


def sigungu_from_pnu(x):
    if x is None or pd.isna(x):
        return np.nan

    s = str(x)
    m = re.search(r"\d{10,}", s)
    if not m:
        return np.nan

    code5 = m.group(0)[:5]
    return PNU_SIGUNGU_MAP.get(code5, np.nan)


def minmax_0_100(s):
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(np.nan, index=s.index)

    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(50.0, index=s.index)

    return ((s - mn) / (mx - mn) * 100).clip(0, 100)


def parse_year_from_name(name):
    dates = re.findall(r"20\d{6}", name)
    if dates:
        return int(max(dates)[:4])

    years = re.findall(r"20\d{2}", name)
    if years:
        return max(int(y) for y in years)

    return np.nan


def parse_snapshot_score(name):
    dates = re.findall(r"20\d{6}", name)
    if dates:
        return max(int(d) for d in dates)

    years = re.findall(r"20\d{2}", name)
    if years:
        return max(int(y) for y in years) * 10000 + 101

    return 0


def source_type_from_name(name):
    low = name.lower()

    if "지하수" in name or "well_info" in low:
        return "groundwater_well"

    if "시추" in name or "plan_result" in low or "agri_plan" in low:
        return "drilling_developed_well"

    return "unknown_well"


def safe_stem(name):
    return re.sub(r"[^0-9A-Za-z가-힣_]+", "_", Path(name).stem)


def extract_zip(zip_path):
    target_dir = EXTRACT_DIR / safe_stem(zip_path.name)
    target_dir.mkdir(parents=True, exist_ok=True)

    if list(target_dir.rglob("*.dbf")):
        return target_dir

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(target_dir)

    return target_dir


def read_dbf(dbf_path):
    last_error = None

    for enc in ENCODINGS:
        try:
            table = DBF(
                str(dbf_path),
                encoding=enc,
                char_decode_errors="ignore",
                ignore_missing_memofile=True,
                load=False
            )
            return table, enc
        except Exception as e:
            last_error = e

    raise RuntimeError(f"DBF read failed: {dbf_path.name} / {last_error}")


def init_accumulator():
    return {
        sg: {
            "sigungu": sg,
            "well_count": 0,
            "groundwater_well_count": 0,
            "drilling_developed_well_count": 0,
            "unknown_well_count": 0,
            "total_pump_capacity": 0.0,
            "pump_capacity_valid_count": 0,
            "well_depth_sum": 0.0,
            "well_depth_valid_count": 0,
            "public_like_well_count": 0,
            "record_used_count": 0,
        }
        for sg in SIGUNGU_LIST
    }


def get_sigungu_by_schema(row, source_type):
    if source_type == "groundwater_well":
        sido = str(row.get("si_do", "")).strip()
        if sido != "충청남도":
            return np.nan

        sg = normalize_sigungu(row.get("si_gun_gu"))
        if pd.notna(sg):
            return sg

        return sigungu_from_pnu(row.get("pnu"))

    if source_type == "drilling_developed_well":
        pnu = row.get("pnu", row.get("PNU"))
        return sigungu_from_pnu(pnu)

    sg = normalize_sigungu(row.get("si_gun_gu"))
    if pd.notna(sg):
        return sg

    return sigungu_from_pnu(row.get("pnu", row.get("PNU")))


def process_dbf(dbf_path, source_zip, acc, sample_rows, sample_limit=1000):
    source_type = source_type_from_name(source_zip)
    table, enc = read_dbf(dbf_path)

    total_rows = 0
    used_rows = 0
    last_used_row = None

    EARLY_STOP_AFTER_NO_CHUNGNAM_ROWS = 10000
    CHECKPOINT_EVERY_ROWS = 50000

    checkpoint_summary_path = INTERIM_DIR / f"checkpoint_{safe_stem(source_zip)}_partial_by_sigungu.csv"

    def save_partial_checkpoint():
        partial = pd.DataFrame(list(acc.values()))
        partial["source_zip"] = source_zip
        partial["source_type"] = source_type
        partial["checkpoint_total_rows"] = total_rows
        partial["checkpoint_used_chungnam"] = used_rows
        partial.to_csv(checkpoint_summary_path, index=False, encoding="utf-8-sig")

    for row in table:
        total_rows += 1
        row = dict(row)

        sigungu = get_sigungu_by_schema(row, source_type)

        if pd.isna(sigungu):
            if total_rows % CHECKPOINT_EVERY_ROWS == 0:
                print(f"  progress rows={total_rows:,} used_chungnam={used_rows:,}", flush=True)
                save_partial_checkpoint()

            if (
                source_type == "groundwater_well"
                and used_rows > 0
                and last_used_row is not None
                and total_rows - last_used_row >= EARLY_STOP_AFTER_NO_CHUNGNAM_ROWS
            ):
                print(
                    f"  [EARLY STOP] no additional Chungnam rows for "
                    f"{EARLY_STOP_AFTER_NO_CHUNGNAM_ROWS:,} rows. "
                    f"stop at total={total_rows:,}, used_chungnam={used_rows:,}",
                    flush=True
                )
                save_partial_checkpoint()
                break

            continue

        used_rows += 1
        last_used_row = total_rows

        a = acc[sigungu]
        a["well_count"] += 1
        a["record_used_count"] += 1

        if source_type == "groundwater_well":
            a["groundwater_well_count"] += 1
        elif source_type == "drilling_developed_well":
            a["drilling_developed_well_count"] += 1
        else:
            a["unknown_well_count"] += 1

        pump = clean_number(row.get("pump_abili"))
        if pd.notna(pump):
            a["total_pump_capacity"] += pump
            a["pump_capacity_valid_count"] += 1

        depth = clean_number(row.get("well_depth"))
        if pd.notna(depth):
            a["well_depth_sum"] += depth
            a["well_depth_valid_count"] += 1

        public_value = str(row.get("public_pri", ""))
        if public_value in ["01", "1"] or re.search(r"공공|공용|public|국가|지자체", public_value, flags=re.IGNORECASE):
            a["public_like_well_count"] += 1

        if len(sample_rows) < sample_limit:
            sample_rows.append({
                "sigungu": sigungu,
                "source_zip": source_zip,
                "source_type": source_type,
                "source_dbf": dbf_path.name,
                "well_id": row.get("well_id"),
                "pnu": row.get("pnu", row.get("PNU")),
                "si_do": row.get("si_do"),
                "si_gun_gu": row.get("si_gun_gu"),
                "address": row.get("address"),
                "well_use": row.get("well_use"),
                "well_depth": depth,
                "pump_capacity": pump,
                "public_private": row.get("public_pri"),
                "dev_date": row.get("dev_date"),
            })

        if total_rows % CHECKPOINT_EVERY_ROWS == 0:
            print(f"  progress rows={total_rows:,} used_chungnam={used_rows:,}", flush=True)
            save_partial_checkpoint()

    save_partial_checkpoint()

    print(
        f"[OK] {source_zip} / {dbf_path.name} rows={total_rows} "
        f"used_chungnam={used_rows} encoding={enc}",
        flush=True
    )

def finalize_summary(acc, source_zip, source_type, year, snapshot_score):
    df = pd.DataFrame(list(acc.values()))

    df["source_zip"] = source_zip
    df["source_type"] = source_type
    df["year"] = year
    df["snapshot_score"] = snapshot_score

    df["avg_pump_capacity"] = np.where(
        df["pump_capacity_valid_count"] > 0,
        df["total_pump_capacity"] / df["pump_capacity_valid_count"],
        np.nan
    )

    df["avg_well_depth"] = np.where(
        df["well_depth_valid_count"] > 0,
        df["well_depth_sum"] / df["well_depth_valid_count"],
        np.nan
    )

    df["well_count_index"] = minmax_0_100(df["well_count"])
    df["pump_capacity_index"] = minmax_0_100(df["total_pump_capacity"])

    df["well_support_score"] = (
        df["well_count_index"].fillna(0) * 0.60
        + df["pump_capacity_index"].fillna(0) * 0.40
    )

    df["well_shortage_score"] = 100 - df["well_support_score"]

    df["well_support_level"] = pd.cut(
        df["well_support_score"],
        bins=[-1, 39, 59, 79, 100],
        labels=["낮음", "보통", "높음", "매우높음"]
    ).astype(str)

    return df


def process_one_zip(zip_path):
    source_zip = zip_path.name
    source_type = source_type_from_name(source_zip)
    year = parse_year_from_name(source_zip)
    snapshot_score = parse_snapshot_score(source_zip)

    summary_path = INTERIM_DIR / f"well_{safe_stem(source_zip)}_by_sigungu.csv"
    sample_path = INTERIM_DIR / f"well_{safe_stem(source_zip)}_sample.csv"

    if summary_path.exists():
        print()
        print(f"[SKIP EXISTING] {summary_path.name}")
        return

    print()
    print("=" * 80)
    print(f"[PROCESS ZIP] {source_zip} | type={source_type} | year={year} | snapshot={snapshot_score}")

    try:
        extract_dir = extract_zip(zip_path)
        dbf_files = list(extract_dir.rglob("*.dbf"))

        if not dbf_files:
            print(f"[WARN] no DBF files in {source_zip}")
            return

        acc = init_accumulator()
        sample_rows = []

        for dbf_path in dbf_files:
            process_dbf(
                dbf_path=dbf_path,
                source_zip=source_zip,
                acc=acc,
                sample_rows=sample_rows,
                sample_limit=1000
            )

        summary = finalize_summary(
            acc=acc,
            source_zip=source_zip,
            source_type=source_type,
            year=year,
            snapshot_score=snapshot_score
        )

        sample = pd.DataFrame(sample_rows)
        if not sample.empty:
            sample["source_zip"] = source_zip
            sample["source_type"] = source_type
            sample["year"] = year
            sample["snapshot_score"] = snapshot_score

        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        sample.to_csv(sample_path, index=False, encoding="utf-8-sig")

        print(f"[SAVED YEARLY] {summary_path}")
        print(f"[SAVED SAMPLE] {sample_path}")

    except Exception as e:
        print(f"[ERROR] zip failed but continue: {source_zip} / {e}")


def load_yearly_summaries():
    files = sorted(INTERIM_DIR.glob("well_*_by_sigungu.csv"))
    parts = []

    for p in files:
        try:
            parts.append(pd.read_csv(p))
        except Exception as e:
            print(f"[WARN] cannot read {p.name}: {e}")

    if not parts:
        return pd.DataFrame()

    return pd.concat(parts, ignore_index=True)


def load_samples():
    files = sorted(INTERIM_DIR.glob("well_*_sample.csv"))
    parts = []

    for p in files:
        try:
            parts.append(pd.read_csv(p))
        except Exception:
            pass

    if not parts:
        return pd.DataFrame()

    return pd.concat(parts, ignore_index=True)


def build_latest_snapshot(yearly):
    if yearly.empty:
        return pd.DataFrame({"sigungu": SIGUNGU_LIST})

    latest_parts = []

    for source_type, part in yearly.groupby("source_type"):
        part = part.copy()
        part["snapshot_score"] = pd.to_numeric(part["snapshot_score"], errors="coerce").fillna(0)
        max_score = part["snapshot_score"].max()
        latest_parts.append(part[part["snapshot_score"] == max_score])

    latest_all = pd.concat(latest_parts, ignore_index=True)

    summary = latest_all.groupby("sigungu", as_index=False).agg(
        well_count=("well_count", "sum"),
        groundwater_well_count=("groundwater_well_count", "sum"),
        drilling_developed_well_count=("drilling_developed_well_count", "sum"),
        unknown_well_count=("unknown_well_count", "sum"),
        total_pump_capacity=("total_pump_capacity", "sum"),
        pump_capacity_valid_count=("pump_capacity_valid_count", "sum"),
        well_depth_sum=("well_depth_sum", "sum"),
        well_depth_valid_count=("well_depth_valid_count", "sum"),
        public_like_well_count=("public_like_well_count", "sum"),
        record_used_count=("record_used_count", "sum"),
    )

    base = pd.DataFrame({"sigungu": SIGUNGU_LIST})
    summary = base.merge(summary, on="sigungu", how="left").fillna(0)

    summary["avg_pump_capacity"] = np.where(
        summary["pump_capacity_valid_count"] > 0,
        summary["total_pump_capacity"] / summary["pump_capacity_valid_count"],
        np.nan
    )

    summary["avg_well_depth"] = np.where(
        summary["well_depth_valid_count"] > 0,
        summary["well_depth_sum"] / summary["well_depth_valid_count"],
        np.nan
    )

    summary["well_count_index"] = minmax_0_100(summary["well_count"])
    summary["pump_capacity_index"] = minmax_0_100(summary["total_pump_capacity"])

    summary["well_support_score"] = (
        summary["well_count_index"].fillna(0) * 0.60
        + summary["pump_capacity_index"].fillna(0) * 0.40
    )

    summary["well_shortage_score"] = 100 - summary["well_support_score"]

    summary["well_support_level"] = pd.cut(
        summary["well_support_score"],
        bins=[-1, 39, 59, 79, 100],
        labels=["낮음", "보통", "높음", "매우높음"]
    ).astype(str)

    return summary.sort_values("well_support_score", ascending=False).reset_index(drop=True)


def build_trend(yearly):
    if yearly.empty:
        return pd.DataFrame({"sigungu": SIGUNGU_LIST})

    valid = yearly[pd.notna(yearly["year"])].copy()
    valid["year"] = pd.to_numeric(valid["year"], errors="coerce")
    valid["snapshot_score"] = pd.to_numeric(valid["snapshot_score"], errors="coerce").fillna(0)

    latest_per_type_year = []

    for (source_type, year), part in valid.groupby(["source_type", "year"]):
        max_score = part["snapshot_score"].max()
        latest_per_type_year.append(part[part["snapshot_score"] == max_score])

    valid = pd.concat(latest_per_type_year, ignore_index=True)

    by_year = valid.groupby(["sigungu", "year"], as_index=False).agg(
        yearly_well_count=("well_count", "sum"),
        yearly_pump_capacity=("total_pump_capacity", "sum")
    )

    rows = []

    for sg, part in by_year.groupby("sigungu"):
        part = part.sort_values("year")
        first = part.iloc[0]
        last = part.iloc[-1]

        first_count = first["yearly_well_count"]
        last_count = last["yearly_well_count"]

        growth_rate = np.nan
        if first_count and first_count > 0:
            growth_rate = (last_count - first_count) / first_count * 100

        rows.append({
            "sigungu": sg,
            "well_trend_first_year": int(first["year"]),
            "well_trend_latest_year": int(last["year"]),
            "well_count_first": first_count,
            "well_count_latest": last_count,
            "well_count_growth_rate": growth_rate,
            "pump_capacity_first": first["yearly_pump_capacity"],
            "pump_capacity_latest": last["yearly_pump_capacity"],
        })

    trend = pd.DataFrame(rows)
    base = pd.DataFrame({"sigungu": SIGUNGU_LIST})
    trend = base.merge(trend, on="sigungu", how="left")

    trend["well_growth_index"] = minmax_0_100(trend["well_count_growth_rate"])

    return trend


def main():
    print("[AquaGuard AI] Step 06 - Well preprocessing fixed schema")
    print(f"RAW_DIR: {RAW_DIR}")

    zip_files = sorted(
        [p for p in RAW_DIR.rglob("*.zip") if p.is_file()],
        key=lambda p: (source_type_from_name(p.name), parse_snapshot_score(p.name), p.name)
    )

    print(f"[INFO] zip files found: {len(zip_files)}")
    for p in zip_files:
        print(" -", p.name)

    for zip_path in zip_files:
        process_one_zip(zip_path)

    yearly = load_yearly_summaries()
    samples = load_samples()

    latest = build_latest_snapshot(yearly)
    trend = build_trend(yearly)

    yearly_path = OUT_DIR / "02_well_yearly_by_sigungu.csv"
    latest_path = OUT_DIR / "02_well_by_sigungu.csv"
    trend_path = OUT_DIR / "02_well_trend_by_sigungu.csv"
    sample_path = OUT_DIR / "02_well_chungnam_sample.csv"

    yearly.to_csv(yearly_path, index=False, encoding="utf-8-sig")
    latest.to_csv(latest_path, index=False, encoding="utf-8-sig")
    trend.to_csv(trend_path, index=False, encoding="utf-8-sig")
    samples.to_csv(sample_path, index=False, encoding="utf-8-sig")

    print()
    print("[Saved]")
    print(f"- {yearly_path} rows={len(yearly)}")
    print(f"- {latest_path} rows={len(latest)}")
    print(f"- {trend_path} rows={len(trend)}")
    print(f"- {sample_path} rows={len(samples)}")

    print()
    print("[Preview: latest well summary]")
    print(latest[[
        "sigungu",
        "well_count",
        "groundwater_well_count",
        "drilling_developed_well_count",
        "total_pump_capacity",
        "well_support_score",
        "well_shortage_score",
        "well_support_level"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()

