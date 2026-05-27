from pathlib import Path
import zipfile
import re
import traceback
import pandas as pd
import numpy as np
from dbfread import DBF

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "02_well"
DEBUG_DIR = ROOT / "data" / "interim" / "well_debug_2020"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

TARGET_KEYWORD = "컬럼정의서포함20200928"

SIGUNGU_LIST = [
    "천안시", "공주시", "보령시", "아산시", "서산시", "논산시", "계룡시", "당진시",
    "금산군", "부여군", "서천군", "청양군", "홍성군", "예산군", "태안군"
]


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

    if "천안" in s:
        return "천안시"

    for sg in SIGUNGU_LIST:
        if sg in s:
            return sg

    return np.nan


def init_acc():
    return {
        sg: {
            "sigungu": sg,
            "well_count": 0,
            "total_pump_capacity": 0.0,
            "pump_capacity_valid_count": 0,
            "well_depth_sum": 0.0,
            "well_depth_valid_count": 0,
            "public_like_well_count": 0,
        }
        for sg in SIGUNGU_LIST
    }


def save_checkpoint(acc, row_idx, used_count):
    df = pd.DataFrame(list(acc.values()))
    df["checkpoint_row"] = row_idx
    df["checkpoint_used_chungnam"] = used_count

    out_path = DEBUG_DIR / "debug_2020_partial_summary.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    marker_path = DEBUG_DIR / "debug_2020_last_checkpoint.txt"
    marker_path.write_text(
        f"last_row={row_idx}\\nused_chungnam={used_count}\\n",
        encoding="utf-8"
    )


def find_target_zip():
    zips = [p for p in RAW_DIR.rglob("*.zip") if TARGET_KEYWORD in p.name]
    if not zips:
        raise FileNotFoundError(f"Cannot find zip with keyword: {TARGET_KEYWORD}")
    return zips[0]


def extract_target(zip_path):
    extract_dir = DEBUG_DIR / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    if not list(extract_dir.rglob("*.dbf")):
        print(f"[UNZIP] {zip_path.name}", flush=True)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)

    dbfs = list(extract_dir.rglob("*.dbf"))
    if not dbfs:
        raise FileNotFoundError("No DBF found in extracted zip.")

    return dbfs[0]


def main():
    print("[DEBUG] 2020 groundwater DBF single-file test", flush=True)

    zip_path = find_target_zip()
    print(f"[TARGET ZIP] {zip_path}", flush=True)

    dbf_path = extract_target(zip_path)
    print(f"[TARGET DBF] {dbf_path}", flush=True)

    table = DBF(
        str(dbf_path),
        encoding="cp949",
        char_decode_errors="ignore",
        ignore_missing_memofile=True,
        load=False
    )

    print("[FIELDS]", list(table.field_names), flush=True)

    acc = init_acc()
    used = 0
    total = 0
    bad_rows = 0

    iterator = iter(table)

    while True:
        try:
            row = next(iterator)
        except StopIteration:
            break
        except BaseException as e:
            bad_rows += 1
            print(f"[ROW ERROR] total={total} bad_rows={bad_rows} error={repr(e)}", flush=True)
            traceback.print_exc()

            save_checkpoint(acc, total, used)

            # 너무 많은 오류면 중단
            if bad_rows >= 20:
                print("[STOP] too many bad rows", flush=True)
                break

            continue

        total += 1

        row = dict(row)

        sido = str(row.get("si_do", "")).strip()
        if sido != "충청남도":
            if total % 100000 == 0:
                print(f"progress total={total:,} used_chungnam={used:,}", flush=True)
                save_checkpoint(acc, total, used)
            continue

        sigungu = normalize_sigungu(row.get("si_gun_gu"))

        if pd.isna(sigungu):
            if total % 100000 == 0:
                print(f"progress total={total:,} used_chungnam={used:,}", flush=True)
                save_checkpoint(acc, total, used)
            continue

        used += 1
        a = acc[sigungu]
        a["well_count"] += 1

        pump = clean_number(row.get("pump_abili"))
        if pd.notna(pump):
            a["total_pump_capacity"] += pump
            a["pump_capacity_valid_count"] += 1

        depth = clean_number(row.get("well_depth"))
        if pd.notna(depth):
            a["well_depth_sum"] += depth
            a["well_depth_valid_count"] += 1

        public_value = str(row.get("public_pri", ""))
        if public_value in ["01", "1"]:
            a["public_like_well_count"] += 1

        if total % 100000 == 0:
            print(f"progress total={total:,} used_chungnam={used:,}", flush=True)
            save_checkpoint(acc, total, used)

    save_checkpoint(acc, total, used)

    final = pd.DataFrame(list(acc.values()))
    final["total_rows_read"] = total
    final["used_chungnam"] = used
    final["bad_rows"] = bad_rows
    final_path = DEBUG_DIR / "debug_2020_final_summary.csv"
    final.to_csv(final_path, index=False, encoding="utf-8-sig")

    print("[DONE]", flush=True)
    print(f"total_rows={total:,}", flush=True)
    print(f"used_chungnam={used:,}", flush=True)
    print(f"bad_rows={bad_rows}", flush=True)
    print(f"saved={final_path}", flush=True)


if __name__ == "__main__":
    main()
