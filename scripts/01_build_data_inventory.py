from pathlib import Path
from datetime import datetime
import zipfile
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "metadata"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORY_MAP = {
    "01_reservoir": "① 농업용저수지 수위조회",
    "02_well": "② 관정현황",
    "03_crop": "③ 재배작물별 농가현황",
    "04_weather_drought": "④ 시·군별 강우량/가뭄 관련 데이터",
    "05_agri_stats": "⑤ 농축어업 통계",
}

TEXT_ENCODINGS = ["utf-8-sig", "cp949", "euc-kr", "utf-8"]


def safe_size_mb(path: Path) -> float:
    try:
        return round(path.stat().st_size / (1024 * 1024), 3)
    except Exception:
        return 0.0


def try_read_csv_columns(path: Path):
    last_error = ""
    for enc in TEXT_ENCODINGS:
        try:
            df = pd.read_csv(path, encoding=enc, nrows=3)
            return list(df.columns), enc, ""
        except Exception as e:
            last_error = str(e)[:150]
    return [], "", last_error


def try_read_excel_columns(path: Path):
    try:
        xls = pd.ExcelFile(path)
        sheet = xls.sheet_names[0]
        df = pd.read_excel(path, sheet_name=sheet, nrows=3)
        return list(df.columns), sheet, ""
    except Exception as e:
        return [], "", str(e)[:150]


def inspect_zip(path: Path):
    try:
        with zipfile.ZipFile(path, "r") as z:
            names = z.namelist()
            return len(names), "; ".join(names[:15])
    except Exception as e:
        return 0, str(e)[:150]


def detect_category(path: Path):
    try:
        rel_parts = path.relative_to(RAW_DIR).parts
    except ValueError:
        return "", ""
    if not rel_parts:
        return "", ""
    folder = rel_parts[0]
    return folder, CATEGORY_MAP.get(folder, "UNKNOWN")


def main():
    records = []

    if not RAW_DIR.exists():
        raise FileNotFoundError(f"RAW_DIR not found: {RAW_DIR}")

    files = [
        p for p in RAW_DIR.rglob("*")
        if p.is_file() and p.name != ".gitkeep"
    ]

    for path in files:
        category_code, category_name = detect_category(path)
        ext = path.suffix.lower()

        columns = []
        preview_info = ""
        error = ""
        zip_count = ""
        zip_preview = ""

        if ext in [".csv", ".txt"]:
            columns, preview_info, error = try_read_csv_columns(path)
        elif ext in [".xlsx", ".xls"]:
            columns, preview_info, error = try_read_excel_columns(path)
        elif ext == ".zip":
            zip_count, zip_preview = inspect_zip(path)
            preview_info = "zip"
        elif ext in [".shp", ".dbf", ".shx", ".prj", ".cpg"]:
            preview_info = "shapefile_component"
        else:
            preview_info = "not_previewed"

        records.append({
            "category_code": category_code,
            "category_name": category_name,
            "relative_path": str(path.relative_to(ROOT)),
            "file_name": path.name,
            "extension": ext,
            "size_mb": safe_size_mb(path),
            "modified_time": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "preview_info": preview_info,
            "columns_preview": " | ".join(map(str, columns[:40])),
            "zip_file_count": zip_count,
            "zip_preview": zip_preview,
            "error": error,
        })

    df = pd.DataFrame(records)

    if len(df) == 0:
        print("[WARN] No files found under data/raw")
        return

    df = df.sort_values(["category_code", "relative_path"]).reset_index(drop=True)

    csv_path = OUT_DIR / "data_inventory.csv"
    xlsx_path = OUT_DIR / "data_inventory.xlsx"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df.to_excel(xlsx_path, index=False)

    print(f"[OK] scanned files: {len(df)}")
    print(f"[OK] saved: {csv_path}")
    print(f"[OK] saved: {xlsx_path}")

    print()
    print("[Summary by category]")
    print(df.groupby(["category_code", "category_name"]).size())


if __name__ == "__main__":
    main()
