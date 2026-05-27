from pathlib import Path
import zipfile
import re
import argparse
import pandas as pd
import numpy as np
from dbfread import DBF

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / 'data' / 'raw' / '02_well'
OUT_DIR = ROOT / 'data' / 'interim' / 'well_segments'
EXTRACT_DIR = ROOT / 'data' / 'interim' / 'well_segment_extract'

OUT_DIR.mkdir(parents=True, exist_ok=True)
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

SIGUNGU_LIST = [
    '천안시', '공주시', '보령시', '아산시', '서산시', '논산시', '계룡시', '당진시',
    '금산군', '부여군', '서천군', '청양군', '홍성군', '예산군', '태안군'
]

ENCODINGS = ['cp949', 'euc-kr', 'utf-8', 'utf-8-sig']


def clean_number(x):
    if x is None or pd.isna(x):
        return np.nan
    if isinstance(x, str):
        x = (
            x.replace(',', '')
             .replace('%', '')
             .replace(' ', '')
             .replace('㎥', '')
             .replace('m3', '')
             .replace('톤', '')
             .replace('일', '')
             .strip()
        )
        if x in ['', '-', 'X', 'nan', 'None', 'NULL', '…']:
            return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan


def normalize_sigungu(x):
    if x is None or pd.isna(x):
        return np.nan

    s = str(x).strip()

    if '천안' in s:
        return '천안시'

    for sg in SIGUNGU_LIST:
        if sg in s:
            return sg

    return np.nan


def init_acc():
    return {
        sg: {
            'sigungu': sg,
            'well_count': 0,
            'groundwater_well_count': 0,
            'drilling_developed_well_count': 0,
            'unknown_well_count': 0,
            'total_pump_capacity': 0.0,
            'pump_capacity_valid_count': 0,
            'well_depth_sum': 0.0,
            'well_depth_valid_count': 0,
            'public_like_well_count': 0,
            'record_used_count': 0,
        }
        for sg in SIGUNGU_LIST
    }


def find_zip(keyword):
    files = [p for p in RAW_DIR.rglob('*.zip') if keyword in p.name]
    if not files:
        raise FileNotFoundError(f'No zip found with keyword: {keyword}')
    return files[0]


def safe_stem(name):
    return re.sub(r'[^0-9A-Za-z가-힣_]+', '_', Path(name).stem)


def extract_zip(zip_path):
    target_dir = EXTRACT_DIR / safe_stem(zip_path.name)
    target_dir.mkdir(parents=True, exist_ok=True)

    if not list(target_dir.rglob('*.dbf')):
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(target_dir)

    dbfs = list(target_dir.rglob('*.dbf'))
    if not dbfs:
        raise FileNotFoundError('No DBF found in zip.')

    return dbfs[0]


def read_dbf(dbf_path):
    last_error = None

    for enc in ENCODINGS:
        try:
            table = DBF(
                str(dbf_path),
                encoding=enc,
                char_decode_errors='ignore',
                ignore_missing_memofile=True,
                load=False
            )
            return table, enc
        except Exception as e:
            last_error = e

    raise RuntimeError(f'DBF read failed: {dbf_path.name} / {last_error}')


def process_segment(keyword, start_row, end_row):
    zip_path = find_zip(keyword)
    dbf_path = extract_zip(zip_path)

    print(f'[TARGET ZIP] {zip_path.name}', flush=True)
    print(f'[TARGET DBF] {dbf_path}', flush=True)
    print(f'[SEGMENT] {start_row:,} ~ {end_row:,}', flush=True)

    table, enc = read_dbf(dbf_path)
    print(f'[ENCODING] {enc}', flush=True)

    acc = init_acc()

    total_seen = 0
    segment_seen = 0
    used = 0
    bad = 0

    for row in table:
        total_seen += 1

        if total_seen < start_row:
            continue

        if total_seen > end_row:
            break

        segment_seen += 1

        try:
            row = dict(row)

            sido = str(row.get('si_do', '')).strip()
            if sido != '충청남도':
                if segment_seen % 10000 == 0:
                    print(f'  progress global_row={total_seen:,} segment_seen={segment_seen:,} used={used:,}', flush=True)
                continue

            sigungu = normalize_sigungu(row.get('si_gun_gu'))

            if pd.isna(sigungu):
                continue

            used += 1
            a = acc[sigungu]

            a['well_count'] += 1
            a['groundwater_well_count'] += 1
            a['record_used_count'] += 1

            pump = clean_number(row.get('pump_abili'))
            if pd.notna(pump):
                a['total_pump_capacity'] += pump
                a['pump_capacity_valid_count'] += 1

            depth = clean_number(row.get('well_depth'))
            if pd.notna(depth):
                a['well_depth_sum'] += depth
                a['well_depth_valid_count'] += 1

            public_value = str(row.get('public_pri', ''))
            if public_value in ['01', '1']:
                a['public_like_well_count'] += 1

            if segment_seen % 10000 == 0:
                print(f'  progress global_row={total_seen:,} segment_seen={segment_seen:,} used={used:,}', flush=True)

        except BaseException as e:
            bad += 1
            print(f'[ROW ERROR] global_row={total_seen} error={repr(e)}', flush=True)
            if bad >= 20:
                print('[STOP] too many bad rows', flush=True)
                break

    out = pd.DataFrame(list(acc.values()))
    out['source_zip'] = zip_path.name
    out['source_type'] = 'groundwater_well'
    out['year'] = 2023
    out['snapshot_score'] = 20231231
    out['segment_start_row'] = start_row
    out['segment_end_row'] = end_row
    out['segment_seen_rows'] = segment_seen
    out['segment_used_chungnam'] = used
    out['segment_bad_rows'] = bad

    out_path = OUT_DIR / f'segment_gdb_well_info_new_20231231_{start_row}_{end_row}.csv'
    out.to_csv(out_path, index=False, encoding='utf-8-sig')

    print('[DONE]', flush=True)
    print(f'total_seen={total_seen:,}', flush=True)
    print(f'segment_seen={segment_seen:,}', flush=True)
    print(f'used_chungnam={used:,}', flush=True)
    print(f'bad_rows={bad}', flush=True)
    print(f'saved={out_path}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--keyword', default='gdb_well_info_new_20231231')
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--end', type=int, required=True)
    args = parser.parse_args()

    process_segment(args.keyword, args.start, args.end)


if __name__ == '__main__':
    main()
