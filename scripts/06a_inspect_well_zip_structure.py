from pathlib import Path
import zipfile
import shutil
from dbfread import DBF

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / 'data' / 'raw' / '02_well'
PROBE_DIR = ROOT / 'data' / 'interim' / 'well_probe'
PROBE_DIR.mkdir(parents=True, exist_ok=True)

ENCODINGS = ['cp949', 'euc-kr', 'utf-8', 'utf-8-sig']


def safe_name(name):
    return (
        name.replace('/', '_')
            .replace('\\\\', '_')
            .replace(':', '_')
            .replace('*', '_')
            .replace('?', '_')
            .replace('"', '_')
            .replace('<', '_')
            .replace('>', '_')
            .replace('|', '_')
    )


def read_dbf_preview(dbf_path):
    for enc in ENCODINGS:
        try:
            table = DBF(
                str(dbf_path),
                encoding=enc,
                char_decode_errors='ignore',
                ignore_missing_memofile=True,
                load=False
            )
            fields = list(table.field_names)

            rows = []
            for i, row in enumerate(table):
                rows.append(dict(row))
                if i >= 2:
                    break

            return enc, fields, rows, None
        except Exception as e:
            last_error = str(e)

    return None, [], [], last_error


def main():
    zip_files = sorted(RAW_DIR.rglob('*.zip'))

    print('[WELL ZIP STRUCTURE INSPECTION]')
    print('zip count =', len(zip_files))
    print()

    for zip_path in zip_files:
        print('=' * 100)
        print('ZIP:', zip_path.name)

        with zipfile.ZipFile(zip_path, 'r') as z:
            names = z.namelist()

            shp_names = [n for n in names if n.lower().endswith('.shp')]
            dbf_names = [n for n in names if n.lower().endswith('.dbf')]
            xlsx_names = [n for n in names if n.lower().endswith(('.xlsx', '.xls'))]

            print('files total:', len(names))
            print('shp:', shp_names)
            print('dbf:', dbf_names)
            print('excel:', xlsx_names)

            for dbf_name in dbf_names:
                out_path = PROBE_DIR / (safe_name(zip_path.stem) + '__' + safe_name(Path(dbf_name).name))
                with z.open(dbf_name) as src, open(out_path, 'wb') as dst:
                    shutil.copyfileobj(src, dst)

                enc, fields, rows, error = read_dbf_preview(out_path)

                print()
                print('  DBF:', dbf_name)
                print('  encoding:', enc)
                print('  field_count:', len(fields))
                print('  fields:', fields)

                if rows:
                    print('  sample_row_keys:', list(rows[0].keys()))
                    print('  sample_row:', rows[0])
                else:
                    print('  ERROR:', error)

        print()

if __name__ == '__main__':
    main()
