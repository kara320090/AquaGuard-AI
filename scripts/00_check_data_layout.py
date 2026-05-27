from pathlib import Path

REQUIRED_DIRS = {
    '01_reservoir': '① 농업용저수지 수위조회',
    '02_well': '② 관정현황',
    '03_crop': '③ 재배작물별 농가현황',
    '04_weather_drought': '④ 시·군별 강우량/가뭄 관련 데이터',
    '05_agri_stats': '⑤ 농축어업 통계',
}

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'

def count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob('*') if p.is_file() and p.name != '.gitkeep')

def main():
    print('[AquaGuard AI] Data layout check')
    print('-' * 70)

    ok = True
    for folder, label in REQUIRED_DIRS.items():
        path = RAW / folder
        n = count_files(path)
        status = 'OK' if path.exists() and n > 0 else 'CHECK'
        print(f'{status:6} {label:35} -> files={n}')
        if n == 0:
            ok = False

    print('-' * 70)
    if ok:
        print('All required raw data folders contain files.')
    else:
        print('Some raw data folders are empty. Place files under data/raw/01~05 folders.')

if __name__ == '__main__':
    main()
