import pandas as pd

files = [
    'data/processed/01_reservoir_sigungu_daily.csv',
    'data/processed/04_weather_drought_latest_by_sigungu.csv',
    'data/processed/03_crop_vulnerability_by_sigungu.csv',
]

for f in files:
    df = pd.read_csv(f)
    print()
    print(f)
    print('shape =', df.shape)
    print('columns =', df.columns.tolist()[:15])
    if 'sigungu' in df.columns:
        print('sigungu_count =', df['sigungu'].nunique())
        print('sigungu =', sorted(df['sigungu'].dropna().unique()))
