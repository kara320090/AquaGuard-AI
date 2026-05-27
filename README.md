# AquaGuard AI

올담 및 공공데이터포털 기반 충남 농업용수 위험도 산정·대체 수원 후보 추천 MVP.

## 목표

충남 시·군별 농업용수 부족 위험을 산정하고, 위험 원인과 대체 수원 후보를 행정 담당자가 이해할 수 있는 형태로 제공한다.

## 신청서류 기준 활용 데이터

1. 농업용저수지 수위조회
2. 관정현황
3. 재배작물별 농가현황
4. 시·군별 강우량/가뭄 관련 데이터
5. 농축어업 통계

## 핵심 산출물

- 시·군별 농업용수 부족 위험도
- 위험 원인 분석
- 대체 수원 후보 TOP 5
- 행정 리포트
- Streamlit 대시보드

## 폴더 구조

`	ext
data/raw/01_reservoir
data/raw/02_well
data/raw/03_crop
data/raw/04_weather_drought
data/raw/05_agri_stats
data/processed
scripts
src/aquaguard
app
docs
reports
outputs
python scripts/00_check_data_layout.py
streamlit run app/streamlit_app.py

