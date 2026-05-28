# AquaGuard AI 운영 메모

## 1. 매일 자동 갱신

.github/workflows/daily_refresh.yml 파일이 매일 09:00 KST에 실행된다.

실행 내용은 다음과 같다.

1. scripts/07_build_final_features.py
2. scripts/09_generate_visuals.py
3. scripts/10_recommend_alternative_sources.py
4. scripts/12_build_reservoir_watchlist.py
5. scripts/11_final_validation.py

## 2. 자동 갱신의 목적

사용자가 Streamlit 앱에 접속할 때마다 무거운 전처리나 AI 학습을 수행하지 않도록 한다.

GitHub Actions가 미리 결과물을 생성하고, Streamlit은 저장된 CSV와 이미지 파일을 읽어 표시한다.

## 3. 올담 API 연동 예정 구조

향후 scripts/14_fetch_oldam_api.py를 추가한다.

이 스크립트의 역할은 다음과 같다.

1. 올담 API 오늘 또는 최근 1개월 데이터 호출
2. data/raw/api_snapshots/YYYYMMDD 폴더에 원본 저장
3. data/processed/latest_api_features.csv 생성
4. 기존 위험도 산식과 AI 추론에 최신 데이터 반영

## 4. AI 모델 연동 예정 구조

과거 공공데이터는 학습용으로 사용하고, 올담 API 최신 데이터는 추론용 입력으로 사용한다.

과거 데이터:
AI 기준 패턴 학습

올담 API 최신 데이터:
현재 상태 입력

AI 모델:
저수율 예측, 이상탐지, 위험도 보정

Streamlit:
AI 결과와 위험도 결과를 대시보드에 표시

## 5. FastAPI를 쓰지 않는 이유

해커톤 MVP에서는 별도 서버를 계속 켜둘 필요가 없도록 Streamlit Cloud 중심으로 배포한다.

FastAPI는 다음 상황에서 추가한다.

1. 모바일 앱 또는 별도 웹 프론트가 생기는 경우
2. 외부 기관이 API로 결과를 조회해야 하는 경우
3. 실시간 모델 추론 서버가 필요한 경우
4. API 키를 서버에 숨겨야 하는 경우
5. DB 기반 운영 서비스로 확장하는 경우

현재 MVP에서는 Streamlit Cloud와 GitHub Actions 조합이 더 안정적이다.

## 6. 운영 원칙

사용자 접속 시점:
저장된 분석 결과 조회

매일 갱신 시점:
GitHub Actions가 데이터 재계산

향후 API 연동 시점:
올담 API 데이터를 snapshot으로 저장 후 재계산

향후 AI 고도화 시점:
GPU 환경에서 모델 학습 후 결과 파일과 모델 파일을 저장

## 7. 제출 전 확인

아래 항목을 확인한다.

1. streamlit run app.py 정상 실행
2. 위험지도 표시
3. 대체 수원 후보 TOP 5 표시
4. 저수지 Watchlist 페이지 표시
5. HTML 행정 리포트 다운로드 가능
6. scripts/11_final_validation.py 실행 시 FAIL 없음
7. git status 결과가 clean 상태
