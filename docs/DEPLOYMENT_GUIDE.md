# AquaGuard AI 배포 가이드

## 1. 배포 구조

AquaGuard AI는 별도 FastAPI 서버 없이 Streamlit Cloud로 배포한다.

기본 구조는 다음과 같다.

GitHub Repository
-> Streamlit Cloud
-> 여러 사용자 웹 접속

데이터 갱신은 GitHub Actions가 담당한다.

GitHub Actions
-> 매일 09:00 KST 실행
-> feature table, 시각화, 후보 추천, watchlist 재생성
-> 변경 사항이 있으면 자동 commit

## 2. Streamlit Cloud 배포 설정

Streamlit Cloud에서 새 앱을 만들 때 아래 값으로 설정한다.

Repository:
kara320090/AquaGuard-AI

Branch:
main

Main file path:
app.py

Python version:
3.11 권장

## 3. 로컬 실행 확인

배포 전 로컬에서 아래 명령어로 실행한다.

streamlit run app.py

정상 확인 항목은 다음과 같다.

1. 메인 KPI 카드 4개 표시
2. 위험지도 표시
3. 지역 상세 분석 표시
4. 대체 수원 후보 TOP 5 표시
5. 저수지 Watchlist 페이지 표시
6. HTML 행정 리포트 다운로드 가능

## 4. 운영 방식

본 MVP는 사용자가 접속할 때마다 AI 학습이나 API 수집을 수행하지 않는다.

대신 사전에 생성된 분석 결과 CSV와 이미지 파일을 읽어 대시보드에 표시한다.

이 방식은 해커톤 시연과 공공데이터 기반 행정 참고 서비스에 적합하다.

## 5. FastAPI를 사용하지 않는 이유

FastAPI를 로컬 또는 개인 서버에서 운영하면 서버 컴퓨터를 계속 켜두어야 한다.

해커톤 MVP에서는 운영 안정성이 중요하므로, 별도 백엔드 서버 없이 Streamlit Cloud와 GitHub Actions 중심으로 구성한다.

FastAPI는 향후 모바일 앱, 별도 웹 프론트, 외부 기관 연동이 필요할 때 서비스화 단계에서 추가한다.

## 6. 여러 사용자 접속 방식

Streamlit Cloud에 배포하면 사용자는 웹 주소로 접속할 수 있다.

로컬 PC에서 streamlit run app.py로 실행하는 경우에는 내 컴퓨터가 켜져 있어야 하므로 다수 사용자용 운영 방식으로 적합하지 않다.

## 7. 향후 올담 API 연동 방식

올담 API가 오늘 데이터 또는 최근 1개월 데이터를 제공하는 경우, API 데이터는 모델 학습용이 아니라 최신 상태 반영용으로 사용한다.

과거 공공데이터:
AI 모델 학습 및 기준 패턴 생성

올담 API 오늘 또는 최근 1개월 데이터:
최신 위험도 갱신 및 AI 추론 입력

따라서 최종 구조는 다음과 같다.

과거 공공데이터
-> 전처리
-> AI 기준 패턴 학습

올담 API 최신 데이터
-> snapshot 저장
-> 최신 feature 생성
-> 위험도 갱신
-> Streamlit 표시

## 8. 발표 시 설명 문장

본 MVP는 별도 서버를 상시 운영하지 않고도 여러 사용자가 접근할 수 있도록 Streamlit Cloud 기반 웹앱으로 구성했습니다.

AI 모델 학습과 올담 API 수집은 사전에 배치 작업으로 수행하고, 대시보드는 저장된 최신 분석 결과를 읽어 보여주는 구조입니다.

따라서 사용자가 접속할 때마다 무거운 AI 학습이나 API 호출이 반복되지 않아, 해커톤 시연과 행정 담당자용 조회 서비스에 적합합니다.
