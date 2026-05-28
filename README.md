# AquaGuard AI

충남 농업용수 부족 위험 예측 및 대체 수원 후보 추천 시스템입니다.

## 1. 프로젝트 개요

AquaGuard AI는 충청남도 올담 및 공공데이터를 활용하여 충남 15개 시·군의 농업용수 부족 위험도를 산정하고, 위험지역에 대해 대체 수원 후보 TOP 5를 추천하는 행정 의사결정 지원 시스템입니다.

농업용수 부족은 단순히 강우량만으로 판단하기 어렵습니다. 실제 현장에서는 저수지 저수율, 관정 현황, 작물 구조, 농가 규모, 수혜면적, 대체 수원 접근성 등을 함께 고려해야 합니다.

## 2. 핵심 기능

| 구분 | 기능 |
|---|---|
| 위험도 산정 | 충남 15개 시·군별 농업용수 부족 위험도 계산 |
| 원인 분석 | 강우 부족도, 저수율 위험도, 관정 의존도, 작물 물수요, 대체 수원 부족도 분석 |
| 우선순위 | 최종 위험도 기준 점검 우선순위 산정 |
| 후보 추천 | 위험지역별 대체 수원 후보 TOP 5 추천 |
| 시각화 | 위험도 순위, 구성요소 기여도, 산점도, TOP 5 표 생성 |
| 대시보드 | Streamlit 기반 위험지도, 상세 분석, 후보 추천, 리포트 다운로드 제공 |

## 3. 활용 데이터

| 번호 | 데이터 | 활용 목적 |
|---:|---|---|
| 1 | 농업용저수지 수위조회 | 저수율 위험도 산정 |
| 2 | 관정현황 | 관정 의존도 및 대체 수원 부족도 산정 |
| 3 | 재배작물별 농가현황 | 작물 물수요 및 취약성 산정 |
| 4 | 강우량·가뭄 관련 데이터 | 강우 부족도 및 기상·가뭄 위험 보정 |
| 5 | 농축어업 통계 | 피해 규모 보조지표 및 우선순위 보정 |

## 4. 최종 위험도 산식

제출 제안서의 기본 위험지수 구조에 맞춰 최종 MVP 산식은 아래와 같이 구성했습니다.

final_water_risk_score =
0.25 * rain_shortage_score
+ 0.25 * reservoir_risk_score
+ 0.20 * groundwater_dependency_score
+ 0.20 * crop_water_demand_score
+ 0.10 * alternative_source_access_shortage_score

| 제안서 지표 | 구현 컬럼 | 설명 |
|---|---|---|
| 강우 부족도 | rain_shortage_score | 평년 대비 강우·저수율 부족 지표 기반 |
| 저수율 위험도 | reservoir_risk_score | 농업용저수지 수위조회 기반 |
| 관정 의존도 | groundwater_dependency_score | 관정 수·양수능력 기반 |
| 작물 물수요 지수 | crop_water_demand_score | 작물·논벼·농가 구조 기반 |
| 대체 수원 접근성 부족도 | alternative_source_access_shortage_score | 관정 기반 대체 수원 부족도 기반 |

## 5. 대체 수원 후보 추천 산식

위험지역 주변 저수지를 거리, 저수율, 수혜면적 기준으로 평가합니다.

candidate_score =
0.40 * distance_score
+ 0.35 * reservoir_surplus_score
+ 0.25 * benefit_area_score

MVP 1차에서는 저수지 개별 좌표가 제한적이므로 시·군 대표좌표 기반 거리로 계산합니다. 실제 공급 가능성은 관로, 수리권, 수질, 현장 접근성, 행정 협의를 추가 검토해야 합니다.

## 6. 주요 산출물

- data/processed/aquaguard_sigungu_features.csv
- data/processed/aquaguard_priority_top15.csv
- data/processed/alternative_source_candidates.csv
- reports/figures/01_final_risk_ranking.png
- reports/figures/02_risk_components_stacked.png
- reports/figures/03_reservoir_vs_alternative_shortage_scatter.png
- reports/figures/04_top5_priority_table.png
- reports/figures/05_alternative_source_top1_by_risk_area.png
- reports/tables/alternative_source_top5_by_sigungu.csv
- reports/tables/top_priority_summary.csv

## 7. 실행 방법

가상환경 활성화:

    .\.venv\Scripts\Activate.ps1

패키지 설치:

    pip install -r requirements.txt

최종 feature 생성:

    python scripts\07_build_final_features.py

시각화 생성:

    python scripts\09_generate_visuals.py

대체 수원 후보 추천:

    python scripts\10_recommend_alternative_sources.py

대시보드 실행:

    streamlit run app.py

접속 주소:

    http://localhost:8501

## 8. 대시보드 구성

| 탭 | 내용 |
|---|---|
| 위험지도 | 충남 15개 시·군 위험도 지도 |
| 지역 상세 | 선택 시·군 위험도, 위험 단계, 주요 원인, 5개 지표 |
| 대체 수원 후보 | 선택 시·군의 후보 저수지 TOP 5 |
| 전체 순위 | 충남 전체 위험도 순위 |
| 보고서용 이미지 | 발표·보고서 삽입용 그래프 |
| 리포트 다운로드 | 선택 지역 HTML 행정 리포트 다운로드 |

## 9. 현재 MVP 결과 요약

- 충남 15개 시·군 위험도 산정 완료
- 제출 PDF 기준 25:25:20:20:10 산식 반영 완료
- 대체 수원 후보 TOP 5 추천 완료
- 후보 중복 제거 완료
- Streamlit 기반 대시보드 MVP 구현 완료

## 10. 한계와 고도화 방향

| 한계 | 고도화 방향 |
|---|---|
| 저수지 개별 좌표 일부 제한 | 좌표 보강 후 실제 거리 기반 추천 |
| 후보 추천이 공급 가능성 확정은 아님 | 관로, 수리권, 수질, 현장 접근성 데이터 추가 |
| 일부 중간 연도 관정 데이터 과소집계 가능성 | 최신 관정 현황 중심 분석 유지, 장기 추세는 별도 검증 |
| 위험도 산식은 규칙 기반 | 향후 실제 피해·급수 제한 이력 확보 시 지도학습 모델로 확장 |
