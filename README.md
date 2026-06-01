# AquaGuard AI

## Live Demo

Streamlit Cloud 배포 URL: https://aquaguard-aibranchmainmainfilepathapppy-dwpqbnvzhdnpnyhifqvvvv.streamlit.app/

AquaGuard AI는 충남 농업용수 부족 위험 예측, 저수지 Watchlist, 대체 수원 후보 추천, Live 데이터 갱신, Deep AI 예측·이상탐지를 함께 제공하는 의사결정 지원 대시보드입니다.

## 1. 프로젝트 개요

AquaGuard AI는 충청남도 올담 및 공공데이터를 활용하여 충남 15개 시·군의 농업용수 부족 위험도를 산정하고, 위험지역에 대해 대체 수원 후보 TOP 5와 점검 우선순위를 제시하는 행정 의사결정 지원 시스템입니다.

농업용수 부족은 단순히 강우량만으로 판단하기 어렵습니다. 실제 현장에서는 저수지 저수율, 관정 현황, 작물 구조, 농가 규모, 수혜면적, 대체 수원 접근성, 최신 공개 데이터 상태를 함께 고려해야 합니다.

## 2. 핵심 기능

| 구분 | 기능 |
|---|---|
| 최종 위험도 산정 | 충남 15개 시·군별 농업용수 부족 위험도 계산 |
| 원인 분석 | 강우 부족도, 저수율 위험도, 관정 의존도, 작물 물수요, 대체 수원 접근성 부족도 분석 |
| 점검 우선순위 | 위험도와 시설 규모를 결합한 시·군 및 저수지 시설 점검 우선순위 제공 |
| Reservoir Watchlist | 저수율 위험도, Watch 단계, 시설별 점검 우선순위 확인 |
| 대체 수원 추천 | 위험지역별 대체 수원 후보 TOP 5 추천 |
| Live 데이터 갱신 | 올담, 기상청 AWS/ASOS, ADMS 토양수분, ADMS 저수율 보조자료 반영 |
| Deep AI Insights | GRU 기반 저수율 예측과 AutoEncoder 이상탐지 결과 제공 |
| 보고서용 시각화 | 위험도 순위, 구성요소 기여도, 산점도, TOP 5 표, 대체 수원 후보 이미지 생성 |
| Streamlit 대시보드 | 비기술 검토자가 핵심 결과를 빠르게 이해할 수 있는 데모용 화면 제공 |

## 3. 활용 데이터

| 번호 | 데이터 | 활용 목적 |
|---:|---|---|
| 1 | 농업용저수지 수위조회 | 저수율 위험도 산정 |
| 2 | 관정현황 | 관정 의존도 및 대체 수원 부족도 산정 |
| 3 | 재배작물별 농가현황 | 작물 물수요 및 취약성 산정 |
| 4 | 강우량·가뭄 관련 데이터 | 강우 부족도 및 기상·가뭄 위험 보정 |
| 5 | 농축어업 통계 | 피해 규모 보조지표 및 우선순위 보정 |
| 6 | 올담 최신 공개 저수지 데이터 | Live 기준 저수지 상태 및 교차검증 |
| 7 | 기상청 AWS/ASOS 최근 강수 데이터 | Live 강우 부족도 갱신 |
| 8 | ADMS 토양수분 및 저수율 보조자료 | Live 위험도 보정 및 원천 데이터 비교 |

## 4. 최종 위험도 산식

제출 제안서의 기본 위험지수 구조에 맞춰 최종 MVP 산식은 아래와 같이 구성했습니다.

```text
final_water_risk_score =
0.25 * rain_shortage_score
+ 0.25 * reservoir_risk_score
+ 0.20 * groundwater_dependency_score
+ 0.20 * crop_water_demand_score
+ 0.10 * alternative_source_access_shortage_score
```

| 제안서 지표 | 구현 컬럼 | 설명 |
|---|---|---|
| 강우 부족도 | `rain_shortage_score` | 평년 대비 강우 및 기상 위험 지표 기반 |
| 저수율 위험도 | `reservoir_risk_score` | 농업용저수지 수위조회 기반 |
| 관정 의존도 | `groundwater_dependency_score` | 관정 수·양수능력 기반 |
| 작물 물수요 지수 | `crop_water_demand_score` | 작물·논벼·농가 구조 기반 |
| 대체 수원 접근성 부족도 | `alternative_source_access_shortage_score` | 관정 및 대체 수원 접근성 기반 |

계룡시처럼 원천 저수지 기준일 또는 시설 매칭 정보가 부족한 지역은 원본 결측을 유지하되, 최종 산정에서는 확보 가능한 강우·관정·작물·대체수원 지표 중심으로 해석합니다.

## 5. 대체 수원 후보 추천 산식

위험지역 주변 저수지를 거리, 저수율, 수혜면적 기준으로 평가합니다.

```text
candidate_score =
0.40 * distance_score
+ 0.35 * reservoir_surplus_score
+ 0.25 * benefit_area_score
```

MVP 1차에서는 저수지 개별 좌표가 제한적이므로 시·군 대표좌표 기반 거리로 계산합니다. 실제 공급 가능성은 관로, 수리권, 수질, 현장 접근성, 행정 협의를 추가 검토해야 합니다.

## 6. Reservoir Watchlist와 시설 점검 우선순위

Reservoir Watchlist는 시·군 단위 저수율 위험도와 시설 제원 정보를 함께 보여줍니다. 시·군 평균 저수율, 최저 저수율, 저수율 위험도는 지역 공통 지표이며, 시설별 행에 반복 표시하지 않습니다.

시설 점검 우선점수는 시설별 실시간 저수율이 아니라, 시·군 저수율 위험도와 시설 규모 정보를 결합한 행정 점검 우선순위입니다.

```text
facility_scale_score =
0.45 * benefit_area_rank_score
+ 0.35 * effective_capacity_rank_score
+ 0.20 * total_capacity_rank_score

inspection_priority_score =
0.40 * sigungu_reservoir_risk_score
+ 0.60 * facility_scale_score
```

시설별 최신 공개 저수율은 OLDAM 원천 데이터에서 시설명이 정확히 매칭되는 경우에만 표시하며, 매칭되지 않는 경우에는 `자료 없음`으로 표시합니다.

## 7. 주요 산출물

| 경로 | 설명 |
|---|---|
| `data/processed/aquaguard_sigungu_features.csv` | 최종 시·군 단위 위험도 feature |
| `data/processed/aquaguard_priority_top15.csv` | 최종 점검 우선순위 |
| `data/processed/alternative_source_candidates.csv` | 대체 수원 후보 |
| `data/processed/reservoir_facility_status_for_dashboard.csv` | 저수지 시설별 점검 우선순위 |
| `reports/tables/reservoir_watchlist.csv` | 시·군 저수율 Watchlist |
| `reports/tables/alternative_source_top5_by_sigungu.csv` | 시·군별 대체 수원 후보 TOP 5 |
| `reports/figures/01_final_risk_ranking.png` | 최종 위험도 순위 이미지 |
| `reports/figures/02_risk_components_stacked.png` | 위험도 구성요소 기여도 이미지 |
| `reports/figures/03_reservoir_vs_alternative_shortage_scatter.png` | 저수율 위험도와 대체 수원 부족도 산점도 |
| `reports/figures/04_top5_priority_table.png` | 우선 점검 대상 TOP 5 표 이미지 |
| `reports/figures/05_alternative_source_top1_by_risk_area.png` | 위험지역별 1순위 대체 수원 후보 이미지 |

## 8. 실행 방법

가상환경 활성화:

```powershell
.\.venv\Scripts\Activate.ps1
```

패키지 설치:

```powershell
pip install -r requirements.txt
```

최종 feature 생성:

```powershell
python scripts\07_build_final_features.py
```

보고서용 시각화 생성:

```powershell
python scripts\09_generate_visuals.py
```

대체 수원 후보 추천:

```powershell
python scripts\10_recommend_alternative_sources.py
```

저수지 Watchlist 생성:

```powershell
python scripts\12_build_reservoir_watchlist.py
```

최종 검증:

```powershell
python scripts\11_final_validation.py
```

대시보드 실행:

```powershell
streamlit run app.py
```

접속 주소:

```text
http://localhost:8501
```

## 9. 대시보드 구성

| 페이지 | 내용 |
|---|---|
| 메인 대시보드 | 핵심 KPI, 점검 우선순위, 성능 검증 요약, 위험도 분석, 상세 원본 확인 |
| Reservoir Watchlist | 시·군 저수율 Watchlist, 선택 지역 요약, 시설별 점검 우선순위 |
| Deep AI Insights | Live 기준월과 AI 비교 기준월 분리, GRU 예측, AutoEncoder 이상탐지 |
| Live Data Update | 최신 공개 데이터 수집 상태, Live 위험도, 원천 데이터 교차검증 |
| 보고서용 이미지 확인 | 생성된 PNG 보고서 이미지 렌더링 및 누락 경로 안내 |

## 10. 현재 MVP 결과 요약

- 충남 15개 시·군 위험도 산정 완료
- 제출 PDF 기준 25:25:20:20:10 산식 반영 완료
- 대체 수원 후보 TOP 5 추천 완료
- Reservoir Watchlist 및 시설별 점검 우선순위 구현 완료
- Deep AI 예측·이상탐지 결과 검증 완료
- Live 데이터 업데이트 및 교차검증 화면 구현 완료
- Streamlit 기반 데모/경진대회 제출용 대시보드 구현 완료

## 11. 한계와 고도화 방향

| 한계 | 고도화 방향 |
|---|---|
| 저수지 개별 좌표 일부 제한 | 좌표 보강 후 실제 거리 기반 추천 |
| 후보 추천이 공급 가능성 확정은 아님 | 관로, 수리권, 수질, 현장 접근성 데이터 추가 |
| 일부 지역은 저수지 기준일 또는 시설 매칭 정보 부족 | 원천 데이터 보강 및 지자체 시설 DB 매칭 고도화 |
| 시설별 실시간 저수율은 일부 공개 데이터에만 존재 | 정확히 매칭되는 시설만 표시하고, 미매칭 시설은 시·군 위험도와 시설 규모 기반으로 우선순위 산정 |
| 위험도 산식은 규칙 기반 | 실제 피해·급수 제한 이력 확보 시 지도학습 모델로 확장 |
