# AquaGuard AI 최종 위험도 산식

## 제출 제안서 기준 산식 반영

제출 PDF의 기본 위험지수 구조에 맞춰 최종 MVP 산식을 아래와 같이 적용한다.

final_water_risk_score =
0.25 * rain_shortage_score
+ 0.25 * reservoir_risk_score
+ 0.20 * groundwater_dependency_score
+ 0.20 * crop_water_demand_score
+ 0.10 * alternative_source_access_shortage_score

## 지표 매핑

| PDF 지표 | 구현 컬럼 | 설명 |
|---|---|---|
| 강우 부족도 | rain_shortage_score | 평년 대비 강우/저수율 부족 지표 기반 |
| 저수율 위험도 | reservoir_risk_score | 농업용저수지 수위조회 기반 |
| 관정 의존도 | groundwater_dependency_score | 관정 수·양수능력 기반 well_support_score 활용 |
| 작물 물수요 지수 | crop_water_demand_score | 재배작물·논벼·농가 구조 기반 crop_vulnerability_index 활용 |
| 대체 수원 접근성 부족도 | alternative_source_access_shortage_score | MVP 1차에서는 well_shortage_score 활용 |

## 처리 원칙

- 제출 PDF와의 정합성을 위해 최종 위험도 산식은 PDF의 25:25:20:20:10 구조를 따른다.
- 농축어업 통계 기반 agri_impact_index는 최종 점수 직접 가중치가 아니라 피해 규모 보조 지표 및 동점 보정 지표로 사용한다.
- 02_well_trend_by_sigungu.csv는 일부 중간 연도 과소집계 경고가 있어 최종 산식에서 제외한다.
- 계룡시처럼 농업용저수지 데이터가 없는 지역은 reservoir_risk_score 원본은 결측으로 유지하고, 최종 산식에서는 중립값 50으로 보정한다.
- 대체 수원 접근성 부족도는 다음 단계에서 거리·저수율·수혜면적 기반 후보 추천 알고리즘으로 고도화한다.
