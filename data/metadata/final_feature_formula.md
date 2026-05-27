# AquaGuard AI 최종 위험도 산식

## 최종 점수

final_water_risk_score =
0.35 * reservoir_risk_score
+ 0.25 * weather_drought_risk_score
+ 0.15 * crop_vulnerability_index
+ 0.15 * agri_impact_index
+ 0.10 * well_shortage_score

## 해석

- reservoir_risk_score: 저수율 부족 위험
- weather_drought_risk_score: 기상·가뭄 위험
- crop_vulnerability_index: 작물 구조상 물 부족 취약성
- agri_impact_index: 농가·농가인구·수혜면적 기반 영향 규모
- well_shortage_score: 관정 기반 대체 수원 부족도

## 처리 원칙

- 02_well_trend_by_sigungu.csv는 일부 중간 연도 과소집계 경고가 있어 최종 산식에서 제외한다.
- 계룡시처럼 농업용 저수지 데이터가 없는 지역은 reservoir_risk_score 원본은 결측으로 유지하고, 최종 산식에서는 중립값 50으로 보정한다.
- 최종 우선순위는 final_water_risk_score 내림차순으로 산정한다.
