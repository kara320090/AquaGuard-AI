# AquaGuard AI Live Feature Method

## 목적

올담 최신 저수지 수위 데이터와 기상청 ASOS 최근 30일 강수량 데이터를 결합해 현재 기준 농업용수 위험도를 갱신한다.

## Live 위험도 산식

final_live_water_risk_score =
0.25 * live_rain_shortage_score
+ 0.25 * live_reservoir_risk_score
+ 0.20 * groundwater_dependency_score
+ 0.20 * crop_water_demand_score
+ 0.10 * alternative_source_access_shortage_score

## 데이터 소스 우선순위

- 저수율 위험도: 올담 오늘 저수율 데이터가 있으면 OLDAM_TODAY 사용, 없으면 기존 2025 기준 baseline 사용
- 강우 부족도: 기상청 ASOS 최근 30일 강수량이 있으면 KMA_ASOS_30D 사용, 없으면 기존 기상 baseline 사용
- 관정, 작물, 대체 수원 접근성: 정적·반정적 공공데이터 기반 기존 feature 사용

## 해석 주의

올담 저수지 데이터는 현재 하루치 snapshot이므로 GRU 30일 시계열 추론 입력으로는 바로 사용할 수 없다.
대신 현재 저수율 상태 갱신에는 사용할 수 있으며, 매일 snapshot을 누적하면 향후 30일 이상 누적 후 GRU/AutoEncoder 현재 추론으로 확장할 수 있다.
