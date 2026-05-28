# AquaGuard AI Live Feature Method

## 목적

올담 최신 저수지 수위 데이터, 기상청 AWS/ASOS 최근 30일 강수량, ADMS 토양수분현황을 결합해 현재 기준 농업용수 위험도를 갱신한다.

## Live 위험도 산식

final_live_water_risk_score =
0.20 * live_rain_shortage_score
+ 0.20 * live_reservoir_risk_score
+ 0.20 * live_soil_moisture_drought_score
+ 0.15 * groundwater_dependency_score
+ 0.15 * crop_water_demand_score
+ 0.10 * alternative_source_access_shortage_score

## 데이터 소스 우선순위

- 강우 부족도: 기상청 AWS/ASOS 최근 30일 강수량
- 저수율 위험도: 올담 최신 저수율 데이터
- 토양수분 가뭄도: ADMS 밭토양수분현황
- 관정, 작물, 대체 수원 접근성: 정적·반정적 공공데이터 기반 기존 feature

## 해석 주의

ADMS 토양유효수분 값이 충분한 상태이면 soil_moisture_drought_score는 낮게 계산된다.
이는 위험을 과장하지 않고 실제 농지 건조 상태를 보정하기 위한 것이다.
