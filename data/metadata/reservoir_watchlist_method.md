# AquaGuard AI 저수지 현황 및 이상탐지 Watchlist

## 목적

제출 제안서의 MVP 범위에 포함된 저수지 현황 테이블과 저수율 이상탐지 대시보드 기능을 구현하기 위한 보조 산출물이다.

## 산출물

- reports/tables/reservoir_watchlist.csv
- reports/tables/reservoir_facility_status_by_sigungu.csv
- data/processed/reservoir_facility_status_for_dashboard.csv

## Watchlist 판정 기준

- 심각후보: 최저 저수율 30% 이하, 30% 이하 저수지 존재, 또는 저수율 위험도 80 이상
- 경계후보: 최저 저수율 40% 이하, 40% 이하 저수지 존재, 또는 저수율 위험도 60 이상
- 주의후보: 평균 저수율 70% 미만 또는 저수율 위험도 40 이상
- 정상: 위 조건에 해당하지 않는 경우

## 해석 주의

MVP 단계에서는 공개데이터 기반 규칙형 Watchlist로 구현한다.
이는 실제 현장 이상 여부를 확정하는 기능이 아니라, 행정 담당자가 우선 점검할 후보를 좁히는 참고 지표이다.
시설 점검 우선점수는 시설별 실시간 저수율이 아니라, 시·군 저수율 위험도와 시설 규모 정보를 결합한 행정 점검 우선순위이다.
향후 장기 시계열이 충분히 확보되면 Isolation Forest, RandomForest, LightGBM 기반 이상탐지로 고도화할 수 있다.
