# AquaGuard AI 시각화 생성 요약

## 생성 파일

- final_risk_ranking: reports\figures\01_final_risk_ranking.png
- risk_components_stacked: reports\figures\02_risk_components_stacked.png
- risk_component_contributions: reports\tables\risk_component_contributions.csv
- reservoir_vs_alternative_shortage_scatter: reports\figures\03_reservoir_vs_alternative_shortage_scatter.png
- top5_priority_table_png: reports\figures\04_top5_priority_table.png
- top_priority_summary_csv: reports\tables\top_priority_summary.csv
- main_driver_summary: reports\tables\main_driver_summary.csv

## 우선 점검 대상 TOP 5

| 순위 | 시·군 | 위험도 | 단계 | 주요 원인 |
|---:|---|---:|---|---|
| 1 | 당진시 | 41.0 | 주의 | 대체 수원 접근성 부족도 |
| 2 | 부여군 | 33.9 | 낮음 | 관정 의존도 |
| 3 | 서산시 | 33.6 | 낮음 | 대체 수원 접근성 부족도 |
| 4 | 논산시 | 30.9 | 낮음 | 관정 의존도 |
| 5 | 보령시 | 28.7 | 낮음 | 대체 수원 접근성 부족도 |

## 해석 기준

- 최종 위험도는 제출 PDF의 25:25:20:20:10 산식을 따른다.
- 구성요소 기여도 그래프는 각 지표 점수에 PDF 가중치를 곱한 값이다.
- 산점도는 저수율 위험과 대체 수원 접근성 부족이 동시에 높은 지역을 식별하기 위한 보조 시각화다.
- TOP 5 표는 발표자료와 보고서에 바로 삽입 가능한 요약표다.