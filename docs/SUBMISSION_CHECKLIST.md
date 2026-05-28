# AquaGuard AI 제출 전 체크리스트

## 1. 코드 실행 체크

아래 명령어가 정상 실행되어야 한다.

    python scripts\07_build_final_features.py
    python scripts\09_generate_visuals.py
    python scripts\10_recommend_alternative_sources.py
    streamlit run app.py

확인 기준:
- aquaguard_sigungu_features.csv 생성
- alternative_source_candidates.csv 생성
- reports/figures 이미지 5개 생성
- Streamlit 대시보드 정상 실행
- HTML 리포트 다운로드 가능

## 2. 데이터 산출물 체크

| 파일 | 상태 |
|---|---|
| data/processed/aquaguard_sigungu_features.csv | 필수 |
| data/processed/aquaguard_priority_top15.csv | 필수 |
| data/processed/alternative_source_candidates.csv | 필수 |
| reports/tables/alternative_source_top5_by_sigungu.csv | 필수 |
| reports/tables/top_priority_summary.csv | 필수 |

## 3. 시각화 체크

| 파일 | 용도 |
|---|---|
| 01_final_risk_ranking.png | 시·군별 위험도 순위 |
| 02_risk_components_stacked.png | 지표별 기여도 |
| 03_reservoir_vs_alternative_shortage_scatter.png | 저수율 위험 vs 대체 수원 부족 |
| 04_top5_priority_table.png | 우선 점검 대상 TOP 5 |
| 05_alternative_source_top1_by_risk_area.png | 위험지역별 1순위 후보 |

## 4. 발표에서 반드시 말할 문장

1. 제출 제안서의 25:25:20:20:10 위험도 산식을 구현했습니다.
2. 전처리 감사 결과를 반영해 관정 장기 추세는 산식에서 제외하고 최신 관정 현황 중심으로 사용했습니다.
3. 대체 수원 후보는 거리, 저수율, 수혜면적 기준으로 TOP 5를 추천합니다.
4. 추천은 공급 확정이 아니라 행정 검토 후보를 좁히는 기능입니다.
5. 선택 지역별 HTML 행정 리포트를 다운로드할 수 있습니다.

## 5. 최종 제출 전 Git 상태

    git status

정상 기준:

    nothing to commit, working tree clean
