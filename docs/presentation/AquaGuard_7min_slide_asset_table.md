# AquaGuard AI 7분 발표 슬라이드 자산표

## 1. 필수 슬라이드 자산

| 슬라이드 | 자산 파일명 | 자산 유형 | 제작/녹화 위치 | 사용 목적 |
| ---: | --- | --- | --- | --- |
| 1 | `01_main_dashboard_overview.png` | 스크린샷 | Main dashboard 첫 화면 | 서비스 첫인상과 KPI 구조 제시 |
| 2 | `02_problem_structure.png` | 도식 | 별도 제작 | 농업용수 위험 판단 구조 설명 |
| 3 | `03_data_pipeline.png` | 도식 | 별도 제작 | OLDAM, KMA, ADMS, 관정, 작물, 대체수원 데이터 흐름 설명 |
| 4 | `04_final_validation_pass.png` | 스크린샷 | `python scripts/11_final_validation.py` 실행 결과 | 최종 검증 PASS 근거 제시 |
| 5 | `clip_01_main_map.mp4` | 영상 | Main dashboard / `지도 보기` | 어디가 위험한지 지도에서 확인 |
| 6 | `clip_02_risk_analysis.mp4` | 영상 | Main dashboard / `위험도 분석` | 왜 위험한지 구성요소로 설명 |
| 7 | `clip_03_priority_alternative.mp4` | 영상 | Main dashboard / `점검·대체수원` | 우선 점검 순위와 대체 수원 후보 설명 |
| 8 | `clip_04_watchlist_facility.mp4` | 영상 | Reservoir Watchlist | 저수지 시설별 점검 우선순위 설명 |
| 9 | `clip_05_live_ai_operability.mp4` | 영상 | Live Data Update + Deep AI Insights | Live 갱신과 AI 비교 기준 분리 설명 |
| 10 | `10_expected_effects_closing.png` | 도식 | 별도 제작 | 기대 효과와 마무리 |

## 2. 영상 녹화 설정

| 항목 | 권장값 |
| --- | --- |
| 화면 비율 | 16:9 |
| 브라우저 zoom | 80~90% |
| 녹화 해상도 | 1920x1080 또는 1600x900 |
| 마우스 이동 | 느리게, 클릭 위치가 보이도록 |
| 불필요 영역 | 터미널, 파일 경로, API key, 개인 경로 숨김 |
| Streamlit 상태 | 로딩 완료 후 녹화 시작 |
| 주소 | `http://127.0.0.1:8501/` 기준 |

## 3. 영상별 녹화 파일 체크

| 파일명 | 길이 | 필수 화면 | 실패 조건 |
| --- | ---: | --- | --- |
| `clip_01_main_map.mp4` | 35초 | `지도 보기` 탭, 밝은 지도, 위험 마커 | 지도가 잘리거나 마커가 화면 밖으로 나감 |
| `clip_02_risk_analysis.mp4` | 40초 | 위험도 순위, 분포, 주요 위험 원인, 지역 구성요소 selectbox | 상세 지역을 사이드바에서 고르는 것처럼 보임 |
| `clip_03_priority_alternative.mp4` | 45초 | 우선 점검 순위, 대체 수원 후보 selectbox와 TOP 5 | 대체 수원 표가 비어 있음 |
| `clip_04_watchlist_facility.mp4` | 45초 | Watchlist 요약, 저수율·시설 점검, 시설별 점검 우선순위 | 지역 공통 저수율이 시설 행마다 반복 표시됨 |
| `clip_05_live_ai_operability.mp4` | 55초 | Live 기준일, 원천 데이터 상태, AI 비교 기준월, 성능/이상탐지 | AI가 2026년 Live 데이터를 직접 예측한 것처럼 보임 |

## 4. Q&A 백업 자산

| 자산 파일명 | 유형 | 용도 |
| --- | --- | --- |
| `backup_data_sources.png` | 스크린샷/도식 | 데이터 출처 질문 대응 |
| `backup_risk_formula.png` | 도식 | 최종 위험도 산식 질문 대응 |
| `backup_validation_report.png` | 스크린샷 | 최종 검증 PASS 질문 대응 |
| `backup_github_actions_success.png` | 스크린샷 | 자동 갱신 운영 질문 대응 |
| `backup_deep_ai_basis.png` | 스크린샷 | Live 기준과 AI 비교 기준 분리 질문 대응 |
| `backup_watchlist_facility_logic.png` | 스크린샷/도식 | 시설 점검 우선점수 질문 대응 |

## 5. 발표 중 UI 표현 가이드

사용할 표현:

- "사이드바에서는 분석 기준, 지역 범위, 위험등급을 조정합니다."
- "상세 확인 대상은 각 탭 내부 selectbox에서 선택합니다."
- "`위험 구성요소 확인 대상`은 `위험도 분석` 탭 안에 있습니다."
- "`대체 수원 확인 대상`은 `점검·대체수원` 탭 안에 있습니다."
- "시설 점검 우선점수는 시설별 실시간 저수율이 아니라, 시·군 저수율 위험도와 시설 규모 정보를 결합한 행정 점검 우선순위입니다."
- "Live 데이터와 AI 해석 기준은 분리해서 표시합니다."

피할 표현:

- "사이드바에서 당진시를 선택합니다."
- "AI가 2026년 5월 데이터를 직접 예측했습니다."
- "시설별 실시간 저수율을 기반으로 시설 점검 순위를 만들었습니다."
- "모든 기능을 하나씩 설명하겠습니다."

## 6. 슬라이드 제작 체크리스트

- 각 슬라이드에는 제목 1개, 핵심 문장 1개, 자산 영역 1개만 둡니다.
- 큰 원본 표는 본문 슬라이드에 넣지 않습니다.
- 영상 슬라이드에는 영상 아래에 한 줄 해석만 둡니다.
- Q&A 백업 슬라이드는 발표 본문 뒤에 숨김 슬라이드로 둡니다.
- 한글 폰트는 `맑은 고딕`, `NanumGothic`, `Noto Sans CJK KR` 중 하나를 사용합니다.
- 영상 파일명은 이 문서의 파일명과 정확히 일치시킵니다.
