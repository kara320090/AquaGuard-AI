from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px

ROOT = Path(__file__).resolve().parents[1]
REPORT_TABLES = ROOT / "reports" / "tables"
PROCESSED = ROOT / "data" / "processed"

WATCHLIST_PATH = REPORT_TABLES / "reservoir_watchlist.csv"
FACILITY_STATUS_PATH = PROCESSED / "reservoir_facility_status_for_dashboard.csv"
OLDAM_TODAY_PATH = PROCESSED / "latest_oldam_reservoir_today.csv"

st.set_page_config(
    page_title="저수지 현황 및 이상탐지",
    page_icon="🚰",
    layout="wide",
)

st.title("🚰 저수지 현황 및 이상탐지 Watchlist")
st.caption("시·군별 저수율 위험 후보와 저수지 시설 현황을 확인하는 MVP 보조 화면입니다.")

st.info(
    "MVP 단계에서는 공개데이터 기반 규칙형 Watchlist로 구현했습니다. "
    "실제 현장 이상 여부를 확정하는 기능이 아니라 우선 점검 후보를 좁히는 행정 참고자료입니다."
)


@st.cache_data
def load_data():
    if not WATCHLIST_PATH.exists():
        raise FileNotFoundError(f"Watchlist 파일이 없습니다: {WATCHLIST_PATH}")
    if not FACILITY_STATUS_PATH.exists():
        raise FileNotFoundError(f"저수지 현황 파일이 없습니다: {FACILITY_STATUS_PATH}")

    watch = pd.read_csv(WATCHLIST_PATH)
    facility = pd.read_csv(FACILITY_STATUS_PATH)

    if OLDAM_TODAY_PATH.exists():
        oldam_today = pd.read_csv(OLDAM_TODAY_PATH)
    else:
        oldam_today = pd.DataFrame()

    return watch, facility, oldam_today


watch, facility, oldam_today = load_data()

sigungu_list = watch.sort_values("watch_rank")["sigungu"].tolist()

selected = st.sidebar.selectbox("시·군 선택", sigungu_list, index=0)

selected_watch = watch[watch["sigungu"] == selected].iloc[0]
selected_facility = facility[facility["sigungu"] == selected].copy()

# 화면 표시 전 중복 제거: 동일 시군 + 저수지명 + 주소가 반복되는 원천 데이터 방어
removed_duplicates = 0
if not selected_facility.empty:
    for c in ["facility_name", "address", "sigungu"]:
        if c in selected_facility.columns:
            selected_facility[c] = selected_facility[c].fillna("").astype(str).str.strip()

    dedup_cols = [c for c in ["sigungu", "facility_name", "address"] if c in selected_facility.columns]
    if dedup_cols:
        before_rows = len(selected_facility)
        selected_facility = selected_facility.drop_duplicates(subset=dedup_cols, keep="first").copy()
        removed_duplicates = before_rows - len(selected_facility)

if not selected_facility.empty:
    dedup_cols = [c for c in ["facility_name", "address"] if c in selected_facility.columns]
    if dedup_cols:
        before_rows = len(selected_facility)
        selected_facility = selected_facility.drop_duplicates(dedup_cols, keep="first").copy()
        removed_duplicates = before_rows - len(selected_facility)
    else:
        removed_duplicates = 0
else:
    removed_duplicates = 0

if not oldam_today.empty and "sigungu" in oldam_today.columns:
    selected_oldam = oldam_today[oldam_today["sigungu"] == selected].copy()
else:
    selected_oldam = pd.DataFrame()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Watch 순위", f"{int(selected_watch['watch_rank'])}위")
c2.metric("Watch 단계", str(selected_watch["watch_level"]))
c3.metric("평균 저수율", f"{selected_watch.get('avg_reservoir_rate', 0):.1f}")
c4.metric("저수율 위험도", f"{selected_watch.get('reservoir_risk_score', 0):.1f}")

st.markdown(f"### {selected} Watch 판단 근거")

reason = str(selected_watch.get("watch_reason", ""))
min_rate = selected_watch.get("min_reservoir_rate", None)
avg_rate = selected_watch.get("avg_reservoir_rate", None)

st.info(
    f"{selected}의 Watch 단계는 '{selected_watch['watch_level']}'입니다. "
    f"판정 사유는 '{reason}'입니다. "
    f"시·군 평균 저수율은 {avg_rate:.1f}%, 최저 저수율은 {min_rate:.1f}%입니다."
    if pd.notna(avg_rate) and pd.notna(min_rate)
    else f"{selected}의 Watch 단계는 '{selected_watch['watch_level']}'입니다. 판정 사유는 '{reason}'입니다."
)

if not selected_oldam.empty:
    st.markdown("#### 올담 최신 공개 저수지 기준 현황")
    oldam_cols = [
        "facility_name",
        "reservoir_rate",
        "date",
        "location_raw",
        "facility_match_status",
    ]
    oldam_cols = [c for c in oldam_cols if c in selected_oldam.columns]
    oldam_view = selected_oldam[oldam_cols].copy().rename(columns={
        "facility_name": "저수지명",
        "reservoir_rate": "최신 공개 저수율",
        "date": "기준일",
        "location_raw": "위치",
        "facility_match_status": "매칭 상태",
    })
    st.dataframe(
        oldam_view.sort_values("최신 공개 저수율", ascending=True),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.caption("이 시·군은 올담 최신 공개 저수지 snapshot에는 직접 포함되지 않아, 정적 저수지 시설 데이터와 시·군 평균 지표로 Watch 단계를 해석합니다.")

st.markdown("### 시·군별 저수율 Watchlist")

view_cols = [
    "watch_rank",
    "sigungu",
    "watch_level",
    "watch_reason",
    "avg_reservoir_rate",
    "min_reservoir_rate",
    "low_reservoir_count_40",
    "low_reservoir_count_30",
    "reservoir_risk_score",
    "final_water_risk_score",
]

view_cols = [c for c in view_cols if c in watch.columns]

st.dataframe(
    watch[view_cols].rename(columns={
        "watch_rank": "순위",
        "sigungu": "시·군",
        "watch_level": "Watch 단계",
        "watch_reason": "판정 사유",
        "avg_reservoir_rate": "평균 저수율",
        "min_reservoir_rate": "최저 저수율",
        "low_reservoir_count_40": "40% 이하 저수지 수",
        "low_reservoir_count_30": "30% 이하 저수지 수",
        "reservoir_risk_score": "저수율 위험도",
        "final_water_risk_score": "최종 위험도",
    }),
    use_container_width=True,
    hide_index=True,
)

fig = px.bar(
    watch.sort_values("reservoir_risk_score", ascending=False),
    x="sigungu",
    y="reservoir_risk_score",
    color="watch_level",
    hover_data=["watch_reason", "avg_reservoir_rate", "min_reservoir_rate"],
    title="시·군별 저수율 위험도 Watchlist",
)
st.plotly_chart(fig, use_container_width=True)

st.markdown(f"### {selected} 저수지 시설 현황")

if selected_facility.empty:
    st.warning("선택한 시·군의 저수지 시설 정보가 없습니다.")
else:
    table_cols = [
        "facility_name",
        "address",
        "benefit_area",
        "effective_capacity",
        "total_capacity",
        "sigungu_avg_reservoir_rate",
        "sigungu_min_reservoir_rate",
        "sigungu_reservoir_risk_score",
        "inspection_priority_score",
        "reservoir_status_note",
    ]
    table_cols = [c for c in table_cols if c in selected_facility.columns]

    display = selected_facility[table_cols].copy()
    display = display.rename(columns={
        "facility_name": "저수지명",
        "address": "주소",
        "benefit_area": "수혜면적",
        "effective_capacity": "유효저수량",
        "total_capacity": "총저수량",
        "sigungu_avg_reservoir_rate": "시군 평균 저수율",
        "sigungu_min_reservoir_rate": "시군 최저 저수율",
        "sigungu_reservoir_risk_score": "시군 저수율 위험도",
        "inspection_priority_score": "시설 점검 우선점수",
        "reservoir_status_note": "상태 메모",
    })

    st.caption(f"중복 제거 후 표시 시설 수: {len(display)}개" + (f" · 제거된 중복 행: {removed_duplicates}개" if removed_duplicates else ""))

    top_display = display.sort_values("시설 점검 우선점수", ascending=False).head(10)
    st.markdown("#### 점검 우선 시설 TOP 10")
    st.dataframe(
        top_display,
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("전체 저수지 시설 현황 보기"):
        st.dataframe(
            display.sort_values("시설 점검 우선점수", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    st.download_button(
        label=f"{selected} 저수지 현황 CSV 다운로드",
        data=selected_facility.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{selected}_reservoir_facility_status.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.warning(
    "저수지 개별 실시간 저수율은 원천 데이터 한계로 일부 제한될 수 있으므로, "
    "MVP에서는 시·군 평균 저수율과 시설 현황을 함께 표시합니다."
)
