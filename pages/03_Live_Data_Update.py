from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORT_TABLES = ROOT / "reports" / "tables"
META = ROOT / "data" / "metadata"

LIVE_FEATURE_PATH = PROCESSED / "latest_live_sigungu_features.csv"
LIVE_SUMMARY_PATH = REPORT_TABLES / "latest_live_risk_summary.csv"
LIVE_STATUS_PATH = REPORT_TABLES / "latest_live_data_status.csv"
OLDAM_STATUS_PATH = REPORT_TABLES / "latest_oldam_status_summary.csv"
KMA_STATUS_PATH = REPORT_TABLES / "latest_kma_weather_status.csv"
METHOD_PATH = META / "live_feature_method.md"

st.set_page_config(
    page_title="Live 데이터 갱신",
    page_icon="🔄",
    layout="wide",
)

st.title("🔄 Live 데이터 갱신 위험도")
st.caption("올담 최신 저수지 수위 데이터와 기상청 ASOS 최근 30일 강수량을 반영한 현재 기준 위험도입니다.")


@st.cache_data
def load_data():
    live = pd.read_csv(LIVE_FEATURE_PATH) if LIVE_FEATURE_PATH.exists() else pd.DataFrame()
    summary = pd.read_csv(LIVE_SUMMARY_PATH) if LIVE_SUMMARY_PATH.exists() else pd.DataFrame()
    status = pd.read_csv(LIVE_STATUS_PATH) if LIVE_STATUS_PATH.exists() else pd.DataFrame()
    oldam_status = pd.read_csv(OLDAM_STATUS_PATH) if OLDAM_STATUS_PATH.exists() else pd.DataFrame()
    kma_status = pd.read_csv(KMA_STATUS_PATH) if KMA_STATUS_PATH.exists() else pd.DataFrame()
    method = METHOD_PATH.read_text(encoding="utf-8") if METHOD_PATH.exists() else ""
    return live, summary, status, oldam_status, kma_status, method


live, summary, status, oldam_status, kma_status, method = load_data()

if summary.empty:
    st.error("Live 위험도 결과가 없습니다. 먼저 scripts/16_build_live_features.py를 실행하세요.")
    st.stop()

top = summary.sort_values("final_live_water_risk_score", ascending=False).iloc[0]
avg_score = summary["final_live_water_risk_score"].mean()
warning_count = summary[summary["final_live_water_risk_level"].isin(["주의", "경계", "심각"])].shape[0]

oldam_count = int((live["live_reservoir_source"] == "OLDAM_TODAY").sum()) if "live_reservoir_source" in live.columns else 0
kma_count = int((live["live_weather_source"] != "BASELINE_WEATHER").sum()) if "live_weather_source" in live.columns else 0
soil_count = int((live["live_soil_source"] == "ADMS_SOIL_AUTO").sum()) if "live_soil_source" in live.columns else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Live 평균 위험도", f"{avg_score:.1f}점")
c2.metric("Live 주의 이상", f"{warning_count}곳")
c3.metric("Live 최고 위험 지역", str(top["sigungu"]))
c4.metric("최신 데이터 반영", f"저수지 {oldam_count} / 기상 {kma_count} / 토양 {soil_count}곳")

st.markdown("### 데이터 수집 상태")

s1, s2, s3 = st.columns(3)

with s1:
    st.markdown("#### Live Feature")
    if not status.empty:
        st.dataframe(status, use_container_width=True, hide_index=True)
    else:
        st.info("Live feature status 없음")

with s2:
    st.markdown("#### 올담 저수지")
    if not oldam_status.empty:
        st.dataframe(oldam_status, use_container_width=True, hide_index=True)
    else:
        st.info("올담 status 없음")

with s3:
    st.markdown("#### 기상청 ASOS")
    if not kma_status.empty:
        st.dataframe(kma_status, use_container_width=True, hide_index=True)
    else:
        st.info("기상청 status 없음")

st.markdown("### 현재 기준 Live 위험도 순위")

fig = px.bar(
    summary.sort_values("final_live_water_risk_score", ascending=False),
    x="sigungu",
    y="final_live_water_risk_score",
    color="final_live_water_risk_level",
    hover_data=[
        "live_score_delta_from_baseline",
        "live_main_risk_driver",
        "live_weather_source",
        "live_reservoir_source",
        "live_soil_source",
    ],
    title="올담 저수지 + 기상청 최근 30일 강수량 반영 Live 위험도",
)
fig.update_layout(
    xaxis_title="시·군",
    yaxis_title="Live 위험도",
    height=520,
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("### 기존 위험도 대비 Live 위험도 변화")

if "live_score_delta_from_baseline" in summary.columns:
    delta_fig = px.bar(
        summary.sort_values("live_score_delta_from_baseline", ascending=False),
        x="sigungu",
        y="live_score_delta_from_baseline",
        color="final_live_water_risk_level",
        hover_data=["final_water_risk_score", "final_live_water_risk_score"],
        title="기존 기준 대비 Live 위험도 변화",
    )
    delta_fig.update_layout(
        xaxis_title="시·군",
        yaxis_title="Live - Baseline",
        height=500,
    )
    st.plotly_chart(delta_fig, use_container_width=True)

display = summary.rename(columns={
    "final_live_priority_rank": "Live 순위",
    "sigungu": "시·군",
    "final_live_water_risk_score": "Live 위험도",
    "final_live_water_risk_level": "Live 단계",
    "live_score_delta_from_baseline": "기존 대비 변화",
    "live_main_risk_driver": "주요 원인",
    "live_weather_source": "기상 데이터 소스",
    "live_reservoir_source": "저수지 데이터 소스",
    "live_soil_source": "토양수분 데이터 소스",
    "soil_data_date": "토양수분 기준일",
    "soil_moisture_avg": "토양유효수분",
    "soil_moisture_drought_score": "토양수분 가뭄도",
    "today_avg_reservoir_rate": "오늘 평균 저수율",
    "today_min_reservoir_rate": "오늘 최저 저수율",
    "today_reservoir_count": "올담 반영 저수지 수",
    "rainfall_30d": "최근 30일 강수량",
    "rainfall_7d": "최근 7일 강수량",
    "latest_rain_shortage_score": "최신 강우 부족도",
    "final_water_risk_score": "기존 위험도",
    "final_water_risk_level": "기존 단계",
})

st.dataframe(display, use_container_width=True, hide_index=True)

st.download_button(
    label="Live 위험도 결과 CSV 다운로드",
    data=summary.to_csv(index=False).encode("utf-8-sig"),
    file_name="latest_live_risk_summary.csv",
    mime="text/csv",
    use_container_width=True,
)

st.markdown("### 산식 및 해석 기준")

if method:
    st.markdown(method)
else:
    st.info("Live feature method 문서가 없습니다.")

st.warning(
    "올담 저수지 데이터는 하루치 snapshot이므로 현재 저수율 갱신에는 사용 가능하지만, "
    "GRU 30일 시계열 추론에는 매일 snapshot이 30일 이상 누적된 이후 연결하는 것이 안전합니다."
)


st.markdown("### 저수율 교차검증: 올담 vs ADMS")

crosscheck_path = REPORT_TABLES / "latest_reservoir_source_crosscheck.csv"
adms_reservoir_status_path = REPORT_TABLES / "latest_adms_reservoir_support_status.csv"

if adms_reservoir_status_path.exists():
    adms_status = pd.read_csv(adms_reservoir_status_path)
    st.caption("ADMS My 지역정보 기반 시·군별 저수율 보조 데이터 수집 상태")
    st.dataframe(adms_status, use_container_width=True, hide_index=True)

if crosscheck_path.exists():
    cross = pd.read_csv(crosscheck_path)
    show_cols = [
        "sigungu",
        "adms_rvow",
        "adms_normal_rvow",
        "adms_normal_ratio",
        "oldam_avg_reservoir_rate",
        "rvow_diff_oldam_minus_adms",
        "crosscheck_status",
    ]
    show_cols = [c for c in show_cols if c in cross.columns]
    renamed = cross[show_cols].rename(columns={
        "sigungu": "시·군",
        "adms_rvow": "ADMS 저수율",
        "adms_normal_rvow": "ADMS 평년저수율",
        "adms_normal_ratio": "평년대비 저수율",
        "oldam_avg_reservoir_rate": "올담 평균 저수율",
        "rvow_diff_oldam_minus_adms": "올담-ADMS 차이",
        "crosscheck_status": "교차검증 상태",
    })
    st.dataframe(renamed, use_container_width=True, hide_index=True)
else:
    st.info("저수율 교차검증 파일이 없습니다. scripts/18_fetch_adms_reservoir_support.py를 실행하세요.")
