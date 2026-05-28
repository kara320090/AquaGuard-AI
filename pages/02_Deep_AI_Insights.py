from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

ROOT = Path(__file__).resolve().parents[1]
REPORT_TABLES = ROOT / "reports" / "tables"
PROCESSED = ROOT / "data" / "processed"
META = ROOT / "data" / "metadata"

SUMMARY_PATH = REPORT_TABLES / "ai_sigungu_deep_summary.csv"
FORECAST_PATH = PROCESSED / "ai_gru_reservoir_forecast_by_sigungu.csv"
ANOMALY_PATH = PROCESSED / "ai_autoencoder_anomaly_by_sigungu.csv"
REPORT_PATH = META / "deep_ai_model_report.md"

st.set_page_config(
    page_title="Deep AI 예측·이상탐지",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Deep AI 저수율 예측·이상탐지")
st.caption("PyTorch GRU 기반 7일 후 저수율 예측과 AutoEncoder 기반 이상 패턴 탐지 결과입니다.")

st.info(
    "과거 공공데이터는 AI 모델의 기준 패턴 학습에 사용하고, "
    "향후 올담 API의 오늘 또는 최근 1개월 데이터는 최신 추론 입력으로 사용할 수 있습니다."
)


@st.cache_data
def load_data():
    if not SUMMARY_PATH.exists():
        return None, None, None, ""

    summary = pd.read_csv(SUMMARY_PATH)
    forecast = pd.read_csv(FORECAST_PATH) if FORECAST_PATH.exists() else pd.DataFrame()
    anomaly = pd.read_csv(ANOMALY_PATH) if ANOMALY_PATH.exists() else pd.DataFrame()
    report = REPORT_PATH.read_text(encoding="utf-8") if REPORT_PATH.exists() else ""

    return summary, forecast, anomaly, report


summary, forecast, anomaly, report = load_data()

if summary is None:
    st.error("AI 결과 파일이 없습니다. 먼저 python scripts\\13_train_deep_reservoir_ai.py 를 실행하세요.")
    st.stop()

top = summary.sort_values("deep_ai_risk_score", ascending=False).iloc[0]
avg_ai = summary["deep_ai_risk_score"].mean()
warning_count = summary[summary["deep_ai_risk_level"].isin(["주의", "경계", "심각"])].shape[0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("AI 평균 위험도", f"{avg_ai:.1f}점")
c2.metric("AI 주의 이상", f"{warning_count}곳")
c3.metric("AI 최고 위험 지역", str(top["sigungu"]))
c4.metric("AI 분석 시·군", f"{len(summary)}곳")

st.markdown("### 시·군별 Deep AI 위험도")

fig = px.bar(
    summary.sort_values("deep_ai_risk_score", ascending=False),
    x="sigungu",
    y="deep_ai_risk_score",
    color="deep_ai_risk_level",
    hover_data=[
        "current_avg_reservoir_rate",
        "pred_avg_reservoir_rate_7d",
        "forecast_drop_7d",
        "forecast_risk_score",
        "autoencoder_anomaly_score",
    ],
    title="GRU 예측 + AutoEncoder 이상탐지 기반 Deep AI 위험도",
)

fig.update_layout(
    xaxis_title="시·군",
    yaxis_title="Deep AI 위험도",
    height=500,
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("### Deep AI 시·군 요약")

display_summary = summary.rename(columns={
    "deep_ai_rank": "AI 순위",
    "sigungu": "시·군",
    "deep_ai_risk_score": "Deep AI 위험도",
    "deep_ai_risk_level": "Deep AI 단계",
    "current_avg_reservoir_rate": "현재 평균 저수율",
    "pred_avg_reservoir_rate_7d": "7일 후 예측 저수율",
    "forecast_drop_7d": "예상 하락폭",
    "forecast_risk_score": "예측 위험도",
    "forecast_risk_level": "예측 단계",
    "autoencoder_anomaly_score": "AutoEncoder 이상점수",
    "autoencoder_anomaly_level": "이상탐지 단계",
    "base_date": "기준일",
    "target_date": "예측일",
})

st.dataframe(display_summary, use_container_width=True, hide_index=True)

st.markdown("### GRU 7일 후 저수율 예측 결과")

if not forecast.empty:
    display_forecast = forecast.rename(columns={
        "sigungu": "시·군",
        "base_date": "기준일",
        "target_date": "예측일",
        "current_avg_reservoir_rate": "현재 평균 저수율",
        "pred_avg_reservoir_rate_7d": "7일 후 예측 저수율",
        "forecast_drop_7d": "예상 하락폭",
        "forecast_risk_score": "예측 위험도",
        "forecast_risk_level": "예측 단계",
    })

    st.dataframe(display_forecast, use_container_width=True, hide_index=True)

    st.download_button(
        label="GRU 예측 결과 CSV 다운로드",
        data=forecast.to_csv(index=False).encode("utf-8-sig"),
        file_name="ai_gru_reservoir_forecast_by_sigungu.csv",
        mime="text/csv",
        use_container_width=True,
    )
else:
    st.warning("GRU 예측 결과 파일이 없습니다.")

st.markdown("### AutoEncoder 이상탐지 결과")

if not anomaly.empty:
    display_anomaly = anomaly.rename(columns={
        "sigungu": "시·군",
        "base_date": "기준일",
        "current_avg_reservoir_rate": "현재 평균 저수율",
        "reconstruction_error": "재구성 오차",
        "autoencoder_anomaly_score": "이상점수",
        "autoencoder_anomaly_level": "이상 단계",
    })

    st.dataframe(display_anomaly, use_container_width=True, hide_index=True)

    st.download_button(
        label="AutoEncoder 이상탐지 결과 CSV 다운로드",
        data=anomaly.to_csv(index=False).encode("utf-8-sig"),
        file_name="ai_autoencoder_anomaly_by_sigungu.csv",
        mime="text/csv",
        use_container_width=True,
    )
else:
    st.warning("AutoEncoder 이상탐지 결과 파일이 없습니다.")

st.markdown("### AI 모델 리포트")

if report:
    st.markdown(report)
else:
    st.info("AI 모델 리포트가 없습니다.")

st.warning(
    "Deep AI 결과는 공개데이터 기반 예측·이상탐지 참고자료입니다. "
    "실제 농업용수 대응 여부는 현장 저수율, 관로, 수리권, 수질, 행정 협의를 함께 검토해야 합니다."
)
