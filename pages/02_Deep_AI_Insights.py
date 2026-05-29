from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
REPORT_TABLES = ROOT / "reports" / "tables"
PROCESSED = ROOT / "data" / "processed"
META = ROOT / "data" / "metadata"

SUMMARY_PATH = REPORT_TABLES / "ai_sigungu_deep_summary.csv"
FORECAST_PATH = PROCESSED / "ai_gru_reservoir_forecast_by_sigungu.csv"
ANOMALY_PATH = PROCESSED / "ai_autoencoder_anomaly_by_sigungu.csv"
GRU_HISTORY_PATH = REPORT_TABLES / "ai_gru_training_history.csv"
AE_HISTORY_PATH = REPORT_TABLES / "ai_autoencoder_training_history.csv"
REPORT_PATH = META / "deep_ai_model_report.md"

RISK_COLORS = {
    "심각": "#c62828",
    "심각후보": "#d84315",
    "경계": "#ef6c00",
    "주의": "#f9a825",
    "낮음": "#2e7d32",
}

st.set_page_config(
    page_title="Deep AI 예측·이상탐지",
    page_icon="🧠",
    layout="wide",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.6rem; padding-bottom: 3rem; }
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 14px 16px;
        }
        .ag-section {
            margin-top: 1.6rem;
            padding-top: 0.5rem;
            border-top: 1px solid #eef2f7;
        }
        .ag-filter {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 0.75rem 0.9rem;
            margin: 0.4rem 0 1rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(title: str, service_definition: str, latest_basis: str) -> None:
    st.title(title)
    st.caption(service_definition)
    st.markdown(f"**최신 분석 기준:** {latest_basis}")


def render_section_header(title: str, description: str | None = None) -> None:
    st.markdown('<div class="ag-section"></div>', unsafe_allow_html=True)
    st.subheader(title)
    if description:
        st.caption(description)


def render_empty_state(message: str = "조건에 맞는 데이터가 없습니다") -> None:
    st.info(message)


def render_kpi_cards(cards: list[tuple[str, str, str | None]]) -> None:
    cols = st.columns(len(cards))
    for col, (label, value, help_text) in zip(cols, cards):
        col.metric(label, value, help=help_text)


def format_value(value, suffix: str = "", decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}{suffix}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):,.{decimals}f}{suffix}"
    return str(value) if str(value) else "N/A"


@st.cache_data(show_spinner=False)
def safe_read_data(path_text: str, required: bool = False) -> tuple[pd.DataFrame, str | None]:
    path = Path(path_text)
    if not path.exists():
        level = "필수" if required else "선택"
        return pd.DataFrame(), f"{level} 데이터 파일이 없습니다: {path}"
    try:
        return pd.read_csv(path), None
    except Exception as exc:  # noqa: BLE001
        return pd.DataFrame(), f"데이터 파일을 읽지 못했습니다: {path} ({exc})"


@st.cache_data(show_spinner=False)
def read_text_file(path_text: str) -> tuple[str, str | None]:
    path = Path(path_text)
    if not path.exists():
        return "", f"선택 문서 파일이 없습니다: {path}"
    try:
        return path.read_text(encoding="utf-8-sig"), None
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8"), None
    except Exception as exc:  # noqa: BLE001
        return "", f"문서 파일을 읽지 못했습니다: {path} ({exc})"


def show_messages(messages: list[str | None], stop_on_required: bool = False) -> None:
    has_required = False
    for message in messages:
        if not message:
            continue
        if message.startswith("필수") or "읽지 못했습니다" in message:
            has_required = has_required or message.startswith("필수")
            st.warning(message)
        else:
            st.info(message)
    if stop_on_required and has_required:
        st.stop()


def normalize_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def latest_date(df: pd.DataFrame, columns: list[str]) -> str:
    dates = []
    for col in columns:
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            dates.extend(parsed.dropna().tolist())
    if not dates:
        return "N/A"
    return max(dates).strftime("%Y-%m-%d")


def parse_report_metrics(report: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    mae = re.search(r"MAE:\s*([-+]?\d+(?:\.\d+)?)", report, flags=re.IGNORECASE)
    r2 = re.search(r"R2:\s*([-+]?\d+(?:\.\d+)?)", report, flags=re.IGNORECASE)
    sample_count = re.search(r"검증\s*샘플\s*수:\s*([\d,]+)", report)
    if mae:
        metrics["검증 MAE"] = float(mae.group(1))
    if r2:
        metrics["검증 R2"] = float(r2.group(1))
    if sample_count:
        metrics["검증 샘플 수"] = float(sample_count.group(1).replace(",", ""))
    return metrics


def build_performance_cards(report: str, gru_history: pd.DataFrame, ae_history: pd.DataFrame) -> tuple[str, str, list[tuple[str, str, str]]]:
    report_metrics = parse_report_metrics(report)
    cards: list[tuple[str, str, str]] = []
    if "검증 MAE" in report_metrics:
        cards.append(("GRU 검증 MAE", f"{report_metrics['검증 MAE']:.4f}", "낮을수록 7일 후 저수율 예측 오차가 작습니다."))
    if "검증 R2" in report_metrics:
        cards.append(("GRU 검증 R2", f"{report_metrics['검증 R2']:.4f}", "1에 가까울수록 검증 데이터의 변동을 잘 설명합니다."))
    if "검증 샘플 수" in report_metrics:
        cards.append(("검증 샘플 수", f"{int(report_metrics['검증 샘플 수']):,}", "성능 확인에 사용된 시계열 샘플 수입니다."))
    if not gru_history.empty and "valid_mae" in gru_history.columns:
        best_mae = pd.to_numeric(gru_history["valid_mae"], errors="coerce").min()
        if pd.notna(best_mae):
            cards.append(("학습 이력 최저 MAE", f"{best_mae:.4f}", "epoch별 검증 MAE 중 최저값입니다."))
    if not gru_history.empty and "valid_r2" in gru_history.columns:
        best_r2 = pd.to_numeric(gru_history["valid_r2"], errors="coerce").max()
        if pd.notna(best_r2):
            cards.append(("학습 이력 최고 R2", f"{best_r2:.4f}", "epoch별 검증 R2 중 최고값입니다."))
    if not ae_history.empty and "valid_recon_loss" in ae_history.columns:
        best_loss = pd.to_numeric(ae_history["valid_recon_loss"], errors="coerce").min()
        if pd.notna(best_loss):
            cards.append(("AutoEncoder 최저 재구성 손실", f"{best_loss:.4f}", "정상 패턴 복원 오차의 최저값입니다."))
    best_model = "PyTorch GRU" if "검증 MAE" in report_metrics or not gru_history.empty else ("PyTorch Sequence AutoEncoder" if not ae_history.empty else "N/A")
    main_metric = f"MAE {report_metrics['검증 MAE']:.4f}" if "검증 MAE" in report_metrics else (cards[0][1] if cards else "N/A")
    return best_model, main_metric, cards


def available_levels(summary: pd.DataFrame) -> list[str]:
    if summary.empty or "deep_ai_risk_level" not in summary.columns:
        return []
    found = summary["deep_ai_risk_level"].dropna().astype(str).unique().tolist()
    order = ["심각", "경계", "주의", "낮음"]
    return [x for x in order if x in found] + sorted([x for x in found if x not in order])


def render_sidebar(summary: pd.DataFrame) -> tuple[list[str], list[str], int, str]:
    regions = summary.sort_values("deep_ai_rank")["sigungu"].dropna().astype(str).tolist() if "deep_ai_rank" in summary.columns else sorted(summary["sigungu"].dropna().astype(str).tolist())
    levels = available_levels(summary)
    if "ai_region_filter" not in st.session_state:
        st.session_state.ai_region_filter = regions
    if "ai_level_filter" not in st.session_state:
        st.session_state.ai_level_filter = levels

    st.session_state.ai_region_filter = [x for x in st.session_state.ai_region_filter if x in regions] or regions
    st.session_state.ai_level_filter = [x for x in st.session_state.ai_level_filter if x in levels] or levels

    with st.sidebar:
        st.header("필터")
        if st.button("기본값으로 초기화", use_container_width=True):
            st.session_state.ai_region_filter = regions
            st.session_state.ai_level_filter = levels
            st.session_state.ai_top_n = 10
            st.rerun()

        st.markdown("#### 기간")
        st.caption(f"기준일/예측일: {latest_date(summary, ['base_date', 'target_date'])}")

        st.markdown("#### 지역/대상")
        selected_regions = st.multiselect("시·군", regions, key="ai_region_filter")
        focus = st.selectbox("상세 확인 대상", selected_regions or regions, index=0 if (selected_regions or regions) else None)

        st.markdown("#### 모델/위험등급")
        selected_levels = st.multiselect("Deep AI 등급", levels, key="ai_level_filter")
        max_top = max(5, min(14, len(summary))) if not summary.empty else 5
        top_n = st.slider("우선순위 표시 수", 5, max_top, min(st.session_state.get("ai_top_n", 10), max_top), key="ai_top_n")

    return selected_regions, selected_levels, top_n, focus


def build_priority_table(summary: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    cols = [
        "deep_ai_rank",
        "sigungu",
        "deep_ai_risk_score",
        "deep_ai_risk_level",
        "forecast_drop_7d",
        "forecast_risk_score",
        "autoencoder_anomaly_score",
        "autoencoder_anomaly_level",
    ]
    cols = [c for c in cols if c in summary.columns]
    out = summary.sort_values("deep_ai_rank").head(top_n)[cols].copy()
    return out.rename(
        columns={
            "deep_ai_rank": "AI 순위",
            "sigungu": "시·군",
            "deep_ai_risk_score": "Deep AI 위험도",
            "deep_ai_risk_level": "Deep AI 등급",
            "forecast_drop_7d": "예상 하락폭",
            "forecast_risk_score": "예측 위험도",
            "autoencoder_anomaly_score": "이상점수",
            "autoencoder_anomaly_level": "이상탐지 등급",
        }
    )


def make_ai_ranking_chart(summary: pd.DataFrame, top_n: int):
    if summary.empty or "deep_ai_risk_score" not in summary.columns:
        return None
    plot_df = summary.sort_values("deep_ai_risk_score", ascending=False).head(top_n)
    fig = px.bar(
        plot_df.sort_values("deep_ai_risk_score"),
        x="deep_ai_risk_score",
        y="sigungu",
        orientation="h",
        color="deep_ai_risk_level" if "deep_ai_risk_level" in plot_df.columns else None,
        color_discrete_map=RISK_COLORS,
        hover_data=[c for c in ["forecast_risk_score", "autoencoder_anomaly_score", "forecast_drop_7d"] if c in plot_df.columns],
        title="Deep AI 위험도 순위: 예측·이상탐지 기준 어디가 먼저인가?",
    )
    fig.update_layout(xaxis_title="Deep AI 위험도", yaxis_title="시·군", height=max(380, 34 * len(plot_df) + 140))
    return fig


def make_forecast_scatter(summary: pd.DataFrame):
    needed = {"current_avg_reservoir_rate", "pred_avg_reservoir_rate_7d", "sigungu"}
    if summary.empty or not needed.issubset(summary.columns):
        return None
    fig = px.scatter(
        summary,
        x="current_avg_reservoir_rate",
        y="pred_avg_reservoir_rate_7d",
        color="deep_ai_risk_level" if "deep_ai_risk_level" in summary.columns else None,
        size="deep_ai_risk_score" if "deep_ai_risk_score" in summary.columns else None,
        hover_name="sigungu",
        color_discrete_map=RISK_COLORS,
        title="7일 후 저수율 예측: 현재보다 내려가는 지역은 어디인가?",
    )
    fig.add_shape(type="line", x0=0, y0=0, x1=100, y1=100, line=dict(color="#94a3b8", dash="dash"))
    fig.update_layout(xaxis_title="현재 평균 저수율", yaxis_title="7일 후 예측 저수율", height=420)
    return fig


def make_anomaly_chart(summary: pd.DataFrame, top_n: int):
    if summary.empty or "autoencoder_anomaly_score" not in summary.columns:
        return None
    plot_df = summary.sort_values("autoencoder_anomaly_score", ascending=False).head(top_n)
    fig = px.bar(
        plot_df.sort_values("autoencoder_anomaly_score"),
        x="autoencoder_anomaly_score",
        y="sigungu",
        orientation="h",
        color="autoencoder_anomaly_level" if "autoencoder_anomaly_level" in plot_df.columns else None,
        color_discrete_map=RISK_COLORS,
        title="AutoEncoder 이상점수: 평소 패턴과 다른 지역은 어디인가?",
    )
    fig.update_layout(xaxis_title="이상점수", yaxis_title="시·군", height=max(360, 32 * len(plot_df) + 120))
    return fig


def make_training_chart(gru_history: pd.DataFrame):
    if gru_history.empty or not {"epoch", "valid_mae"}.issubset(gru_history.columns):
        return None
    fig = px.line(
        gru_history,
        x="epoch",
        y="valid_mae",
        title="GRU 검증 MAE 추이: 학습 중 예측 오차는 어떻게 변했는가?",
    )
    fig.update_layout(xaxis_title="Epoch", yaxis_title="검증 MAE", height=320)
    return fig


def main() -> None:
    inject_css()

    summary, summary_msg = safe_read_data(str(SUMMARY_PATH), required=True)
    forecast, forecast_msg = safe_read_data(str(FORECAST_PATH), required=False)
    anomaly, anomaly_msg = safe_read_data(str(ANOMALY_PATH), required=False)
    gru_history, gru_msg = safe_read_data(str(GRU_HISTORY_PATH), required=False)
    ae_history, ae_msg = safe_read_data(str(AE_HISTORY_PATH), required=False)
    report, report_msg = read_text_file(str(REPORT_PATH))

    show_messages([summary_msg], stop_on_required=True)
    if summary.empty:
        render_empty_state("Deep AI 요약 결과가 비어 있습니다.")
        st.stop()

    summary = normalize_numeric(
        summary,
        [
            "deep_ai_rank",
            "deep_ai_risk_score",
            "current_avg_reservoir_rate",
            "pred_avg_reservoir_rate_7d",
            "forecast_drop_7d",
            "forecast_risk_score",
            "autoencoder_anomaly_score",
        ],
    )
    forecast = normalize_numeric(forecast, ["current_avg_reservoir_rate", "pred_avg_reservoir_rate_7d", "forecast_drop_7d", "forecast_risk_score"])
    anomaly = normalize_numeric(anomaly, ["reconstruction_error", "autoencoder_anomaly_score"])
    gru_history = normalize_numeric(gru_history, ["epoch", "train_loss", "valid_mae", "valid_r2"])
    ae_history = normalize_numeric(ae_history, ["epoch", "train_recon_loss", "valid_recon_loss"])

    best_model, main_metric, performance_cards = build_performance_cards(report, gru_history, ae_history)
    latest_basis = latest_date(summary, ["base_date", "target_date"])

    render_page_header(
        "Deep AI 저수율 예측·이상탐지",
        "GRU 7일 후 저수율 예측과 AutoEncoder 이상 패턴 탐지를 결합해 학습 기반 위험 신호를 확인합니다.",
        latest_basis,
    )

    selected_regions, selected_levels, top_n, focus = render_sidebar(summary)
    filtered = summary.copy()
    if selected_regions:
        filtered = filtered[filtered["sigungu"].isin(selected_regions)]
    if selected_levels and "deep_ai_risk_level" in filtered.columns:
        filtered = filtered[filtered["deep_ai_risk_level"].isin(selected_levels)]

    st.markdown(
        f"""
        <div class="ag-filter">
        현재 필터: 지역 <b>{len(selected_regions) if selected_regions else 0}개</b> ·
        Deep AI 등급 <b>{", ".join(selected_levels) if selected_levels else "전체"}</b> · 상세 대상 <b>{focus or "N/A"}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    warning_count = int(filtered["deep_ai_risk_level"].astype(str).isin(["주의", "경계", "심각"]).sum()) if "deep_ai_risk_level" in filtered.columns and not filtered.empty else 0
    top = filtered.sort_values("deep_ai_risk_score", ascending=False).iloc[0] if not filtered.empty and "deep_ai_risk_score" in filtered.columns else None

    render_kpi_cards(
        [
            ("AI 분석 시·군", f"{len(filtered):,}곳", "현재 필터 기준 Deep AI 결과 수입니다."),
            ("주의 이상 대상", f"{warning_count:,}곳", "AI 위험등급이 주의 이상인 대상 수입니다."),
            ("AI 최고 위험 지역", str(top["sigungu"]) if top is not None else "N/A", "현재 필터 기준 Deep AI 위험도가 가장 높은 지역입니다."),
            ("최고 성능 모델", best_model, "성능 검증 파일에서 확인 가능한 모델입니다."),
            ("주요 성능 지표", main_metric, "존재하는 성능 지표만 표시합니다."),
        ]
    )

    if filtered.empty:
        render_empty_state()
        st.stop()

    render_section_header("점검 우선순위", "AI 예측과 이상탐지 결과를 결합해 먼저 확인할 지역입니다.")
    priority = build_priority_table(filtered, top_n)
    if priority.empty:
        render_empty_state()
    else:
        st.dataframe(priority, use_container_width=True, hide_index=True)

    render_section_header("성능 검증 요약", "리포트와 학습 이력에 실제 존재하는 성능 지표만 표시합니다.")
    if performance_cards:
        render_kpi_cards(performance_cards[:5])
        if len(performance_cards) > 5:
            render_kpi_cards(performance_cards[5:])
    else:
        st.info("성능 검증 지표 컬럼 또는 리포트 항목이 없어 표시할 수 있는 지표가 없습니다.")

    train_fig = make_training_chart(gru_history)
    if train_fig:
        st.plotly_chart(train_fig, use_container_width=True)
        st.caption("검증 MAE가 낮을수록 7일 후 저수율 예측 오차가 작습니다.")
    else:
        st.info("epoch 또는 valid_mae 컬럼이 없어 GRU 학습 추이 차트를 건너뛰었습니다.")

    render_section_header("예측·이상탐지 분석", "각 차트는 위험 순위, 예측 변화, 이상 패턴을 분리해서 보여줍니다.")
    left, right = st.columns(2)
    with left:
        ranking_fig = make_ai_ranking_chart(filtered, top_n)
        if ranking_fig:
            st.plotly_chart(ranking_fig, use_container_width=True)
            st.caption("Deep AI 위험도는 GRU 예측 위험도와 AutoEncoder 이상점수를 결합한 결과입니다.")
        else:
            st.info("deep_ai_risk_score 컬럼이 없어 순위 차트를 건너뛰었습니다.")
    with right:
        forecast_fig = make_forecast_scatter(filtered)
        if forecast_fig:
            st.plotly_chart(forecast_fig, use_container_width=True)
            st.caption("점선보다 아래에 있으면 7일 후 예측 저수율이 현재보다 낮습니다.")
        else:
            st.info("현재/예측 저수율 컬럼이 없어 예측 산점도를 건너뛰었습니다.")

    anomaly_fig = make_anomaly_chart(filtered, top_n)
    if anomaly_fig:
        st.plotly_chart(anomaly_fig, use_container_width=True)
        st.caption("이상점수가 높을수록 과거 정상 패턴과 다른 저수율 흐름입니다.")
    else:
        st.info("autoencoder_anomaly_score 컬럼이 없어 이상탐지 차트를 건너뛰었습니다.")

    render_section_header(f"{focus} 상세 해석", "선택한 지역의 예측값과 이상탐지 결과를 한 줄로 설명합니다.")
    focus_row = summary[summary["sigungu"] == focus]
    if focus_row.empty:
        render_empty_state()
    else:
        row = focus_row.iloc[0]
        st.success(
            f"{focus}의 Deep AI 위험도는 {format_value(row.get('deep_ai_risk_score'), '점', 1)}이며 "
            f"등급은 {row.get('deep_ai_risk_level', 'N/A')}입니다. "
            f"현재 평균 저수율 {format_value(row.get('current_avg_reservoir_rate'), '%', 1)}에서 "
            f"7일 후 {format_value(row.get('pred_avg_reservoir_rate_7d'), '%', 1)}로 예측되었습니다."
        )

    render_section_header("상세 데이터 및 원본 결과 확인", "요약 이후 필요한 예측·이상탐지 원본 결과를 확인합니다.")
    with st.expander("Deep AI 시·군 요약", expanded=True):
        st.dataframe(filtered.sort_values("deep_ai_rank"), use_container_width=True, hide_index=True)
    with st.expander("GRU 7일 후 저수율 예측 결과"):
        if forecast.empty:
            st.info(f"선택 데이터 파일이 없습니다: {FORECAST_PATH}")
        else:
            st.dataframe(forecast, use_container_width=True, hide_index=True)
            st.download_button(
                label="GRU 예측 결과 CSV 다운로드",
                data=forecast.to_csv(index=False).encode("utf-8-sig"),
                file_name="ai_gru_reservoir_forecast_by_sigungu.csv",
                mime="text/csv",
                use_container_width=True,
            )
    with st.expander("AutoEncoder 이상탐지 결과"):
        if anomaly.empty:
            st.info(f"선택 데이터 파일이 없습니다: {ANOMALY_PATH}")
        else:
            st.dataframe(anomaly, use_container_width=True, hide_index=True)
            st.download_button(
                label="AutoEncoder 이상탐지 결과 CSV 다운로드",
                data=anomaly.to_csv(index=False).encode("utf-8-sig"),
                file_name="ai_autoencoder_anomaly_by_sigungu.csv",
                mime="text/csv",
                use_container_width=True,
            )
    with st.expander("AI 모델 리포트 및 학습 이력"):
        if report:
            st.markdown(report)
        else:
            st.info(f"선택 문서 파일이 없습니다: {REPORT_PATH}")
        if not gru_history.empty:
            st.markdown("#### GRU 학습 이력")
            st.dataframe(gru_history, use_container_width=True, hide_index=True)
        if not ae_history.empty:
            st.markdown("#### AutoEncoder 학습 이력")
            st.dataframe(ae_history, use_container_width=True, hide_index=True)

    st.info(
        "Deep AI 결과는 공개데이터 기반 예측·이상탐지 참고자료입니다. 실제 농업용수 대응 여부는 현장 저수율, "
        "관로, 수리권, 수질, 행정 협의를 함께 검토해야 합니다."
    )
    show_messages([forecast_msg, anomaly_msg, gru_msg, ae_msg, report_msg])


if __name__ == "__main__":
    main()
