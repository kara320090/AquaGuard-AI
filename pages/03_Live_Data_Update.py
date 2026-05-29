from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORT_TABLES = ROOT / "reports" / "tables"
META = ROOT / "data" / "metadata"

LIVE_FEATURE_PATH = PROCESSED / "latest_live_sigungu_features.csv"
LIVE_SUMMARY_PATH = REPORT_TABLES / "latest_live_risk_summary.csv"
LIVE_STATUS_PATH = REPORT_TABLES / "latest_live_data_status.csv"
OLDAM_STATUS_PATH = REPORT_TABLES / "latest_oldam_status_summary.csv"
KMA_STATUS_PATH = REPORT_TABLES / "latest_kma_weather_status.csv"
SOIL_STATUS_PATH = REPORT_TABLES / "latest_adms_soil_moisture_status.csv"
ADMS_RESERVOIR_STATUS_PATH = REPORT_TABLES / "latest_adms_reservoir_support_status.csv"
CROSSCHECK_PATH = REPORT_TABLES / "latest_reservoir_source_crosscheck.csv"
METHOD_PATH = META / "live_feature_method.md"

RISK_COLORS = {
    "심각": "#c62828",
    "심각후보": "#d84315",
    "경계": "#ef6c00",
    "주의": "#f9a825",
    "낮음": "#2e7d32",
}

st.set_page_config(
    page_title="Live 데이터 갱신 위험도",
    page_icon="📡",
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


def latest_date(frames: list[pd.DataFrame], columns: list[str]) -> str:
    dates = []
    for df in frames:
        if df.empty:
            continue
        for col in columns:
            if col in df.columns:
                parsed = pd.to_datetime(df[col], errors="coerce")
                dates.extend(parsed.dropna().tolist())
    if not dates:
        return "N/A"
    return max(dates).strftime("%Y-%m-%d")


def available_levels(summary: pd.DataFrame) -> list[str]:
    if summary.empty or "final_live_water_risk_level" not in summary.columns:
        return []
    found = summary["final_live_water_risk_level"].dropna().astype(str).unique().tolist()
    order = ["심각", "경계", "주의", "낮음"]
    return [x for x in order if x in found] + sorted([x for x in found if x not in order])


def render_sidebar(summary: pd.DataFrame) -> tuple[list[str], list[str], int, str, bool]:
    regions = summary.sort_values("final_live_priority_rank")["sigungu"].dropna().astype(str).tolist() if "final_live_priority_rank" in summary.columns else sorted(summary["sigungu"].dropna().astype(str).tolist())
    levels = available_levels(summary)
    if "live_region_filter" not in st.session_state:
        st.session_state.live_region_filter = regions
    if "live_level_filter" not in st.session_state:
        st.session_state.live_level_filter = levels
    st.session_state.live_region_filter = [x for x in st.session_state.live_region_filter if x in regions] or regions
    st.session_state.live_level_filter = [x for x in st.session_state.live_level_filter if x in levels] or levels

    with st.sidebar:
        st.header("필터")
        if st.button("기본값으로 초기화", use_container_width=True):
            st.session_state.live_region_filter = regions
            st.session_state.live_level_filter = levels
            st.session_state.live_top_n = 10
            st.session_state.only_updated_sources = False
            st.rerun()

        st.markdown("#### 기간")
        st.caption(f"Live 기준일: {latest_date([summary], ['soil_data_date'])}")

        st.markdown("#### 지역/대상")
        selected_regions = st.multiselect("시·군", regions, key="live_region_filter")
        focus = st.selectbox("상세 확인 대상", selected_regions or regions, index=0 if (selected_regions or regions) else None)

        st.markdown("#### 모델/위험등급")
        selected_levels = st.multiselect("Live 위험등급", levels, key="live_level_filter")
        only_updated = st.checkbox("최신 원천데이터 반영 지역만 보기", key="only_updated_sources")
        max_top = max(5, min(15, len(summary))) if not summary.empty else 5
        top_n = st.slider("우선순위 표시 수", 5, max_top, min(st.session_state.get("live_top_n", 10), max_top), key="live_top_n")

    return selected_regions, selected_levels, top_n, focus, only_updated


def build_priority_table(summary: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    cols = [
        "final_live_priority_rank",
        "sigungu",
        "final_live_water_risk_score",
        "final_live_water_risk_level",
        "live_score_delta_from_baseline",
        "live_main_risk_driver",
        "live_weather_source",
        "live_reservoir_source",
        "live_soil_source",
    ]
    cols = [c for c in cols if c in summary.columns]
    out = summary.sort_values("final_live_priority_rank").head(top_n)[cols].copy()
    return out.rename(
        columns={
            "final_live_priority_rank": "Live 순위",
            "sigungu": "시·군",
            "final_live_water_risk_score": "Live 위험점수",
            "final_live_water_risk_level": "Live 등급",
            "live_score_delta_from_baseline": "기준 대비 변화",
            "live_main_risk_driver": "주요 위험 원인",
            "live_weather_source": "기상 데이터",
            "live_reservoir_source": "저수지 데이터",
            "live_soil_source": "토양수분 데이터",
        }
    )


def make_live_ranking_chart(summary: pd.DataFrame, top_n: int):
    if summary.empty or "final_live_water_risk_score" not in summary.columns:
        return None
    plot_df = summary.sort_values("final_live_water_risk_score", ascending=False).head(top_n)
    fig = px.bar(
        plot_df.sort_values("final_live_water_risk_score"),
        x="final_live_water_risk_score",
        y="sigungu",
        orientation="h",
        color="final_live_water_risk_level" if "final_live_water_risk_level" in plot_df.columns else None,
        color_discrete_map=RISK_COLORS,
        hover_data=[c for c in ["live_main_risk_driver", "live_weather_source", "live_reservoir_source", "live_soil_source"] if c in plot_df.columns],
        title="Live 위험도 순위: 오늘 기준 어디를 먼저 확인할 것인가?",
    )
    fig.update_layout(xaxis_title="Live 위험점수", yaxis_title="시·군", height=max(380, 34 * len(plot_df) + 140))
    return fig


def make_delta_chart(summary: pd.DataFrame, top_n: int):
    if summary.empty or "live_score_delta_from_baseline" not in summary.columns:
        return None
    plot_df = summary.reindex(summary["live_score_delta_from_baseline"].abs().sort_values(ascending=False).index).head(top_n)
    fig = px.bar(
        plot_df.sort_values("live_score_delta_from_baseline"),
        x="live_score_delta_from_baseline",
        y="sigungu",
        orientation="h",
        color="final_live_water_risk_level" if "final_live_water_risk_level" in plot_df.columns else None,
        color_discrete_map=RISK_COLORS,
        hover_data=[c for c in ["final_water_risk_score", "final_live_water_risk_score"] if c in plot_df.columns],
        title="기준 위험도 대비 Live 변화: 어디가 가장 달라졌는가?",
    )
    fig.update_layout(xaxis_title="Live - 기준 위험점수", yaxis_title="시·군", height=max(360, 32 * len(plot_df) + 120))
    return fig


def make_source_coverage_chart(live: pd.DataFrame):
    source_cols = [c for c in ["live_weather_source", "live_reservoir_source", "live_soil_source"] if c in live.columns]
    if live.empty or not source_cols:
        return None
    rows = []
    for col in source_cols:
        for source, count in live[col].fillna("N/A").astype(str).value_counts().items():
            rows.append({"구분": col, "원천": source, "대상 수": count})
    source_df = pd.DataFrame(rows)
    source_df["구분"] = source_df["구분"].map(
        {
            "live_weather_source": "기상",
            "live_reservoir_source": "저수지",
            "live_soil_source": "토양수분",
        }
    )
    fig = px.bar(source_df, x="구분", y="대상 수", color="원천", title="Live 원천데이터 반영 현황: 어떤 데이터가 최신으로 들어왔는가?")
    fig.update_layout(xaxis_title="데이터 구분", yaxis_title="시·군 수", height=360)
    return fig


def make_crosscheck_chart(cross: pd.DataFrame):
    needed = {"sigungu", "adms_rvow", "oldam_avg_reservoir_rate"}
    if cross.empty or not needed.issubset(cross.columns):
        return None
    fig = px.scatter(
        cross,
        x="adms_rvow",
        y="oldam_avg_reservoir_rate",
        hover_name="sigungu",
        color="crosscheck_status" if "crosscheck_status" in cross.columns else None,
        title="저수율 교차검증: ADMS와 올담 값은 얼마나 가까운가?",
    )
    fig.add_shape(type="line", x0=0, y0=0, x1=120, y1=120, line=dict(color="#94a3b8", dash="dash"))
    fig.update_layout(xaxis_title="ADMS 저수율", yaxis_title="올담 평균 저수율", height=420)
    return fig


def status_summary_table(status_frames: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for name, df in status_frames:
        if df.empty:
            rows.append({"구분": name, "상태": "N/A", "행 수": 0, "대표 값": "파일 없음"})
            continue
        status_value = df["status"].iloc[0] if "status" in df.columns else "확인"
        rows.append({"구분": name, "상태": status_value, "행 수": len(df), "대표 값": ", ".join(df.columns[:4])})
    return pd.DataFrame(rows)


def main() -> None:
    inject_css()

    live, live_msg = safe_read_data(str(LIVE_FEATURE_PATH), required=False)
    summary, summary_msg = safe_read_data(str(LIVE_SUMMARY_PATH), required=True)
    status, status_msg = safe_read_data(str(LIVE_STATUS_PATH), required=False)
    oldam_status, oldam_msg = safe_read_data(str(OLDAM_STATUS_PATH), required=False)
    kma_status, kma_msg = safe_read_data(str(KMA_STATUS_PATH), required=False)
    soil_status, soil_msg = safe_read_data(str(SOIL_STATUS_PATH), required=False)
    adms_status, adms_msg = safe_read_data(str(ADMS_RESERVOIR_STATUS_PATH), required=False)
    cross, cross_msg = safe_read_data(str(CROSSCHECK_PATH), required=False)
    method, method_msg = read_text_file(str(METHOD_PATH))

    show_messages([summary_msg], stop_on_required=True)
    if summary.empty:
        render_empty_state("Live 위험도 결과가 비어 있습니다.")
        st.stop()

    summary = normalize_numeric(
        summary,
        [
            "final_live_priority_rank",
            "final_live_water_risk_score",
            "live_score_delta_from_baseline",
            "today_avg_reservoir_rate",
            "today_min_reservoir_rate",
            "rainfall_30d",
            "rainfall_7d",
            "latest_rain_shortage_score",
            "soil_moisture_avg",
            "soil_moisture_drought_score",
            "final_water_risk_score",
        ],
    )
    live = normalize_numeric(live, ["final_live_water_risk_score", "live_score_delta_from_baseline"])
    cross = normalize_numeric(cross, ["adms_rvow", "oldam_avg_reservoir_rate", "rvow_diff_oldam_minus_adms"])

    latest_basis = latest_date([summary, live], ["soil_data_date", "date", "base_date"])
    render_page_header(
        "Live 데이터 갱신 위험도",
        "올담 저수지 snapshot, 기상청 AWS/ASOS, ADMS 토양수분을 반영해 최신 위험도 변화를 확인합니다.",
        latest_basis,
    )

    selected_regions, selected_levels, top_n, focus, only_updated = render_sidebar(summary)
    filtered = summary.copy()
    if selected_regions:
        filtered = filtered[filtered["sigungu"].isin(selected_regions)]
    if selected_levels and "final_live_water_risk_level" in filtered.columns:
        filtered = filtered[filtered["final_live_water_risk_level"].isin(selected_levels)]
    if only_updated:
        source_cols = [c for c in ["live_weather_source", "live_reservoir_source", "live_soil_source"] if c in filtered.columns]
        if source_cols:
            mask = filtered[source_cols].apply(lambda row: any("BASELINE" not in str(v) for v in row), axis=1)
            filtered = filtered[mask]

    st.markdown(
        f"""
        <div class="ag-filter">
        현재 필터: 지역 <b>{len(selected_regions) if selected_regions else 0}개</b> ·
        Live 등급 <b>{", ".join(selected_levels) if selected_levels else "전체"}</b> · 상세 대상 <b>{focus or "N/A"}</b> ·
        최신 원천만 <b>{"ON" if only_updated else "OFF"}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    top = filtered.sort_values("final_live_water_risk_score", ascending=False).iloc[0] if not filtered.empty else None
    warning_count = int(filtered["final_live_water_risk_level"].astype(str).isin(["주의", "경계", "심각"]).sum()) if "final_live_water_risk_level" in filtered.columns and not filtered.empty else 0
    oldam_count = int((summary["live_reservoir_source"] == "OLDAM_TODAY").sum()) if "live_reservoir_source" in summary.columns else 0
    kma_count = int((summary["live_weather_source"].astype(str) != "BASELINE_WEATHER").sum()) if "live_weather_source" in summary.columns else 0
    soil_count = int((summary["live_soil_source"] == "ADMS_SOIL_AUTO").sum()) if "live_soil_source" in summary.columns else 0

    render_kpi_cards(
        [
            ("Live 분석 시·군", f"{len(filtered):,}곳", "현재 필터 기준 Live 결과 수입니다."),
            ("주의 이상 대상", f"{warning_count:,}곳", "Live 위험등급이 주의 이상인 대상 수입니다."),
            ("최고 위험 지역", str(top["sigungu"]) if top is not None else "N/A", "현재 필터 기준 Live 위험점수가 가장 높은 지역입니다."),
            ("최신 데이터 반영", f"저수지 {oldam_count} / 기상 {kma_count} / 토양 {soil_count}", "최신 원천 데이터가 반영된 시·군 수입니다."),
            ("최신 기준일", latest_basis, "Live 입력 데이터에서 확인 가능한 최신 날짜입니다."),
        ]
    )

    if filtered.empty:
        render_empty_state()
        st.stop()

    render_section_header("점검 우선순위", "Live 업데이트 후 가장 먼저 확인할 시·군입니다.")
    priority = build_priority_table(filtered, top_n)
    if priority.empty:
        render_empty_state()
    else:
        st.dataframe(priority, use_container_width=True, hide_index=True)

    render_section_header("데이터 수집 상태", "Live 위험도에 들어온 원천 데이터의 정상 수집 여부를 요약합니다.")
    status_table = status_summary_table(
        [
            ("Live Feature", status),
            ("올담 저수지", oldam_status),
            ("기상청 AWS/ASOS", kma_status),
            ("ADMS 토양수분", soil_status),
            ("ADMS 저수율 보조", adms_status),
        ]
    )
    st.dataframe(status_table, use_container_width=True, hide_index=True)

    render_section_header("Live 위험도 분석", "현재 위험도와 기준 대비 변화량을 분리해서 확인합니다.")
    left, right = st.columns(2)
    with left:
        ranking_fig = make_live_ranking_chart(filtered, top_n)
        if ranking_fig:
            st.plotly_chart(ranking_fig, use_container_width=True)
            st.caption("막대가 길수록 오늘 기준 점검 우선순위가 높습니다.")
        else:
            st.info("final_live_water_risk_score 컬럼이 없어 위험도 차트를 건너뛰었습니다.")
    with right:
        delta_fig = make_delta_chart(filtered, top_n)
        if delta_fig:
            st.plotly_chart(delta_fig, use_container_width=True)
            st.caption("0보다 크면 기준 산정 대비 Live 위험도가 상승한 지역입니다.")
        else:
            st.info("live_score_delta_from_baseline 컬럼이 없어 변화량 차트를 건너뛰었습니다.")

    left, right = st.columns(2)
    with left:
        coverage_fig = make_source_coverage_chart(summary)
        if coverage_fig:
            st.plotly_chart(coverage_fig, use_container_width=True)
            st.caption("데이터 구분별 최신 원천 반영 범위를 확인합니다.")
        else:
            st.info("Live 원천 데이터 컬럼이 없어 수집 범위 차트를 건너뛰었습니다.")
    with right:
        cross_fig = make_crosscheck_chart(cross)
        if cross_fig:
            st.plotly_chart(cross_fig, use_container_width=True)
            st.caption("점선에 가까울수록 ADMS와 올담 저수율이 비슷합니다.")
        else:
            st.info(f"저수율 교차검증 차트를 만들 수 없습니다: {CROSSCHECK_PATH}")

    render_section_header(f"{focus} 상세 해석", "선택한 지역의 Live 변화 원인을 짧게 확인합니다.")
    focus_row = summary[summary["sigungu"] == focus]
    if focus_row.empty:
        render_empty_state()
    else:
        row = focus_row.iloc[0]
        st.success(
            f"{focus}의 Live 위험점수는 {format_value(row.get('final_live_water_risk_score'), '점', 1)}이며 "
            f"기준 산정 대비 {format_value(row.get('live_score_delta_from_baseline'), '점', 1)} 변했습니다. "
            f"주요 원인은 {row.get('live_main_risk_driver', 'N/A')}입니다."
        )

    render_section_header("상세 데이터 및 원본 결과 확인", "요약 이후 필요한 원본 Live 결과와 검증 자료를 확인합니다.")
    with st.expander("Live 위험도 결과", expanded=True):
        st.dataframe(filtered.sort_values("final_live_priority_rank"), use_container_width=True, hide_index=True)
        st.download_button(
            label="Live 위험도 결과 CSV 다운로드",
            data=filtered.to_csv(index=False).encode("utf-8-sig"),
            file_name="latest_live_risk_summary_filtered.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with st.expander("Live feature 원본"):
        if live.empty:
            st.info(f"선택 데이터 파일이 없습니다: {LIVE_FEATURE_PATH}")
        else:
            st.dataframe(live, use_container_width=True, hide_index=True)
    with st.expander("올담 vs ADMS 교차검증 원본"):
        if cross.empty:
            st.info(f"선택 데이터 파일이 없습니다: {CROSSCHECK_PATH}")
        else:
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
            st.dataframe(cross[show_cols], use_container_width=True, hide_index=True)
    with st.expander("산식 및 해석 기준"):
        if method:
            st.markdown(method)
        else:
            st.info(f"선택 문서 파일이 없습니다: {METHOD_PATH}")

    st.info(
        "Live 결과는 최신 공개 snapshot을 반영한 참고자료입니다. 실시간 운영 판단에는 현장 저수율, 관로 상태, "
        "수리권, 수질, 행정 협의 정보를 함께 확인해야 합니다."
    )

    show_messages([live_msg, status_msg, oldam_msg, kma_msg, soil_msg, adms_msg, cross_msg, method_msg])


if __name__ == "__main__":
    main()
