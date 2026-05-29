from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
REPORT_TABLES = ROOT / "reports" / "tables"
PROCESSED = ROOT / "data" / "processed"

WATCHLIST_PATH = REPORT_TABLES / "reservoir_watchlist.csv"
FACILITY_STATUS_PATH = PROCESSED / "reservoir_facility_status_for_dashboard.csv"
OLDAM_TODAY_PATH = PROCESSED / "latest_oldam_reservoir_today.csv"

WATCH_COLORS = {
    "심각": "#c62828",
    "심각후보": "#d84315",
    "경계": "#ef6c00",
    "주의": "#f9a825",
    "낮음": "#2e7d32",
}

st.set_page_config(
    page_title="저수지 현황 및 이상징후 Watchlist",
    page_icon="💧",
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


def prepare_facility(facility: pd.DataFrame, sigungu: str) -> tuple[pd.DataFrame, int]:
    if facility.empty or "sigungu" not in facility.columns:
        return pd.DataFrame(), 0
    selected = facility[facility["sigungu"] == sigungu].copy()
    if selected.empty:
        return selected, 0
    for col in ["facility_name", "address", "sigungu"]:
        if col in selected.columns:
            selected[col] = selected[col].fillna("").astype(str).str.strip()
    dedup_cols = [c for c in ["sigungu", "facility_name", "address"] if c in selected.columns]
    before = len(selected)
    if dedup_cols:
        selected = selected.drop_duplicates(subset=dedup_cols, keep="first")
    return selected, before - len(selected)


def make_watch_priority_table(watch: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if watch.empty:
        return pd.DataFrame()
    cols = [
        "watch_rank",
        "sigungu",
        "watch_level",
        "watch_reason",
        "avg_reservoir_rate",
        "min_reservoir_rate",
        "low_reservoir_count_30",
        "reservoir_risk_score",
    ]
    cols = [c for c in cols if c in watch.columns]
    out = watch.sort_values("watch_rank").head(top_n)[cols].copy()
    return out.rename(
        columns={
            "watch_rank": "순위",
            "sigungu": "시·군",
            "watch_level": "Watch 등급",
            "watch_reason": "판단 근거",
            "avg_reservoir_rate": "평균 저수율",
            "min_reservoir_rate": "최저 저수율",
            "low_reservoir_count_30": "30% 이하 저수지",
            "reservoir_risk_score": "저수율 위험점수",
        }
    )


def make_facility_priority_table(facility: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if facility.empty:
        return pd.DataFrame()
    cols = [
        "facility_name",
        "address",
        "benefit_area",
        "effective_capacity",
        "sigungu_avg_reservoir_rate",
        "sigungu_min_reservoir_rate",
        "inspection_priority_score",
        "reservoir_status_note",
    ]
    cols = [c for c in cols if c in facility.columns]
    out = facility[cols].copy()
    if "inspection_priority_score" in out.columns:
        out = out.sort_values("inspection_priority_score", ascending=False)
    return out.head(top_n).rename(
        columns={
            "facility_name": "저수지명",
            "address": "주소",
            "benefit_area": "수혜면적",
            "effective_capacity": "유효저수량",
            "sigungu_avg_reservoir_rate": "시·군 평균 저수율",
            "sigungu_min_reservoir_rate": "시·군 최저 저수율",
            "inspection_priority_score": "시설 점검 우선점수",
            "reservoir_status_note": "상태 메모",
        }
    )


def make_watch_bar(watch: pd.DataFrame, top_n: int):
    if watch.empty or "reservoir_risk_score" not in watch.columns:
        return None
    plot_df = watch.sort_values("reservoir_risk_score", ascending=False).head(top_n)
    fig = px.bar(
        plot_df.sort_values("reservoir_risk_score"),
        x="reservoir_risk_score",
        y="sigungu",
        orientation="h",
        color="watch_level" if "watch_level" in plot_df.columns else None,
        color_discrete_map=WATCH_COLORS,
        hover_data=[c for c in ["watch_reason", "avg_reservoir_rate", "min_reservoir_rate"] if c in plot_df.columns],
        title="시·군별 저수율 위험점수: 어느 지역 저수지를 먼저 볼 것인가?",
    )
    fig.update_layout(xaxis_title="저수율 위험점수", yaxis_title="시·군", height=max(380, 34 * len(plot_df) + 140))
    return fig


def make_rate_compare_chart(watch: pd.DataFrame, top_n: int):
    needed = {"sigungu", "avg_reservoir_rate", "min_reservoir_rate"}
    if watch.empty or not needed.issubset(watch.columns):
        return None
    plot_df = watch.sort_values("watch_rank").head(top_n)
    long_df = plot_df.melt(
        id_vars=["sigungu"],
        value_vars=["avg_reservoir_rate", "min_reservoir_rate"],
        var_name="지표",
        value_name="저수율",
    )
    long_df["지표"] = long_df["지표"].map({"avg_reservoir_rate": "평균 저수율", "min_reservoir_rate": "최저 저수율"})
    fig = px.bar(
        long_df,
        x="sigungu",
        y="저수율",
        color="지표",
        barmode="group",
        title="평균·최저 저수율 비교: 평균은 괜찮아도 최저 저수지가 낮은가?",
    )
    fig.update_layout(xaxis_title="시·군", yaxis_title="저수율(%)", height=420)
    return fig


def render_sidebar(watch: pd.DataFrame) -> tuple[list[str], list[str], int, str]:
    sigungu_options = watch.sort_values("watch_rank")["sigungu"].dropna().astype(str).tolist() if "watch_rank" in watch.columns else sorted(watch["sigungu"].dropna().astype(str).tolist())
    levels = watch["watch_level"].dropna().astype(str).unique().tolist() if "watch_level" in watch.columns else []

    if "watch_sigungu_filter" not in st.session_state:
        st.session_state.watch_sigungu_filter = sigungu_options
    if "watch_level_filter" not in st.session_state:
        st.session_state.watch_level_filter = levels

    st.session_state.watch_sigungu_filter = [x for x in st.session_state.watch_sigungu_filter if x in sigungu_options] or sigungu_options
    st.session_state.watch_level_filter = [x for x in st.session_state.watch_level_filter if x in levels] or levels

    with st.sidebar:
        st.header("필터")
        if st.button("기본값으로 초기화", use_container_width=True):
            st.session_state.watch_sigungu_filter = sigungu_options
            st.session_state.watch_level_filter = levels
            st.session_state.watch_top_n = 10
            st.rerun()

        st.markdown("#### 기간")
        st.caption(f"저수지 기준일: {latest_date(watch, ['reservoir_latest_date'])}")

        st.markdown("#### 지역/대상")
        selected_regions = st.multiselect("시·군", sigungu_options, key="watch_sigungu_filter")
        focus = st.selectbox("상세 확인 대상", selected_regions or sigungu_options, index=0 if (selected_regions or sigungu_options) else None)

        st.markdown("#### 모델/위험등급")
        selected_levels = st.multiselect("Watch 등급", levels, key="watch_level_filter")
        max_top = max(5, min(15, len(watch))) if not watch.empty else 5
        top_n = st.slider("우선순위 표시 수", 5, max_top, min(st.session_state.get("watch_top_n", 10), max_top), key="watch_top_n")

    return selected_regions, selected_levels, top_n, focus


def main() -> None:
    inject_css()

    watch, watch_msg = safe_read_data(str(WATCHLIST_PATH), required=True)
    facility, facility_msg = safe_read_data(str(FACILITY_STATUS_PATH), required=True)
    oldam_today, oldam_msg = safe_read_data(str(OLDAM_TODAY_PATH), required=False)

    show_messages([watch_msg, facility_msg], stop_on_required=True)
    if watch.empty:
        render_empty_state("Watchlist 데이터가 비어 있습니다.")
        st.stop()

    watch = normalize_numeric(
        watch,
        [
            "watch_rank",
            "watch_priority_score",
            "avg_reservoir_rate",
            "min_reservoir_rate",
            "low_reservoir_count_30",
            "low_reservoir_count_40",
            "reservoir_risk_score",
            "final_water_risk_score",
        ],
    )
    facility = normalize_numeric(
        facility,
        [
            "benefit_area",
            "effective_capacity",
            "total_capacity",
            "sigungu_avg_reservoir_rate",
            "sigungu_min_reservoir_rate",
            "inspection_priority_score",
        ],
    )

    render_page_header(
        "저수지 현황 및 이상징후 Watchlist",
        "저수율 위험도와 시설 정보를 결합해 우선 점검할 저수지·시군을 빠르게 확인합니다.",
        latest_date(watch, ["reservoir_latest_date"]),
    )

    selected_regions, selected_levels, top_n, focus = render_sidebar(watch)
    filtered = watch.copy()
    if selected_regions:
        filtered = filtered[filtered["sigungu"].isin(selected_regions)]
    if selected_levels and "watch_level" in filtered.columns:
        filtered = filtered[filtered["watch_level"].isin(selected_levels)]

    st.markdown(
        f"""
        <div class="ag-filter">
        현재 필터: 지역 <b>{len(selected_regions) if selected_regions else 0}개</b> ·
        Watch 등급 <b>{", ".join(selected_levels) if selected_levels else "전체"}</b> · 상세 대상 <b>{focus or "N/A"}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    top_watch = filtered.sort_values("watch_rank").iloc[0] if not filtered.empty and "watch_rank" in filtered.columns else None
    severe_count = int(filtered["watch_level"].astype(str).str.contains("심각|경계|주의", regex=True).sum()) if "watch_level" in filtered.columns and not filtered.empty else 0
    facility_count = len(facility)

    render_kpi_cards(
        [
            ("Watch 대상 시·군", f"{len(filtered):,}곳", "현재 필터 기준 Watchlist 대상 수입니다."),
            ("주의 이상 대상", f"{severe_count:,}곳", "주의·경계·심각후보 등급 대상 수입니다."),
            ("전체 저수지 시설", f"{facility_count:,}개", "대시보드에 연결된 저수지 시설 수입니다."),
            ("최우선 시·군", str(top_watch["sigungu"]) if top_watch is not None else "N/A", "현재 필터 기준 Watch 1순위입니다."),
            ("저수지 기준일", latest_date(filtered, ["reservoir_latest_date"]), "Watchlist 산정에 사용된 최신 기준일입니다."),
        ]
    )

    if filtered.empty:
        render_empty_state()
        st.stop()

    render_section_header("점검 우선순위", "저수율 위험점수가 높고 Watch 순위가 빠른 시·군을 먼저 확인합니다.")
    st.dataframe(make_watch_priority_table(filtered, top_n), use_container_width=True, hide_index=True)

    render_section_header("저수율 위험도 해석", "평균과 최저 저수율을 함께 보며 지역 단위 위험을 확인합니다.")
    left, right = st.columns(2)
    with left:
        watch_fig = make_watch_bar(filtered, top_n)
        if watch_fig:
            st.plotly_chart(watch_fig, use_container_width=True)
            st.caption("막대가 길수록 저수율 관점에서 우선 확인할 필요가 큽니다.")
        else:
            st.info("reservoir_risk_score 컬럼이 없어 위험도 차트를 건너뛰었습니다.")
    with right:
        rate_fig = make_rate_compare_chart(filtered, top_n)
        if rate_fig:
            st.plotly_chart(rate_fig, use_container_width=True)
            st.caption("최저 저수율이 평균보다 크게 낮으면 특정 시설 중심의 현장 확인이 필요합니다.")
        else:
            st.info("평균·최저 저수율 컬럼이 없어 비교 차트를 건너뛰었습니다.")

    selected_facility, removed_duplicates = prepare_facility(facility, focus)
    render_section_header(f"{focus} 시설 점검 우선순위", "선택한 시·군 안에서 먼저 확인할 저수지 시설입니다.")
    facility_top = make_facility_priority_table(selected_facility, top_n)
    if facility_top.empty:
        render_empty_state("선택한 시·군의 저수지 시설 정보가 없습니다.")
    else:
        st.caption(f"표시 시설 {len(selected_facility):,}개" + (f" · 중복 제거 {removed_duplicates:,}개" if removed_duplicates else ""))
        st.dataframe(facility_top, use_container_width=True, hide_index=True)

    if not oldam_today.empty and "sigungu" in oldam_today.columns:
        render_section_header("올담 최신 공개 저수지 현황", "최신 공개 snapshot에 포함된 시설만 표시합니다.")
        selected_oldam = oldam_today[oldam_today["sigungu"] == focus].copy()
        if selected_oldam.empty:
            st.info("선택한 시·군은 올담 최신 snapshot에 직접 포함되지 않았습니다.")
        else:
            oldam_cols = [c for c in ["facility_name", "reservoir_rate", "date", "location_raw", "facility_match_status"] if c in selected_oldam.columns]
            oldam_view = selected_oldam[oldam_cols].rename(
                columns={
                    "facility_name": "저수지명",
                    "reservoir_rate": "최신 공개 저수율",
                    "date": "기준일",
                    "location_raw": "위치",
                    "facility_match_status": "매칭 상태",
                }
            )
            sort_col = "최신 공개 저수율" if "최신 공개 저수율" in oldam_view.columns else oldam_view.columns[0]
            st.dataframe(oldam_view.sort_values(sort_col), use_container_width=True, hide_index=True)
    else:
        st.info(f"선택 데이터 파일이 없습니다: {OLDAM_TODAY_PATH}")

    render_section_header("상세 데이터 및 원본 결과 확인", "요약 이후 필요한 원본 Watchlist와 시설 목록을 확인합니다.")
    with st.expander("Watchlist 전체 보기", expanded=True):
        st.dataframe(filtered.sort_values("watch_rank"), use_container_width=True, hide_index=True)
        st.download_button(
            label="Watchlist CSV 다운로드",
            data=filtered.to_csv(index=False).encode("utf-8-sig"),
            file_name="reservoir_watchlist_filtered.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with st.expander(f"{focus} 저수지 시설 전체 보기"):
        if selected_facility.empty:
            render_empty_state("조건에 맞는 데이터가 없습니다")
        else:
            st.dataframe(selected_facility, use_container_width=True, hide_index=True)
            st.download_button(
                label=f"{focus} 저수지 시설 CSV 다운로드",
                data=selected_facility.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{focus}_reservoir_facility_status.csv",
                mime="text/csv",
                use_container_width=True,
            )

    st.info(
        "저수지 개별 실시간 저수율은 공개 원천 데이터의 한계로 일부 시설만 최신 snapshot과 직접 연결됩니다. "
        "시설 점검 우선순위는 시·군 단위 위험과 시설 제원 정보를 결합한 참고 지표입니다."
    )
    show_messages([oldam_msg])


if __name__ == "__main__":
    main()
