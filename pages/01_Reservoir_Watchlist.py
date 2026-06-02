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
LIVE_FEATURE_PATH = PROCESSED / "latest_live_sigungu_features.csv"
TRAINING_DATA_PATH = PROCESSED / "01_reservoir_sigungu_daily.csv"

EXCLUDED_SIGUNGU = {"계룡시"}
SIGUNGU_FILTER_COLUMNS = (
    "sigungu",
    "target_sigungu",
    "candidate_sigungu",
    "시·군",
    "시군",
    "시군명",
    "adms_sigun_name",
)

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
        .ag-page-title {
            margin: 0 0 0.35rem 0;
            font-size: 2.1rem;
            line-height: 1.2;
            letter-spacing: 0;
            font-weight: 750;
        }
        .ag-basis-card {
            background: #f8fafc;
            border: 1px solid #dbe4f0;
            border-radius: 8px;
            padding: 0.85rem 1rem;
            margin-top: 0.3rem;
        }
        .ag-basis-label {
            color: #475569;
            font-size: 0.78rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }
        .ag-basis-value {
            color: #0f172a;
            font-size: 0.95rem;
            line-height: 1.45;
            font-weight: 650;
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
    left, right = st.columns([3.4, 1.15])
    with left:
        st.markdown(f'<h1 class="ag-page-title">{title}</h1>', unsafe_allow_html=True)
        st.caption(service_definition)
    with right:
        st.markdown(
            f"""
            <div class="ag-basis-card">
                <div class="ag-basis-label">최신 분석 기준</div>
                <div class="ag-basis-value">{latest_basis}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


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


RAW_MISSING_TEXT = {"", "-", "none", "nan", "nat", "<na>", "n/a"}
DATE_DISPLAY_COLUMNS = {
    "기준일",
    "reservoir_latest_date",
    "latest_measurement_date",
    "soil_data_date",
    "date",
    "base_date",
    "analysis_date",
}
RESERVOIR_DATE_COLUMNS = {"기준일", "reservoir_latest_date", "latest_measurement_date"}
ANALYSIS_BASIS_COLUMNS = ["분석 기준", "analysis_mode", "analysis_basis"]
SCORE_DISPLAY_COLUMNS = {
    "위험점수",
    "risk_score",
    "score",
    "priority_score",
    "facility_scale_score",
    "reservoir_risk_score",
    "inspection_priority_score",
    "저수율 위험점수",
    "시설 규모점수",
    "시설 점검 우선점수",
}
MISSING_RESERVOIR_NOTE = (
    "해당 시·군은 저수지 기준일 또는 시설 매칭 정보가 부족하여, "
    "최종 위험도는 확보 가능한 강우·관정·작물·대체수원 지표 중심으로 산정했습니다."
)


def is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in RAW_MISSING_TEXT
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(result, (bool, np.bool_)):
        return bool(result)
    return False


def format_display_value(value, missing_label: str = "자료 없음") -> str:
    if is_missing(value):
        return missing_label
    text = str(value).strip()
    return text if text else missing_label


def format_date_display(value, missing_label: str = "자료 없음") -> str:
    if is_missing(value):
        return missing_label
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.notna(parsed):
            return parsed.strftime("%Y-%m-%d")
    return format_display_value(value, missing_label)


def build_analysis_basis(row) -> str:
    basis_value = next((row.get(col) for col in ANALYSIS_BASIS_COLUMNS if col in row.index), "최종 산정")
    basis_text = format_display_value(basis_value, "최종 산정")
    final_basis = is_missing(basis_value) or basis_text == "최종 산정"
    has_missing_reservoir_date = any(
        col in row.index and is_missing(row.get(col))
        for col in ["reservoir_latest_date", "latest_measurement_date"]
    )
    has_missing_final_basis_date = any(
        col in row.index and is_missing(row.get(col))
        for col in ["기준일", "basis_date"]
    ) and final_basis
    if has_missing_reservoir_date or has_missing_final_basis_date:
        return "최종 산정 · 저수지 기준일 미확보"
    return basis_text if not is_missing(basis_text) else "최종 산정"


def is_date_display_column(column: str) -> bool:
    column_text = str(column)
    lower = column_text.lower()
    return column_text in DATE_DISPLAY_COLUMNS or lower in DATE_DISPLAY_COLUMNS or lower.endswith("_date")


def is_reservoir_date_column(column: str) -> bool:
    column_text = str(column)
    return column_text in RESERVOIR_DATE_COLUMNS or column_text.lower() in RESERVOIR_DATE_COLUMNS


def should_round_display_column(column: str) -> bool:
    column_text = str(column)
    lower = column_text.lower()
    return (
        column_text in SCORE_DISPLAY_COLUMNS
        or lower in SCORE_DISPLAY_COLUMNS
        or "score" in lower
        or "점수" in column_text
        or "저수율" in column_text
    )


def format_numeric_display(value, column: str) -> str:
    if is_missing(value):
        return "자료 없음"
    if should_round_display_column(column):
        try:
            return f"{float(value):,.2f}"
        except (TypeError, ValueError):
            return format_display_value(value)
    return format_display_value(value)


def row_has_missing_reservoir_context(row) -> bool:
    basis_value = next((row.get(col) for col in ANALYSIS_BASIS_COLUMNS if col in row.index), "최종 산정")
    final_basis = is_missing(basis_value) or format_display_value(basis_value, "최종 산정") == "최종 산정"
    if any(col in row.index and is_missing(row.get(col)) for col in ["reservoir_latest_date", "latest_measurement_date"]):
        return True
    if any(col in row.index and is_missing(row.get(col)) for col in ["기준일", "basis_date"]) and final_basis:
        return True
    return any(col in row.index and is_missing(row.get(col)) for col in ["reservoir_count", "저수지 수", "저수지수"])


def has_missing_reservoir_context(df: pd.DataFrame) -> bool:
    if df.empty:
        return False
    return bool(df.apply(row_has_missing_reservoir_context, axis=1).any())


def render_missing_reservoir_note(df: pd.DataFrame) -> None:
    if has_missing_reservoir_context(df):
        st.info(MISSING_RESERVOIR_NOTE)


def format_display_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    display_df = df.copy()
    if display_df.empty:
        return display_df

    for basis_col in ANALYSIS_BASIS_COLUMNS:
        if basis_col in display_df.columns:
            display_df[basis_col] = df.apply(build_analysis_basis, axis=1)

    for col in display_df.columns:
        if col in ANALYSIS_BASIS_COLUMNS:
            continue
        if is_date_display_column(col):
            missing_label = "저수지 기준일 없음" if is_reservoir_date_column(col) else "자료 없음"
            display_df[col] = df[col].map(lambda value: format_date_display(value, missing_label))
        elif should_round_display_column(col):
            display_df[col] = df[col].map(lambda value: format_numeric_display(value, col))
        else:
            display_df[col] = df[col].map(format_display_value)
    return display_df


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


def exclude_unavailable_regions(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    mask = pd.Series(False, index=df.index)
    for col in SIGUNGU_FILTER_COLUMNS:
        if col in df.columns:
            mask = mask | df[col].astype(str).str.strip().isin(EXCLUDED_SIGUNGU)
    return df.loc[~mask].reset_index(drop=True)


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


def months_from_columns(frames: list[pd.DataFrame], columns: list[str]) -> list[str]:
    months: set[str] = set()
    for df in frames:
        if df.empty:
            continue
        for col in columns:
            if col in df.columns:
                parsed = pd.to_datetime(df[col], errors="coerce").dropna()
                months.update(parsed.dt.to_period("M").astype(str).tolist())
    return sorted(months)


def select_default_ai_comparison_month(current_month: str, available_months: list[str]) -> str:
    if not available_months or current_month == "N/A":
        return "N/A"
    current = pd.Period(current_month, freq="M")
    candidates = [pd.Period(month, freq="M") for month in available_months]
    same_month_previous_years = [month for month in candidates if month.month == current.month and month.year < current.year]
    if same_month_previous_years:
        return str(max(same_month_previous_years))
    previous_months = [month for month in candidates if month < current]
    return str(max(previous_months)) if previous_months else str(max(candidates))


def build_basis_text(live: pd.DataFrame, training_history: pd.DataFrame, fallback_live_basis: str) -> str:
    live_basis = latest_date(live, ["weather_end_date", "soil_data_date", "date", "base_date"]) if not live.empty else fallback_live_basis
    live_basis = live_basis if live_basis != "N/A" else fallback_live_basis
    live_month = pd.to_datetime(live_basis, errors="coerce")
    live_month_text = live_month.to_period("M").strftime("%Y-%m") if pd.notna(live_month) else "N/A"
    training_months = months_from_columns([training_history], ["date"])
    ai_comparison_month = select_default_ai_comparison_month(live_month_text, training_months)
    training_final_month = training_months[-1] if training_months else "N/A"
    return (
        f"Live 기준일 {live_basis}<br>"
        f"AI 비교 기준월 {ai_comparison_month}<br>"
        f"학습 데이터 최종월 {training_final_month}"
    )


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


def attach_latest_oldam_to_facilities(facility: pd.DataFrame, oldam_today: pd.DataFrame, sigungu: str) -> pd.DataFrame:
    out = facility.copy()
    out["facility_latest_reservoir_rate"] = np.nan
    out["facility_latest_date"] = np.nan
    if out.empty or oldam_today.empty or "facility_name" not in out.columns or "facility_name" not in oldam_today.columns:
        return out

    latest = oldam_today.copy()
    if "sigungu" in latest.columns:
        latest = latest[latest["sigungu"] == sigungu].copy()
    if latest.empty:
        return out

    latest["_facility_key"] = latest["facility_name"].fillna("").astype(str).str.strip()
    out["_facility_key"] = out["facility_name"].fillna("").astype(str).str.strip()

    if "date" in latest.columns:
        latest["_date_sort"] = pd.to_datetime(latest["date"], errors="coerce")
        latest = latest.sort_values("_date_sort", ascending=False)
    latest = latest.drop_duplicates("_facility_key", keep="first")

    rate_map = latest.set_index("_facility_key")["reservoir_rate"] if "reservoir_rate" in latest.columns else pd.Series(dtype=float)
    date_map = latest.set_index("_facility_key")["date"] if "date" in latest.columns else pd.Series(dtype=object)
    out["facility_latest_reservoir_rate"] = out["_facility_key"].map(rate_map)
    out["facility_latest_date"] = out["_facility_key"].map(date_map)
    return out.drop(columns=["_facility_key"], errors="ignore")


def render_selected_region_summary(watch: pd.DataFrame, focus: str) -> None:
    selected = watch[watch["sigungu"] == focus] if "sigungu" in watch.columns else pd.DataFrame()
    if selected.empty:
        st.info("선택한 시·군의 Watchlist 요약 정보가 없습니다.")
        return

    row = selected.iloc[0]
    render_kpi_cards(
        [
            ("시·군 평균 저수율", format_value(row.get("avg_reservoir_rate"), "%", 1), "지역 공통 저수율 지표입니다."),
            ("시·군 최저 저수율", format_value(row.get("min_reservoir_rate"), "%", 1), "지역 내 최저 저수율입니다."),
            ("시·군 저수율 위험도", format_value(row.get("reservoir_risk_score"), "점", 1), "지역 단위 저수율 위험 점수입니다."),
            ("Watch 단계", format_display_value(row.get("watch_level")), "Watchlist 판정 단계입니다."),
        ]
    )
    st.markdown(f"**Watch 판정 사유:** {format_display_value(row.get('watch_reason'))}")


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
        "facility_priority_rank",
        "facility_priority_level",
        "facility_name",
        "address",
        "benefit_area",
        "effective_capacity",
        "total_capacity",
        "facility_latest_reservoir_rate",
        "facility_latest_date",
        "facility_scale_score",
        "inspection_priority_score",
        "facility_priority_reason",
        "reservoir_status_note",
    ]
    cols = [c for c in cols if c in facility.columns]
    out = facility[cols].copy()
    if "facility_priority_rank" in out.columns:
        out = out.sort_values("facility_priority_rank")
    elif "inspection_priority_score" in out.columns:
        out = out.sort_values("inspection_priority_score", ascending=False)
    return out.head(top_n).rename(
        columns={
            "facility_priority_rank": "시설 점검 순위",
            "facility_priority_level": "시설 우선등급",
            "facility_name": "저수지명",
            "address": "주소",
            "benefit_area": "수혜면적",
            "effective_capacity": "유효저수량",
            "total_capacity": "총저수량",
            "facility_latest_reservoir_rate": "시설별 최신 공개 저수율",
            "facility_latest_date": "시설별 기준일",
            "facility_scale_score": "시설 규모점수",
            "inspection_priority_score": "시설 점검 우선점수",
            "facility_priority_reason": "우선순위 사유",
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


def render_sidebar(watch: pd.DataFrame) -> tuple[list[str], list[str]]:
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
            st.session_state.watch_summary_top_n = 10
            st.session_state.watch_risk_top_n = 10
            st.session_state.watch_facility_top_n = 10
            st.rerun()

        st.markdown("#### 기간")
        st.caption(f"저수지 기준일: {latest_date(watch, ['reservoir_latest_date'])}")

        st.markdown("#### 지역/대상")
        selected_regions = st.multiselect("시·군", sigungu_options, key="watch_sigungu_filter")

        st.markdown("#### 모델/위험등급")
        selected_levels = st.multiselect("Watch 등급", levels, key="watch_level_filter")

    return selected_regions, selected_levels


def main() -> None:
    inject_css()

    watch, watch_msg = safe_read_data(str(WATCHLIST_PATH), required=True)
    facility, facility_msg = safe_read_data(str(FACILITY_STATUS_PATH), required=True)
    oldam_today, oldam_msg = safe_read_data(str(OLDAM_TODAY_PATH), required=False)
    live, _live_msg = safe_read_data(str(LIVE_FEATURE_PATH), required=False)
    training_history, _training_msg = safe_read_data(str(TRAINING_DATA_PATH), required=False)

    show_messages([watch_msg, facility_msg], stop_on_required=True)
    watch = exclude_unavailable_regions(watch)
    facility = exclude_unavailable_regions(facility)
    oldam_today = exclude_unavailable_regions(oldam_today)
    live = exclude_unavailable_regions(live)
    training_history = exclude_unavailable_regions(training_history)

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
            "facility_priority_rank",
            "benefit_area",
            "effective_capacity",
            "total_capacity",
            "sigungu_avg_reservoir_rate",
            "sigungu_min_reservoir_rate",
            "sigungu_reservoir_risk_score",
            "facility_scale_score",
            "inspection_priority_score",
            "facility_latest_reservoir_rate",
        ],
    )

    render_page_header(
        "저수지 현황 및 이상징후 Watchlist",
        "저수율 위험도와 시설 정보를 결합해 우선 점검할 저수지·시군을 빠르게 확인합니다.",
        build_basis_text(live, training_history, latest_date(watch, ["reservoir_latest_date"])),
    )

    selected_regions, selected_levels = render_sidebar(watch)
    filtered = watch.copy()
    if selected_regions:
        filtered = filtered[filtered["sigungu"].isin(selected_regions)]
    if selected_levels and "watch_level" in filtered.columns:
        filtered = filtered[filtered["watch_level"].isin(selected_levels)]

    st.markdown(
        f"""
        <div class="ag-filter">
        현재 필터: 지역 <b>{len(selected_regions) if selected_regions else 0}개</b> ·
        Watch 등급 <b>{", ".join(selected_levels) if selected_levels else "전체"}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    top_watch = filtered.sort_values("watch_rank").iloc[0] if not filtered.empty and "watch_rank" in filtered.columns else None
    severe_count = int(filtered["watch_level"].astype(str).str.contains("심각|경계|주의", regex=True).sum()) if "watch_level" in filtered.columns and not filtered.empty else 0
    facility_count = len(facility)

    if filtered.empty:
        render_empty_state()
        st.stop()

    summary_tab, risk_tab, raw_tab = st.tabs(
        ["Watchlist 요약", "저수율·시설 점검", "공개·원본 데이터"]
    )

    with summary_tab:
        summary_max_top = max(5, min(15, len(filtered))) if not filtered.empty else 5
        summary_top_n = st.slider(
            "점검 우선순위 표시 수",
            5,
            summary_max_top,
            min(st.session_state.get("watch_summary_top_n", 10), summary_max_top),
            key="watch_summary_top_n",
            help="Watchlist 요약 탭의 점검 우선순위 표에만 적용됩니다.",
        )
        render_kpi_cards(
            [
                ("Watch 대상 시·군", f"{len(filtered):,}곳", "현재 필터 기준 Watchlist 대상 수입니다."),
                ("주의 이상 대상", f"{severe_count:,}곳", "주의·경계·심각후보 등급 대상 수입니다."),
                ("전체 저수지 시설", f"{facility_count:,}개", "대시보드에 연결된 저수지 시설 수입니다."),
                ("최우선 시·군", str(top_watch["sigungu"]) if top_watch is not None else "N/A", "현재 필터 기준 Watch 1순위입니다."),
            ]
        )
        render_section_header("점검 우선순위", "저수율 위험점수가 높고 Watch 순위가 빠른 시·군을 먼저 확인합니다.")
        st.dataframe(format_display_dataframe(make_watch_priority_table(filtered, summary_top_n)), use_container_width=True, hide_index=True)

    with risk_tab:
        risk_max_top = max(5, min(15, len(filtered))) if not filtered.empty else 5
        risk_top_n = st.slider(
            "저수율 차트 표시 수",
            5,
            risk_max_top,
            min(st.session_state.get("watch_risk_top_n", 10), risk_max_top),
            key="watch_risk_top_n",
            help="저수율 위험도 차트에만 적용됩니다.",
        )
        render_section_header("저수율 위험도 해석", "평균과 최저 저수율을 함께 보며 지역 단위 위험을 확인합니다.")
        left, right = st.columns(2)
        with left:
            watch_fig = make_watch_bar(filtered, risk_top_n)
            if watch_fig:
                st.plotly_chart(watch_fig, use_container_width=True)
                st.caption("막대가 길수록 저수율 관점에서 우선 확인할 필요가 큽니다.")
            else:
                st.info("reservoir_risk_score 컬럼이 없어 위험도 차트를 건너뛰었습니다.")
        with right:
            rate_fig = make_rate_compare_chart(filtered, risk_top_n)
            if rate_fig:
                st.plotly_chart(rate_fig, use_container_width=True)
                st.caption("최저 저수율이 평균보다 크게 낮으면 특정 시설 중심의 현장 확인이 필요합니다.")
            else:
                st.info("평균·최저 저수율 컬럼이 없어 비교 차트를 건너뛰었습니다.")

        facility_options = filtered["sigungu"].dropna().astype(str).drop_duplicates().tolist() if "sigungu" in filtered.columns else []
        if st.session_state.get("watch_facility_focus") not in facility_options and facility_options:
            st.session_state.watch_facility_focus = facility_options[0]
        focus = st.selectbox(
            "시설 점검 시·군 선택",
            facility_options,
            key="watch_facility_focus",
            help="이 선택은 시설별 점검 우선순위와 최신 공개 저수지 현황에만 적용됩니다.",
        ) if facility_options else None
        if focus is None:
            render_empty_state()
            return
        selected_facility, removed_duplicates = prepare_facility(facility, focus)
        facility_for_table = attach_latest_oldam_to_facilities(selected_facility, oldam_today, focus)

        render_section_header(f"{focus} 시설별 점검 우선순위", "선택한 시·군 안에서 먼저 확인할 저수지 시설입니다.")
        render_selected_region_summary(watch, focus)
        st.caption(
            "아래 시설 표의 우선순위는 시·군 위험도와 시설 규모 정보를 결합한 점검 우선순위입니다. "
            "시·군 평균/최저 저수율은 지역 공통 지표이므로 시설별 행에 반복 표시하지 않습니다."
        )
        st.info("시설 점검 우선점수는 시설별 실시간 저수율이 아니라, 시·군 저수율 위험도와 시설 규모 정보를 결합한 행정 점검 우선순위입니다.")
        if "facility_latest_reservoir_rate" in facility_for_table.columns and facility_for_table["facility_latest_reservoir_rate"].isna().all():
            st.info("시설별 최신 저수율은 원천 데이터에서 직접 제공되지 않아, 시·군 단위 저수율과 시설 규모를 결합해 우선순위를 산정했습니다.")

        facility_max_top = max(5, min(15, len(facility_for_table))) if not facility_for_table.empty else 5
        facility_top_n = st.slider(
            "시설 우선순위 표시 수",
            5,
            facility_max_top,
            min(st.session_state.get("watch_facility_top_n", 10), facility_max_top),
            key="watch_facility_top_n",
            help="선택한 시·군의 시설별 점검 표에만 적용됩니다.",
        )
        facility_top = make_facility_priority_table(facility_for_table, facility_top_n)
        if facility_top.empty:
            render_empty_state("선택한 시·군의 저수지 시설 정보가 없습니다.")
        else:
            st.caption(f"표시 시설 {len(facility_for_table):,}개" + (f" · 중복 제거 {removed_duplicates:,}개" if removed_duplicates else ""))
            st.dataframe(format_display_dataframe(facility_top), use_container_width=True, hide_index=True)

    with raw_tab:
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
                oldam_sorted = oldam_view.sort_values(sort_col)
                render_missing_reservoir_note(oldam_sorted)
                st.dataframe(format_display_dataframe(oldam_sorted), use_container_width=True, hide_index=True)
        else:
            st.info(f"선택 데이터 파일이 없습니다: {OLDAM_TODAY_PATH}")

        render_section_header("상세 데이터 및 원본 결과 확인", "요약 이후 필요한 원본 Watchlist와 시설 목록을 확인합니다.")
        with st.expander("Watchlist 전체 보기", expanded=True):
            watch_display = filtered.sort_values("watch_rank")
            render_missing_reservoir_note(watch_display)
            st.dataframe(format_display_dataframe(watch_display), use_container_width=True, hide_index=True)
            st.download_button(
                label="Watchlist CSV 다운로드",
                data=filtered.to_csv(index=False).encode("utf-8-sig"),
                file_name="reservoir_watchlist_filtered.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with st.expander(f"{focus} 저수지 시설 전체 보기"):
            if facility_for_table.empty:
                render_empty_state("조건에 맞는 데이터가 없습니다")
            else:
                facility_detail = make_facility_priority_table(facility_for_table, len(facility_for_table))
                st.dataframe(format_display_dataframe(facility_detail), use_container_width=True, hide_index=True)
                st.download_button(
                    label=f"{focus} 저수지 시설 CSV 다운로드",
                    data=facility_for_table.to_csv(index=False).encode("utf-8-sig"),
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
