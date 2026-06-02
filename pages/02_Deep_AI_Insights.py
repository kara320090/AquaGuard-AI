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
LIVE_SUMMARY_PATH = REPORT_TABLES / "latest_live_risk_summary.csv"
TRAINING_DATA_PATH = PROCESSED / "01_reservoir_sigungu_daily.csv"
GRU_HISTORY_PATH = REPORT_TABLES / "ai_gru_training_history.csv"
AE_HISTORY_PATH = REPORT_TABLES / "ai_autoencoder_training_history.csv"
REPORT_PATH = META / "deep_ai_model_report.md"

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


def render_top_n_slider(label: str, total: int, key: str, default: int = 10, max_cap: int = 14, help_text: str | None = None) -> int:
    if total <= 5:
        return max(total, 0)
    max_value = min(max_cap, total)
    value = min(max(st.session_state.get(key, default), 5), max_value)
    return st.slider(label, 5, max_value, value, key=key, help=help_text)


def format_value(value, suffix: str = "", decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}{suffix}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):,.{decimals}f}{suffix}"
    return str(value) if str(value) else "N/A"


def safe_get(row: pd.Series, candidates: list[str], default=np.nan):
    for col in candidates:
        if col in row.index and pd.notna(row.get(col)):
            return row.get(col)
    return default


def format_score(value, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


RAW_MISSING_TEXT = {"", "-", "none", "nan", "nat", "<na>", "n/a"}
DATE_DISPLAY_COLUMNS = {
    "기준일",
    "reservoir_latest_date",
    "latest_measurement_date",
    "soil_data_date",
    "date",
    "base_date",
    "analysis_date",
    "target_date",
    "month",
}
RESERVOIR_DATE_COLUMNS = {"기준일", "reservoir_latest_date", "latest_measurement_date"}
ANALYSIS_BASIS_COLUMNS = ["분석 기준", "analysis_mode", "analysis_basis"]
SCORE_DISPLAY_COLUMNS = {
    "위험점수",
    "risk_score",
    "score",
    "priority_score",
    "autoencoder_anomaly_score",
    "reconstruction_loss",
    "deep_ai_risk_score",
    "forecast_risk_score",
    "Deep AI 위험도",
    "예측 위험도",
    "Anomaly Score",
    "Reconstruction Loss",
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
        or "loss" in lower
        or "점수" in column_text
        or "위험도" in column_text
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


def latest_month_from_columns(frames: list[pd.DataFrame], columns: list[str]) -> str:
    months = months_from_columns(frames, columns)
    return months[-1] if months else "N/A"


def current_month_from_live(live: pd.DataFrame) -> str:
    live_month = latest_month_from_columns([live], ["soil_data_date", "date", "base_date", "target_date"])
    if live_month != "N/A":
        return live_month
    return pd.Timestamp.today().to_period("M").strftime("%Y-%m")


def select_default_ai_comparison_month(current_month: str, available_months: list[str]) -> tuple[str, bool]:
    if not available_months or current_month == "N/A":
        return "N/A", False

    current = pd.Period(current_month, freq="M")
    candidates = [pd.Period(month, freq="M") for month in available_months]
    same_month_previous_years = [month for month in candidates if month.month == current.month and month.year < current.year]
    if same_month_previous_years:
        return str(max(same_month_previous_years)), False

    previous_months = [month for month in candidates if month < current]
    if previous_months:
        nearest = min(previous_months, key=lambda month: abs(month.ordinal - current.ordinal))
        return str(nearest), True

    nearest = min(candidates, key=lambda month: abs(month.ordinal - current.ordinal))
    return str(nearest), True


def get_autoencoder_score_columns(df: pd.DataFrame) -> dict[str, str | None]:
    raw_candidates = ["reconstruction_error", "autoencoder_reconstruction_error", "valid_recon_loss"]
    score_candidates = ["autoencoder_anomaly_score", "anomaly_score"]
    level_candidates = ["autoencoder_anomaly_level", "anomaly_level"]
    return {
        "raw": next((col for col in raw_candidates if col in df.columns), None),
        "score": next((col for col in score_candidates if col in df.columns), None),
        "level": next((col for col in level_candidates if col in df.columns), None),
    }


def attach_autoencoder_reconstruction(summary: pd.DataFrame, anomaly: pd.DataFrame) -> pd.DataFrame:
    if summary.empty or anomaly.empty or "reconstruction_error" not in anomaly.columns or "reconstruction_error" in summary.columns:
        return summary

    join_cols = ["sigungu"]
    if "base_date" in summary.columns and "base_date" in anomaly.columns:
        join_cols.append("base_date")
    raw = anomaly[join_cols + ["reconstruction_error"]].drop_duplicates(join_cols)
    return summary.merge(raw, on=join_cols, how="left")


def build_historical_month_snapshot(history: pd.DataFrame, comparison_month: str) -> pd.DataFrame:
    if history.empty or comparison_month == "N/A" or "date" not in history.columns or "sigungu" not in history.columns:
        return pd.DataFrame()

    df = history.copy()
    df["month"] = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M").astype(str)
    month_df = df[df["month"] == comparison_month]
    if month_df.empty:
        return pd.DataFrame()

    agg_map = {}
    for col in ["avg_reservoir_rate", "min_reservoir_rate", "reservoir_risk_score", "low_reservoir_count_30"]:
        if col in month_df.columns:
            agg_map[col] = "mean"
    if not agg_map:
        return pd.DataFrame()

    return month_df.groupby("sigungu", as_index=False).agg(agg_map)


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


def region_checkbox_key(prefix: str, region: str) -> str:
    return f"{prefix}_region_checkbox_{region}"


def set_all_region_checkboxes(prefix: str, regions: list[str]) -> None:
    select_all = bool(st.session_state.get(f"{prefix}_region_select_all", False))
    for region in regions:
        st.session_state[region_checkbox_key(prefix, region)] = select_all


def update_region_select_all(prefix: str, regions: list[str]) -> None:
    st.session_state[f"{prefix}_region_select_all"] = all(
        bool(st.session_state.get(region_checkbox_key(prefix, region), False))
        for region in regions
    )


def render_region_checkboxes(regions: list[str], prefix: str, selection_key: str) -> list[str]:
    selected_existing = [x for x in st.session_state.get(selection_key, regions) if x in regions]
    all_key = f"{prefix}_region_select_all"

    if all_key not in st.session_state:
        st.session_state[all_key] = True

    for region in regions:
        key = region_checkbox_key(prefix, region)
        if key not in st.session_state:
            st.session_state[key] = bool(st.session_state[all_key]) or region in selected_existing
        if bool(st.session_state[all_key]):
            st.session_state[key] = True

    st.checkbox("전체 지역", key=all_key, on_change=set_all_region_checkboxes, args=(prefix, regions))

    selected_regions: list[str] = []
    cols = st.columns(2)
    for idx, region in enumerate(regions):
        key = region_checkbox_key(prefix, region)
        checked = cols[idx % 2].checkbox(region, key=key, on_change=update_region_select_all, args=(prefix, regions))
        if checked:
            selected_regions.append(region)

    st.session_state[selection_key] = selected_regions
    return selected_regions


def render_sidebar(
    summary: pd.DataFrame,
    live_basis_month: str,
    default_comparison_month: str,
    month_options: list[str],
) -> tuple[list[str], list[str], str]:
    regions = summary.sort_values("deep_ai_rank")["sigungu"].dropna().astype(str).tolist() if "deep_ai_rank" in summary.columns else sorted(summary["sigungu"].dropna().astype(str).tolist())
    levels = available_levels(summary)
    if "ai_region_filter" not in st.session_state:
        st.session_state.ai_region_filter = regions
    if "ai_level_filter" not in st.session_state:
        st.session_state.ai_level_filter = levels

    st.session_state.ai_region_filter = [x for x in st.session_state.ai_region_filter if x in regions]
    st.session_state.ai_level_filter = [x for x in st.session_state.ai_level_filter if x in levels] or levels

    with st.sidebar:
        st.header("필터")
        if st.button("기본값으로 초기화", use_container_width=True):
            st.session_state.ai_region_filter = regions
            st.session_state.ai_region_select_all = True
            for region in regions:
                st.session_state[region_checkbox_key("ai", region)] = True
            st.session_state.ai_level_filter = levels
            st.session_state.ai_top_n = 10
            st.session_state.ai_basis_top_n = 10
            st.session_state.ai_insight_top_n = 10
            st.session_state.ai_comparison_month = default_comparison_month
            st.rerun()

        st.markdown("#### 기간")
        st.caption(f"Live 기준월: {live_basis_month}")
        st.caption(f"모델 출력 기준일: {latest_date(summary, ['base_date', 'target_date'])}")
        if "ai_comparison_month" not in st.session_state:
            st.session_state.ai_comparison_month = default_comparison_month
        st.session_state.ai_comparison_month = (
            st.session_state.ai_comparison_month
            if st.session_state.ai_comparison_month in month_options
            else default_comparison_month
        )
        selected_comparison_month = st.selectbox(
            "AI 비교 기준월",
            month_options or ["N/A"],
            index=(month_options.index(st.session_state.ai_comparison_month) if month_options and st.session_state.ai_comparison_month in month_options else 0),
            key="ai_comparison_month",
            help="Live 기준월과 계절성이 같은 전년도 동일 월을 우선 선택합니다.",
        )

        st.markdown("#### 지역/대상")
        st.caption("시·군")
        selected_regions = render_region_checkboxes(regions, "ai", "ai_region_filter")

        st.markdown("#### 모델/위험등급")
        selected_levels = st.multiselect("Deep AI 등급", levels, key="ai_level_filter")

    return selected_regions, selected_levels, selected_comparison_month


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
        "reconstruction_error",
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
            "reconstruction_error": "Reconstruction Loss",
            "autoencoder_anomaly_score": "Anomaly Score",
            "autoencoder_anomaly_level": "Anomaly Level",
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


def make_historical_comparison_chart(month_snapshot: pd.DataFrame, top_n: int):
    if month_snapshot.empty or "reservoir_risk_score" not in month_snapshot.columns:
        return None
    plot_df = month_snapshot.sort_values("reservoir_risk_score", ascending=False).head(top_n)
    fig = px.bar(
        plot_df.sort_values("reservoir_risk_score"),
        x="reservoir_risk_score",
        y="sigungu",
        orientation="h",
        hover_data=[c for c in ["avg_reservoir_rate", "min_reservoir_rate"] if c in plot_df.columns],
        title="AI 비교 기준월의 저수율 위험도: 같은 계절에 어느 지역이 취약했는가?",
    )
    fig.update_layout(xaxis_title="저수율 위험점수", yaxis_title="시·군", height=max(360, 32 * len(plot_df) + 120))
    return fig


def main() -> None:
    inject_css()

    summary, summary_msg = safe_read_data(str(SUMMARY_PATH), required=True)
    forecast, forecast_msg = safe_read_data(str(FORECAST_PATH), required=False)
    anomaly, anomaly_msg = safe_read_data(str(ANOMALY_PATH), required=False)
    live_summary, live_msg = safe_read_data(str(LIVE_SUMMARY_PATH), required=False)
    training_history, training_msg = safe_read_data(str(TRAINING_DATA_PATH), required=False)
    gru_history, gru_msg = safe_read_data(str(GRU_HISTORY_PATH), required=False)
    ae_history, ae_msg = safe_read_data(str(AE_HISTORY_PATH), required=False)
    report, report_msg = read_text_file(str(REPORT_PATH))

    show_messages([summary_msg], stop_on_required=True)
    summary = exclude_unavailable_regions(summary)
    forecast = exclude_unavailable_regions(forecast)
    anomaly = exclude_unavailable_regions(anomaly)
    live_summary = exclude_unavailable_regions(live_summary)
    training_history = exclude_unavailable_regions(training_history)

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
    live_summary = normalize_numeric(live_summary, ["final_live_water_risk_score"])
    training_history = normalize_numeric(training_history, ["avg_reservoir_rate", "min_reservoir_rate", "reservoir_risk_score", "low_reservoir_count_30"])
    summary = attach_autoencoder_reconstruction(summary, anomaly)
    summary = normalize_numeric(summary, ["reconstruction_error", "autoencoder_anomaly_score"])
    gru_history = normalize_numeric(gru_history, ["epoch", "train_loss", "valid_mae", "valid_r2"])
    ae_history = normalize_numeric(ae_history, ["epoch", "train_recon_loss", "valid_recon_loss"])

    best_model, main_metric, performance_cards = build_performance_cards(report, gru_history, ae_history)
    live_basis_month = current_month_from_live(live_summary)
    live_basis_date = latest_date(live_summary, ["soil_data_date", "date", "base_date", "target_date"])
    training_months = months_from_columns([training_history], ["date"])
    ai_output_month = latest_month_from_columns([summary, forecast, anomaly], ["base_date", "target_date"])
    training_final_month = training_months[-1] if training_months else ai_output_month
    default_comparison_month, comparison_fallback = select_default_ai_comparison_month(live_basis_month, training_months)
    month_options = training_months or ([ai_output_month] if ai_output_month != "N/A" else ["N/A"])

    selected_regions, selected_levels, selected_comparison_month = render_sidebar(
        summary,
        live_basis_month,
        default_comparison_month,
        month_options,
    )
    render_page_header(
        "Deep AI 저수율 예측·이상탐지",
        "GRU 7일 후 저수율 예측과 AutoEncoder 이상 패턴 탐지를 결합해 학습 기반 위험 신호와 계절 비교 기준을 확인합니다.",
        f"Live 기준일 {live_basis_date}<br>AI 비교 기준월 {selected_comparison_month}<br>학습 데이터 최종월 {training_final_month}",
    )
    st.info(
        f"Live 기준일 {live_basis_date}의 최신 공개 데이터와 AI 비교 기준월 {selected_comparison_month}의 학습 데이터를 계절 기준으로 비교합니다."
    )
    if comparison_fallback and selected_comparison_month == default_comparison_month:
        st.warning(f"전년도 동일 월 데이터가 없어 가장 가까운 학습 데이터 월({selected_comparison_month})을 AI 비교 기준월로 사용합니다.")

    filtered = summary.copy()
    filtered = filtered[filtered["sigungu"].isin(selected_regions)] if selected_regions else filtered.iloc[0:0]
    if selected_levels and "deep_ai_risk_level" in filtered.columns:
        filtered = filtered[filtered["deep_ai_risk_level"].isin(selected_levels)]

    st.markdown(
        f"""
        <div class="ag-filter">
        현재 필터: 지역 <b>{len(selected_regions) if selected_regions else 0}개</b> ·
        Deep AI 등급 <b>{", ".join(selected_levels) if selected_levels else "전체"}</b> ·
        AI 비교 기준월 <b>{selected_comparison_month}</b>
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

    basis_tab, insight_tab, raw_tab = st.tabs(["AI 계절·성능", "예측·이상탐지", "원본 결과"])

    with basis_tab:
        basis_top_n = render_top_n_slider(
            "계절 비교 표시 수",
            len(filtered),
            key="ai_basis_top_n",
            help="AI 계절 비교 기준 탭의 학습 데이터 요약 차트에만 적용됩니다.",
        )
        render_section_header(
            "AI 계절 비교 기준",
            "현재 Live 월과 같은 계절의 학습 데이터 월을 골라 AI 해석의 계절 기준을 분리해서 보여줍니다.",
        )
        historical_snapshot = build_historical_month_snapshot(training_history, selected_comparison_month)
        historical_fig = make_historical_comparison_chart(historical_snapshot, basis_top_n)
        if historical_fig:
            st.plotly_chart(historical_fig, use_container_width=True)
            st.caption(
                f"{selected_comparison_month} 학습 데이터의 월평균 저수율 위험도입니다. Deep AI 출력값을 재계산하지 않고 계절 비교 기준으로만 사용합니다."
            )
            with st.expander(f"{selected_comparison_month} 학습 데이터 월별 요약 보기"):
                st.dataframe(format_display_dataframe(historical_snapshot), use_container_width=True, hide_index=True)
        else:
            st.warning(f"AI 비교 기준월에 맞는 학습 데이터가 없습니다: {selected_comparison_month}")

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

    with insight_tab:
        insight_top_n = render_top_n_slider(
            "예측·우선순위 표시 수",
            len(filtered),
            key="ai_insight_top_n",
            help="점검 우선순위와 예측·이상탐지 차트에만 적용됩니다.",
        )
        render_section_header("점검 우선순위", "AI 예측과 이상탐지 결과를 결합해 먼저 확인할 지역입니다.")
        priority = build_priority_table(filtered, insight_top_n)
        if priority.empty:
            render_empty_state()
        else:
            st.dataframe(format_display_dataframe(priority), use_container_width=True, hide_index=True)

        render_section_header("예측·이상탐지 분석", "각 차트는 위험 순위, 예측 변화, 이상 패턴을 분리해서 보여줍니다.")
        left, right = st.columns(2)
        with left:
            ranking_fig = make_ai_ranking_chart(filtered, insight_top_n)
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

        anomaly_fig = make_anomaly_chart(filtered, insight_top_n)
        if anomaly_fig:
            st.plotly_chart(anomaly_fig, use_container_width=True)
            st.caption("이상점수가 높을수록 과거 정상 패턴과 다른 저수율 흐름입니다.")
        else:
            st.info("autoencoder_anomaly_score 컬럼이 없어 이상탐지 차트를 건너뛰었습니다.")

        detail_options = filtered["sigungu"].dropna().astype(str).drop_duplicates().tolist() if "sigungu" in filtered.columns else []
        if not detail_options:
            render_empty_state()
            return
        if st.session_state.get("ai_detail_region_select") not in detail_options:
            st.session_state.ai_detail_region_select = detail_options[0]
        focus = st.selectbox(
            "상세 해석 지역 선택",
            detail_options,
            key="ai_detail_region_select",
            help="이 선택은 현재 탭의 상세 해석에만 적용됩니다.",
        )
        render_section_header(f"{focus} 상세 해석", "선택한 지역의 예측값과 이상탐지 결과를 같은 행 기준으로 설명합니다.")
        focus_row = filtered[filtered["sigungu"] == focus]
        if focus_row.empty:
            st.warning(f"선택한 지역이 현재 필터 결과에 없습니다: {focus}")
        else:
            row = focus_row.iloc[0]
            render_missing_reservoir_note(focus_row)
            ae_cols = get_autoencoder_score_columns(filtered)
            ae_raw = safe_get(row, [ae_cols["raw"]] if ae_cols["raw"] else [], default=np.nan)
            ae_score = safe_get(row, [ae_cols["score"]] if ae_cols["score"] else [], default=np.nan)
            ae_level = safe_get(row, [ae_cols["level"]] if ae_cols["level"] else [], default="N/A")
            ae_cards = []
            if ae_cols["raw"]:
                ae_cards.append(("Reconstruction Loss", format_score(ae_raw, 4), "원본 AutoEncoder 재구성 오차입니다."))
            else:
                st.info("reconstruction_error 컬럼이 없어 Reconstruction Loss 표시는 건너뛰었습니다.")
            if ae_cols["score"]:
                ae_cards.append(("Anomaly Score", format_score(ae_score, 2), "대시보드에서 쓰는 정규화 이상점수입니다."))
            else:
                st.info("autoencoder_anomaly_score 컬럼이 없어 Anomaly Score 표시는 건너뛰었습니다.")
            if ae_cols["level"]:
                ae_cards.append(("Anomaly Level", str(ae_level), "선택한 행의 AutoEncoder 이상탐지 등급입니다."))
            else:
                st.info("autoencoder_anomaly_level 컬럼이 없어 Anomaly Level 표시는 건너뛰었습니다.")
            if ae_cards:
                render_kpi_cards(ae_cards)
                st.caption("AutoEncoder 점수는 학습 데이터 패턴 대비 재구성 오차 기반 이상 신호이며, 수치가 높을수록 평소 패턴과 다름을 의미합니다.")
            st.success(
                f"{focus}의 Deep AI 위험도는 {format_value(row.get('deep_ai_risk_score'), '점', 1)}이며 "
                f"등급은 {row.get('deep_ai_risk_level', 'N/A')}입니다. "
                f"현재 평균 저수율 {format_value(row.get('current_avg_reservoir_rate'), '%', 1)}에서 "
                f"7일 후 {format_value(row.get('pred_avg_reservoir_rate_7d'), '%', 1)}로 예측되었습니다. "
                f"AutoEncoder Anomaly Score는 {format_score(ae_score, 2)}, Anomaly Level은 {ae_level}입니다."
            )

    with raw_tab:
        render_section_header("상세 데이터 및 원본 결과 확인", "요약 이후 필요한 예측·이상탐지 원본 결과를 확인합니다.")
        with st.expander("Deep AI 시·군 요약", expanded=True):
            sorted_filtered = filtered.sort_values("deep_ai_rank")
            render_missing_reservoir_note(sorted_filtered)
            st.dataframe(format_display_dataframe(sorted_filtered), use_container_width=True, hide_index=True)
        with st.expander("GRU 7일 후 저수율 예측 결과"):
            if forecast.empty:
                st.info(f"선택 데이터 파일이 없습니다: {FORECAST_PATH}")
            else:
                st.dataframe(format_display_dataframe(forecast), use_container_width=True, hide_index=True)
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
                st.dataframe(format_display_dataframe(anomaly), use_container_width=True, hide_index=True)
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
                st.dataframe(format_display_dataframe(gru_history), use_container_width=True, hide_index=True)
            if not ae_history.empty:
                st.markdown("#### AutoEncoder 학습 이력")
                st.dataframe(format_display_dataframe(ae_history), use_container_width=True, hide_index=True)

        st.info(
            "Deep AI 결과는 공개데이터 기반 예측·이상탐지 참고자료입니다. 실제 농업용수 대응 여부는 현장 저수율, "
            "관로, 수리권, 수질, 행정 협의를 함께 검토해야 합니다."
        )
        show_messages([forecast_msg, anomaly_msg, live_msg, training_msg, gru_msg, ae_msg, report_msg])


if __name__ == "__main__":
    main()
