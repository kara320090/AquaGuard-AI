from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
REPORT_TABLES = ROOT / "reports" / "tables"
REPORT_FIGURES = ROOT / "reports" / "figures"
META = ROOT / "data" / "metadata"

FEATURE_PATH = PROCESSED / "aquaguard_sigungu_features.csv"
CANDIDATE_PATH = PROCESSED / "alternative_source_candidates.csv"
TOP5_PATH = REPORT_TABLES / "alternative_source_top5_by_sigungu.csv"
LIVE_SUMMARY_PATH = REPORT_TABLES / "latest_live_risk_summary.csv"
AI_SUMMARY_PATH = REPORT_TABLES / "ai_sigungu_deep_summary.csv"
GRU_HISTORY_PATH = REPORT_TABLES / "ai_gru_training_history.csv"
AE_HISTORY_PATH = REPORT_TABLES / "ai_autoencoder_training_history.csv"
AI_REPORT_PATH = META / "deep_ai_model_report.md"
VALIDATION_PATH = META / "final_validation_report.csv"
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

FIG_RANKING = REPORT_FIGURES / "01_final_risk_ranking.png"
FIG_COMPONENTS = REPORT_FIGURES / "02_risk_components_stacked.png"
FIG_SCATTER = REPORT_FIGURES / "03_reservoir_vs_alternative_shortage_scatter.png"
FIG_TOP5 = REPORT_FIGURES / "04_top5_priority_table.png"
FIG_ALT = REPORT_FIGURES / "05_alternative_source_top1_by_risk_area.png"

REPORT_FIGURES_LIST = [
    ("01_final_risk_ranking.png", "시·군별 최종 위험도 순위"),
    ("02_risk_components_stacked.png", "위험도 구성요소별 기여도"),
    ("03_reservoir_vs_alternative_shortage_scatter.png", "저수율 위험도 vs 대체 수원 접근성 부족도"),
    ("04_top5_priority_table.png", "우선 평가 대상 TOP 5"),
    ("05_alternative_source_top1_by_risk_area.png", "위험지역별 1순위 대체 수원 후보"),
]

SIGUNGU_CENTROIDS = {
    "천안시": (36.8151, 127.1139),
    "공주시": (36.4465, 127.1190),
    "보령시": (36.3334, 126.6128),
    "아산시": (36.7898, 127.0026),
    "서산시": (36.7849, 126.4503),
    "논산시": (36.1871, 127.0987),
    "계룡시": (36.2746, 127.2486),
    "당진시": (36.8930, 126.6280),
    "금산군": (36.1089, 127.4880),
    "부여군": (36.2756, 126.9098),
    "서천군": (36.0803, 126.6910),
    "청양군": (36.4592, 126.8023),
    "홍성군": (36.6013, 126.6608),
    "예산군": (36.6828, 126.8489),
    "태안군": (36.7456, 126.2980),
}

RISK_ORDER = ["심각", "심각후보", "경계", "주의", "낮음"]
RISK_COLORS = {
    "심각": "#c62828",
    "심각후보": "#d84315",
    "경계": "#ef6c00",
    "주의": "#f9a825",
    "낮음": "#2e7d32",
}

COMPONENTS = [
    ("rain_shortage_score", "강우 부족도", 0.25),
    ("reservoir_risk_score", "저수율 위험도", 0.25),
    ("groundwater_dependency_score", "관정 의존도", 0.20),
    ("crop_water_demand_score", "작물 물수요 지수", 0.20),
    ("alternative_source_access_shortage_score", "대체 수원 접근성 부족도", 0.10),
]

SOURCE_CONFIG = {
    "최종 산정": {
        "score": "final_water_risk_score",
        "level": "final_water_risk_level",
        "rank": "final_priority_rank",
        "driver": "main_risk_driver",
        "action": "recommended_action",
        "date": "reservoir_latest_date",
    },
    "Live 업데이트": {
        "score": "final_live_water_risk_score",
        "level": "final_live_water_risk_level",
        "rank": "final_live_priority_rank",
        "driver": "live_main_risk_driver",
        "action": "recommended_action",
        "date": "soil_data_date",
    },
    "Deep AI 예측": {
        "score": "deep_ai_risk_score",
        "level": "deep_ai_risk_level",
        "rank": "deep_ai_rank",
        "driver": "autoencoder_anomaly_level",
        "action": "recommended_action",
        "date": "target_date",
    },
}

st.set_page_config(
    page_title="충남 AquaGuard AI",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
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
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        [data-testid="stMetricLabel"] { color: #475569; }
        div[data-testid="stCaptionContainer"] { color: #64748b; }
        .ag-page-title {
            margin: 0 0 0.35rem 0;
            font-size: 2.35rem;
            line-height: 1.18;
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


def format_value(value, suffix: str = "", decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}{suffix}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):,.{decimals}f}{suffix}"
    text = str(value)
    return text if text else "N/A"


RAW_MISSING_TEXT = {"", "-", "none", "nan", "nat", "<na>", "n/a"}
DATE_DISPLAY_COLUMNS = {
    "기준일",
    "reservoir_latest_date",
    "latest_measurement_date",
    "soil_data_date",
    "date",
    "base_date",
    "analysis_date",
    "basis_date",
    "target_date",
}
RESERVOIR_DATE_COLUMNS = {"기준일", "reservoir_latest_date", "latest_measurement_date", "basis_date"}
ANALYSIS_BASIS_COLUMNS = ["분석 기준", "analysis_mode", "analysis_basis"]
SCORE_DISPLAY_COLUMNS = {
    "위험점수",
    "risk_score",
    "score",
    "priority_score",
    "autoencoder_anomaly_score",
    "reconstruction_loss",
    "후보점수",
    "Deep AI 위험도",
    "예측 위험도",
    "Live 위험점수",
    "기준 대비 변화",
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
    if isinstance(value, (datetime, pd.Timestamp, np.datetime64)):
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
        or "변화" in column_text
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


def prepare_main_detail_display(df: pd.DataFrame) -> pd.DataFrame:
    display_df = format_display_dataframe(df)
    priority_columns = ["시·군", "위험점수", "위험등급", "우선순위", "주요 위험 원인", "권고 조치", "기준일", "분석 기준"]
    ordered = [col for col in priority_columns if col in display_df.columns]
    ordered.extend([col for col in display_df.columns if col not in ordered])
    return display_df[ordered]


def safe_float(value, default: float | None = None) -> float | None:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def render_kpi_cards(cards: list[tuple[str, str, str | None]]) -> None:
    cols = st.columns(len(cards))
    for col, (label, value, help_text) in zip(cols, cards):
        col.metric(label, value, help=help_text)


@st.cache_data(show_spinner=False)
def safe_read_data(path_text: str, required: bool = False) -> tuple[pd.DataFrame, str | None]:
    path = Path(path_text)
    if not path.exists():
        level = "필수" if required else "선택"
        return pd.DataFrame(), f"{level} 데이터 파일이 없습니다: {path}"

    try:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path), None
        if suffix == ".parquet":
            return pd.read_parquet(path), None
        if suffix == ".json":
            return pd.read_json(path), None
        return pd.DataFrame(), f"지원하지 않는 데이터 형식입니다: {path}"
    except Exception as exc:  # noqa: BLE001 - dashboard should show a readable state.
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


def show_load_messages(messages: list[str | None], required_stop: bool = False) -> None:
    missing_required = False
    for message in messages:
        if not message:
            continue
        if message.startswith("필수"):
            missing_required = True
            st.warning(message)
        elif "읽지 못했습니다" in message or "지원하지 않는" in message:
            st.warning(message)
        else:
            st.info(message)
    if required_stop and missing_required:
        st.stop()


def normalize_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def latest_date_from_columns(frames: list[pd.DataFrame], columns: list[str]) -> str:
    dates: list[pd.Timestamp] = []
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


def resolve_project_root() -> Path:
    return Path(__file__).resolve().parent


def resolve_report_figure_path(filename: str) -> Path:
    return resolve_project_root() / "reports" / "figures" / filename


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


def available_levels(df: pd.DataFrame, level_col: str = "risk_level") -> list[str]:
    if df.empty or level_col not in df.columns:
        return []
    found = [x for x in df[level_col].dropna().astype(str).unique().tolist()]
    known = [x for x in RISK_ORDER if x in found]
    unknown = sorted([x for x in found if x not in RISK_ORDER])
    return known + unknown


def is_risky(level: str | None) -> bool:
    return str(level) in {"주의", "경계", "심각", "심각후보"}


def risk_color(level: str) -> str:
    return RISK_COLORS.get(str(level), "#64748b")


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


def build_performance_summary(
    report: str,
    gru_history: pd.DataFrame,
    ae_history: pd.DataFrame,
) -> tuple[str, str, list[tuple[str, str, str]]]:
    report_metrics = parse_report_metrics(report)
    metric_cards: list[tuple[str, str, str]] = []

    if "검증 MAE" in report_metrics:
        metric_cards.append(
            (
                "GRU 검증 MAE",
                f"{report_metrics['검증 MAE']:.4f}",
                "낮을수록 7일 후 평균 저수율 예측값과 실제값의 차이가 작습니다.",
            )
        )
    if "검증 R2" in report_metrics:
        metric_cards.append(
            (
                "GRU 검증 R2",
                f"{report_metrics['검증 R2']:.4f}",
                "1에 가까울수록 검증 데이터의 저수율 변동을 더 잘 설명합니다.",
            )
        )
    if "검증 샘플 수" in report_metrics:
        metric_cards.append(
            (
                "검증 샘플 수",
                f"{int(report_metrics['검증 샘플 수']):,}",
                "성능 확인에 사용된 시계열 샘플 수입니다.",
            )
        )

    if not gru_history.empty and "valid_mae" in gru_history.columns:
        best_mae = pd.to_numeric(gru_history["valid_mae"], errors="coerce").min()
        if pd.notna(best_mae):
            metric_cards.append(
                (
                    "학습 이력 최저 MAE",
                    f"{best_mae:.4f}",
                    "학습 로그에 기록된 epoch별 검증 MAE 중 최저값입니다.",
                )
            )
    if not ae_history.empty and "valid_recon_loss" in ae_history.columns:
        best_loss = pd.to_numeric(ae_history["valid_recon_loss"], errors="coerce").min()
        if pd.notna(best_loss):
            metric_cards.append(
                (
                    "AutoEncoder 최저 재구성 손실",
                    f"{best_loss:.4f}",
                    "낮을수록 정상 패턴 복원이 안정적입니다.",
                )
            )

    best_model = "PyTorch GRU" if "검증 MAE" in report_metrics or not gru_history.empty else "N/A"
    main_metric = (
        f"MAE {report_metrics['검증 MAE']:.4f}"
        if "검증 MAE" in report_metrics
        else (metric_cards[0][1] if metric_cards else "N/A")
    )
    return best_model, main_metric, metric_cards


def make_source_view(
    mode: str,
    features: pd.DataFrame,
    live: pd.DataFrame,
    ai_summary: pd.DataFrame,
) -> pd.DataFrame:
    if mode == "Live 업데이트" and not live.empty:
        source = live.copy()
    elif mode == "Deep AI 예측" and not ai_summary.empty:
        source = ai_summary.copy()
    else:
        source = features.copy()
        mode = "최종 산정"

    config = SOURCE_CONFIG[mode]
    base_actions = pd.DataFrame()
    if not features.empty and {"sigungu", "recommended_action"}.issubset(features.columns):
        base_actions = features[["sigungu", "recommended_action"]].drop_duplicates()

    if mode != "최종 산정" and not base_actions.empty and "recommended_action" not in source.columns:
        source = source.merge(base_actions, on="sigungu", how="left")

    out = pd.DataFrame()
    out["sigungu"] = source["sigungu"] if "sigungu" in source.columns else pd.Series(dtype=str)
    out["risk_score"] = pd.to_numeric(source.get(config["score"], np.nan), errors="coerce")
    out["risk_level"] = source.get(config["level"], "N/A").astype(str) if config["level"] in source.columns else "N/A"
    out["priority_rank"] = pd.to_numeric(source.get(config["rank"], np.nan), errors="coerce")
    out["main_risk_driver"] = source.get(config["driver"], "N/A") if config["driver"] in source.columns else "N/A"
    out["recommended_action"] = source.get(config["action"], np.nan) if config["action"] in source.columns else np.nan
    out["basis_date"] = source.get(config["date"], np.nan) if config["date"] in source.columns else np.nan
    out["analysis_mode"] = mode

    if mode == "Live 업데이트":
        for col in ["live_score_delta_from_baseline", "live_weather_source", "live_reservoir_source", "live_soil_source"]:
            if col in source.columns:
                out[col] = source[col]
    if mode == "Deep AI 예측":
        for col in ["current_avg_reservoir_rate", "pred_avg_reservoir_rate_7d", "forecast_drop_7d", "forecast_risk_score", "autoencoder_anomaly_score"]:
            if col in source.columns:
                out[col] = source[col]

    return out.dropna(subset=["sigungu"]).reset_index(drop=True)


def build_rule_based_reason(row: pd.Series) -> str:
    values = []
    for col, label, _weight in COMPONENTS:
        value = safe_float(row.get(col), None)
        if value is not None:
            values.append((label, value))
    if not values:
        driver = row.get("main_risk_driver", None)
        return str(driver) if driver and pd.notna(driver) else "규칙 기반 설명을 만들 수 있는 지표가 없습니다"
    label, value = max(values, key=lambda item: item[1])
    return f"규칙 기반 설명: {label} 점수가 가장 높음({value:.1f}점)"


def build_priority_table(source_df: pd.DataFrame, features: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if source_df.empty:
        return pd.DataFrame()

    base = source_df.copy()
    if not features.empty:
        component_cols = ["sigungu", *[c for c, _label, _weight in COMPONENTS]]
        component_cols = [c for c in component_cols if c in features.columns]
        if component_cols:
            base = base.merge(features[component_cols].drop_duplicates("sigungu"), on="sigungu", how="left")

    base = base.sort_values(["priority_rank", "risk_score"], ascending=[True, False]).head(top_n)

    rows = []
    for _, row in base.iterrows():
        reason = row.get("main_risk_driver", None)
        if reason is None or pd.isna(reason) or str(reason).strip() in {"", "N/A", "nan"}:
            reason = build_rule_based_reason(row)

        action = row.get("recommended_action", None)
        if action is None or pd.isna(action) or str(action).strip() in {"", "nan"}:
            action = "현장 점검 및 대체 수원 후보 확인" if is_risky(row.get("risk_level")) else "정기 모니터링 유지"

        rows.append(
            {
                "순위": format_value(row.get("priority_rank"), decimals=0),
                "대상": row.get("sigungu", "N/A"),
                "위험점수": format_value(row.get("risk_score"), "점", 1),
                "위험등급": row.get("risk_level", "N/A"),
                "판단 근거": reason,
                "권고 조치": action,
            }
        )
    return pd.DataFrame(rows)


def make_ranking_chart(df: pd.DataFrame, title: str, top_n: int) -> go.Figure | None:
    if df.empty:
        return None
    plot_df = df.sort_values("risk_score", ascending=False).head(top_n).sort_values("risk_score")
    fig = go.Figure(
        go.Bar(
            x=plot_df["risk_score"],
            y=plot_df["sigungu"],
            orientation="h",
            text=plot_df["risk_score"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"),
            textposition="outside",
            marker_color=[risk_color(x) for x in plot_df["risk_level"]],
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="위험점수",
        yaxis_title="시·군",
        height=max(380, 34 * len(plot_df) + 140),
        margin=dict(l=20, r=40, t=70, b=40),
        xaxis=dict(range=[0, max(100, float(plot_df["risk_score"].max()) * 1.15)]),
    )
    return fig


def make_distribution_chart(df: pd.DataFrame) -> go.Figure | None:
    if df.empty or "risk_score" not in df.columns:
        return None
    fig = px.histogram(
        df,
        x="risk_score",
        color="risk_level",
        nbins=10,
        color_discrete_map=RISK_COLORS,
        title="위험점수 분포: 전체 대상은 어느 구간에 몰려 있는가?",
    )
    fig.update_layout(xaxis_title="위험점수", yaxis_title="대상 수", height=360)
    return fig


def make_driver_chart(features: pd.DataFrame) -> go.Figure | None:
    if features.empty or "main_risk_driver" not in features.columns or "final_water_risk_score" not in features.columns:
        return None
    driver_df = (
        features.groupby("main_risk_driver", dropna=False)
        .agg(대상수=("sigungu", "count"), 평균위험점수=("final_water_risk_score", "mean"))
        .reset_index()
        .sort_values("평균위험점수", ascending=False)
    )
    fig = px.pie(
        driver_df,
        names="main_risk_driver",
        values="평균위험점수",
        hole=0.55,
        custom_data=["대상수"],
        title="주요 위험 원인별 평균 위험점수: 어떤 요인이 우선인가?",
        color_discrete_sequence=["#2563eb", "#0f766e", "#f59e0b", "#7c3aed", "#dc2626", "#64748b"],
    )
    fig.update_traces(
        texttemplate="%{value:.1f}점",
        textposition="inside",
        insidetextorientation="horizontal",
        textfont=dict(size=13, color="#ffffff"),
        hovertemplate="주요 위험 원인=%{label}<br>평균 위험점수=%{value:.2f}점<br>대상 수=%{customdata[0]}곳<extra></extra>",
    )
    fig.update_layout(
        height=390,
        margin=dict(l=20, r=20, t=70, b=30),
        legend_title_text="주요 위험 원인",
        uniformtext=dict(minsize=12, mode="show"),
    )
    return fig


def make_component_chart(features: pd.DataFrame, sigungu: str) -> go.Figure | None:
    if features.empty or "sigungu" not in features.columns:
        return None
    selected = features[features["sigungu"] == sigungu]
    if selected.empty:
        return None
    row = selected.iloc[0]
    labels = []
    scores = []
    weighted = []
    for col, label, weight in COMPONENTS:
        if col not in features.columns:
            continue
        value = safe_float(row.get(col), None)
        if value is None:
            continue
        labels.append(label)
        scores.append(value)
        weighted.append(value * weight)
    if not labels:
        return None

    fig = go.Figure()
    fig.add_bar(x=labels, y=scores, name="원점수", text=[f"{v:.1f}" for v in scores], textposition="outside")
    fig.add_scatter(
        x=labels,
        y=weighted,
        name="가중 기여점수",
        mode="lines+markers+text",
        text=[f"{v:.1f}" for v in weighted],
        textposition="top center",
    )
    fig.update_layout(
        title=f"{sigungu} 위험 구성요소: 어떤 지표가 점수를 끌어올리는가?",
        yaxis_title="점수",
        xaxis_title="위험 구성요소",
        height=390,
        yaxis=dict(range=[0, max(100, max(scores) * 1.15)]),
    )
    return fig


def make_map_chart(df: pd.DataFrame, map_style: str = "carto-positron") -> go.Figure | None:
    if df.empty:
        return None
    map_df = df.copy()
    map_df["lat"] = map_df["sigungu"].map(lambda x: SIGUNGU_CENTROIDS.get(str(x), (np.nan, np.nan))[0])
    map_df["lon"] = map_df["sigungu"].map(lambda x: SIGUNGU_CENTROIDS.get(str(x), (np.nan, np.nan))[1])
    map_df = map_df.dropna(subset=["lat", "lon"])
    if map_df.empty:
        return None
    map_df["marker_size"] = 12 + map_df["risk_score"].fillna(0).clip(lower=0) * 0.55
    fig = px.scatter_mapbox(
        map_df,
        lat="lat",
        lon="lon",
        size="marker_size",
        color="risk_level",
        hover_name="sigungu",
        hover_data={"risk_score": ":.1f", "main_risk_driver": True, "lat": False, "lon": False, "marker_size": False},
        color_discrete_map=RISK_COLORS,
        zoom=8,
        height=680,
        center={"lat": 36.55, "lon": 126.95},
        title="지도 보기: 위험 대상은 어디에 집중되어 있는가?",
    )
    fig.update_layout(mapbox_style=map_style, margin=dict(l=0, r=0, t=60, b=0), legend_title_text="위험등급")
    return fig


def make_ai_scatter(ai_summary: pd.DataFrame) -> go.Figure | None:
    needed = {"current_avg_reservoir_rate", "pred_avg_reservoir_rate_7d", "sigungu", "deep_ai_risk_level"}
    if ai_summary.empty or not needed.issubset(ai_summary.columns):
        return None
    fig = px.scatter(
        ai_summary,
        x="current_avg_reservoir_rate",
        y="pred_avg_reservoir_rate_7d",
        color="deep_ai_risk_level",
        hover_name="sigungu",
        size="deep_ai_risk_score" if "deep_ai_risk_score" in ai_summary.columns else None,
        color_discrete_map=RISK_COLORS,
        title="Deep AI 예측: 현재 저수율과 7일 후 예측은 어떻게 다른가?",
    )
    fig.add_shape(type="line", x0=0, y0=0, x1=100, y1=100, line=dict(color="#94a3b8", dash="dash"))
    fig.update_layout(xaxis_title="현재 평균 저수율", yaxis_title="7일 후 예측 저수율", height=420)
    return fig


def selected_candidates(candidates: pd.DataFrame, sigungu: str, top_n: int = 5) -> pd.DataFrame:
    if candidates.empty or "target_sigungu" not in candidates.columns:
        return pd.DataFrame()
    out = candidates[candidates["target_sigungu"] == sigungu].copy()
    if "candidate_rank" in out.columns:
        out = out.sort_values("candidate_rank")
    return out.head(top_n)


def candidate_display(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    cols = [
        "candidate_rank",
        "candidate_reservoir_name",
        "candidate_sigungu",
        "distance_km",
        "candidate_reservoir_rate",
        "benefit_area",
        "candidate_score",
        "recommendation_reason",
    ]
    cols = [c for c in cols if c in candidates.columns]
    return candidates[cols].rename(
        columns={
            "candidate_rank": "순위",
            "candidate_reservoir_name": "대체 수원 후보",
            "candidate_sigungu": "소속 시·군",
            "distance_km": "거리(km)",
            "candidate_reservoir_rate": "후보 저수율",
            "benefit_area": "수혜면적",
            "candidate_score": "후보점수",
            "recommendation_reason": "추천 사유",
        }
    )


def build_html_report(row: pd.Series, candidates: pd.DataFrame) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    sigungu = row.get("sigungu", "-")
    score = format_value(row.get("risk_score"), "점", 1)
    level = row.get("risk_level", "N/A")
    reason = row.get("main_risk_driver", "N/A")
    action = row.get("recommended_action", "현장 점검 및 대체 수원 후보 확인")

    candidate_rows = ""
    if candidates.empty:
        candidate_rows = "<tr><td colspan='6'>대체 수원 후보 없음</td></tr>"
    else:
        for _, c in candidates.iterrows():
            candidate_rows += f"""
            <tr>
              <td>{format_value(c.get('candidate_rank'), decimals=0)}</td>
              <td>{c.get('candidate_reservoir_name', '-')}</td>
              <td>{c.get('candidate_sigungu', '-')}</td>
              <td>{format_value(c.get('distance_km'), ' km', 1)}</td>
              <td>{format_value(c.get('candidate_reservoir_rate'), '%', 1)}</td>
              <td>{format_value(c.get('candidate_score'), '점', 1)}</td>
            </tr>
            """

    return f"""
    <!doctype html>
    <html lang="ko">
    <head>
      <meta charset="utf-8">
      <title>AquaGuard AI 의사결정 리포트 - {sigungu}</title>
      <style>
        body {{ font-family: Arial, sans-serif; margin: 36px; line-height: 1.55; color: #222; }}
        h1 {{ color: #0f3d4f; }}
        h2 {{ margin-top: 28px; color: #145374; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
        th {{ background: #eef4f7; }}
        .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 12px 0; }}
        .note {{ background: #fff7e6; border-left: 5px solid #ffb000; padding: 12px; }}
      </style>
    </head>
    <body>
      <h1>충남 AquaGuard AI 의사결정 리포트</h1>
      <p>생성 시각: {generated_at}</p>
      <div class="card">
        <h2>{sigungu} 농업용수 위험 요약</h2>
        <p><b>위험점수:</b> {score}</p>
        <p><b>위험등급:</b> {level}</p>
        <p><b>판단 근거:</b> {reason}</p>
        <p><b>권고 조치:</b> {action}</p>
      </div>
      <h2>대체 수원 후보 TOP 5</h2>
      <table>
        <tr><th>순위</th><th>후보 저수지</th><th>소속 시·군</th><th>거리</th><th>후보 저수율</th><th>후보점수</th></tr>
        {candidate_rows}
      </table>
      <div class="note">
        본 리포트는 공개데이터 기반 의사결정 참고자료입니다. 실제 급수 대응은 현장 접근성, 관로, 수리권, 수질, 행정 협의를 함께 검토해야 합니다.
      </div>
    </body>
    </html>
    """


def render_sidebar(source_df: pd.DataFrame) -> str:
    all_modes = ["최종 산정", "Live 업데이트", "Deep AI 예측"]
    all_regions = sorted(source_df["sigungu"].dropna().astype(str).unique().tolist()) if not source_df.empty else []

    if "analysis_mode" not in st.session_state:
        st.session_state.analysis_mode = "최종 산정"
    if "selected_regions" not in st.session_state:
        st.session_state.selected_regions = all_regions

    with st.sidebar:
        st.header("필터")
        if st.button("기본값으로 초기화", use_container_width=True):
            st.session_state.analysis_mode = "최종 산정"
            st.session_state.selected_regions = all_regions
            st.session_state.main_region_select_all = True
            for region in all_regions:
                st.session_state[region_checkbox_key("main", region)] = True
            st.session_state.selected_levels = available_levels(source_df)
            st.session_state.main_chart_top_n = 10
            st.session_state.main_priority_top_n = 10
            st.rerun()

        st.markdown("#### 기간")
        mode = st.radio("분석 기준", all_modes, key="analysis_mode", help="현재 파일에 저장된 최신 기준 결과를 선택합니다.")

    return mode


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


def render_filters_for_mode(source_df: pd.DataFrame, latest_basis: str) -> tuple[list[str], list[str]]:
    all_regions = sorted(source_df["sigungu"].dropna().astype(str).unique().tolist()) if not source_df.empty else []
    levels = available_levels(source_df)
    if "selected_regions" not in st.session_state:
        st.session_state.selected_regions = all_regions
    if "selected_levels" not in st.session_state:
        st.session_state.selected_levels = levels
    st.session_state.selected_regions = [r for r in st.session_state.selected_regions if r in all_regions]
    st.session_state.selected_levels = [r for r in st.session_state.selected_levels if r in levels] or levels

    with st.sidebar:
        st.caption(f"최신 기준: {latest_basis}")
        st.markdown("#### 지역/대상")
        st.caption("시·군")
        selected_regions = render_region_checkboxes(all_regions, "main", "selected_regions")

        st.markdown("#### 모델/위험등급")
        selected_levels = st.multiselect("위험등급", levels, key="selected_levels")

    return selected_regions, selected_levels


def main() -> None:
    inject_css()

    features, feature_msg = safe_read_data(str(FEATURE_PATH), required=True)
    candidates, candidate_msg = safe_read_data(str(TOP5_PATH if TOP5_PATH.exists() else CANDIDATE_PATH), required=False)
    live_summary, live_msg = safe_read_data(str(LIVE_SUMMARY_PATH), required=False)
    ai_summary, ai_msg = safe_read_data(str(AI_SUMMARY_PATH), required=False)
    gru_history, gru_msg = safe_read_data(str(GRU_HISTORY_PATH), required=False)
    ae_history, ae_msg = safe_read_data(str(AE_HISTORY_PATH), required=False)
    validation, validation_msg = safe_read_data(str(VALIDATION_PATH), required=False)
    training_history, training_msg = safe_read_data(str(TRAINING_DATA_PATH), required=False)
    report_text, report_msg = read_text_file(str(AI_REPORT_PATH))

    show_load_messages([feature_msg], required_stop=True)
    if features.empty:
        render_empty_state("필수 위험도 산정 데이터가 비어 있습니다.")
        st.stop()

    features = exclude_unavailable_regions(features)
    candidates = exclude_unavailable_regions(candidates)
    live_summary = exclude_unavailable_regions(live_summary)
    ai_summary = exclude_unavailable_regions(ai_summary)
    training_history = exclude_unavailable_regions(training_history)

    if features.empty:
        render_empty_state("분석 제외 지역을 제거한 뒤 표시할 위험도 산정 데이터가 없습니다.")
        st.stop()

    numeric_cols = [
        "final_priority_rank",
        "final_water_risk_score",
        "rain_shortage_score",
        "reservoir_risk_score",
        "groundwater_dependency_score",
        "crop_water_demand_score",
        "alternative_source_access_shortage_score",
        "avg_reservoir_rate",
        "min_reservoir_rate",
        "reservoir_count",
    ]
    features = normalize_numeric(features, numeric_cols)
    live_summary = normalize_numeric(
        live_summary,
        ["final_live_priority_rank", "final_live_water_risk_score", "live_score_delta_from_baseline"],
    )
    ai_summary = normalize_numeric(
        ai_summary,
        ["deep_ai_rank", "deep_ai_risk_score", "current_avg_reservoir_rate", "pred_avg_reservoir_rate_7d"],
    )

    live_basis_month = current_month_from_live(live_summary)
    training_months = months_from_columns([training_history], ["date"])
    ai_output_month = latest_month_from_columns([ai_summary], ["base_date", "target_date"])
    training_final_month = training_months[-1] if training_months else ai_output_month
    ai_comparison_month, ai_comparison_fallback = select_default_ai_comparison_month(live_basis_month, training_months)

    latest_basis = latest_date_from_columns(
        [features, live_summary, ai_summary],
        ["reservoir_latest_date", "weather_latest_date", "soil_data_date", "base_date", "target_date"],
    )
    best_model, main_metric, performance_cards = build_performance_summary(report_text, gru_history, ae_history)

    render_page_header(
        "충남 AquaGuard AI 농업용수 위험도 의사결정 대시보드",
        "기상·저수율·관정·작물·대체 수원 데이터를 기반으로 고위험 시·군과 점검 우선순위를 한눈에 확인합니다.",
        f"Live 기준일 {latest_basis}<br>AI 비교 기준월 {ai_comparison_month}<br>학습 데이터 최종월 {training_final_month}",
    )

    if ai_comparison_fallback:
        st.warning(f"전년도 동일 월 데이터가 없어 가장 가까운 학습 데이터 월({ai_comparison_month})을 AI 비교 기준월로 사용합니다.")

    mode = render_sidebar(features)
    source_df = make_source_view(mode, features, live_summary, ai_summary)
    selected_regions, selected_levels = render_filters_for_mode(source_df, latest_basis)

    filtered = source_df.copy()
    filtered = filtered[filtered["sigungu"].isin(selected_regions)] if selected_regions else filtered.iloc[0:0]
    if selected_levels:
        filtered = filtered[filtered["risk_level"].isin(selected_levels)]

    high_risk_count = int(filtered["risk_level"].map(is_risky).sum()) if not filtered.empty else 0
    latest_period = latest_date_from_columns([filtered], ["basis_date"])
    total_records = len(filtered)

    st.markdown(
        f"""
        <div class="ag-filter">
        현재 필터: <b>{mode}</b> · 지역 <b>{len(selected_regions) if selected_regions else 0}개</b> ·
        위험등급 <b>{", ".join(selected_levels) if selected_levels else "전체"}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_kpi_cards(
        [
            ("분석 데이터 수", f"{total_records:,}건", "현재 필터 기준으로 표시되는 시·군 단위 결과 수입니다."),
            ("고위험 대상 수", f"{high_risk_count:,}곳", "위험등급이 주의·경계·심각·심각후보인 대상 수입니다."),
            (
                "우선 점검 대상",
                "N/A" if filtered.empty else str(filtered.sort_values(["priority_rank", "risk_score"], ascending=[True, False]).iloc[0]["sigungu"]),
                "현재 필터 기준으로 가장 먼저 확인할 대상입니다.",
            ),
            ("최고 성능 모델", best_model, "성능 검증 파일에서 확인 가능한 모델명입니다."),
            ("주요 성능 지표", main_metric, "존재하는 검증 지표만 표시합니다."),
        ]
    )

    if filtered.empty:
        render_empty_state()
        st.stop()

    map_tab, analysis_tab, priority_tab, raw_tab = st.tabs(
        ["지도 보기", "위험도 분석", "점검·대체수원", "원본·기타"]
    )

    with map_tab:
        render_section_header("지도 보기", "시·군별 위험점수를 지도에서 먼저 확인합니다.")
        map_style_label = st.radio(
            "지도 유형",
            ["밝은 지도", "기본 지도"],
            index=0,
            horizontal=True,
            help="밝은 지도는 흰색 배경의 Carto Positron 스타일로 발표 화면에서 더 선명하게 보입니다.",
        )
        map_style = {"밝은 지도": "carto-positron", "기본 지도": "open-street-map"}[map_style_label]
        map_fig = make_map_chart(filtered, map_style)
        if map_fig:
            st.plotly_chart(map_fig, use_container_width=True)
            st.caption("마커 크기는 위험점수, 색상은 위험등급을 의미합니다. 밝은 지도는 발표·리뷰 화면에서 행정구역과 마커를 더 또렷하게 확인할 때 사용합니다.")
        else:
            render_empty_state("지도에 표시할 좌표 또는 위험도 데이터가 없습니다.")

    with analysis_tab:
        chart_max_top = max(5, min(15, len(filtered))) if not filtered.empty else 5
        chart_top_n = st.slider(
            "위험도 차트 표시 수",
            5,
            chart_max_top,
            min(st.session_state.get("main_chart_top_n", 10), chart_max_top),
            key="main_chart_top_n",
            help="위험도 분석 탭의 순위 차트에만 적용됩니다.",
        )
        render_section_header("위험도 분석 그래프", "위험도 순위와 분포를 나누어 확인합니다.")
        left, right = st.columns([1.25, 1])
        with left:
            ranking_fig = make_ranking_chart(filtered, f"{mode}: 어느 시·군을 먼저 볼 것인가?", chart_top_n)
            if ranking_fig:
                st.plotly_chart(ranking_fig, use_container_width=True)
                st.caption("위험점수가 높고 순위가 빠를수록 점검 우선순위가 높습니다.")
            else:
                render_empty_state()
        with right:
            dist_fig = make_distribution_chart(filtered)
            if dist_fig:
                st.plotly_chart(dist_fig, use_container_width=True)
                st.caption("위험등급별 점수 분포를 보면 특정 구간에 대상이 몰려 있는지 확인할 수 있습니다.")
            else:
                render_empty_state()

        left, right = st.columns(2)
        with left:
            render_section_header("주요 위험 원인별 평균 위험점수")
            driver_fig = make_driver_chart(features)
            if driver_fig:
                st.plotly_chart(driver_fig, use_container_width=True)
                st.caption("도넛 조각은 주요 위험 원인별 평균 위험점수의 상대적 비중을 보여줍니다.")
            else:
                st.info("main_risk_driver 또는 final_water_risk_score 컬럼이 없어 위험 원인 차트를 건너뛰었습니다.")
        with right:
            component_options = filtered["sigungu"].dropna().astype(str).drop_duplicates().tolist() if "sigungu" in filtered.columns else []
            if st.session_state.get("main_component_focus") not in component_options and component_options:
                st.session_state.main_component_focus = component_options[0]
            component_focus = st.selectbox(
                "위험 구성요소 확인 대상",
                component_options,
                key="main_component_focus",
                help="이 선택은 위험 구성요소 그래프에만 적용됩니다.",
            ) if component_options else None
            render_section_header(f"{component_focus or 'N/A'} 위험 구성요소 그래프")
            component_fig = make_component_chart(features, component_focus) if component_focus else None
            if component_fig:
                st.plotly_chart(component_fig, use_container_width=True)
                st.caption("원점수와 가중 기여점수를 함께 보면 해당 시·군의 위험 원인을 빠르게 설명할 수 있습니다.")
            else:
                st.info("구성요소 점수 컬럼이 없어 상세 구성 차트를 건너뛰었습니다.")

        focus_row = filtered[filtered["sigungu"] == component_focus] if component_focus else pd.DataFrame()
        if not focus_row.empty and has_missing_reservoir_context(focus_row):
            render_missing_reservoir_note(focus_row)
            row = focus_row.iloc[0]
            st.caption(
                " · ".join(
                    [
                        f"기준일: {format_date_display(row.get('basis_date'), '저수지 기준일 없음')}",
                        f"분석 기준: {build_analysis_basis(row)}",
                    ]
                )
            )

    with priority_tab:
        priority_max_top = max(5, min(15, len(filtered))) if not filtered.empty else 5
        priority_top_n = st.slider(
            "점검 우선순위 표시 수",
            5,
            priority_max_top,
            min(st.session_state.get("main_priority_top_n", 10), priority_max_top),
            key="main_priority_top_n",
            help="점검·대체수원 탭의 우선순위 표에만 적용됩니다.",
        )
        render_section_header(
            "우선 점검 순위",
            "현재 필터 기준으로 가장 먼저 확인할 시·군과 권고 조치를 정리했습니다.",
        )
        priority_table = build_priority_table(filtered, features, priority_top_n)
        if priority_table.empty:
            render_empty_state()
        else:
            st.dataframe(format_display_dataframe(priority_table), use_container_width=True, hide_index=True)

        render_section_header(
            "대체 수원 후보",
            "선택한 시·군 기준으로 이미 생성된 후보 결과를 보여줍니다.",
        )
        candidate_options = filtered["sigungu"].dropna().astype(str).drop_duplicates().tolist() if "sigungu" in filtered.columns else []
        if st.session_state.get("main_candidate_focus") not in candidate_options and candidate_options:
            st.session_state.main_candidate_focus = candidate_options[0]
        candidate_focus = st.selectbox(
            "대체 수원 확인 대상",
            candidate_options,
            key="main_candidate_focus",
            help="이 선택은 대체 수원 후보 표에만 적용됩니다.",
        ) if candidate_options else None
        focus_candidates = selected_candidates(candidates, candidate_focus) if candidate_focus else pd.DataFrame()
        if focus_candidates.empty:
            st.info("조건에 맞는 대체 수원 후보 데이터가 없습니다.")
        else:
            st.dataframe(format_display_dataframe(candidate_display(focus_candidates)), use_container_width=True, hide_index=True)
            st.download_button(
                label=f"{candidate_focus} 대체 수원 후보 CSV 다운로드",
                data=focus_candidates.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{candidate_focus}_alternative_source_top5.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with raw_tab:
        render_section_header(
            "성능 검증 요약",
            "모델 성능 파일에 실제 존재하는 지표만 보여줍니다. 없는 지표는 표시하지 않습니다.",
        )
        if performance_cards:
            render_kpi_cards(performance_cards[:5])
            if len(performance_cards) > 5:
                render_kpi_cards(performance_cards[5:])
        else:
            st.info("성능 검증 지표 컬럼 또는 리포트 항목이 없어 표시할 수 있는 지표가 없습니다.")

        render_section_header("상세 데이터 및 원본 결과 확인", "요약 판단 이후 필요한 원본 산정 결과를 확인합니다.")
        detailed = filtered.rename(
            columns={
                "priority_rank": "우선순위",
                "sigungu": "시·군",
                "risk_score": "위험점수",
                "risk_level": "위험등급",
                "main_risk_driver": "주요 위험 원인",
                "recommended_action": "권고 조치",
                "basis_date": "기준일",
                "analysis_mode": "분석 기준",
            }
        )
        render_missing_reservoir_note(detailed)
        st.dataframe(prepare_main_detail_display(detailed), use_container_width=True, hide_index=True)

        with st.expander("원본 최종 산정 결과 보기"):
            st.dataframe(format_display_dataframe(features), use_container_width=True, hide_index=True)
            st.download_button(
                label="최종 산정 결과 CSV 다운로드",
                data=features.to_csv(index=False).encode("utf-8-sig"),
                file_name="aquaguard_sigungu_features.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with st.expander("성능 검증·학습 이력 원본 보기"):
            if not gru_history.empty:
                st.markdown("#### GRU 학습 이력")
                st.dataframe(format_display_dataframe(gru_history), use_container_width=True, hide_index=True)
            else:
                st.info("GRU 학습 이력 파일이 없습니다.")
            if not ae_history.empty:
                st.markdown("#### AutoEncoder 학습 이력")
                st.dataframe(format_display_dataframe(ae_history), use_container_width=True, hide_index=True)
            else:
                st.info("AutoEncoder 학습 이력 파일이 없습니다.")
            if report_text:
                st.markdown("#### AI 모델 리포트")
                st.markdown(report_text)
            if not validation.empty:
                st.markdown("#### 최종 검증 체크 결과")
                st.dataframe(format_display_dataframe(validation), use_container_width=True, hide_index=True)

        with st.expander("보고서용 이미지 확인"):
            for filename, caption in REPORT_FIGURES_LIST:
                path = resolve_report_figure_path(filename)
                if path.exists():
                    st.image(str(path), caption=caption, use_container_width=True)
                else:
                    st.warning(f"이미지 파일이 없습니다: {path}")

        render_section_header("설명 notes", "비기술 검토자가 볼 때 필요한 해석 기준만 짧게 남겼습니다.")
        st.info(
            "AquaGuard AI 결과는 공개데이터 기반 의사결정 참고자료입니다. 실제 대응은 현장 저수율, 관로 연결성, "
            "수리권, 수질, 행정 협의를 함께 확인해야 합니다."
        )

        report_options = filtered["sigungu"].dropna().astype(str).drop_duplicates().tolist() if "sigungu" in filtered.columns else []
        if st.session_state.get("main_report_focus") not in report_options and report_options:
            st.session_state.main_report_focus = report_options[0]
        report_focus = st.selectbox(
            "리포트 다운로드 대상",
            report_options,
            key="main_report_focus",
            help="이 선택은 HTML 의사결정 리포트 다운로드에만 적용됩니다.",
        ) if report_options else None
        selected_row = filtered[filtered["sigungu"] == report_focus] if report_focus else pd.DataFrame()
        if selected_row.empty:
            selected_row = filtered.sort_values(["priority_rank", "risk_score"], ascending=[True, False]).head(1)
            report_focus = str(selected_row.iloc[0]["sigungu"]) if not selected_row.empty and "sigungu" in selected_row.columns else "AquaGuard"
        if not selected_row.empty:
            report_candidates = selected_candidates(candidates, report_focus)
            report_html = build_html_report(selected_row.iloc[0], report_candidates)
            st.download_button(
                label=f"{report_focus} HTML 의사결정 리포트 다운로드",
                data=report_html.encode("utf-8-sig"),
                file_name=f"AquaGuard_AI_{report_focus}_decision_report.html",
                mime="text/html",
                use_container_width=True,
            )

    show_load_messages([candidate_msg, live_msg, ai_msg, gru_msg, ae_msg, validation_msg, training_msg, report_msg])


if __name__ == "__main__":
    main()
