from pathlib import Path
from datetime import datetime
import base64

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
REPORT_TABLES = ROOT / "reports" / "tables"
REPORT_FIGURES = ROOT / "reports" / "figures"
META = ROOT / "data" / "metadata"

FEATURE_PATH = PROCESSED / "aquaguard_sigungu_features.csv"
CANDIDATE_PATH = PROCESSED / "alternative_source_candidates.csv"
TOP5_PATH = REPORT_TABLES / "alternative_source_top5_by_sigungu.csv"

FIG_RANKING = REPORT_FIGURES / "01_final_risk_ranking.png"
FIG_COMPONENTS = REPORT_FIGURES / "02_risk_components_stacked.png"
FIG_SCATTER = REPORT_FIGURES / "03_reservoir_vs_alternative_shortage_scatter.png"
FIG_TOP5 = REPORT_FIGURES / "04_top5_priority_table.png"
FIG_ALT = REPORT_FIGURES / "05_alternative_source_top1_by_risk_area.png"

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

PDF_COMPONENTS = [
    ("rain_shortage_score", "강우 부족도", 0.25),
    ("reservoir_risk_score", "저수율 위험도", 0.25),
    ("groundwater_dependency_score", "관정 의존도", 0.20),
    ("crop_water_demand_score", "작물 물수요 지수", 0.20),
    ("alternative_source_access_shortage_score", "대체 수원 접근성 부족도", 0.10),
]


st.set_page_config(
    page_title="충남 AquaGuard AI",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)


def risk_color(level: str) -> str:
    if level == "심각":
        return "#d73027"
    if level == "경계":
        return "#fc8d59"
    if level == "주의":
        return "#fee08b"
    if level == "낮음":
        return "#91cf60"
    return "#cccccc"


def risk_badge(level: str) -> str:
    color = risk_color(level)
    text_color = "#111111" if level in ["주의", "낮음"] else "#ffffff"
    return (
        f"<span style='background:{color}; color:{text_color}; padding:6px 12px; "
        f"border-radius:999px; font-weight:700;'>{level}</span>"
    )


@st.cache_data
def load_data():
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(f"최종 feature 파일이 없습니다: {FEATURE_PATH}")

    features = pd.read_csv(FEATURE_PATH)

    if TOP5_PATH.exists():
        candidates = pd.read_csv(TOP5_PATH)
    elif CANDIDATE_PATH.exists():
        candidates = pd.read_csv(CANDIDATE_PATH)
    else:
        candidates = pd.DataFrame()

    features["final_water_risk_score"] = pd.to_numeric(
        features["final_water_risk_score"], errors="coerce"
    )

    features = features.sort_values("final_priority_rank").reset_index(drop=True)

    for col in [
        "rain_shortage_score",
        "reservoir_risk_score",
        "groundwater_dependency_score",
        "crop_water_demand_score",
        "alternative_source_access_shortage_score",
        "water_supply_pressure_score",
        "agricultural_demand_score",
        "alternative_water_risk_score",
        "agri_impact_index",
        "well_support_score",
        "well_shortage_score",
    ]:
        if col in features.columns:
            features[col] = pd.to_numeric(features[col], errors="coerce")

    return features, candidates


def make_map_df(features):
    map_df = features.copy()
    map_df["lat"] = map_df["sigungu"].map(lambda x: SIGUNGU_CENTROIDS.get(x, (np.nan, np.nan))[0])
    map_df["lon"] = map_df["sigungu"].map(lambda x: SIGUNGU_CENTROIDS.get(x, (np.nan, np.nan))[1])
    map_df = map_df.dropna(subset=["lat", "lon"]).copy()
    map_df["marker_size"] = 12 + map_df["final_water_risk_score"].fillna(0) * 0.7
    return map_df


def make_risk_map(features, selected_sigungu):
    map_df = make_map_df(features)

    fig = px.scatter_mapbox(
        map_df,
        lat="lat",
        lon="lon",
        size="marker_size",
        color="final_water_risk_level",
        hover_name="sigungu",
        hover_data={
            "final_water_risk_score": ":.1f",
            "main_risk_driver": True,
            "lat": False,
            "lon": False,
            "marker_size": False,
        },
        color_discrete_map={
            "낮음": "#91cf60",
            "주의": "#fee08b",
            "경계": "#fc8d59",
            "심각": "#d73027",
        },
        zoom=7,
        height=520,
        center={"lat": 36.55, "lon": 126.95},
    )

    selected = map_df[map_df["sigungu"] == selected_sigungu]
    if not selected.empty:
        fig.add_trace(
            go.Scattermapbox(
                lat=selected["lat"],
                lon=selected["lon"],
                mode="markers",
                marker=dict(size=28, color="black", opacity=0.35),
                name="선택 지역",
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        mapbox_style="open-street-map",
        margin=dict(l=0, r=0, t=0, b=0),
        legend_title_text="위험 단계",
    )

    return fig


def make_component_chart(row):
    labels = []
    scores = []
    weighted = []

    for col, label, weight in PDF_COMPONENTS:
        value = row.get(col, np.nan)
        value = 50 if pd.isna(value) else float(value)
        labels.append(f"{label} ({int(weight * 100)}%)")
        scores.append(value)
        weighted.append(value * weight)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=labels,
            y=scores,
            name="원점수",
            text=[f"{v:.1f}" for v in scores],
            textposition="outside",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=labels,
            y=weighted,
            name="가중 기여점수",
            mode="lines+markers+text",
            text=[f"{v:.1f}" for v in weighted],
            textposition="top center",
        )
    )

    fig.update_layout(
        title="PDF 기준 5개 위험지표 점수",
        yaxis_title="점수",
        xaxis_title="지표",
        yaxis=dict(range=[0, max(100, max(scores) * 1.15)]),
        height=430,
        margin=dict(l=30, r=30, t=60, b=60),
    )

    return fig


def make_ranking_chart(features):
    plot_df = features.sort_values("final_water_risk_score", ascending=True).copy()

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=plot_df["final_water_risk_score"],
            y=plot_df["sigungu"],
            orientation="h",
            text=plot_df["final_water_risk_score"].map(lambda x: f"{x:.1f}"),
            textposition="outside",
            marker_color=[risk_color(x) for x in plot_df["final_water_risk_level"]],
        )
    )

    fig.update_layout(
        title="충남 시·군별 농업용수 부족 위험도 순위",
        xaxis_title="최종 위험도 점수",
        yaxis_title="시·군",
        height=560,
        margin=dict(l=40, r=50, t=60, b=40),
        xaxis=dict(range=[0, max(100, plot_df["final_water_risk_score"].max() * 1.15)]),
    )

    return fig


def format_score(x):
    if pd.isna(x):
        return "-"
    return f"{float(x):.1f}"


def selected_candidates(candidates, sigungu):
    if candidates.empty:
        return candidates

    if "target_sigungu" not in candidates.columns:
        return pd.DataFrame()

    df = candidates[candidates["target_sigungu"] == sigungu].copy()

    if "candidate_rank" in df.columns:
        df = df.sort_values("candidate_rank")

    return df.head(5)


def build_html_report(row, candidates):
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    component_rows = ""
    for col, label, weight in PDF_COMPONENTS:
        value = row.get(col, np.nan)
        component_rows += (
            f"<tr><td>{label}</td><td>{format_score(value)}</td>"
            f"<td>{int(weight * 100)}%</td><td>{format_score((0 if pd.isna(value) else value) * weight)}</td></tr>"
        )

    candidate_rows = ""
    if candidates.empty:
        candidate_rows = "<tr><td colspan='7'>추천 후보 없음</td></tr>"
    else:
        for _, c in candidates.iterrows():
            candidate_rows += f"""
            <tr>
              <td>{int(c.get('candidate_rank', 0))}</td>
              <td>{c.get('candidate_reservoir_name', '-')}</td>
              <td>{c.get('candidate_sigungu', '-')}</td>
              <td>{format_score(c.get('distance_km', np.nan))} km</td>
              <td>{format_score(c.get('candidate_reservoir_rate', np.nan))}</td>
              <td>{format_score(c.get('benefit_area', np.nan))}</td>
              <td>{format_score(c.get('candidate_score', np.nan))}</td>
            </tr>
            """

    html = f"""
    <!doctype html>
    <html lang="ko">
    <head>
      <meta charset="utf-8">
      <title>AquaGuard AI 행정 리포트 - {row['sigungu']}</title>
      <style>
        body {{ font-family: Arial, sans-serif; margin: 36px; line-height: 1.55; color: #222; }}
        h1 {{ color: #0b3d5c; }}
        h2 {{ margin-top: 28px; color: #145374; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
        th {{ background: #eef4f7; }}
        .card {{ border: 1px solid #ddd; border-radius: 10px; padding: 16px; margin: 12px 0; }}
        .warning {{ background: #fff7e6; border-left: 5px solid #ffb000; padding: 12px; }}
      </style>
    </head>
    <body>
      <h1>충남 AquaGuard AI 행정 참고 리포트</h1>
      <p>생성 시각: {generated_at}</p>

      <div class="card">
        <h2>{row['sigungu']} 농업용수 부족 위험 요약</h2>
        <p><b>최종 위험도:</b> {format_score(row.get('final_water_risk_score'))}점</p>
        <p><b>위험 단계:</b> {row.get('final_water_risk_level')}</p>
        <p><b>주요 위험 원인:</b> {row.get('main_risk_driver')}</p>
        <p><b>권고 조치:</b> {row.get('recommended_action')}</p>
      </div>

      <h2>PDF 기준 5개 위험지표</h2>
      <table>
        <tr><th>지표</th><th>점수</th><th>가중치</th><th>가중 기여점수</th></tr>
        {component_rows}
      </table>

      <h2>대체 수원 후보 TOP 5</h2>
      <table>
        <tr>
          <th>순위</th><th>후보 저수지</th><th>소속 시·군</th>
          <th>거리</th><th>후보 저수율</th><th>수혜면적</th><th>후보점수</th>
        </tr>
        {candidate_rows}
      </table>

      <div class="warning">
        본 리포트는 공개데이터 기반 행정 참고자료입니다. 실제 대체 수원 공급 가능 여부는 관로, 수리권,
        수질, 현장 접근성, 행정 협의 등을 추가 검토해야 합니다.
      </div>
    </body>
    </html>
    """

    return html


def download_html_button(html, filename):
    st.download_button(
        label="HTML 행정 리포트 다운로드",
        data=html.encode("utf-8-sig"),
        file_name=filename,
        mime="text/html",
        use_container_width=True,
    )


def render_image_if_exists(path, caption):
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"이미지 파일이 아직 없습니다: {path}")


def main():
    features, candidates = load_data()

    st.title("💧 충남 AquaGuard AI")
    st.caption(
        "올담 공공데이터 기반 농업용수 부족 위험 예측 · 대체 수원 후보 추천 · 행정 리포트 MVP"
    )

    st.warning(
        "본 서비스는 공개데이터 기반 행정 참고자료입니다. 실제 대체 수원 공급 가능 여부는 "
        "관로, 수리권, 수질, 현장 접근성, 행정 협의 등을 추가 검토해야 합니다."
    )

    avg_score = features["final_water_risk_score"].mean()
    top_row = features.sort_values("final_water_risk_score", ascending=False).iloc[0]
    caution_or_above = features[features["final_water_risk_level"].isin(["주의", "경계", "심각"])].shape[0]
    candidate_count = len(candidates)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("충남 평균 위험도", f"{avg_score:.1f}점")
    c2.metric("주의 이상 시·군", f"{caution_or_above}곳")
    c3.metric("최고 위험 지역", f"{top_row['sigungu']}")
    c4.metric("대체 수원 후보", f"{candidate_count}건")

    st.sidebar.header("지역 선택")
    sigungu_options = features.sort_values("final_priority_rank")["sigungu"].tolist()
    default_idx = 0
    selected_sigungu = st.sidebar.selectbox("시·군 선택", sigungu_options, index=default_idx)

    selected_row = features[features["sigungu"] == selected_sigungu].iloc[0]
    selected_cands = selected_candidates(candidates, selected_sigungu)

    st.sidebar.markdown("---")
    st.sidebar.subheader("선택 지역 요약")
    st.sidebar.write(f"**순위:** {int(selected_row['final_priority_rank'])}위")
    st.sidebar.write(f"**위험도:** {selected_row['final_water_risk_score']:.1f}점")
    st.sidebar.write(f"**단계:** {selected_row['final_water_risk_level']}")
    st.sidebar.write(f"**주요 원인:** {selected_row['main_risk_driver']}")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["위험지도", "지역 상세", "대체 수원 후보", "전체 순위", "보고서용 이미지"]
    )

    with tab1:
        st.subheader("시·군별 농업용수 부족 위험지도")
        st.plotly_chart(make_risk_map(features, selected_sigungu), use_container_width=True)

        st.dataframe(
            features[[
                "final_priority_rank",
                "sigungu",
                "final_water_risk_score",
                "final_water_risk_level",
                "main_risk_driver",
            ]].rename(columns={
                "final_priority_rank": "순위",
                "sigungu": "시·군",
                "final_water_risk_score": "위험도",
                "final_water_risk_level": "단계",
                "main_risk_driver": "주요 원인",
            }),
            use_container_width=True,
            hide_index=True,
        )

    with tab2:
        st.subheader(f"{selected_sigungu} 상세 분석")

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("최종 위험도", f"{selected_row['final_water_risk_score']:.1f}점")
        d2.markdown("위험 단계")
        d2.markdown(risk_badge(selected_row["final_water_risk_level"]), unsafe_allow_html=True)
        d3.metric("우선순위", f"{int(selected_row['final_priority_rank'])}위")
        d4.metric("주요 원인", str(selected_row["main_risk_driver"]))

        st.plotly_chart(make_component_chart(selected_row), use_container_width=True)

        st.markdown("#### 주요 지표")
        indicator_cols = [
            "rain_shortage_score",
            "reservoir_risk_score",
            "groundwater_dependency_score",
            "crop_water_demand_score",
            "alternative_source_access_shortage_score",
            "agri_impact_index",
            "well_support_score",
            "well_shortage_score",
            "avg_reservoir_rate",
            "reservoir_count",
            "groundwater_well_count",
            "drilling_developed_well_count",
        ]

        available_cols = [c for c in indicator_cols if c in features.columns]
        indicator_table = selected_row[available_cols].to_frame("값").reset_index()
        indicator_table = indicator_table.rename(columns={"index": "지표"})
        st.dataframe(indicator_table, use_container_width=True, hide_index=True)

    with tab3:
        st.subheader(f"{selected_sigungu} 대체 수원 후보 TOP 5")

        if selected_cands.empty:
            st.info("해당 지역의 대체 수원 후보가 없습니다.")
        else:
            view_cols = [
                "candidate_rank",
                "candidate_reservoir_name",
                "candidate_sigungu",
                "distance_km",
                "candidate_reservoir_rate",
                "benefit_area",
                "effective_capacity",
                "candidate_score",
                "recommendation_reason",
            ]
            view_cols = [c for c in view_cols if c in selected_cands.columns]

            display = selected_cands[view_cols].copy()
            display = display.rename(columns={
                "candidate_rank": "순위",
                "candidate_reservoir_name": "후보 저수지",
                "candidate_sigungu": "소속 시·군",
                "distance_km": "거리(km)",
                "candidate_reservoir_rate": "후보 저수율",
                "benefit_area": "수혜면적",
                "effective_capacity": "유효저수량",
                "candidate_score": "후보점수",
                "recommendation_reason": "추천 사유",
            })

            st.dataframe(display, use_container_width=True, hide_index=True)

            st.download_button(
                label="선택 지역 후보 CSV 다운로드",
                data=selected_cands.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{selected_sigungu}_alternative_source_top5.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.markdown("#### 추천 산식")
        st.code(
            "candidate_score = 0.40 * distance_score + 0.35 * reservoir_surplus_score + 0.25 * benefit_area_score",
            language="text",
        )

    with tab4:
        st.subheader("전체 위험도 순위")
        st.plotly_chart(make_ranking_chart(features), use_container_width=True)

        st.download_button(
            label="전체 위험도 테이블 CSV 다운로드",
            data=features.to_csv(index=False).encode("utf-8-sig"),
            file_name="aquaguard_sigungu_features.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with tab5:
        st.subheader("보고서·발표자료용 이미지")
        render_image_if_exists(FIG_RANKING, "시·군별 최종 위험도 순위")
        render_image_if_exists(FIG_COMPONENTS, "위험도 구성요소별 기여도")
        render_image_if_exists(FIG_SCATTER, "저수율 위험도 vs 대체 수원 접근성 부족도")
        render_image_if_exists(FIG_TOP5, "우선 점검 대상 TOP 5")
        render_image_if_exists(FIG_ALT, "위험지역별 1순위 대체 수원 후보")

    st.markdown("---")
    st.subheader("행정 리포트 다운로드")

    report_html = build_html_report(selected_row, selected_cands)
    download_html_button(
        report_html,
        f"AquaGuard_AI_{selected_sigungu}_행정리포트.html",
    )


if __name__ == "__main__":
    main()
