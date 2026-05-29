from pathlib import Path
import pandas as pd
import textwrap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
FIG_DIR = ROOT / "reports" / "figures"
TABLE_DIR = ROOT / "reports" / "tables"
META_DIR = ROOT / "data" / "metadata"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_PATH = PROCESSED / "aquaguard_sigungu_features.csv"

PDF_COMPONENTS = [
    ("rain_shortage_score", "강우 부족도", 0.25),
    ("reservoir_risk_score", "저수율 위험도", 0.25),
    ("groundwater_dependency_score", "관정 의존도", 0.20),
    ("crop_water_demand_score", "작물 물수요", 0.20),
    ("alternative_source_access_shortage_score", "대체 수원 부족도", 0.10),
]

LEVEL_ORDER = ["낮음", "주의", "경계", "심각"]


def setup_korean_font() -> str | None:
    candidates = [
        "Malgun Gothic",
        "NanumGothic",
        "Nanum Gothic",
        "Noto Sans CJK KR",
        "AppleGothic",
    ]

    def normalize(name: str) -> str:
        return "".join(ch for ch in name.lower() if ch.isalnum())

    installed_fonts: dict[str, str] = {}
    for font in fm.fontManager.ttflist:
        font_name = getattr(font, "name", "")
        if font_name:
            installed_fonts.setdefault(normalize(font_name), font_name)

    plt.rcParams["axes.unicode_minus"] = False
    for candidate in candidates:
        selected = installed_fonts.get(normalize(candidate))
        if selected:
            plt.rcParams["font.family"] = selected
            print(f"[FONT] Korean font selected: {selected}")
            return selected

    print("[WARN] No Korean font found. Korean text may render as tofu boxes.")
    return None


def read_features():
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(f"Feature file not found: {FEATURE_PATH}")

    df = pd.read_csv(FEATURE_PATH)

    required = [
        "final_priority_rank",
        "sigungu",
        "final_water_risk_score",
        "final_water_risk_level",
        "main_risk_driver",
    ]

    for col in required:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    df["final_water_risk_score"] = pd.to_numeric(df["final_water_risk_score"], errors="coerce")
    df = df.sort_values("final_priority_rank").reset_index(drop=True)

    return df


def risk_level_color(level):
    if level == "심각":
        return "#d73027"
    if level == "경계":
        return "#fc8d59"
    if level == "주의":
        return "#fee08b"
    if level == "낮음":
        return "#91cf60"
    return "#cccccc"


def save_final_risk_ranking(df):
    plot_df = df.sort_values("final_water_risk_score", ascending=True).copy()

    colors = [risk_level_color(x) for x in plot_df["final_water_risk_level"]]

    fig, ax = plt.subplots(figsize=(11, 8))
    bars = ax.barh(plot_df["sigungu"], plot_df["final_water_risk_score"], color=colors)

    ax.set_title("충남 시·군별 농업용수 부족 위험도 순위", fontsize=17, fontweight="bold", pad=16)
    ax.set_xlabel("최종 위험도 점수 (0~100)")
    ax.set_ylabel("시·군")
    ax.set_xlim(0, max(100, plot_df["final_water_risk_score"].max() * 1.12))
    ax.grid(axis="x", alpha=0.25)

    for bar, score, level in zip(bars, plot_df["final_water_risk_score"], plot_df["final_water_risk_level"]):
        ax.text(
            score + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.1f}점 / {level}",
            va="center",
            fontsize=9,
        )

    legend_handles = [
        Patch(facecolor=risk_level_color(level), label=level)
        for level in LEVEL_ORDER
    ]
    ax.legend(handles=legend_handles, title="위험 단계", loc="lower right")

    fig.tight_layout()
    out = FIG_DIR / "01_final_risk_ranking.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def save_components_stacked(df):
    plot_df = df.sort_values("final_water_risk_score", ascending=False).copy()

    contribution_cols = []
    for col, label, weight in PDF_COMPONENTS:
        if col not in plot_df.columns:
            plot_df[col] = np.nan

        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce").fillna(50).clip(0, 100)
        contrib_col = f"{col}_contribution"
        plot_df[contrib_col] = plot_df[col] * weight
        contribution_cols.append((contrib_col, label, weight))

    fig, ax = plt.subplots(figsize=(13, 8))

    bottom = np.zeros(len(plot_df))

    for contrib_col, label, weight in contribution_cols:
        values = plot_df[contrib_col].values
        ax.bar(
            plot_df["sigungu"],
            values,
            bottom=bottom,
            label=f"{label} ({int(weight * 100)}%)",
        )
        bottom += values

    ax.set_title("최종 위험도 구성요소별 기여도", fontsize=17, fontweight="bold", pad=16)
    ax.set_ylabel("위험도 기여 점수")
    ax.set_xlabel("시·군")
    ax.set_ylim(0, max(100, bottom.max() * 1.15))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)

    plt.xticks(rotation=45, ha="right")

    for i, score in enumerate(plot_df["final_water_risk_score"]):
        ax.text(i, score + 1, f"{score:.1f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    out = FIG_DIR / "02_risk_components_stacked.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    long_rows = []
    for _, row in plot_df.iterrows():
        for col, label, weight in PDF_COMPONENTS:
            long_rows.append({
                "sigungu": row["sigungu"],
                "component": label,
                "raw_score": row[col],
                "weight": weight,
                "contribution_score": row[col] * weight,
                "final_water_risk_score": row["final_water_risk_score"],
            })

    long_df = pd.DataFrame(long_rows)
    long_path = TABLE_DIR / "risk_component_contributions.csv"
    long_df.to_csv(long_path, index=False, encoding="utf-8-sig")

    return out, long_path


def save_well_vs_reservoir_scatter(df):
    needed = [
        "sigungu",
        "reservoir_risk_score",
        "groundwater_dependency_score",
        "alternative_source_access_shortage_score",
        "final_water_risk_score",
        "final_water_risk_level",
    ]

    for col in needed:
        if col not in df.columns:
            df[col] = np.nan

    plot_df = df.copy()
    for col in [
        "reservoir_risk_score",
        "groundwater_dependency_score",
        "alternative_source_access_shortage_score",
        "final_water_risk_score",
    ]:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

    plot_df["reservoir_risk_score"] = plot_df["reservoir_risk_score"].fillna(50)
    plot_df["alternative_source_access_shortage_score"] = plot_df["alternative_source_access_shortage_score"].fillna(50)
    plot_df["final_water_risk_score"] = plot_df["final_water_risk_score"].fillna(0)

    sizes = 80 + plot_df["final_water_risk_score"] * 4
    colors = [risk_level_color(x) for x in plot_df["final_water_risk_level"]]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        plot_df["reservoir_risk_score"],
        plot_df["alternative_source_access_shortage_score"],
        s=sizes,
        c=colors,
        alpha=0.75,
        edgecolors="black",
        linewidths=0.5,
    )

    for _, row in plot_df.iterrows():
        ax.text(
            row["reservoir_risk_score"] + 1,
            row["alternative_source_access_shortage_score"] + 1,
            row["sigungu"],
            fontsize=9,
        )

    ax.set_title("저수율 위험도 vs 대체 수원 접근성 부족도", fontsize=17, fontweight="bold", pad=16)
    ax.set_xlabel("저수율 위험도")
    ax.set_ylabel("대체 수원 접근성 부족도")
    ax.set_xlim(-3, 103)
    ax.set_ylim(-3, 103)
    ax.grid(alpha=0.25)

    ax.axvline(50, linestyle="--", alpha=0.35)
    ax.axhline(50, linestyle="--", alpha=0.35)

    legend_handles = [
        Patch(facecolor=risk_level_color(level), label=level)
        for level in LEVEL_ORDER
    ]
    ax.legend(handles=legend_handles, title="위험 단계", loc="lower right")

    fig.tight_layout()
    out = FIG_DIR / "03_reservoir_vs_alternative_shortage_scatter.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def save_top5_table_image(df):
    top = df.sort_values("final_priority_rank").head(5).copy()

    cols = [
        "final_priority_rank",
        "sigungu",
        "final_water_risk_score",
        "final_water_risk_level",
        "main_risk_driver",
        "recommended_action",
    ]

    top = top[cols].copy()
    top["final_water_risk_score"] = top["final_water_risk_score"].map(lambda x: f"{x:.1f}")

    rename = {
        "final_priority_rank": "순위",
        "sigungu": "시·군",
        "final_water_risk_score": "위험도",
        "final_water_risk_level": "단계",
        "main_risk_driver": "주요 원인",
        "recommended_action": "권고 조치",
    }

    top = top.rename(columns=rename)
    display_top = top.copy()
    for col in ["주요 원인", "권고 조치"]:
        if col in display_top.columns:
            display_top[col] = display_top[col].map(lambda value: textwrap.fill(str(value), width=18))

    fig, ax = plt.subplots(figsize=(16, 5.2))
    ax.axis("off")

    table = ax.table(
        cellText=display_top.values,
        colLabels=display_top.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.07, 0.10, 0.10, 0.10, 0.25, 0.38],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.9)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#e6e6e6")
        else:
            level = display_top.iloc[row - 1]["단계"]
            if col == 3:
                cell.set_facecolor(risk_level_color(level))

    ax.set_title("AquaGuard AI 우선 점검 대상 TOP 5", fontsize=17, fontweight="bold", pad=18)

    fig.tight_layout()
    out = FIG_DIR / "04_top5_priority_table.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)

    csv_out = TABLE_DIR / "top_priority_summary.csv"
    top.to_csv(csv_out, index=False, encoding="utf-8-sig")

    return out, csv_out


def save_alternative_source_top1_figure():
    source = TABLE_DIR / "alternative_source_top5_by_sigungu.csv"
    if not source.exists():
        print(f"[WARN] Alternative source table missing; skipped figure: {source}")
        return None

    recommendations = pd.read_csv(source)
    required = [
        "target_sigungu",
        "target_final_water_risk_score",
        "candidate_rank",
        "candidate_reservoir_name",
        "candidate_score",
    ]
    missing = [col for col in required if col not in recommendations.columns]
    if missing:
        print(f"[WARN] Alternative source table missing columns; skipped figure: {missing}")
        return None

    top_targets = (
        recommendations[pd.to_numeric(recommendations["candidate_rank"], errors="coerce") == 1]
        .copy()
    )
    top_targets["target_final_water_risk_score"] = pd.to_numeric(
        top_targets["target_final_water_risk_score"], errors="coerce"
    )
    top_targets["candidate_score"] = pd.to_numeric(top_targets["candidate_score"], errors="coerce")
    top_targets = top_targets.sort_values("target_final_water_risk_score", ascending=False).head(5)

    if top_targets.empty:
        print(f"[WARN] Alternative source table has no rank-1 rows; skipped figure: {source}")
        return None

    fig, ax = plt.subplots(figsize=(12, 6))

    labels = [
        f"{row['target_sigungu']} → {row['candidate_reservoir_name']}"
        for _, row in top_targets.iterrows()
    ]
    scores = top_targets["candidate_score"].astype(float)

    ax.barh(labels[::-1], scores[::-1])
    ax.set_title("위험지역별 1순위 대체 수원 후보", fontsize=16, fontweight="bold", pad=14)
    ax.set_xlabel("후보 적합도 점수")
    ax.set_xlim(0, max(100, scores.max() * 1.15))
    ax.grid(axis="x", alpha=0.25)

    for idx, score in enumerate(scores[::-1]):
        ax.text(score + 1, idx, f"{score:.1f}", va="center", fontsize=9)

    fig.tight_layout()

    out = FIG_DIR / "05_alternative_source_top1_by_risk_area.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return out


def save_main_driver_summary(df):
    summary = (
        df.groupby(["main_risk_driver", "final_water_risk_level"], as_index=False)
          .agg(
              sigungu_count=("sigungu", "count"),
              avg_final_risk=("final_water_risk_score", "mean"),
          )
          .sort_values(["sigungu_count", "avg_final_risk"], ascending=[False, False])
    )

    out = TABLE_DIR / "main_driver_summary.csv"
    summary.to_csv(out, index=False, encoding="utf-8-sig")
    return out


def save_markdown_summary(df, outputs):
    top = df.sort_values("final_priority_rank").head(5).copy()

    lines = []
    lines.append("# AquaGuard AI 시각화 생성 요약")
    lines.append("")
    lines.append("## 생성 파일")
    lines.append("")

    for name, path in outputs.items():
        lines.append(f"- {name}: {path.relative_to(ROOT).as_posix()}")

    lines.append("")
    lines.append("## 우선 점검 대상 TOP 5")
    lines.append("")
    lines.append("| 순위 | 시·군 | 위험도 | 단계 | 주요 원인 |")
    lines.append("|---:|---|---:|---|---|")

    for _, row in top.iterrows():
        lines.append(
            f"| {int(row['final_priority_rank'])} | {row['sigungu']} | "
            f"{row['final_water_risk_score']:.1f} | {row['final_water_risk_level']} | "
            f"{row['main_risk_driver']} |"
        )

    lines.append("")
    lines.append("## 해석 기준")
    lines.append("")
    lines.append("- 최종 위험도는 제출 PDF의 25:25:20:20:10 산식을 따른다.")
    lines.append("- 구성요소 기여도 그래프는 각 지표 점수에 PDF 가중치를 곱한 값이다.")
    lines.append("- 산점도는 저수율 위험과 대체 수원 접근성 부족이 동시에 높은 지역을 식별하기 위한 보조 시각화다.")
    lines.append("- TOP 5 표는 발표자료와 보고서에 바로 삽입 가능한 요약표다.")

    out = META_DIR / "visual_generation_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main():
    setup_korean_font()
    df = read_features()

    outputs = {}

    outputs["final_risk_ranking"] = save_final_risk_ranking(df)

    stacked_png, component_csv = save_components_stacked(df)
    outputs["risk_components_stacked"] = stacked_png
    outputs["risk_component_contributions"] = component_csv

    outputs["reservoir_vs_alternative_shortage_scatter"] = save_well_vs_reservoir_scatter(df)

    top5_png, top5_csv = save_top5_table_image(df)
    outputs["top5_priority_table_png"] = top5_png
    outputs["top_priority_summary_csv"] = top5_csv

    alt_png = save_alternative_source_top1_figure()
    if alt_png:
        outputs["alternative_source_top1_by_risk_area"] = alt_png

    outputs["main_driver_summary"] = save_main_driver_summary(df)

    summary_md = save_markdown_summary(df, outputs)
    outputs["visual_generation_summary"] = summary_md

    print("[Saved visual outputs]")
    for name, path in outputs.items():
        print(f"- {name}: {path}")

    print()
    print("[Top 5]")
    print(df.sort_values("final_priority_rank")[[
        "final_priority_rank",
        "sigungu",
        "final_water_risk_score",
        "final_water_risk_level",
        "main_risk_driver",
    ]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
