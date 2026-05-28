from pathlib import Path
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORT_TABLES = ROOT / "reports" / "tables"
REPORT_FIGURES = ROOT / "reports" / "figures"
META = ROOT / "data" / "metadata"

REPORT_TABLES.mkdir(parents=True, exist_ok=True)
REPORT_FIGURES.mkdir(parents=True, exist_ok=True)
META.mkdir(parents=True, exist_ok=True)

FEATURE_PATH = PROCESSED / "aquaguard_sigungu_features.csv"
FACILITY_PATH = PROCESSED / "01_reservoir_facility_clean.csv"

# 시·군 대표 좌표. MVP 1차에서는 저수지 개별 좌표가 부족하므로 시·군 대표 좌표 기반 거리로 계산한다.
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

CANDIDATE_WEIGHTS = {
    "distance_score": 0.40,
    "reservoir_surplus_score": 0.35,
    "benefit_area_score": 0.25,
}


def setup_korean_font():
    candidates = [
        "Malgun Gothic",
        "맑은 고딕",
        "AppleGothic",
        "NanumGothic",
        "Noto Sans CJK KR",
        "Noto Sans KR",
    ]

    installed = {f.name for f in fm.fontManager.ttflist}

    for font in candidates:
        if font in installed:
            plt.rcParams["font.family"] = font
            break

    plt.rcParams["axes.unicode_minus"] = False


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0

    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)

    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return r * c


def minmax_0_100(s):
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(50.0, index=s.index)

    mn = s.min()
    mx = s.max()

    if mx == mn:
        return pd.Series(50.0, index=s.index)

    return ((s - mn) / (mx - mn) * 100).clip(0, 100)


def read_inputs():
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(f"Missing feature table: {FEATURE_PATH}")

    if not FACILITY_PATH.exists():
        raise FileNotFoundError(f"Missing reservoir facility file: {FACILITY_PATH}")

    features = pd.read_csv(FEATURE_PATH)
    facilities = pd.read_csv(FACILITY_PATH)

    required_feature_cols = [
        "sigungu",
        "final_priority_rank",
        "final_water_risk_score",
        "final_water_risk_level",
        "avg_reservoir_rate",
        "reservoir_risk_score",
    ]

    for col in required_feature_cols:
        if col not in features.columns:
            raise KeyError(f"Missing required feature column: {col}")

    required_facility_cols = [
        "facility_name",
        "sigungu",
        "address",
        "benefit_area",
        "effective_capacity",
    ]

    for col in required_facility_cols:
        if col not in facilities.columns:
            raise KeyError(f"Missing required facility column: {col}")

    return features, facilities


def prepare_facilities(features, facilities):
    sigungu_rate = features[[
        "sigungu",
        "avg_reservoir_rate",
        "reservoir_risk_score",
        "final_water_risk_score",
    ]].copy()

    sigungu_rate["avg_reservoir_rate"] = pd.to_numeric(
        sigungu_rate["avg_reservoir_rate"], errors="coerce"
    )

    facilities = facilities.copy()

    facilities["benefit_area"] = pd.to_numeric(
        facilities["benefit_area"], errors="coerce"
    ).fillna(0)

    facilities["effective_capacity"] = pd.to_numeric(
        facilities["effective_capacity"], errors="coerce"
    ).fillna(0)

    facilities = facilities.merge(
        sigungu_rate.rename(columns={
            "avg_reservoir_rate": "candidate_sigungu_avg_reservoir_rate",
            "reservoir_risk_score": "candidate_sigungu_reservoir_risk_score",
            "final_water_risk_score": "candidate_sigungu_final_risk_score",
        }),
        on="sigungu",
        how="left"
    )

    # 시설별 실시간 저수율이 없으므로 후보 저수지의 시·군 평균 저수율을 MVP 1차 후보 여유도 proxy로 사용한다.
    facilities["candidate_reservoir_rate"] = facilities["candidate_sigungu_avg_reservoir_rate"]

    # 시·군 대표 좌표 부여
    facilities["candidate_lat"] = facilities["sigungu"].map(lambda x: SIGUNGU_CENTROIDS.get(x, (np.nan, np.nan))[0])
    facilities["candidate_lon"] = facilities["sigungu"].map(lambda x: SIGUNGU_CENTROIDS.get(x, (np.nan, np.nan))[1])

    # 후보로 쓸 수 없는 데이터 제거
    facilities = facilities[
        facilities["sigungu"].notna()
        & facilities["facility_name"].notna()
        & facilities["candidate_lat"].notna()
        & facilities["candidate_lon"].notna()
    ].copy()

    return facilities


def build_candidates_for_target(target_row, facilities, max_distance_km=120, top_n=5):
    target_sigungu = target_row["sigungu"]

    if target_sigungu not in SIGUNGU_CENTROIDS:
        return pd.DataFrame()

    target_lat, target_lon = SIGUNGU_CENTROIDS[target_sigungu]

    cand = facilities.copy()
    cand["target_sigungu"] = target_sigungu
    cand["target_final_water_risk_score"] = target_row["final_water_risk_score"]
    cand["target_final_water_risk_level"] = target_row["final_water_risk_level"]
    cand["target_main_risk_driver"] = target_row.get("main_risk_driver", "")

    cand["target_lat"] = target_lat
    cand["target_lon"] = target_lon

    cand["distance_km"] = cand.apply(
        lambda r: haversine_km(
            target_lat,
            target_lon,
            r["candidate_lat"],
            r["candidate_lon"]
        ),
        axis=1
    )

    cand = cand[cand["distance_km"] <= max_distance_km].copy()

    if cand.empty:
        return cand

    # 같은 시·군 후보는 실제 행정 검토에서 가장 가까운 후보일 수 있으므로 제외하지 않는다.
    # 다만 보고서에서 “우선 검토 후보”라고 표현한다.

    cand["distance_score"] = (100 - (cand["distance_km"] / max_distance_km * 100)).clip(0, 100)

    # 저수율 여유도: 50% 이상부터 후보 여유로 간주.
    # 50 미만이면 점수를 낮게, 100에 가까울수록 높게.
    cand["candidate_reservoir_rate"] = pd.to_numeric(
        cand["candidate_reservoir_rate"], errors="coerce"
    )

    cand["reservoir_surplus_score"] = (
        (cand["candidate_reservoir_rate"].fillna(50) - 50) / 50 * 100
    ).clip(0, 100)

    cand["benefit_area_score"] = minmax_0_100(cand["benefit_area"])

    cand["candidate_score"] = (
        cand["distance_score"] * CANDIDATE_WEIGHTS["distance_score"]
        + cand["reservoir_surplus_score"] * CANDIDATE_WEIGHTS["reservoir_surplus_score"]
        + cand["benefit_area_score"] * CANDIDATE_WEIGHTS["benefit_area_score"]
    ).clip(0, 100)

    cand["recommendation_reason"] = cand.apply(build_reason, axis=1)

    # 같은 저수지가 원천 데이터의 연도/버전 차이로 여러 번 들어오는 문제 제거
    # 동일 target에 대해 같은 후보 저수지명 + 후보 시군 조합은 1개만 유지한다.
    cand = cand.sort_values(
        ["candidate_score", "candidate_reservoir_rate", "benefit_area", "effective_capacity"],
        ascending=[False, False, False, False]
    ).reset_index(drop=True)

    before_dedup = len(cand)

    cand = cand.drop_duplicates(
        subset=["target_sigungu", "facility_name", "sigungu"],
        keep="first"
    ).reset_index(drop=True)

    after_dedup = len(cand)

    if before_dedup != after_dedup:
        print(
            f"  [DEDUP] {target_sigungu}: removed {before_dedup - after_dedup} duplicate reservoir rows",
            flush=True
        )

    cand["candidate_rank"] = np.arange(1, len(cand) + 1)

    return cand.head(top_n)


def build_reason(row):
    reasons = []

    if row["distance_km"] <= 20:
        reasons.append("근거리")
    elif row["distance_km"] <= 50:
        reasons.append("중거리")

    rate = row.get("candidate_reservoir_rate")
    if pd.notna(rate):
        if rate >= 80:
            reasons.append("저수율 여유 높음")
        elif rate >= 60:
            reasons.append("저수율 여유 보통")

    if row.get("benefit_area", 0) > 0:
        reasons.append("수혜면적 정보 보유")

    if not reasons:
        reasons.append("기본 후보")

    return " / ".join(reasons)


def build_all_recommendations(features, facilities):
    features = features.copy()
    features["final_water_risk_score"] = pd.to_numeric(
        features["final_water_risk_score"], errors="coerce"
    )

    features = features.sort_values("final_priority_rank").reset_index(drop=True)

    all_parts = []

    for _, target_row in features.iterrows():
        cand = build_candidates_for_target(target_row, facilities, max_distance_km=120, top_n=5)
        if not cand.empty:
            all_parts.append(cand)

    if not all_parts:
        raise RuntimeError("No alternative source candidates generated.")

    result = pd.concat(all_parts, ignore_index=True)

    keep_cols = [
        "target_sigungu",
        "target_final_water_risk_score",
        "target_final_water_risk_level",
        "target_main_risk_driver",
        "candidate_rank",
        "facility_name",
        "sigungu",
        "address",
        "distance_km",
        "candidate_reservoir_rate",
        "benefit_area",
        "effective_capacity",
        "distance_score",
        "reservoir_surplus_score",
        "benefit_area_score",
        "candidate_score",
        "recommendation_reason",
    ]

    keep_cols = [c for c in keep_cols if c in result.columns]
    result = result[keep_cols].copy()

    result = result.rename(columns={
        "sigungu": "candidate_sigungu",
        "facility_name": "candidate_reservoir_name",
    })

    return result


def save_top_target_figure(recommendations):
    top_targets = (
        recommendations[recommendations["candidate_rank"] == 1]
        .sort_values("target_final_water_risk_score", ascending=False)
        .head(5)
        .copy()
    )

    if top_targets.empty:
        return None

    fig, ax = plt.subplots(figsize=(12, 6))

    labels = [
        f"{r['target_sigungu']} → {r['candidate_reservoir_name']}"
        for _, r in top_targets.iterrows()
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

    out_path = REPORT_FIGURES / "05_alternative_source_top1_by_risk_area.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return out_path


def write_method_doc():
    text = """# AquaGuard AI 대체 수원 후보 추천 방식

## 목적

위험지역 주변의 저수지를 거리, 저수율, 수혜면적 기준으로 평가하여 행정 담당자가 우선 검토할 후보 TOP 5를 제시한다.

## 후보 점수

candidate_score =
0.40 * distance_score
+ 0.35 * reservoir_surplus_score
+ 0.25 * benefit_area_score

## 지표 정의

- distance_score: 위험 시·군 대표좌표와 후보 저수지 소속 시·군 대표좌표 간 거리 기반 점수
- reservoir_surplus_score: 후보 저수지 소속 시·군의 평균 저수율 기반 여유도 점수
- benefit_area_score: 후보 저수지의 수혜면적 기반 점수

## MVP 1차 한계

- 저수지 개별 좌표가 없거나 불완전한 경우가 있어 시·군 대표좌표 기반 거리로 계산한다.
- 후보 추천은 실제 공급 가능성을 확정하는 기능이 아니라 행정 검토 후보를 좁히는 기능이다.
- 실제 공급 가능 여부는 관로, 수리권, 수질, 현장 접근성, 행정 협의를 추가 검토해야 한다.

## 제출 PDF와의 정합성

제출 제안서에서 제시한 거리·저수율·수혜면적 기반 대체 수원 후보 TOP 5 추천 구조를 MVP 수준에서 구현한다.
"""

    path = META / "alternative_source_recommendation_method.md"
    path.write_text(text, encoding="utf-8")
    return path


def main():
    setup_korean_font()

    features, facilities = read_inputs()
    facilities = prepare_facilities(features, facilities)

    recommendations = build_all_recommendations(features, facilities)

    output_path = PROCESSED / "alternative_source_candidates.csv"
    table_path = REPORT_TABLES / "alternative_source_top5_by_sigungu.csv"

    recommendations.to_csv(output_path, index=False, encoding="utf-8-sig")
    recommendations.to_csv(table_path, index=False, encoding="utf-8-sig")

    figure_path = save_top_target_figure(recommendations)
    method_path = write_method_doc()

    print("[Saved]")
    print(f"- {output_path} rows={len(recommendations)}")
    print(f"- {table_path} rows={len(recommendations)}")
    if figure_path:
        print(f"- {figure_path}")
    print(f"- {method_path}")

    print()
    print("[Preview: top candidates]")
    preview_cols = [
        "target_sigungu",
        "candidate_rank",
        "candidate_reservoir_name",
        "candidate_sigungu",
        "distance_km",
        "candidate_reservoir_rate",
        "benefit_area",
        "candidate_score",
        "recommendation_reason",
    ]

    print(
        recommendations[preview_cols]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
