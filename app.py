# app.py
import io
import itertools
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import font_manager


# =========================
# 기본 설정
# =========================
ROAD_START = 0.0
ROAD_END = 106.8

IC_POINTS = {
    "서영암IC": 0,
    "강진IC": 20,
    "장흥IC": 40,
    "보성IC": 60,
    "벌교IC": 80,
    "남순천": 100,
    "해룡IC": 106.8,
}

LANES = ["1차로", "2차로", "갓길"]


# =========================
# 한글 폰트 설정
# =========================
def set_korean_font():
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}

    font_candidates = [
        "Malgun Gothic",      # Windows
        "AppleGothic",        # Mac
        "NanumGothic",        # Linux
        "Noto Sans CJK KR",
        "Noto Sans KR",
        "DejaVu Sans",
    ]

    for font_name in font_candidates:
        if font_name in available_fonts:
            plt.rcParams["font.family"] = font_name
            break

    plt.rcParams["axes.unicode_minus"] = False


set_korean_font()


# =========================
# 데이터 처리 함수
# =========================
def parse_lanes(text):
    """
    차로 입력 예시
    - 1차로
    - 2차로
    - 갓길
    - 1차로,2차로
    - 2차로,갓길
    - 전체
    """
    if pd.isna(text) or str(text).strip() == "":
        return LANES.copy()

    text = str(text).replace(" ", "")

    if "전체" in text:
        return LANES.copy()

    result = []
    for lane in LANES:
        if lane in text:
            result.append(lane)

    return result if result else LANES.copy()
    
def clean_group_name(value):
    """
    그룹명 빈칸, NaN, nan, None 등을 모두 빈 문자열로 처리.
    실제로 입력된 그룹명만 다공종 그룹으로 사용.
    """
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text.lower() in ["nan", "none", "null", ""]:
        return ""

    return text

def normalize_df(df):
    """
    사용자가 입력한 표를 프로그램이 쓰기 쉬운 형태로 정리
    """
    rows = []

    for _, row in df.iterrows():
        if pd.isna(row.get("번호")) or pd.isna(row.get("시점")) or pd.isna(row.get("종점")):
            continue

        try:
            start = float(row["시점"])
            end = float(row["종점"])
        except ValueError:
            continue

        start2 = min(start, end)
        end2 = max(start, end)

        if end2 <= ROAD_START or start2 >= ROAD_END:
            continue

        start2 = max(start2, ROAD_START)
        end2 = min(end2, ROAD_END)

        if end2 <= start2:
            continue

        direction = str(row.get("방향", "")).strip()
        if direction not in ["순천", "영암"]:
            continue

        no = str(row["번호"]).replace("#", "").strip()
        name = str(row.get("공사명", "")).strip()
        group = clean_group_name(row.get("그룹명", ""))

        lanes = parse_lanes(row.get("차로", "전체"))

        rows.append({
            "번호": no,
            "공사명": name,
            "방향": direction,
            "시점": start2,
            "종점": end2,
            "차로": lanes,
            "차로표시": ",".join(lanes),
            "그룹명": group,
        })

    return pd.DataFrame(rows)


def build_work_units(df, use_group=True):
    """
    개별 작업을 실제 검토 단위로 변환.

    그룹명이 있으면 같은 그룹명을 하나의 다공종 작업으로 묶음.
    그룹명이 없으면 개별 작업 그대로 사용.
    """
    if df.empty:
        return pd.DataFrame()

    units = {}

    for idx, row in df.iterrows():
        group_name = clean_group_name(row.get("그룹명", ""))

        if use_group and group_name != "":
            key = ("GROUP", group_name, row["방향"])
        else:
            key = ("SINGLE", idx, row["방향"])

        if key not in units:
            units[key] = {
                "unit_id": "|".join(map(str, key)),
                "번호목록": [],
                "공사명목록": [],
                "방향": row["방향"],
                "시점": row["시점"],
                "종점": row["종점"],
                "차로": set(),
                "그룹명": group_name if group_name != "" else "",
                "원본인덱스": [],
            }

        units[key]["번호목록"].append(row["번호"])
        units[key]["공사명목록"].append(row["공사명"])
        units[key]["시점"] = min(units[key]["시점"], row["시점"])
        units[key]["종점"] = max(units[key]["종점"], row["종점"])
        units[key]["차로"].update(row["차로"])
        units[key]["원본인덱스"].append(idx)

    result = []

    for _, unit in units.items():
        lane_list = [lane for lane in LANES if lane in unit["차로"]]
        no_list = unit["번호목록"]
        name_list = [n for n in unit["공사명목록"] if n]

        is_group = len(no_list) >= 2 and unit["그룹명"] != ""

        if is_group:
            display_no = ",".join([f"#{n}" for n in no_list])
            display_name = "다공종작업"
            detail_name = " / ".join(name_list)
        else:
            display_no = f"#{no_list[0]}"
            display_name = name_list[0] if name_list else ""
            detail_name = display_name

        result.append({
            "unit_id": unit["unit_id"],
            "번호표시": display_no,
            "공사명": display_name,
            "상세공사명": detail_name,
            "방향": unit["방향"],
            "시점": unit["시점"],
            "종점": unit["종점"],
            "차로": lane_list,
            "차로표시": ",".join(lane_list),
            "그룹명": unit["그룹명"],
            "다공종여부": is_group,
            "원본인덱스": unit["원본인덱스"],
        })

    return pd.DataFrame(result)


def interval_relation(a_start, a_end, b_start, b_end):
    """
    두 이정 구간의 겹침 또는 이격거리 계산
    """
    overlap_start = max(a_start, b_start)
    overlap_end = min(a_end, b_end)
    overlap_len = overlap_end - overlap_start

    if overlap_len > 0:
        return {
            "type": "overlap",
            "distance": 0.0,
            "start": overlap_start,
            "end": overlap_end,
        }

    gap = max(a_start, b_start) - min(a_end, b_end)

    gap_start = min(a_end, b_end)
    gap_end = max(a_start, b_start)

    return {
        "type": "near",
        "distance": gap,
        "start": gap_start,
        "end": gap_end,
    }


def find_conflicts(units, threshold_km=5.0, same_direction_only=True, consider_lane=False):
    """
    겹침 또는 5km 이내 인접 구간 찾기.

    units는 이미 다공종 그룹이 묶인 상태.
    따라서 같은 그룹 내부 작업끼리는 여기서 비교되지 않음.
    """
    if units.empty:
        return pd.DataFrame()

    conflicts = []

    for i, j in itertools.combinations(units.index, 2):
        a = units.loc[i]
        b = units.loc[j]

        if same_direction_only and a["방향"] != b["방향"]:
            continue

        if consider_lane:
            if len(set(a["차로"]) & set(b["차로"])) == 0:
                continue

        relation = interval_relation(a["시점"], a["종점"], b["시점"], b["종점"])

        if relation["type"] == "overlap":
            conflicts.append({
                "작업1": f"{a['번호표시']} {a['공사명']}",
                "작업2": f"{b['번호표시']} {b['공사명']}",
                "방향": a["방향"] if a["방향"] == b["방향"] else "양방향",
                "구분": "구간 겹침",
                "문제구간": f"{relation['start']:.1f}k ~ {relation['end']:.1f}k",
                "이격거리(km)": 0.0,
                "start": relation["start"],
                "end": relation["end"],
                "type": "overlap",
                "unit_id1": a["unit_id"],
                "unit_id2": b["unit_id"],
            })

        elif relation["distance"] <= threshold_km:
            conflicts.append({
                "작업1": f"{a['번호표시']} {a['공사명']}",
                "작업2": f"{b['번호표시']} {b['공사명']}",
                "방향": a["방향"] if a["방향"] == b["방향"] else "양방향",
                "구분": f"{threshold_km:g}km 이내 인접",
                "문제구간": f"{relation['start']:.1f}k ~ {relation['end']:.1f}k",
                "이격거리(km)": round(relation["distance"], 2),
                "start": relation["start"],
                "end": relation["end"],
                "type": "near",
                "unit_id1": a["unit_id"],
                "unit_id2": b["unit_id"],
            })

    return pd.DataFrame(conflicts)


# =========================
# 도식 그리기 함수
# =========================
def get_lane_y_range(direction, lanes):
    """
    중앙 굵은선을 기준으로
    위쪽: 영암방향
    아래쪽: 순천방향

    중앙선 가까운 순서:
    1차로 → 2차로 → 갓길
    """
    lane_idx = {
        "1차로": 0,
        "2차로": 1,
        "갓길": 2,
    }

    idxs = [lane_idx[lane] for lane in lanes if lane in lane_idx]

    if not idxs:
        idxs = [0, 1, 2]

    min_idx = min(idxs)
    max_idx = max(idxs)

    if direction == "영암":
        y0 = min_idx
        y1 = max_idx + 1
    else:
        y0 = -(max_idx + 1)
        y1 = -min_idx

    return y0, y1


def draw_diagram(units, conflicts, show_warnings=True, submit_mode=False):
    """
    공사구간 도식 생성.

    show_warnings=True:
        검토용. 빨강/주황 음영, 빨간 테두리 표시.

    show_warnings=False:
        제출용. 경고 표시 없이 회색 박스만 표시.
    """
    fig, ax = plt.subplots(figsize=(15, 4.8), dpi=160)

    ax.set_xlim(-2, 108.5)
    ax.set_ylim(-3.85, 4.05)
    ax.axis("off")

    # 문제구간 음영 표시: 검토용에서만 표시
    if show_warnings and conflicts is not None and not conflicts.empty:
        for _, c in conflicts.iterrows():
            if c["type"] == "overlap":
                ax.axvspan(c["start"], c["end"], color="red", alpha=0.18, zorder=0)
            else:
                ax.axvspan(c["start"], c["end"], color="orange", alpha=0.13, zorder=0)

    # 세로 격자선
    x_ticks = list(range(0, 101, 10)) + [106.8]
    for x in x_ticks:
        major = x in [0, 20, 40, 60, 80, 100] or abs(x - 106.8) < 0.01
        lw = 1.2 if major else 0.8
        ax.plot([x, x], [-3, 3], color="black", linewidth=lw, alpha=0.85)

    # 가로 차로선
    for y in [-3, -2, -1, 0, 1, 2, 3]:
        if y == 0:
            ax.plot([0, ROAD_END], [y, y], color="black", linewidth=3.2)
        else:
            ax.plot([0, ROAD_END], [y, y], color="black", linewidth=0.9, alpha=0.8)

    # 거리 숫자
    for x in range(0, 101, 10):
        label = "0k" if x == 0 else f"{x}"
        ax.text(x + 0.4, 3.08, label, fontsize=10, ha="left", va="bottom")

    ax.text(106.8, 3.08, "107k", fontsize=10, ha="right", va="bottom")

    # IC 표시
    for name, x in IC_POINTS.items():
        ax.text(
            x,
            3.55,
            name,
            fontsize=10,
            fontweight="bold",
            color="blue",
            ha="center",
            va="bottom",
        )

    # 방향 라벨
    ax.text(-1.2, 1.5, "영암\n방향", fontsize=9, ha="right", va="center")
    ax.text(-1.2, -1.5, "순천\n방향", fontsize=9, ha="right", va="center")

    # 차로 라벨
    ax.text(108.0, 0.5, "1차로", fontsize=8, va="center")
    ax.text(108.0, 1.5, "2차로", fontsize=8, va="center")
    ax.text(108.0, 2.5, "갓길", fontsize=8, va="center")
    ax.text(108.0, -0.5, "1차로", fontsize=8, va="center")
    ax.text(108.0, -1.5, "2차로", fontsize=8, va="center")
    ax.text(108.0, -2.5, "갓길", fontsize=8, va="center")

    # 충돌 대상 unit_id
    conflict_unit_ids = set()
    if show_warnings and conflicts is not None and not conflicts.empty:
        for _, c in conflicts.iterrows():
            conflict_unit_ids.add(c["unit_id1"])
            conflict_unit_ids.add(c["unit_id2"])

    # 작업 박스 표시
    for _, row in units.iterrows():
        x0 = row["시점"]
        width = row["종점"] - row["시점"]
        y0, y1 = get_lane_y_range(row["방향"], row["차로"])
        height = y1 - y0

        is_warning = show_warnings and row["unit_id"] in conflict_unit_ids

        edge_color = "red" if is_warning else "black"
        line_width = 2.0 if is_warning else 1.0

        rect = Rectangle(
            (x0, y0),
            width,
            height,
            facecolor="#BFBFBF",
            edgecolor=edge_color,
            linewidth=line_width,
            zorder=3,
        )
        ax.add_patch(rect)

        # 박스 안 텍스트
        if row["다공종여부"]:
            if width >= 8:
                label = f"{row['그룹명']}\n{row['번호표시']}\n다공종"
                fontsize = 7
            else:
                label = f"{row['그룹명']}\n{row['번호표시']}"
                fontsize = 7
        else:
            if width >= 8 and not submit_mode:
                label = f"{row['번호표시']}\n{row['공사명']}"
                fontsize = 7
            else:
                label = f"{row['번호표시']}"
                fontsize = 9

        ax.text(
            x0 + width / 2,
            y0 + height / 2,
            label,
            fontsize=fontsize,
            ha="center",
            va="center",
            zorder=4,
        )

    # 검토용 범례
    if show_warnings:
        ax.text(
            0,
            -3.55,
            "빨간 음영: 구간 겹침 / 주황 음영: 5km 이내 인접 / 빨간 테두리: 검토 대상 작업",
            fontsize=9,
            ha="left",
            va="center",
        )

    return fig


# =========================
# Streamlit 화면
# =========================
st.set_page_config(
    page_title="보성지사 공사구간 도식 생성기",
    layout="wide",
)

st.title("보성지사 공사구간 도식 생성기")
st.caption("영암순천선 0k ~ 106.8k 기준 / 다공종 그룹 처리 포함")

with st.sidebar:
    st.header("설정")

    output_mode = st.radio(
        "출력 모드",
        ["검토용", "제출용"],
        index=0,
        help="검토용은 겹침/인접 경고를 표시하고, 제출용은 경고 표시를 숨깁니다.",
    )

    threshold = st.number_input(
        "인접 판단 거리(km)",
        min_value=0.0,
        max_value=20.0,
        value=5.0,
        step=0.5,
    )

    use_group = st.checkbox(
        "그룹명 기준으로 다공종 작업 묶기",
        value=True,
        help="같은 그룹명과 같은 방향의 작업을 하나의 다공종 작업으로 묶습니다.",
    )

    same_direction_only = st.checkbox(
        "같은 방향끼리만 검토",
        value=True,
    )

    consider_lane = st.checkbox(
        "차로까지 고려해서 검토",
        value=False,
        help="체크하면 차로가 겹치는 작업끼리만 겹침/인접 여부를 검토합니다.",
    )

    st.markdown("---")
    st.markdown("""
    **방향 기준**
    - 순천방향: 0k → 106.8k
    - 영암방향: 106.8k → 0k

    **차로 기준**
    - 중앙선 가까운 쪽: 1차로
    - 그다음: 2차로
    - 바깥쪽: 갓길
    """)


# 예시 데이터
default_df = pd.DataFrame([
    {"번호": 1, "공사명": "포장 보수", "방향": "순천", "시점": 30.0, "종점": 38.0, "차로": "2차로,갓길", "그룹명": "A"},
    {"번호": 3, "공사명": "응력완화줄눈", "방향": "순천", "시점": 39.5, "종점": 43.0, "차로": "2차로", "그룹명": "A"},
    {"번호": 5, "공사명": "교량 점검", "방향": "순천", "시점": 34.5, "종점": 36.5, "차로": "1차로", "그룹명": ""},
    {"번호": 2, "공사명": "시설물 점검", "방향": "순천", "시점": 17.0, "종점": 24.0, "차로": "1차로", "그룹명": ""},
    {"번호": 4, "공사명": "통신 작업", "방향": "영암", "시점": 103.0, "종점": 106.5, "차로": "갓길", "그룹명": ""},
])

st.subheader("1. 공사구간 입력")

input_df = st.data_editor(
    default_df,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "번호": st.column_config.NumberColumn("번호", min_value=1, step=1),
        "공사명": st.column_config.TextColumn("공사명"),
        "방향": st.column_config.SelectboxColumn("방향", options=["순천", "영암"]),
        "시점": st.column_config.NumberColumn("시점(km)", min_value=0.0, max_value=106.8, step=0.1),
        "종점": st.column_config.NumberColumn("종점(km)", min_value=0.0, max_value=106.8, step=0.1),
        "차로": st.column_config.TextColumn("차로 예: 1차로 / 2차로,갓길 / 전체"),
        "그룹명": st.column_config.TextColumn("다공종 그룹명 예: A, B"),
    },
)

work_df = normalize_df(input_df)
units_df = build_work_units(work_df, use_group=use_group)

show_warnings = output_mode == "검토용"

st.subheader("2. 다공종 묶음 결과")

if units_df.empty:
    st.warning("입력된 공사구간이 없습니다.")
else:
    display_cols = [
        "번호표시",
        "공사명",
        "상세공사명",
        "방향",
        "시점",
        "종점",
        "차로표시",
        "그룹명",
        "다공종여부",
    ]
    st.dataframe(units_df[display_cols], use_container_width=True)

    conflicts = find_conflicts(
        units_df,
        threshold_km=threshold,
        same_direction_only=same_direction_only,
        consider_lane=consider_lane,
    )

    st.subheader("3. 겹침 / 인접 판정")

    if output_mode == "검토용":
        if conflicts.empty:
            st.success("겹치는 구간 또는 기준 거리 이내 인접 구간이 없습니다.")
        else:
            st.error(f"주의가 필요한 구간이 {len(conflicts)}건 확인되었습니다.")

            show_cols = [
                "작업1",
                "작업2",
                "방향",
                "구분",
                "문제구간",
                "이격거리(km)",
            ]
            st.dataframe(conflicts[show_cols], use_container_width=True)
    else:
        st.info("제출용 모드입니다. 겹침/인접 경고 표시는 도식에서 숨김 처리됩니다.")

    st.subheader("4. 공사구간 도식")

    fig = draw_diagram(
        units_df,
        conflicts,
        show_warnings=show_warnings,
        submit_mode=(output_mode == "제출용"),
    )

    st.pyplot(fig, use_container_width=True)

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)

    file_name = "bosung_work_diagram_review.png" if output_mode == "검토용" else "bosung_work_diagram_submit.png"

    st.download_button(
        label="PNG 이미지 다운로드",
        data=buffer,
        file_name=file_name,
        mime="image/png",
    )
