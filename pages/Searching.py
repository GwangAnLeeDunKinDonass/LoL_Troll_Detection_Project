import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import base64
import random
import string
from tools.for_display import *

# Streamlit wide mode 설정
st.set_page_config(page_title="Searching",page_icon="./img/screenshot/troll.webp", layout="wide")

# 세션 스테이트 초기화
if 'page_number' not in st.session_state:
    st.session_state.page_number = 1

if 'selected_user' not in st.session_state:
    st.session_state.selected_user = None

if 'metric1' not in st.session_state:
    st.session_state.metric1 = "KDA"

if 'metric2' not in st.session_state:
    st.session_state.metric2 = "Deaths per Minute"

current_page = "Searching"
if 'last_page' in st.session_state:
    if st.session_state.last_page != current_page:
        st.session_state.selected_user = None
        st.session_state.page_number = 1
else:
    st.session_state.selected_user = None
    st.session_state.page_number = 1
st.session_state.last_page = current_page

def update_page(new_page):
    st.session_state.page_number = new_page

def anonymize_nickname():
    length = random.randint(3, 8)
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def anonymize_puuid():
    return ''.join(random.choices(string.ascii_letters + string.digits, k=60))

@st.cache_resource
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

# 헬퍼 함수: 기본 계산 함수 (예: deaths_per_min 등)
def compute_metric(row, metric):
    if metric == "deaths_per_min":
        return row["deaths"] / row["timePlayed"] if row["timePlayed"] else 0
    elif metric == "cs_per_min":
        return row["cs"] / row["timePlayed"] if row["timePlayed"] else 0
    elif metric == "damage_taken_per_min":
        return row["totalDamageTaken"] / row["timePlayed"] if row["timePlayed"] else 0
    elif metric == "damage_dealt_per_min":
        return row["totalDamageDealtToChampions"] / row["timePlayed"] if row["timePlayed"] else 0
    elif metric == "vision_score_per_min":
        return row["visionScore"] / row["timePlayed"] if row["timePlayed"] else 0
    else:
        return 0

# 헬퍼 함수: 유저 데이터(row)에서 해당 지표 값을 반환. 없으면 계산하여 반환
def get_metric_value(row, metric):
    try:
        return row[metric]
    except KeyError:
        return compute_metric(row, metric)

# Custom CSS
st.markdown(
    """
    <style>
    .stApp {
        background-color: #262626;
        color: #D8A953;
    }
    .stTextInput, .stSelectbox, .stButton {
        background-color: #171717;
        border: 1px solid #D8A953;
        color: #D8A953;
    }
    .stButton button {
        background-color: #007BFF;
        color: white;
        border: none;
        padding: 10px 0;
        border-radius: 5px;
        cursor: pointer;
        width: 100%;
    }
    hr {
        border: 1px solid #D8A953;
        margin: 20px 0;
    }
    .custom-label {
        font-size: 16px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .api-key-label {
        font-size: 16px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .api-key-label small {
        font-size: 12px;
        font-weight: normal;
        margin-left: 5px;
    }
    .custom-title {
        color: #D8A953;
        font-size: 50px;
        font-weight: bold;
    }
    .stLinkButton {
        text-align: right;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 상단 입력 UI
st.markdown("<h1 class='custom-title'>매칭 검색</h1>", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns([4, 3, 1, 1])
with col1:
    st.markdown(
        "<div class='api-key-label'>API Key <small>(<a href='https://developer.riotgames.com/' target='_blank' style='color: #D8A953;'>Riot Developer에서 발급</a>)</small></div>",
        unsafe_allow_html=True
    )
    api_key = st.text_input("", type="password", key="api_key_input", label_visibility="collapsed")
with col2:
    st.markdown("<div class='custom-label'>Nickname</div>", unsafe_allow_html=True)
    nickname = st.text_input("", key="nickname_input", label_visibility="collapsed")
with col3:
    st.markdown("<div class='custom-label'>Tag</div>", unsafe_allow_html=True)
    tag = st.text_input("", key="tag_input", label_visibility="collapsed")
with col4:
    st.markdown("<div class='custom-label'>Match Count</div>", unsafe_allow_html=True)
    match_count = st.selectbox("", [1, 2, 3, 4, 5], key="match_count_selectbox", label_visibility="collapsed")

st.write("")
st.write("")

# 검색 버튼 클릭 시 결과를 session_state에 저장
if st.button("Searching"):
    if api_key and nickname and tag:
        match_data = for_display(api_key, nickname, tag, num=match_count)
        combined_df = pd.concat(match_data).reset_index(drop=True)
        if "anomal" in combined_df.columns:
            filtered_df = combined_df[combined_df["anomal"] == True].reset_index(drop=True)
        else:
            st.warning("의심유저가 없습니다.")
            filtered_df = combined_df

        st.session_state.combined_df = combined_df
        st.session_state.filtered_df = filtered_df
        st.session_state.page_number = 1
        st.session_state.selected_user = None

        # 익명화 적용
        st.session_state.combined_df['summonerName'] = st.session_state.combined_df['summonerName'].apply(lambda x: anonymize_nickname())
        st.session_state.combined_df['puuid'] = st.session_state.combined_df['puuid'].apply(lambda x: anonymize_puuid())

        st.session_state.filtered_df['summonerName'] = st.session_state.filtered_df['summonerName'].apply(lambda x: anonymize_nickname())
        st.session_state.filtered_df['puuid'] = st.session_state.filtered_df['puuid'].apply(lambda x: anonymize_puuid())

    else:
        st.warning("모든 필드를 입력해주세요.")

# 검색 결과가 session_state에 있다면 하단 레이아웃 렌더링
if "combined_df" in st.session_state and "filtered_df" in st.session_state:
    combined_df = st.session_state.combined_df
    filtered_df = st.session_state.filtered_df
    total_suspicious = filtered_df.shape[0]

    # 페이지네이션 (한 페이지당 4명)
    items_per_page = 4
    total_pages = (total_suspicious - 1) // items_per_page + 1
    page_number = st.session_state.page_number
    start_idx = (page_number - 1) * items_per_page
    end_idx = start_idx + items_per_page

    st.markdown("<hr>", unsafe_allow_html=True)
    left_col, right_col = st.columns([1, 2], gap="large")

    # 좌측: 의심유저 리스트
    with left_col:
        st.markdown(f"### <span style='color:#D8A953;'>의심유저 수: {total_suspicious}명</span>", unsafe_allow_html=True)
        selected_user = st.session_state.selected_user

        for idx, row in filtered_df.iloc[start_idx:end_idx].iterrows():
            colA, colB = st.columns([5, 1])
            with colA:
                st.markdown(f"**{row['summonerName']}** | 라인: {row['teamPosition']}")
            with colB:
                if st.button("View", key=f"view_{idx}"):
                    st.session_state.selected_user = row
                    st.session_state.metric1 = "KDA"
                    st.session_state.metric2 = "Deaths per Minute"

        st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)
        if 'refresh' not in st.session_state:
            st.session_state.refresh = False
        pagination_col1, pagination_col2, pagination_col3 = st.columns([1, 4, 1])
        with pagination_col1:
            if st.button("⬅️", key="prev_button", on_click=lambda: update_page(page_number - 1) if page_number > 1 else None):
                pass
        with pagination_col2:
            st.markdown(f"<div style='text-align: center;'>{page_number} / {total_pages}</div>", unsafe_allow_html=True)
        with pagination_col3:
            if st.button("➡️", key="next_button", on_click=lambda: update_page(page_number + 1) if page_number < total_pages else None):
                pass

    # 우측: 선택된 유저의 상세 매치 정보 레이아웃
    with right_col:
        if st.session_state.selected_user is not None and isinstance(st.session_state.selected_user, pd.Series):
            user_data = st.session_state.selected_user
            # AFK 처리: 원본 summonerName은 그대로 두고 display_name에만 처리
            display_name = user_data['summonerName']
            nickname_style = "color: #D8A953;"  # 기본 색상
            if user_data.get("afk", False):
                display_name = f"{display_name} (탈주)"
                nickname_style = "color: red;"
            st.markdown(f"<h1 style='{nickname_style}'>{display_name}</h1>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div style="display: flex; justify-content: space-between;">
                    <div style="font-size: 14px;">라인: {user_data.get('teamPosition', '')}</div>
                    <div style="font-size: 14px;">PUUID: {user_data.get('puuid', '')}</div>
                </div>
                """, unsafe_allow_html=True
            )
            st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)

            # 상세 정보 섹션: 챔피언, 스펠, 아이템
            champ_col, spell_col, item_col = st.columns([1, 2, 6])
            with champ_col:
                st.markdown(
                    f"""
                    <div style="border: 1px solid #D8A953; padding: 10px; border-radius: 5px; text-align: center;">
                        <strong>챔피언</strong>
                        <hr style='width: 100%; margin: 5px 0;'>
                        <img src="data:image/png;base64,{image_to_base64(f'./img/championName/{user_data["championName"]}.png')}" style="width:65px;height:65px;">
                        <div>{user_data['championName']}</div>
                    </div>
                    """, unsafe_allow_html=True
                )
            with spell_col:
                st.markdown(
                    f"""
                    <div style="border: 1px solid #D8A953; padding: 10px; border-radius: 5px; text-align: center;">
                        <strong>스펠</strong>
                        <hr style='width: 100%; margin: 5px 0;'>
                        <div style="display: flex; justify-content: space-evenly;">
                            <div>
                                <img src="data:image/png;base64,{image_to_base64(f'./img/summoner/{user_data["summoner1Id"]}.png')}" style="width:65px;height:65px;">
                                <div>{user_data['summoner1Id']}</div>
                            </div>
                            <div>
                                <img src="data:image/png;base64,{image_to_base64(f'./img/summoner/{user_data["summoner2Id"]}.png')}" style="width:65px;height:65px;">
                                <div>{user_data['summoner2Id']}</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True
                )
            with item_col:
                st.markdown(
                    f"""
                    <div style="border: 1px solid #D8A953; padding: 10px; border-radius: 5px; text-align: center;">
                        <strong>아이템</strong>
                        <hr style='width: 100%; margin: 5px 0;'>
                        <div style="display: flex; justify-content: space-evenly;">
                            <div>
                                <img src="data:image/png;base64,{image_to_base64(f'./img/item/{user_data["item0"]}.png')}" style="width:65px;height:65px;">
                                <div>1번</div>
                            </div>
                            <div>
                                <img src="data:image/png;base64,{image_to_base64(f'./img/item/{user_data["item1"]}.png')}" style="width:65px;height:65px;">
                                <div>2번</div>
                            </div>
                            <div>
                                <img src="data:image/png;base64,{image_to_base64(f'./img/item/{user_data["item2"]}.png')}" style="width:65px;height:65px;">
                                <div>3번</div>
                            </div>
                            <div>
                                <img src="data:image/png;base64,{image_to_base64(f'./img/item/{user_data["item3"]}.png')}" style="width:65px;height:65px;">
                                <div>4번</div>
                            </div>
                            <div>
                                <img src="data:image/png;base64,{image_to_base64(f'./img/item/{user_data["item4"]}.png')}" style="width:65px;height:65px;">
                                <div>5번</div>
                            </div>
                            <div>
                                <img src="data:image/png;base64,{image_to_base64(f'./img/item/{user_data["item5"]}.png')}" style="width:65px;height:65px;">
                                <div>6번</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True
                )

            # 지표 비교 섹션
            st.write("")
            st.write("")
            st.markdown("<h4>지표 비교</h4>", unsafe_allow_html=True)
            st.markdown("<hr style='margin: 3px 0;'>", unsafe_allow_html=True)
            
            # display_df는 st.session_state.combined_df를 사용 (평균 계산용)
            display_df = st.session_state.combined_df.copy()
            # 필요한 분당 지표 계산 (없으면 계산)
            if "deaths_per_min" not in display_df.columns:
                display_df["deaths_per_min"] = display_df["deaths"] / display_df["timePlayed"]
            if "cs_per_min" not in display_df.columns:
                display_df["cs_per_min"] = display_df["cs"] / display_df["timePlayed"]
            if "damage_taken_per_min" not in display_df.columns:
                display_df["damage_taken_per_min"] = display_df["totalDamageTaken"] / display_df["timePlayed"]
            if "damage_dealt_per_min" not in display_df.columns:
                display_df["damage_dealt_per_min"] = display_df["totalDamageDealtToChampions"] / display_df["timePlayed"]
            if "vision_score_per_min" not in display_df.columns:
                display_df["vision_score_per_min"] = display_df["visionScore"] / display_df["timePlayed"]

            left_plot_col, right_plot_col = st.columns(2)
            metrics = {
                "KDA": "kda",
                "Deaths per Minute": "deaths_per_min",
                "Kill Involvement Ratio": "kill_involve_ratio",
                "Tower Involvement Ratio": "tower_involve_ratio",
                "Object Involvement Ratio": "object_involve_ratio",
                "CS per Minute": "cs_per_min",
                "Damage Taken per Minute": "damage_taken_per_min",
                "Damage Dealt per Minute": "damage_dealt_per_min",
                "Vision Score per Minute": "vision_score_per_min"
            }
            average_metrics = {
                "kda": display_df[display_df["anomal"] == False]["kda"].mean(),
                "deaths_per_min": display_df[display_df["anomal"] == False]["deaths_per_min"].mean(),
                "kill_involve_ratio": display_df[display_df["anomal"] == False]["kill_involve_ratio"].mean(),
                "tower_involve_ratio": display_df[display_df["anomal"] == False]["tower_involve_ratio"].mean(),
                "object_involve_ratio": display_df[display_df["anomal"] == False]["object_involve_ratio"].mean(),
                "cs_per_min": display_df[display_df["anomal"] == False]["cs_per_min"].mean(),
                "damage_taken_per_min": display_df[display_df["anomal"] == False]["damage_taken_per_min"].mean(),
                "damage_dealt_per_min": display_df[display_df["anomal"] == False]["damage_dealt_per_min"].mean(),
                "vision_score_per_min": display_df[display_df["anomal"] == False]["vision_score_per_min"].mean(),
            }
            with left_plot_col:
                selected_metric1 = st.selectbox("첫 번째 지표 선택", list(metrics.keys()), key="metric1", label_visibility="collapsed")
                selected_metric_key1 = metrics[selected_metric1]
                normal_avg1 = average_metrics[selected_metric_key1]
                user_metric_value1 = get_metric_value(user_data, selected_metric_key1)
                fig, ax = plt.subplots(figsize=(6,4))
                ax.bar(["Normal AVG", "User"], [normal_avg1, user_metric_value1], color=["#C9C9C9", "#D8A953"])
                ax.set_title(f"{selected_metric1}", fontsize=14)
                fig.tight_layout()
                st.pyplot(fig)
            with right_plot_col:
                selected_metric2 = st.selectbox("두 번째 지표 선택", list(metrics.keys()), key="metric2", label_visibility="collapsed")
                selected_metric_key2 = metrics[selected_metric2]
                normal_avg2 = average_metrics[selected_metric_key2]
                user_metric_value2 = get_metric_value(user_data, selected_metric_key2)
                fig, ax = plt.subplots(figsize=(6,4))
                ax.bar(["Normal AVG", "User"], [normal_avg2, user_metric_value2], color=["#C9C9C9", "#D8A953"])
                ax.set_title(f"{selected_metric2}", fontsize=14)
                fig.tight_layout()
                st.pyplot(fig)
                
st.markdown("<hr>", unsafe_allow_html=True)
