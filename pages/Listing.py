import os
import numpy as np
import pandas as pd
import streamlit as st
import base64
import time
import matplotlib.pyplot as plt

# Streamlit 앱을 wide 모드로 설정
st.set_page_config(page_title="Listing",page_icon="./img/screenshot/troll.webp", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv('./data/final.csv')

# 세션 스테이트 초기화
if 'page_number' not in st.session_state:
    st.session_state.page_number = 1

if 'selected_user' not in st.session_state:
    st.session_state.selected_user = None

if 'metric1' not in st.session_state:
    st.session_state.metric1 = "KDA"

if 'metric2' not in st.session_state:
    st.session_state.metric2 = "Deaths per Minute"

current_page = "Listing"
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

# 이미지 파일을 Base64로 인코딩하는 함수 정의
@st.cache_resource
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

st.markdown(
    """
    <style>
    /* 전체 배경 색상 설정 */
    .stApp {
        background-color: #262626;
        color: #D8A953;
    }
    /* 기본 텍스트 및 기타 요소 색상 */
    body, div, h1, h2, h3, h4, h5, h6, p, label {
        color: #D8A953;
    }
    /* Multiselect 테두리 및 배경색 */
    .stMultiSelect > div {
        border: 1px solid #D8A953 !important;
    }
    .stMultiSelect > div > div {
        background-color: #171717 !important;
        color: #D8A953 !important;
    }
    /* Multiselect 드롭다운 아이템 배경 및 텍스트 색상 */
    .stMultiSelect > div > div > div[role="listbox"] {
        background-color: #171717 !important;
    }
    .stMultiSelect > div > div > div[role="listbox"] > div {
        color: #D8A953 !important;
    }
    /* 버튼 색상 및 중앙 배치 */
    .stButton button {
        background-color: #007BFF;
        color: white;
        border: none;
        padding: 5px 10px;
        border-radius: 5px;
        cursor: pointer;
        margin-top: auto;
        margin-bottom: auto;
    }
    /* 구분선 색상 */
    hr {
        border: 1px solid #D8A953;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 데이터 불러오기
# df_origin = pd.read_csv('./data/final.csv')
df_origin = load_data()

# 데이터 전처리
display_df = df_origin.copy()
display_df['win'] = display_df['win'].map({True: 'Win', False: 'Lose'})
display_df['tier'] = display_df['tier'].apply(lambda x: round(x))

# 분당 지표 계산
display_df['cs_per_min'] = display_df['cs'] / display_df['timePlayed']
display_df['deaths_per_min'] = display_df['deaths'] / display_df['timePlayed']
display_df['damage_taken_per_min'] = display_df['totalDamageTaken'] / display_df['timePlayed']
display_df['damage_dealt_per_min'] = display_df['totalDamageDealtToChampions'] / display_df['timePlayed']
display_df['vision_score_per_min'] = display_df['visionScore'] / display_df['timePlayed']

tier_mapping = {
    1: 'Iron',
    2: 'Bronze',
    3: 'Silver',
    4: 'Gold',
    5: 'Platinum',
    6: 'Diamond',
    7: 'Master',
    8: 'Grandmaster',
    9: 'Challenger',
    10: 'Challenger'
}
display_df['tier'] = display_df['tier'].map(tier_mapping)

with st.container():
    st.title('의심유저 리스트')

    # 레이아웃 설정
    left_col, right_col = st.columns([1, 2], gap="large")

    # 좌측 섹션
    with left_col:
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            selected_tiers = st.multiselect('티어 선택', list(tier_mapping.values()))
        with filter_col2:
            selected_positions = st.multiselect('팀 포지션 선택', display_df['teamPosition'].unique().tolist())

        # 필터 적용
        filtered_df = display_df[(display_df['anomal'] == True)]
        if selected_tiers:
            filtered_df = filtered_df[filtered_df['tier'].isin(selected_tiers)]
        if selected_positions:
            filtered_df = filtered_df[filtered_df['teamPosition'].isin(selected_positions)]

        # 페이지네이션 설정
        items_per_page = 8
        total_items = filtered_df.shape[0]
        total_pages = (total_items - 1) // items_per_page + 1

        page_number = st.session_state.page_number
        start_idx = (page_number - 1) * items_per_page
        end_idx = start_idx + items_per_page

        selected_user = st.session_state.selected_user

        # 유저 리스트 출력
        for idx, row in filtered_df.iloc[start_idx:end_idx].iterrows():
            col1, col2 = st.columns([5, 1])  # col1은 유저 정보, col2는 View 버튼

            with col1:
                win_emoji = "🔵" if row['win'] == "Win" else "🔴"
                st.markdown(
                    f"""
                    <div style="display: flex; align-items: center;">
                        <div style="font-size: 24px; font-weight: bold; margin-right: 10px;">
                            {win_emoji}
                        </div>
                        <div>
                            <div style="font-size: 20px; font-weight: bold;">{row['summonerName']}</div>
                            <div style="font-size: 14px; color: #666;">티어: {row['tier']} | 라인: {row['teamPosition']}</div>
                        </div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

            with col2:
                if st.button("View", key=f"view_{idx}"):
                    st.session_state.selected_user = row
                    st.session_state.metric1 = "KDA"
                    st.session_state.metric2 = "Deaths per Minute"

            st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)

        # 페이지 이동 버튼
        st.write("###")
        if 'refresh' not in st.session_state:
            st.session_state.refresh = False
        st.write("###")
        pagination_col1, pagination_col2, pagination_col3 = st.columns([1, 4, 1])
        with pagination_col1:
            if st.button("⬅️", key="prev_button", on_click=lambda: update_page(page_number - 1) if page_number > 1 else None):
                pass
        with pagination_col2:
            st.markdown(f"<div style='text-align: center;'>{page_number} / {total_pages}</div>", unsafe_allow_html=True)
        with pagination_col3:
            if st.button("➡️", key="next_button", on_click=lambda: update_page(page_number + 1) if page_number < total_pages else None):
                pass

    # 우측 섹션: 유저 상세 정보
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

            # 티어와 포지션, PUUID
            st.markdown(
                f"""
                <div style="display: flex; justify-content: space-between;">
                    <div style="font-size: 14px;">티어: {user_data['tier']} | 포지션: {user_data['teamPosition']}</div>
                    <div style="font-size: 14px;">PUUID: {user_data['puuid']}</div>
                </div>
                """, 
                unsafe_allow_html=True
            )

            st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)

            # 챔피언, 스펠, 아이템 섹션
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
                    """, 
                    unsafe_allow_html=True
                )

            with spell_col:
                st.markdown(
                    f"""
                    <div style="border: 1px solid #D8A953; padding: 10px; border-radius: 5px; text-align: center;">
                        <strong>스펠</strong>
                        <hr style='width: 100%; margin: 5px 0;'>
                        <div style="display: flex; justify-content: space-evenly; align-items: space-between;">
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
                    """, 
                    unsafe_allow_html=True
                )

            with item_col:
                st.markdown(
                    f"""
                    <div style="border: 1px solid #D8A953; padding: 10px; border-radius: 5px; text-align: center;">
                        <strong>아이템</strong>
                        <hr style='width: 100%; margin: 5px 0;'>
                        <div style='display: flex; justify-content: space-evenly; align-items: space-between;'>
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
                    """, 
                    unsafe_allow_html=True
                )

            # 지표 비교
            st.write('')
            st.write('')
            st.markdown("<h4>지표 비교</h4>", unsafe_allow_html=True)
            st.markdown("<hr style='margin: 3px 0;'>", unsafe_allow_html=True)

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

            # 정상 유저 평균값 계산
            average_metrics = {
                "kda": display_df[display_df['anomal'] == False]['kda'].mean(),
                "deaths_per_min": display_df[display_df['anomal'] == False]['deaths_per_min'].mean(),
                "kill_involve_ratio": display_df[display_df['anomal'] == False]['kill_involve_ratio'].mean(),
                "tower_involve_ratio": display_df[display_df['anomal'] == False]['tower_involve_ratio'].mean(),
                "object_involve_ratio": display_df[display_df['anomal'] == False]['object_involve_ratio'].mean(),
                "cs_per_min": display_df[display_df['anomal'] == False]['cs_per_min'].mean(),
                "damage_taken_per_min": display_df[display_df['anomal'] == False]['damage_taken_per_min'].mean(),
                "damage_dealt_per_min": display_df[display_df['anomal'] == False]['damage_dealt_per_min'].mean(),
                "vision_score_per_min": display_df[display_df['anomal'] == False]['vision_score_per_min'].mean(),
            }

            # 좌측 그래프
            with left_plot_col:
                selected_metric1 = st.selectbox("첫 번째 지표 선택", list(metrics.keys()), key="metric1", label_visibility="collapsed")
                selected_metric_key1 = metrics[selected_metric1]

                normal_avg1 = average_metrics[selected_metric_key1]
                user_metric_value1 = user_data[selected_metric_key1]

                fig, ax = plt.subplots(figsize=(6,4))
                ax.bar(["Normal AVG", "User"], [normal_avg1, user_metric_value1], color=["#C9C9C9", "#D8A953"])
                ax.set_title(f"{selected_metric1}", fontsize=14)
                fig.tight_layout()
                st.pyplot(fig)

            # 우측 그래프
            with right_plot_col:
                selected_metric2 = st.selectbox("두 번째 지표 선택", list(metrics.keys()), key="metric2", label_visibility="collapsed")
                selected_metric_key2 = metrics[selected_metric2]

                normal_avg2 = average_metrics[selected_metric_key2]
                user_metric_value2 = user_data[selected_metric_key2]

                fig, ax = plt.subplots(figsize=(6,4))
                ax.bar(["Normal AVG", "User"], [normal_avg2, user_metric_value2], color=["#C9C9C9", "#D8A953"])
                ax.set_title(f"{selected_metric2}", fontsize=14)
                fig.tight_layout()
                st.pyplot(fig)
