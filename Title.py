import streamlit as st
import base64

st.set_page_config(page_title="Title",page_icon="./img/screenshot/troll.webp", layout="wide")

# Custom CSS: 전체 배경, 글자 색상, 사이드바 너비 및 기타 스타일 강제 적용
st.markdown(
    """
    <style>
    html, body, .stApp {
        background-color: #262626 !important;
        color: #D8A953 !important;
    }
    [data-testid="stSidebar"] {
        width: 100px !important;
    }
    .sidebar .sidebar-content {
        background-color: #262626 !important;
        color: #D8A953 !important;
        border: 1px solid #D8A953;
    }
    .info-box {
        border: 2px solid #D8A953;
        border-radius: 10px;
        padding-top: 10px;   /* 위쪽 여백 축소 */
        padding-bottom: 5px;  /* 아래쪽 여백 축소 */
        padding-left: 10px;
        padding-right: 10px;
        margin-bottom: 5px;   /* 하단 공백 최소화 */
        text-align: center;
    }
    img.screenshot {
        border: 2px solid #D8A953;
        border-radius: 10px;
        margin-top: 5px;    /* 이미지 위 여백 축소 */
        margin-bottom: 15px; 
        width: 100%;
        height: 280px;      /* 이미지 높이 축소 */
        object-fit: cover;
    }
    .custom-title {
        color: #D8A953;
        font-size: 50px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;  /* 타이틀 하단 여백 축소 */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 사이드바 내용 제거
st.sidebar.markdown("")

# 타이틀
st.markdown("<h1 class='custom-title'>League of Legends 어뷰징 의심 유저 탐지 대시보드</h1>", unsafe_allow_html=True)

# 이미지 파일을 Base64로 인코딩하는 함수
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

# 스크린샷 경로
list_img = "./img/screenshot/list.png"
search_img = "./img/screenshot/search.png"

# 이미지가 존재할 때만 표시
list_img_base64 = image_to_base64(list_img) if list_img else None
search_img_base64 = image_to_base64(search_img) if search_img else None

# 좌우 레이아웃 설정 (1:1 비율)
col1, col2 = st.columns(2)

# 좌측: Listing 설명 및 스크린샷
with col1:
    st.markdown(
        f"""
        <div class="info-box">
            <h2 style="color:#D8A953;">Listing</h2>
            <p style="color:#D8A953; font-size:18px;">
                DB에 적재된 전체 의심 유저의 해당 매치 기록을 확인합니다.
            </p>
            <img class="screenshot" src="data:image/png;base64,{list_img_base64}">
        </div>
        """,
        unsafe_allow_html=True
    )

# 우측: Searching 설명 및 스크린샷
with col2:
    st.markdown(
        f"""
        <div class="info-box">
            <h2 style="color:#D8A953;">Searching</h2>
            <p style="color:#D8A953; font-size:18px;">
                실시간으로 유저의 전적 데이터를 받아 검색 후 의심 유저를 체크합니다.
            </p>
            <img class="screenshot" src="data:image/png;base64,{search_img_base64}">
        </div>
        """,
        unsafe_allow_html=True
    )
