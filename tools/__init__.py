import warnings
from .cleansing import *
from .calculate import *
from .for_display import *

def _import_load_data():
    # 포맷이 개선된 경고 메시지
    warning_message = (
        "\n\n"
        "사용법 안내1: `concat_data` 함수 사용 시, 인자는 다음 순서로 전달해야 합니다:\n"
        "\n"
        "1. participant 테이블 \n"
        "2. position 테이블 \n"
        "3. match_info 테이블 \n"
        "4. objectives 테이블 \n"
        "\n"
        "정확한 인자 순서를 지켜 주세요. \n"
        "예시) participant, position, match_info, objectives = load_data()\n\n"
        "concat_data 시, Dask 데이터프레임으로 추출하려면 to_pandas=False 인자를 추가해주세요 \n"
        "데이터 저장 시, make_file=True 인자를 추가해주세요\n"
        "\n-----------------------------------------------------------------------\n"
        "사용법 안내2: `sql_normalization` 함수 사용 시, 인자는 다음 순서로 전달해야 합니다:\n"
        "\n"
        "1. match_tier 테이블 \n"
        "2. team_result 테이블 \n"
        "3. participant_preset 테이블 \n"
        "4. participant_minute 테이블 \n"
        "5. participant_result 테이블 \n"
        "6. champion 테이블 \n"
        "\n"
        "정확한 인자 순서를 지켜 주세요. \n"
        "데이터 저장 시, make_file=True 인자를 추가해주세요\n"
        "\n-----------------------------------------------------------------------\n"
        "사용법 안내3: `load_final_data` 함수 사용 시, 사용법은 다음과 같습니다:\n"
        "데이터 저장 시, make_file=True 인자를 추가해주세요\n"
        "모델링용 데이터 추출 시, for_model=True 인자를 추가해주세요\n"
    )
    warnings.warn(warning_message, UserWarning)

# _import_load_data 함수 호출
# _import_load_data()