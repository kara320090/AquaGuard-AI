import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title='AquaGuard AI',
    page_icon='💧',
    layout='wide'
)

st.title('AquaGuard AI')
st.subheader('충남 농업용수 위험도 산정 및 대체 수원 후보 추천 MVP')

st.markdown('''
### 신청서류 기준 데이터 구성

1. 농업용저수지 수위조회
2. 관정현황
3. 재배작물별 농가현황
4. 시·군별 강우량/가뭄 관련 데이터
5. 농축어업 통계
''')

root = Path(__file__).resolve().parents[1]
processed = root / 'data' / 'processed'

st.info('전처리 완료 후 시·군별 위험도와 대체 수원 후보를 이 화면에서 표시합니다.')
st.code(str(processed), language='text')
