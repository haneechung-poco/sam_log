import streamlit as st
import pandas as pd
from openai import OpenAI
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from collections import Counter
from wordcloud import WordCloud
from io import BytesIO
from PIL import Image
import openai
import datetime
import re

# --- 기본 설정 (수정 없음) ---
FONT_PATH = "NanumGothic-Regular.ttf"
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
st.set_page_config(page_title="SAM 분석 보고서", layout="wide")  # 넓은 레이아웃으로 변경

# --- 사이드바 설정 ---
st.sidebar.header("⚙️ 분석 설정")

# 1. 기본 파일 업로드
uploaded_file = st.sidebar.file_uploader("1. 질문/답변 데이터 업로드",
                                         type=["csv", "xlsx"])

# df_learning을 세션 상태에 초기화
if 'df_learning' not in st.session_state:
    st.session_state.df_learning = None

# 2. 분석 모드 선택
analysis_mode = st.sidebar.radio("2. 분석 모드 선택",
                                 ('수강 이력 없이 질문 내역만으로 조회', '수강 이력 업로드 후 함께 분석'))

# 3. 조건부로 수강 이력 파일 업로드
if analysis_mode == '수강 이력 업로드 후 함께 분석':
    learning_file_main = st.sidebar.file_uploader("3. 수강 이력 데이터 업로드",
                                                  type=["csv", "xlsx"],
                                                  key="main_learning_uploader")
    if learning_file_main:
        try:
            if learning_file_main.name.endswith(".csv"):
                st.session_state.df_learning = pd.read_csv(learning_file_main)
            else:
                st.session_state.df_learning = pd.read_excel(
                    learning_file_main)
            st.sidebar.success("✅ 수강 이력 파일 로드 완료")
        except Exception as e:
            st.sidebar.error(f"파일 처리 오류: {e}")
            st.session_state.df_learning = None
else:
    st.session_state.df_learning = None

st.sidebar.markdown("---")
st.sidebar.info("모든 설정을 완료한 후, 우측 화면에서 분석 결과를 확인하세요.")

# --- 메인 화면 구성 ---
st.title("📄 SAM 분석 보고서")

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # --- 이 아래부터는 기존 탭 코드와 동일 ---
        # (단, tab5, tab6 내부의 파일 업로더는 제거된 최종 버전 기준)

        # 조사기간 처리
        if 'regymdt' in df.columns:
            df['regymdt'] = pd.to_datetime(df['regymdt'], errors='coerce')
            start_date = df['regymdt'].min().strftime("%Y-%m-%d")
            end_date = df['regymdt'].max().strftime("%Y-%m-%d")
        else:
            start_date = end_date = "날짜 정보 없음"

        # 총 참여자
        if 'user_id' in df.columns:
            total_users = df['user_id'].nunique()
        else:
            total_users = 0

        # 총 질문 수
        if 'question' in df.columns:
            total_questions = df['question'].notnull().sum()
        else:
            total_questions = 0

        # 탭 구성
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
            ["📌 분석 개요", "🏢 조직별 현황", "❓ 질문 현황", "🧠 답변 분석", "👤이용자 분석", "📊 실험실"])

        # (Tab1, Tab2, Tab3, Tab4 코드는 기존과 동일하므로 생략)
        with tab1:
            st.subheader("📌 분석 개요")
            st.markdown(f"- **조사기간**: {start_date} ~ {end_date}")
            st.markdown(f"- **총 참여자**: {total_users}명")
            st.markdown(f"- **총 질문 수**: {total_questions}건")

        with tab2:
            st.subheader("🏢 조직별 현황")

            if all(col in df.columns
                   for col in ['group_1', 'group_2', 'group_3', 'user_id']):

                # 1. 안내 메시지 및 한 줄 레이아웃
                st.info("조직을 순서대로 선택하여 필터링하고, 아래 기준을 선택하여 데이터를 집계합니다.")
                col1, col2, col3 = st.columns(3)

                # --- 계층적 조직 필터 ---
                with col1:
                    group1_options = ['전체'] + sorted(
                        df['group_1'].dropna().unique().tolist())
                    selected_group1 = st.selectbox("1️⃣ 1차 조직 (센터)",
                                                   options=group1_options)

                df_filtered = df.copy()
                if selected_group1 != '전체':
                    df_filtered = df_filtered[df_filtered['group_1'] ==
                                              selected_group1]

                with col2:
                    group2_options = ['전체'] + sorted(
                        df_filtered['group_2'].dropna().unique().tolist())
                    selected_group2 = st.selectbox("2️⃣ 2차 조직 (실)",
                                                   options=group2_options)

                if selected_group2 != '전체':
                    df_filtered = df_filtered[df_filtered['group_2'] ==
                                              selected_group2]

                with col3:
                    group3_options = ['전체'] + sorted(
                        df_filtered['group_3'].dropna().unique().tolist())
                    selected_group3 = st.selectbox("3️⃣ 3차 조직 (팀)",
                                                   options=group3_options)

                if selected_group3 != '전체':
                    df_filtered = df_filtered[df_filtered['group_3'] ==
                                              selected_group3]

                st.markdown("---")

                # 2. 동적 라디오 버튼 로직
                group_labels = {
                    '센터 기준': 'group_1',
                    '실 기준': 'group_2',
                    '팀 기준': 'group_3'
                }

                # 선택된 조직 레벨에 따라 라디오 버튼 옵션과 기본값을 동적으로 결정
                if selected_group3 != '전체':
                    # 3차 조직 선택 시: '팀 기준'만 가능
                    radio_options = ['팀 기준']
                    radio_index = 0
                elif selected_group2 != '전체':
                    # 2차 조직 선택 시: '실 기준', '팀 기준' 가능
                    radio_options = ['실 기준', '팀 기준']
                    radio_index = 0
                else:
                    # 전체 또는 1차 조직 선택 시: 모든 기준 가능
                    radio_options = ['센터 기준', '실 기준', '팀 기준']
                    radio_index = 0

                selected_label = st.radio("📊 어떤 기준으로 볼까요?",
                                          options=radio_options,
                                          index=radio_index,
                                          horizontal=True)

                selected_group_level = group_labels[selected_label]

                # 집계 및 시각화 (기존 로직 유지)
                if selected_group_level in df_filtered.columns:
                    # 필터링된 데이터프레임(df_filtered)에서 시각화 기준(selected_group_level)에 해당하는 조직만 집계
                    view_df = df_filtered.dropna(subset=[selected_group_level])

                    question_counts = view_df[
                        selected_group_level].value_counts().reset_index()
                    question_counts.columns = ['조직명', '질문 수']

                    user_counts = view_df.groupby(selected_group_level)[
                        'user_id'].nunique().reset_index()
                    user_counts.columns = ['조직명', '사용자 수']

                    org_stats = pd.merge(question_counts,
                                         user_counts,
                                         on='조직명',
                                         how='left')
                    org_stats = org_stats.sort_values(
                        by='질문 수', ascending=False).reset_index(drop=True)
                    org_stats.index = org_stats.index + 1

                    st.markdown("### 📊 질문 수")
                    st.bar_chart(org_stats.set_index('조직명')[['질문 수']])

                    st.markdown("#### 📄 조직별 질문 수 및 사용자 수")
                    st.dataframe(org_stats)
                else:
                    st.warning("선택한 그룹 수준이 유효하지 않습니다.")
            else:
                st.warning(
                    "⚠️ 필요한 컬럼이 없습니다 (group_1, group_2, group_3, user_id).")

        with tab3:
            st.subheader("❓ 질문 현황")

            # 1. 월별 질문 수 추이 (신규 추가)
            st.markdown("#### 📈 월별 질문 수 추이")
            if 'regymdt' in df.columns:
                # 월별 집계를 위해 데이터프레임 복사 및 날짜 형식 변환
                df_trend = df.copy()
                df_trend['regymdt'] = pd.to_datetime(df_trend['regymdt'],
                                                     errors='coerce')

                # 날짜 정보가 없는 행 제거
                df_trend.dropna(subset=['regymdt'], inplace=True)

                if not df_trend.empty:
                    # 월(Month)을 기준으로 데이터 리샘플링 및 질문 수 계산
                    monthly_counts = df_trend.resample(
                        'M', on='regymdt').size().reset_index(name='질문 수')

                    # 차트의 x축 레이블을 'YYYY-MM' 형식으로 변경
                    monthly_counts['월'] = monthly_counts[
                        'regymdt'].dt.strftime('%Y-%m')

                    # '월'을 인덱스로 설정하여 차트 데이터 준비
                    chart_data_monthly = monthly_counts.set_index('월')

                    # 라인 차트 표시
                    st.line_chart(chart_data_monthly[['질문 수']])
                else:
                    st.info("표시할 날짜 데이터가 없습니다.")
            else:
                st.info("⚠️ 월별 추이 분석을 위해서는 'regymdt' 날짜 컬럼이 필요합니다.")

            st.markdown("---")

            # 2. 질문 현황 Top 10 (차트)
            #st.markdown("#### 🏆 질문 현황 Top 10 (차트)")
            if 'chat_title' in df.columns:
                # Top 10 데이터 생성 및 바 차트 시각화
                #top_10_chart = df['chat_title'].value_counts().head(10)
                #st.bar_chart(top_10_chart)

                # 3. 질문 현황 Top 20 (표)
                st.markdown("#### 📄 질문 유형별 상세 데이터(Top 20)")
                top_20_table = df['chat_title'].value_counts().head(
                    20).reset_index()
                top_20_table.columns = ['질문 주제', '건수']
                top_20_table.index += 1
                #top_20_table.insert(0, '순위', top_20_table.index)
                st.dataframe(top_20_table)
            else:
                st.warning("⚠️ 'chat_title' 컬럼이 없습니다.")

        with tab4:
            st.subheader("🧠 답변 분석")
            # 응답율 통계 (이전과 동일)
            if 'answer_yn' in df.columns:
                answer_counts = df['answer_yn'].value_counts()
                answered = answer_counts.get('Y', 0)
                unanswered = answer_counts.get('N', 0)
                total = answered + unanswered
                answered_pct = round(answered / total *
                                     100, 1) if total > 0 else 0
                unanswered_pct = round(unanswered / total *
                                       100, 1) if total > 0 else 0

                st.markdown(f"총 질문 수: **{len(df)}**")
                st.markdown(f"✅ 응답: {answered}건 ({answered_pct}%)")
                st.markdown(f"❌ 미응답: {unanswered}건 ({unanswered_pct}%)")
                st.markdown("---")

            # GPT 분석 로직 (버튼 통합)
            if 'answer_yn' in df.columns and 'question' in df.columns:
                answered_df = df[df['answer_yn'] == 'Y']['question'].dropna()
                unanswered_df = df[df['answer_yn'] == 'N']['question'].dropna()

                # GPT 분석 함수 (이전과 동일)
                def run_gpt_analysis(data_list, is_answered=True):
                    # 샘플 수를 30개로 고정하여 API 비용 및 시간 관리
                    samples = data_list.sample(min(30, len(data_list)),
                                               random_state=42).tolist()
                    messages = [{
                        "role":
                        "system",
                        "content":
                        ("아래는 교육 시스템에서 " +
                         ("응답된 질문 목록입니다." if is_answered else "미응답 질문 목록입니다.")
                         + " 이 질문들을 유형별로 분류하고, " +
                         ("응답된 질문의 핵심 특징을" if is_answered else "미응답된 핵심 사유를") +
                         " 요약 분석해주세요. 반드시 명확한 카테고리로 나누어 설명해야 합니다.")
                    }, {
                        "role": "user",
                        "content": "\n".join(samples)
                    }]
                    try:
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=messages,
                            temperature=0.3,
                        )
                        return response.choices[0].message.content
                    except Exception as e:
                        return f"❌ GPT 분석 오류: {e}"

                st.subheader("🤖 응답/미응답 분석하기")

                # 분석할 데이터가 하나라도 있을 경우에만 버튼 표시
                if not answered_df.empty or not unanswered_df.empty:
                    if st.button("ChatGPT로 응답/미응답 내역 동시 분석하기"):
                        with st.spinner(
                                "GPT가 전체 질문 유형을 분석 중입니다... 잠시만 기다려주세요."):

                            # 1. 응답된 질문 분석
                            if not answered_df.empty:
                                st.markdown("### ✅ 응답된 질문 유형 분석 결과")
                                result_answered = run_gpt_analysis(
                                    answered_df, is_answered=True)
                                st.markdown(result_answered)
                            else:
                                st.info("분석할 응답된 질문이 없습니다.")

                            st.markdown("---")  # 분석 결과 구분선

                            # 2. 미응답 질문 분석
                            if not unanswered_df.empty:
                                st.markdown("### ❌ 미응답 질문 유형 분석 결과")
                                result_unanswered = run_gpt_analysis(
                                    unanswered_df, is_answered=False)
                                st.markdown(result_unanswered)
                            else:
                                st.info("분석할 미응답 질문이 없습니다.")

                        st.success("✅ 모든 분석이 완료되었습니다.")
                else:
                    st.info("분석할 질문 데이터가 없습니다.")

            else:
                st.warning("⚠️ 'answer_yn' 또는 'question' 컬럼이 존재하지 않습니다.")

        # --- 수정된 tab5 코드 ---
        with tab5:
            st.subheader("👤 이용자 분석")
            st.markdown("---")

            if st.session_state.df_learning is None:
                st.info(
                    "💡 수강 이력을 포함한 종합 분석을 원하시면, 왼쪽 사이드바에서 '수강 이력 업로드 후 함께 분석'을 선택해주세요."
                )

            if 'user_id' in df.columns:
                # --- ★★★ 핵심 수정 부분 ★★★ ---
                # 1. 플레이스홀더(안내 문구) 정의
                placeholder = "분석할 이용자를 선택하세요."

                # 2. 기존 사용자 목록 생성
                if 'user_name' in df.columns:
                    user_df = df[['user_id', 'user_name'
                                  ]].copy().dropna().drop_duplicates()
                    user_df['user_id'] = user_df['user_id'].astype(str)
                    user_df['display'] = user_df['user_id'] + " / " + user_df[
                        'user_name']
                    options_list = sorted(user_df['display'].unique())
                else:
                    options_list = sorted(
                        df['user_id'].dropna().astype(str).unique())

                # 3. 플레이스홀더를 목록 맨 앞에 추가하여 selectbox 생성
                selected_display = st.selectbox(
                    "👤 이용자 선택",  # 레이블을 더 간결하게 수정
                    options=[placeholder] + options_list)

                # 4. 플레이스홀더가 아닌, 실제 사용자가 선택되었을 때만 아래 분석 로직 실행
                if selected_display != placeholder:
                    selected_user_id = selected_display.split(
                        ' / '
                    )[0] if ' / ' in selected_display else selected_display
                    user_qa = df[df['user_id'].astype(str) == selected_user_id]

                    # --- 2. 질문/응답 요약 ---
                    st.markdown("---")
                    st.markdown("### 📄 질문/응답 요약")
                    if not user_qa.empty:
                        total_q = len(user_qa)
                        answered_q = (user_qa['answer_yn'] == 'Y').sum(
                        ) if 'answer_yn' in user_qa.columns else 0
                        unanswered_q = (user_qa['answer_yn'] == 'N').sum(
                        ) if 'answer_yn' in user_qa.columns else 0
                        st.markdown(f"- 총 질문 수: **{total_q}** 건")
                        st.markdown(f"- 응답된 질문: **{answered_q}** 건")
                        st.markdown(f"- 미응답 질문: **{unanswered_q}** 건")
                        if 'regymdt' in user_qa.columns:
                            st.markdown(
                                f"- 마지막 질문일: **{user_qa['regymdt'].max().strftime('%Y-%m-%d')}**"
                            )
                    else:
                        st.info("해당 사용자의 질문/응답 데이터가 없습니다.")

                    # --- 3. 학습 이력 분석 ---
                    st.markdown("---")
                    st.markdown("### 📚 학습 이력 분석")

                    user_learning = pd.DataFrame()
                    if st.session_state.df_learning is not None:
                        df_learning = st.session_state.df_learning
                        if 'user_id' in df_learning.columns:
                            user_learning = df_learning[
                                df_learning['user_id'].astype(
                                    str) == selected_user_id]
                            if not user_learning.empty:
                                with st.expander(
                                        f"📖 학습 이력 상세보기 ({len(user_learning)}건)"
                                ):
                                    st.dataframe(user_learning)
                            else:
                                st.warning(
                                    f"⚠️ 업로드된 학습 이력 파일에서 {selected_user_id} 님의 데이터를 찾을 수 없습니다."
                                )
                        else:
                            st.error("⚠️ 업로드된 학습 이력 파일에 'user_id' 컬럼이 없습니다.")
                    else:
                        st.info(
                            "표시할 학습 이력 데이터가 없습니다. 종합 분석을 원하시면 사이드바에서 이력 파일을 업로드해주세요."
                        )

                    # --- 4. 학습 성향 종합 분석 (GPT) ---
                    st.markdown("---")
                    st.markdown("### 🧠 학습 성향 종합 분석 (by GPT)")
                    if st.button("🤖 ChatGPT로 분석 실행하기",
                                 key=f"gpt_user_{selected_user_id}"
                                 ):  # 사용자별로 버튼 키를 다르게 하여 상태 유지
                        if user_qa.empty:
                            st.warning("⚠️ 분석할 질문 데이터가 없습니다.")
                        else:
                            with st.spinner("GPT가 사용자의 활동 기록을 분석 중입니다..."):
                                # (이하 GPT 분석 로직은 이전과 동일)
                                questions_list = user_qa['question'].dropna(
                                ).head(10).tolist()
                                base_data_info = "이 분석은 사용자의 [질문/응답 기록]을 기반으로 합니다."
                                learning_titles_text = ""
                                if not user_learning.empty and 'title' in user_learning.columns:
                                    learning_titles = user_learning[
                                        'title'].dropna().head(15).tolist()
                                    if learning_titles:
                                        learning_titles_text, base_data_info = f'\n### 2. 주요 학습 이력 (최대 15개):\n- {"- ".join(learning_titles)}', "이 분석은 사용자의 [질문/응답 기록]과 [학습 이력]을 종합하여 제공됩니다."
                                prompt = f"""다음은 한 직원의 시스템 내 활동 기록입니다.\n\n### 1. 주요 질문 내역 (최대 10개):\n- {"- ".join(questions_list) if questions_list else "질문 기록 없음"}\n{learning_titles_text}\n\n### [분석 요청]\n위의 기록을 바탕으로, 이 직원의 **학습 성향과 주요 관심사**를 분석해주세요.\n분석 결과는 반드시 아래 4가지 항목으로 명확하게 나누고, 각 항목의 제목을 반드시 붙여서 설명해주세요.\n\n1.  **주요 관심 분야**: 어떤 주제에 대해 궁금해하고 학습하는 경향이 있는가? (구체적인 키워드나 영역 제시)\n2.  **학습 태도**: 질문과 학습 기록을 볼 때, 자기주도적으로 문제를 해결하려 하는가, 아니면 주어진 지식을 수동적으로 습득하는가? 적극성, 탐구심 등을 평가.\n3.  **지식 격차(Knowledge Gap) 추정**: (학습 이력이 있다면) 질문 내용과 학습 내용을 비교하여 추가 학습이 필요한 부분을 추정. (학습 이력이 없다면) 질문 내용만으로 파악되는 지식 탐구 영역이나 부족한 점을 기술.\n4.  **종합 요약 및 추천**: 위 1~3번 내용을 바탕으로 이 직원의 학습 성향을 1~2 문장으로 요약하고, 경력 개발에 도움이 될 만한 학습 활동이나 과정을 추천."""
                                try:
                                    response = client.chat.completions.create(
                                        model="gpt-4-turbo-preview",
                                        messages=[{
                                            "role":
                                            "system",
                                            "content":
                                            "당신은 임직원의 활동 데이터를 기반으로 개인의 학습 성향과 역량 수준을 분석하는 전문 HRD 컨설턴트입니다. 반드시 제시된 형식에 맞춰 각 항목을 명확하게 구분하여 답변해야 합니다."
                                        }, {
                                            "role": "user",
                                            "content": prompt
                                        }],
                                        temperature=0.5)
                                    summary = response.choices[
                                        0].message.content
                                    st.success("✅ GPT 분석 완료!")
                                    st.info(base_data_info)
                                    st.markdown(summary)
                                except Exception as e:
                                    st.error(f"❌ GPT 분석 중 오류 발생: {e}")
            else:
                st.warning("⚠️ 이용자 분석을 진행하려면 원본 데이터에 'user_id' 컬럼이 있어야 합니다.")

        # --- 수정된 tab6 코드 ---
        with tab6:
            st.header("🧪 실험실: 조직 및 키워드 기반 심층 분석")
            tab_org_search, tab_keyword_search = st.tabs(
                ["🏢 조직 검색", "🔍 단어 검색"])
            with tab_org_search:
                st.subheader("조직별 관심사 및 학습 방향 분석")
                if all(col in df.columns
                       for col in ['group_1', 'group_2', 'group_3']):
                    org_full_list = df.apply(
                        lambda row:
                        f"{row['group_1']}/{row['group_2']}/{row['group_3']}",
                        axis=1).dropna().unique().tolist()
                    options_list = ['전체'] + sorted(org_full_list)
                    selected_org_full = st.selectbox(
                        "분석할 조직을 선택하세요 (예: A센터/경영지원실/인사팀)",
                        options=options_list)
                    df_filtered = df.copy()
                    if selected_org_full != '전체':
                        g1, g2, g3 = selected_org_full.split('/')
                        df_filtered = df[(df['group_1'] == g1)
                                         & (df['group_2'] == g2) &
                                         (df['group_3'] == g3)]
                    st.markdown("---")
                    if not df_filtered.empty:
                        st.subheader("☁️ 주요 키워드 워드클라우드")
                        text_data = ' '.join(
                            df_filtered['question'].fillna('').tolist())
                        if st.session_state.df_learning is not None:
                            org_user_ids = df_filtered['user_id'].unique()
                            org_learning_df = st.session_state.df_learning[
                                st.session_state.df_learning['user_id'].isin(
                                    org_user_ids)]
                            if not org_learning_df.empty and 'title' in org_learning_df.columns:
                                text_data += ' ' + ' '.join(
                                    org_learning_df['title'].fillna(
                                        '').tolist())
                                st.info("질문 내용과 수강한 강좌명을 바탕으로 생성되었습니다.")
                        if not text_data.strip():
                            st.warning("데이터가 부족하여 워드클라우드를 생성할 수 없습니다.")
                        else:
                            wordcloud = WordCloud(
                                font_path=FONT_PATH,
                                width=800,
                                height=400,
                                background_color='white').generate(text_data)
                            fig, ax = plt.subplots()
                            ax.imshow(wordcloud, interpolation='bilinear')
                            ax.axis("off")
                            st.pyplot(fig)
                        st.markdown("---")
                        if st.button("🤖 GPT로 조직 분석 리포트 생성"):
                            with st.spinner(
                                    "GPT가 조직 데이터를 분석하고 HRD 관점의 인사이트를 도출 중입니다..."
                            ):
                                top_keywords = Counter([
                                    w for w in text_data.split() if len(w) > 1
                                ]).most_common(30)
                                keyword_text = ", ".join(
                                    [w for w, _ in top_keywords])
                                top_questions_text = "\n- ".join(
                                    df_filtered['chat_title'].value_counts(
                                    ).head(5).index.tolist()
                                ) if 'chat_title' in df_filtered else "질문 주제 데이터 없음"
                                learning_summary_text = " (학습 이력 데이터 없음)"
                                if st.session_state.df_learning is not None and not org_learning_df.empty:
                                    top_courses = org_learning_df[
                                        'title'].value_counts().head(
                                            5).index.tolist()
                                    learning_summary_text = f"### 3. 주요 학습 과정 Top 5:\n- " + "\n- ".join(
                                        top_courses)
                                prompt = f"""당신은 데이터 기반의 HRD 전략 컨설턴트입니다. 다음 데이터를 바탕으로 조직의 특성을 심층 분석하고 보고서를 작성해주세요.\n\n### 분석 대상 조직: {selected_org_full}\n\n### 1. 주요 질문/학습 키워드: {keyword_text}\n### 2. 주요 질문 주제: {top_questions_text}\n{learning_summary_text}\n---\n### [분석 요청]\n위 데이터를 HRD 관점에서 종합 분석하여, 반드시 아래 4가지 항목의 제목을 포함하여 보고서를 작성해주세요.\n\n1. **조직의 주요 관심사 및 현황**: 구성원들이 현재 가장 관심을 갖는 업무 분야나 주제는 무엇입니까?\n2. **업무/역량 관련 주요 이슈**: 자주 묻는 질문들을 통해 파악할 수 있는 이 조직의 업무상 어려움(pain point)이나 역량적 공백은 무엇입니까?\n3. **지식 격차 및 필요 역량**: 구성원들이 보유한 지식(학습 이력)과 궁금해하는 지식(질문) 사이의 차이는 무엇이며, 어떤 역량을 추가 개발해야 합니까?\n4. **HRD 관점의 종합 제언**: 이 조직의 성과 향상과 역량 개발을 위해 어떤 교육 프로그램 설계나 학습 문화 조성이 효과적일지 구체적인 액션 아이템 1~2가지를 제안해주세요."""
                                try:
                                    response = client.chat.completions.create(
                                        model="gpt-4-turbo-preview",
                                        messages=[{
                                            "role": "user",
                                            "content": prompt
                                        }],
                                        temperature=0.4)
                                    summary = response.choices[
                                        0].message.content
                                    st.subheader("🧠 GPT 조직 분석 리포트")
                                    st.markdown(summary)
                                except Exception as e:
                                    st.error(f"❌ GPT 분석 중 오류 발생: {e}")
                else:
                    st.warning(
                        "⚠️ 조직 분석을 위해서는 원본 데이터에 'group_1', 'group_2', 'group_3' 컬럼이 모두 필요합니다."
                    )
            with tab_keyword_search:
                st.subheader("키워드 관련 조직 및 학습 분석")
                keyword = st.text_input("검색할 단어를 입력하세요",
                                        key="lab_keyword_input")
                st.markdown("---")
                if keyword:
                    contains_mask = df['question'].str.contains(
                        keyword, na=False) | df['answer'].str.contains(
                            keyword, na=False)
                    df_filtered_keyword = df[contains_mask]
                    if not df_filtered_keyword.empty:
                        st.success(
                            f"'{keyword}' 키워드가 포함된 **{len(df_filtered_keyword)}**건의 대화를 찾았습니다."
                        )
                        if 'group_1' in df_filtered_keyword.columns:
                            st.subheader(
                                f"🏅 '{keyword}' 키워드 언급 조직 Top 10 (센터 기준)")
                            top_orgs = df_filtered_keyword[
                                'group_1'].value_counts().head(10)
                            st.dataframe(top_orgs)
                        st.markdown("---")
                        with st.expander("📂 관련 질문 예시 보기"):
                            st.dataframe(df_filtered_keyword[[
                                'question', 'answer', 'group_1', 'group_2'
                            ]].head(10))
                        st.markdown("---")
                        st.subheader("📚 키워드 언급 구성원의 수강 현황")
                        if st.session_state.df_learning is not None:
                            df_learning = st.session_state.df_learning

                            if 'title' in df_learning.columns:
                                keyword_user_ids = df_filtered_keyword[
                                    'user_id'].unique()
                                related_learning = df_learning[df_learning[
                                    'user_id'].isin(keyword_user_ids)]

                                if not related_learning.empty:
                                    # --- ★★★ 요청하신 요약 지표 계산 및 표시 부분 ★★★ ---
                                    total_courses = related_learning[
                                        'title'].nunique()
                                    total_enrollments = len(related_learning)
                                    total_users = related_learning[
                                        'user_id'].nunique()

                                    # st.columns를 사용하여 지표를 가로로 나열
                                    col1, col2, col3 = st.columns(3)
                                    col1.metric("총 강좌 수", f"{total_courses} 개")
                                    col2.metric("총 수강 횟수",
                                                f"{total_enrollments} 회")
                                    col3.metric("총 수강 인원", f"{total_users} 명")

                                    st.markdown("---")  # 요약 지표와 테이블 사이 구분선

                                    # 표 데이터 가공
                                    course_counts = related_learning[
                                        'title'].value_counts().reset_index()
                                    course_counts.columns = ['강좌명', '총 수강 횟수']

                                    if 'group_1' in df.columns:
                                        user_to_org_map = df[[
                                            'user_id', 'group_1'
                                        ]].drop_duplicates().set_index(
                                            'user_id')['group_1']
                                        related_learning_with_org = related_learning.copy(
                                        )
                                        related_learning_with_org[
                                            'group_1'] = related_learning_with_org[
                                                'user_id'].map(user_to_org_map)
                                        related_learning_with_org.dropna(
                                            subset=['title', 'group_1'],
                                            inplace=True)

                                        org_counts_by_course = related_learning_with_org.groupby(
                                            ['title',
                                             'group_1']).size().reset_index(
                                                 name='org_count')
                                        top_org_by_course = org_counts_by_course.sort_values(
                                            'org_count',
                                            ascending=False).drop_duplicates(
                                                'title')
                                        top_org_by_course[
                                            '최다 수강 조직(횟수)'] = top_org_by_course.apply(
                                                lambda row:
                                                f"{row['group_1']} ({row['org_count']}회)",
                                                axis=1)

                                        final_table = pd.merge(
                                            course_counts,
                                            top_org_by_course[[
                                                'title', '최다 수강 조직(횟수)'
                                            ]].rename(
                                                columns={'title': '강좌명'}),
                                            on='강좌명',
                                            how='left')
                                        st.dataframe(final_table)
                                    else:
                                        st.warning(
                                            "조직별 수강 현황을 보려면 원본 질문/답변 데이터에 'group_1' 컬럼이 필요합니다."
                                        )
                                        st.dataframe(course_counts)
                            else:
                                st.info("키워드를 언급한 구성원들의 수강 이력이 없습니다.")
                        else:
                            st.info("수강 이력을 업로드하면, 키워드와 연관된 학습 현황을 볼 수 있습니다.")
                    else:
                        st.warning(f"'{keyword}'를 포함하는 질문이나 답변을 찾을 수 없습니다.")

    except Exception as e:
        st.error(f"❌ 파일 처리 중 오류 발생: {e}")
else:
    st.info("📂 시작하려면 왼쪽 사이드바에서 분석할 파일을 업로드해주세요.")
