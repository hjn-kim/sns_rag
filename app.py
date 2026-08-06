import sys
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

import streamlit as st

# src/ 를 임포트 경로에 넣는다 (앱은 프로젝트 루트에서 실행한다)
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from query_rewrite import RewriteResult, all_queries, rewrite_query  # noqa: E402


# ---------------------------------------------------------
# 기본 설정
# ---------------------------------------------------------
# set_page_config 는 다른 st.* 호출보다 반드시 먼저 와야 한다.
# (제목은 스타일이 주입된 뒤 아래 "화면" 절에서 그린다)
st.set_page_config(
    page_title="문서 AI 모델 결과",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------
# 선택 항목
# ---------------------------------------------------------
QUESTION_OPTIONS = [
    "When is the next workshop?",
    "How do I extract data from the budget PDF?",
    "Which MDAs are still open to work on?",
    "What tool do they use for visualisation?",
    "Is there a deadline for submission?",
    "다음 워크숍이 언제야?",
    "예산 PDF에서 데이터를 어떻게 추출해?",
    "아직 작업 안 된 부처가 뭐가 남았어?",
]

DOCUMENT_OPTIONS = {
    "한국어": "ko",
    "영어": "en",
    "베트남어": "vi",
    "필리핀어": "fil",
    "러시아어": "ru",
    "중국어": "zh",
}

PIPELINE_STEPS = [
    "질문 입력",
    "질의 재작성·확장",
    "병렬 검색",
    "리랭킹",
    "청크 선정",
    "LLM 답변",
]


# ---------------------------------------------------------
# 검색 요청 모델
# ---------------------------------------------------------
@dataclass
class SearchRequest:
    question: str
    document_name: str
    language_code: str


def run_search(request: SearchRequest) -> dict[str, Any]:
    """
    현재는 화면 검증용 자리표시자 함수입니다.

    이후 다음 모듈을 순서대로 연결하면 됩니다.
    1. rewrite_and_expand_query()
    2. parallel_retrieve()
    3. rerank_results()
    4. select_chunks()
    5. generate_answer()
    """
    return {
        "status": "ready",
        "question": request.question,
        "document_name": request.document_name,
        "language_code": request.language_code,
        "message": "검색 요청이 생성되었습니다. 현재 버전에는 실제 검색 엔진이 연결되어 있지 않습니다.",
    }


@st.cache_data(show_spinner=False)
def cached_rewrite(question: str, language: str) -> RewriteResult:
    """
    같은 (질문, 언어) 조합은 한 번만 호출한다.

    Streamlit 은 위젯을 건드릴 때마다 스크립트를 처음부터 다시 돌린다.
    캐시가 없으면 화면을 조금만 움직여도 Gemini 를 다시 부른다.
    """
    return rewrite_query(question, language=language)


# ---------------------------------------------------------
# 스타일
# ---------------------------------------------------------
st.markdown(
    """
    <style>
        /* Outfit / Inter 는 시스템에 없다. 안 받아오면 sans-serif 로 떨어져
           의도한 모양이 안 나오므로 웹폰트로 가져온다. */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@800&family=Inter:wght@400;500;700&display=swap');

        .block-container {
            max-width: 1080px;
            padding-top: 3rem;
            padding-bottom: 3rem;
        }

        .main-title {
            font-family: 'Outfit', 'Inter', sans-serif;
            background: linear-gradient(90deg, #4A90E2, #8E2DE2);
            -webkit-background-clip: text;
            background-clip: text;              /* 웹킷 아닌 브라우저용 */
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            font-size: 2.8rem;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            font-family: 'Inter', sans-serif;
            color: #7f8c8d;
            font-size: 1.05rem;
            margin-bottom: 1.5rem;
        }

        .section-label {
            font-size: 0.92rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }

        .pipeline-card {
            border: 1px solid #E4E7EC;
            border-radius: 10px;
            padding: 0.5rem 0.6rem;
            text-align: center;
            min-height: 58px;
            background: #FFFFFF;
        }

        .pipeline-number {
            font-size: 0.7rem;
            color: #98A2B3;
            margin-bottom: 0.1rem;
        }

        .pipeline-name {
            font-size: 0.84rem;
            font-weight: 650;
            line-height: 1.3;
        }

        /* 처리 단계와 검색 폼 사이 간격 */
        .section-gap { height: 2.2rem; }

        .result-card {
            border: 1px solid #D0D5DD;
            border-radius: 14px;
            padding: 1.25rem 1.35rem;
            background: #F9FAFB;
            margin-top: 1rem;
        }

        .card-title {
            font-size: 0.95rem;
            font-weight: 700;
            margin-bottom: 0.75rem;
        }

        /* 재작성 카드 */
        .query-origin {
            font-size: 0.8rem;
            color: #98A2B3;
            margin-bottom: 0.4rem;
        }
        .query-main {
            font-size: 1.02rem;
            line-height: 1.5;
            font-weight: 500;
        }
        .query-note {
            font-size: 0.82rem;
            color: #667085;
            margin-top: 0.55rem;
        }

        /* 확장 카드 */
        .query-item {
            font-size: 0.95rem;
            line-height: 1.5;
            padding: 0.42rem 0;
            border-top: 1px solid #EAECF0;
        }
        .query-item:first-of-type { border-top: none; padding-top: 0; }
        .query-num {
            color: #98A2B3;
            font-weight: 700;
            margin-right: 0.5rem;
        }

        .tag {
            display: inline-block;
            font-size: 0.72rem;
            font-weight: 600;
            padding: 0.1rem 0.45rem;
            border-radius: 5px;
            background: #EAECF0;
            color: #475467;
            margin-left: 0.4rem;
            vertical-align: middle;
        }
        .tag-changed {
            background: rgba(76,110,245,.14);
            color: #3B5BDB;
        }

        div.stButton > button {
            height: 3rem;
            font-size: 1rem;
            font-weight: 700;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# 화면
# ---------------------------------------------------------
st.markdown('<h1 class="main-title">문서 AI 모델 결과</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Multi-Query, pgvector, Reranking, topK, LLM</p>',
    unsafe_allow_html=True,
)

st.markdown("#### 처리 단계")

pipeline_columns = st.columns(len(PIPELINE_STEPS), gap="small")
for index, (column, step_name) in enumerate(
    zip(pipeline_columns, PIPELINE_STEPS),
    start=1,
):
    with column:
        st.markdown(
            f"""
            <div class="pipeline-card">
                <div class="pipeline-number">STEP {index}</div>
                <div class="pipeline-name">{step_name}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

with st.form("rag_search_form", clear_on_submit=False):
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown(
            '<div class="section-label">1. 질문 선택</div>',
            unsafe_allow_html=True,
        )
        selected_question = st.selectbox(
            label="질문",
            options=QUESTION_OPTIONS,
            label_visibility="collapsed",
        )

    with right:
        st.markdown(
            '<div class="section-label">2. 검색 문서 선택</div>',
            unsafe_allow_html=True,
        )
        selected_document = st.selectbox(
            label="검색 문서",
            options=list(DOCUMENT_OPTIONS.keys()),
            label_visibility="collapsed",
        )

    st.write("")
    search_clicked = st.form_submit_button(
        "검색",
        type="primary",
        use_container_width=True,
    )


if search_clicked:
    request = SearchRequest(
        question=selected_question,
        document_name=selected_document,
        language_code=DOCUMENT_OPTIONS[selected_document],
    )

    with st.spinner("Gemini 로 질의를 재작성하고 있습니다."):
        rewrite = cached_rewrite(request.question, request.document_name)

    if not rewrite.ok:
        st.error(f"질의 재작성에 실패했습니다.\n\n{rewrite.error}")

    # --- 첫 번째 div : 질의 재작성 ---------------------------------------
    badge = (
        '<span class="tag tag-changed">재작성됨</span>' if rewrite.changed
        else '<span class="tag">변경 없음</span>'
    )
    reason_html = (
        f'<div class="query-note">{escape(rewrite.reason)}</div>'
        if rewrite.reason else ""
    )
    st.markdown(
        f"""
        <div class="result-card">
            <div class="card-title">1. 질의 재작성 {badge}</div>
            <div class="query-origin">원 질문 · {escape(rewrite.question)}</div>
            <div class="query-main">{escape(rewrite.rewritten)}</div>
            {reason_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- 두 번째 div : 질의 확장 -----------------------------------------
    if rewrite.expansions:
        items = "".join(
            f'<div class="query-item">'
            f'<span class="query-num">{i}.</span>{escape(q)}</div>'
            for i, q in enumerate(rewrite.expansions, 1)
        )
    else:
        items = '<div class="query-note">확장 질의를 받지 못했습니다.</div>'

    st.markdown(
        f"""
        <div class="result-card">
            <div class="card-title">2. 질의 확장
                <span class="tag">{len(rewrite.expansions)}개</span></div>
            {items}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

    with st.expander("개발용 데이터 보기"):
        st.json(
            {
                "question": request.question,
                "document_name": request.document_name,
                "language_code": request.language_code,
                "rewritten": rewrite.rewritten,
                "changed": rewrite.changed,
                "expansions": rewrite.expansions,
                "search_queries": all_queries(rewrite),
                "model": rewrite.model,
                "elapsed_sec": round(rewrite.elapsed, 2),
                "error": rewrite.error,
            }
        )