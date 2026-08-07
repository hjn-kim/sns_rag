import sys
from html import escape
from pathlib import Path

import streamlit as st

# src/ 를 임포트 경로에 넣는다 (앱은 프로젝트 루트에서 실행한다)
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from grade import gold_for  # noqa: E402
from main import PipelineResult, run_pipeline  # noqa: E402
from multi_query import RewriteResult  # noqa: E402
from rerank import FINAL_TOP_N  # noqa: E402
from search import DEFAULT_TOP_K  # noqa: E402


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
"1. 두 페이지만 있고 자본 프로젝트가 없다고 언급된 MDA는 무엇인가요?",
"2. 튜토리얼에서 데이터 추출 함수를 만들기 위해 어느 부처의 데이터를 사용했나요?",
"3. 가입 과정에서 직장 또는 학생 이메일을 요구한 소프트웨어는 무엇인가요?",
"1. Which MDA was described as having only two pages and zero capital projects?",
"2. Which ministry's data was used in the tutorials to create the extraction functions?",
"3. Which software asked the user for a work or student email during the sign-up process?",
"1. 哪个 MDA 被描述为只有两页且没有资本项目？",
"2. 教程中使用了哪个部门的数据来创建数据提取函数？",
"3. 哪个软件在注册过程中要求用户提供工作邮箱或学生邮箱？",
"1. MDA nào được mô tả là chỉ có hai trang và không có dự án vốn?",
"2. Dữ liệu của bộ nào đã được sử dụng trong các hướng dẫn để tạo các hàm trích xuất dữ liệu?",
"3. Phần mềm nào yêu cầu người dùng cung cấp email công việc hoặc email sinh viên trong quá trình đăng ký?",
"1. Aling MDA ang inilarawan na mayroon lamang dalawang pahina at walang capital projects?",
"2. Ang datos ng aling ministry ang ginamit sa mga tutorial upang gumawa ng mga extraction function?",
"3. Aling software ang humingi sa user ng work o student email sa proseso ng pag-sign up?",
"1. Какое MDA было описано как состоящее всего из двух страниц и не имеющее капитальных проектов?",
"2. Данные какого министерства использовались в учебных материалах для создания функций извлечения данных?",
"3. Какая программа запросила у пользователя рабочий или студенческий адрес электронной почты при регистрации?"
]

DOCUMENT_OPTIONS = {
    "한국어": "ko",
    "영어": "en",
    "중국어": "zh",
    "필리핀어": "fil",
    "베트남어": "vi",
    "러시아어": "ru",
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
# 파이프라인 호출
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def cached_pipeline(question: str, lang: str,
                    top_k: int = DEFAULT_TOP_K,
                    final_n: int = FINAL_TOP_N) -> PipelineResult:
    """
    src/main.py 의 6단계 파이프라인을 부른다. 화면은 결과만 받아 그린다.

    같은 (질문, 언어) 조합은 한 번만 돈다. Streamlit 은 위젯을 건드릴 때마다
    스크립트를 처음부터 다시 도는데, 캐시가 없으면 화면을 조금만 움직여도
    Gemini 를 세 번(재작성·리랭킹·답변) 다시 부른다.
    """
    return run_pipeline(question, lang=lang, top_k=top_k, final_n=final_n)


def query_labels(rewrite: RewriteResult, queries: list[str]) -> list[str]:
    """
    질의 목록의 각 항목이 어디서 온 것인지 짧은 이름을 붙인다.

    all_queries() 는 [재작성, 확장1~3, 원 질문] 순서로 만들고 중복을 지운다.
    재작성이 원 질문과 같으면 하나로 합쳐지므로 순서만 보고 판단할 수 없다.
    """
    labels = []
    for q in queries:
        if q == rewrite.rewritten and q == rewrite.question:
            labels.append("재작성=원문")
        elif q == rewrite.rewritten:
            labels.append("재작성")
        elif q == rewrite.question:
            labels.append("원 질문")
        elif q in rewrite.expansions:
            labels.append(f"확장 {rewrite.expansions.index(q) + 1}")
        else:
            labels.append("")
    return labels


# ---------------------------------------------------------
# 카드 그리기
#
# 파이프라인이 단계를 끝낼 때마다 하나씩 불린다. 각 함수는 st.markdown 한 번으로
# 카드 하나를 그리고 끝낸다. 계산은 하지 않는다.
# ---------------------------------------------------------

def render_rewrite(rewrite: RewriteResult) -> None:
    """1. 질의 재작성"""
    badge = (
        '<span class="tag tag-changed">재작성됨</span>' if rewrite.changed
        else '<span class="tag">변경 없음</span>'
    )
    reason = (f'<div class="query-note">{escape(rewrite.reason)}</div>'
              if rewrite.reason else "")
    st.markdown(
        f"""
        <div class="result-card">
            <div class="card-title">1. 질의 재작성 {badge}</div>
            <div class="query-origin">원 질문 · {escape(rewrite.question)}</div>
            <div class="query-main">{escape(rewrite.rewritten)}</div>
            {reason}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_expansion(rewrite: RewriteResult) -> None:
    """2. 질의 확장"""
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


def render_query_table(result, rewrite: RewriteResult) -> None:
    """3-1. 질의별 랭킹 — 세로는 순위, 가로는 질의"""
    labels = query_labels(rewrite, result.queries)
    # 두 개 이상의 질의가 함께 뽑은 청크. 색으로 표시해 3-2 에서 몇 개가
    # 합쳐지는지 눈으로 보이게 한다.
    shared = {hit.key for hit in result.pooled if len(hit.found_by) > 1}

    header = "".join(
        f'<th>질의 {i}<div class="qhead-sub">{escape(label)}</div></th>'
        for i, label in enumerate(labels, 1)
    )
    rows = ""
    for rank in range(result.top_k):
        cells = ""
        for hits in result.per_query:
            if rank >= len(hits):
                cells += "<td></td>"
                continue
            hit = hits[rank]
            css = "qcell dup" if hit.key in shared else "qcell"
            cells += (f'<td class="{css}">'
                      f'<span class="qscore">{hit.score:.3f}</span>'
                      f'<span class="qchunk">{hit.key}</span></td>')
        rows += f'<tr><td class="qrank">{rank + 1}</td>{cells}</tr>'

    st.markdown(
        f"""
        <div class="result-card">
            <div class="card-title">3-1. 질의별 랭킹
                <span class="tag">{selected_document} 색인</span>
                <span class="tag">질의 {len(result.queries)}개 x 상위
                    {result.top_k}개 = {result.n_total}개</span></div>
            <div class="qtable-wrap">
                <table class="qtable">
                    <thead><tr><th class="qrank">#</th>{header}</tr></thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>
            <div class="query-note">색칠된 칸은 두 개 이상의 질의가 함께 뽑은
                청크입니다. 아래 3-2 에서 하나로 합쳐집니다.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pooled(result) -> None:
    """
    3-2. 중복 제거 — 리랭킹으로 넘어갈 후보

    한 줄에 하나씩. 20개 넘게 나오는 목록이라 훑어보기 좋아야 한다.
    본문은 글자 수로 자르지 않고 CSS 로 넘치는 만큼 '...' 처리한다. 글자 수로
    자르면 창 너비에 따라 두 줄이 되기도 해서 줄 높이가 들쭉날쭉해진다.
    """
    if result.pooled:
        items = "".join(
            f'<div class="hit-line">'
            f'<span class="hit-rank">{rank}</span>'
            f'<span class="hit-score">{hit.score:.3f}</span>'
            f'<span class="hit-src">{hit.key}</span>'
            f'<span class="hit-found" title="질의 '
            f'{", ".join(str(q + 1) for q in hit.found_by)}번이 뽑음">'
            f'{len(hit.found_by)}/{len(result.queries)}</span>'
            f'<span class="hit-oneline">{escape(hit.preview(220))}</span>'
            f'</div>'
            for rank, hit in enumerate(result.pooled, 1)
        )
    else:
        items = '<div class="query-note">검색된 청크가 없습니다.</div>'
    st.markdown(
        f"""
        <div class="result-card">
            <div class="card-title">3-2. 중복 제거
                <span class="tag">{result.n_total}개 → {len(result.pooled)}개</span>
                <span class="tag">리랭킹 후보</span></div>
            {items}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_rerank(rr) -> None:
    """4. 리랭킹 — 등수가 어떻게 바뀌었는지"""
    method = {
        "llm": "Gemini",
        "cross": f"크로스인코더 {rr.model or 'bge-reranker-v2-m3'}",
        "rrf": "RRF (질의별 등수 합산, 호출 없음)",
    }.get(rr.method, rr.method)
    rows = ""
    for item in rr.ranked:
        selected = item.rank_after <= len(rr.selected)
        if item.moved > 0:
            move = f'<span class="rr-up">▲{item.moved}</span>'
        elif item.moved < 0:
            move = f'<span class="rr-down">▼{-item.moved}</span>'
        else:
            move = '<span class="rr-same">-</span>'
        score = (f'<span class="rr-llm">{item.llm_score}</span>'
                 if item.llm_score is not None else "-")
        rows += (
            f'<tr class="{"rr-picked" if selected else ""}">'
            f'<td class="qrank">{item.rank_after}</td>'
            f'<td class="qrank">{item.rank_before}</td>'
            f'<td>{move}</td>'
            f'<td>{score}</td>'
            f'<td><span class="qchunk">{item.hit.key}</span></td>'
            f'<td class="rr-reason">{escape(item.reason)}</td></tr>'
        )
    st.markdown(
        f"""
        <div class="result-card">
            <div class="card-title">4. 리랭킹
                <span class="tag">{escape(method)}</span>
                <span class="tag">{rr.elapsed:.1f}초</span></div>
            <div class="qtable-wrap">
                <table class="qtable">
                    <thead><tr>
                        <th class="qrank">후</th><th class="qrank">전</th>
                        <th>이동</th><th>점수</th><th>청크</th><th>판단 근거</th>
                    </tr></thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>
            <div class="query-note">점수는 "질문에 대한 답이 실제로 이 청크에
                있는가"를 0~10 으로 매긴 것입니다. 파랗게 칠한 줄이 5번에서
                선정됩니다.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_selected(rr) -> None:
    """5. 최종 청크 선정"""
    items = "".join(
        f'<div class="hit">'
        f'  <div class="hit-head">'
        f'    <span class="hit-rank">{rank}</span>'
        f'    <span class="hit-src">{hit.key}</span>'
        f'    <span>{escape(hit.started_at[:10])} · {hit.n_utterances}발화 · '
        f'{escape(", ".join(hit.participants[:3]))}</span>'
        f'  </div>'
        f'  <div class="hit-text">{escape(hit.preview(260))}</div>'
        f'</div>'
        for rank, hit in enumerate(rr.selected, 1)
    )
    st.markdown(
        f"""
        <div class="result-card">
            <div class="card-title">5. 최종 청크 선정
                <span class="tag">{len(rr.ranked)}개 → {len(rr.selected)}개</span>
                <span class="tag">약 {len(rr.selected) * 500}토큰</span></div>
            {items}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_answer(ans) -> None:
    """6. LLM 답변"""
    if ans is None:
        return
    if not ans.ok:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="card-title">6. LLM 답변
                    <span class="tag">실패</span></div>
                <div class="query-note">{escape(ans.error or "")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    badge = ('<span class="tag tag-changed">근거 존재</span>' if ans.enough
             else '<span class="tag">근거 부족</span>')
    cites = ("".join(f'<span class="cite">{escape(c)}</span>'
                     for c in ans.citations)
             if ans.citations else '<span class="cite-none">없음</span>')
    note = (f'<div class="query-note">{escape(ans.note)}</div>'
            if ans.note else "")
    st.markdown(
        f"""
        <div class="result-card answer-card">
            <div class="card-title">6. LLM 답변 {badge}
                <span class="tag">{ans.elapsed:.1f}초</span></div>
            <div class="answer-text">{escape(ans.answer)}</div>
            <div class="cite-row">근거 {cites}</div>
            {note}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_grade(gr) -> None:
    """
    7. 정답 비교

    data/answer.json 에 적어 둔 실제 정답과 6단계 답변을 나란히 놓는다.
    """
    if gr is None:
        return

    # 맞았으면 겹친 후보만, 틀렸으면 후보 전체가 gold_display 에 담겨 온다.
    if gr.correct:
        style, mark = "grade-ok", "O"
    elif gr.verdict == "오답":
        style, mark = "grade-no", "X"
    else:
        style, mark = "grade-none", "?"

    st.markdown(
        f"""
        <div class="result-card grade-card {style}">
            <div class="card-title">7. 정답 비교
                <span class="tag tag-verdict {style}">{mark} {gr.verdict}</span>
                <span class="tag">문자열 포함 판정</span></div>
            <div class="grade-row">
                <span class="grade-label">LLM 정답</span>
                <span class="grade-value">{escape(gr.llm_answer) or "—"}</span>
            </div>
            <div class="grade-row">
                <span class="grade-label">실제 정답</span>
                <span class="grade-value grade-gold">
                    {escape(gr.gold_display) or "—"}</span>
            </div>
            <div class="grade-row">
                <span class="grade-label">정답 여부</span>
                <span class="grade-value">{mark} {escape(gr.verdict)}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def dev_payload(result: PipelineResult) -> dict:
    """개발용 데이터 보기에 넣을 것들."""
    ms, rr, ans = result.comparison, result.rerank, result.answer
    labels = query_labels(result.rewrite, result.queries)
    return {
        "question": result.question,
        "raw_question": result.raw_question,
        "lang": result.lang,
        "elapsed_sec": round(result.elapsed, 2),
        "1_2_rewrite": {
            "rewritten": result.rewrite.rewritten,
            "changed": result.rewrite.changed,
            "reason": result.rewrite.reason,
            "expansions": result.rewrite.expansions,
            "queries": result.queries,
            "error": result.rewrite.error,
        },
        "3_comparison": {
            "per_query_top_k": ms.top_k,
            "n_before_dedup": ms.n_total,
            "n_after_dedup": len(ms.pooled),
            "per_query": {
                f"질의 {i} ({label})": [f"{h.key} {h.score:.4f}" for h in hits]
                for i, (label, hits) in enumerate(zip(labels, ms.per_query), 1)
            },
        },
        "4_5_rerank": {
            "method": rr.method,
            "error": rr.error,
            "ranked": [
                {
                    "rank_after": x.rank_after,
                    "rank_before": x.rank_before,
                    "llm_score": x.llm_score,
                    "rrf": round(x.rrf_score, 4),
                    "dense": round(x.hit.score, 4),
                    "chunk": x.hit.key,
                    "reason": x.reason,
                }
                for x in rr.ranked
            ],
            "selected": [h.key for h in rr.selected],
        },
        "7_grade": None if result.grade is None else {
            "verdict": result.grade.verdict,
            "correct": result.grade.correct,
            "llm_answer": result.grade.llm_answer,
            "candidates": result.grade.candidates,
            "matched": result.grade.matched,
            "gold_display": result.grade.gold_display,
            "reason": result.grade.reason,
        },
        "6_answer": None if ans is None else {
            "answer": ans.answer,
            "enough": ans.enough,
            "citations": ans.citations,
            "note": ans.note,
            "model": ans.model,
            "error": ans.error,
        },
        "selected_chunks": [
            {"chunk": h.key, "tokens": [h.token_start, h.token_end],
             "started_at": h.started_at, "text": h.text}
            for h in result.selected
        ],
    }


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
            /* 1~7번 카드가 세로로 이어지는 간격. 카드마다 st.markdown 이 따로
               나가므로 카드 사이 여백은 이 margin-top 하나로 결정된다. */
            margin-top: 1.2rem;
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

        /* 3-1 질의별 랭킹 표 : 세로는 순위, 가로는 질의 */
        .qtable-wrap {
            overflow-x: auto;          /* 질의가 늘어도 페이지가 밀리지 않게 */
            margin-top: 0.3rem;
        }
        .qtable {
            border-collapse: collapse;
            width: 100%;
            font-size: 0.8rem;
        }
        .qtable th, .qtable td {
            padding: 0.34rem 0.5rem;
            border-bottom: 1px solid #EAECF0;
            text-align: left;
            white-space: nowrap;
        }
        .qtable th {
            font-weight: 700;
            color: #475467;
            border-bottom: 1px solid #D0D5DD;
        }
        .qhead-sub {
            font-weight: 400;
            font-size: 0.72rem;
            color: #98A2B3;
        }
        .qrank {
            color: #98A2B3;
            font-variant-numeric: tabular-nums;
            width: 2rem;
        }
        .qscore {
            font-variant-numeric: tabular-nums;
            color: #344054;
            margin-right: 0.35rem;
        }
        .qchunk {
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            font-size: 0.74rem;
            color: #667085;
        }
        /* 두 개 이상의 질의가 함께 뽑은 청크 = 중복 제거 대상 */
        .qtable td.dup {
            background: rgba(76, 110, 245, 0.09);
            border-radius: 4px;
        }
        .qtable td.dup .qchunk { color: #3B5BDB; font-weight: 600; }

        /* 검색 결과 카드 */
        .hit {
            padding: 0.6rem 0;
            border-top: 1px solid #EAECF0;
        }
        .hit:first-of-type { border-top: none; padding-top: 0; }

        .hit-head {
            display: flex;
            align-items: baseline;
            gap: 0.5rem;
            font-size: 0.8rem;
            color: #667085;
            margin-bottom: 0.25rem;
        }
        .hit-rank {
            font-weight: 700;
            color: #3B5BDB;
            min-width: 1.4rem;
        }
        .hit-score {
            font-variant-numeric: tabular-nums;
            font-weight: 600;
            color: #344054;
        }
        .hit-src {
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            font-size: 0.76rem;
            background: #EAECF0;
            border-radius: 4px;
            padding: 0.05rem 0.35rem;
        }
        .hit-query {
            font-size: 0.78rem;
            color: #98A2B3;
            margin-bottom: 0.25rem;
        }
        .hit-text {
            font-size: 0.9rem;
            line-height: 1.55;
        }

        /* 3-2 후보 목록 : 한 항목이 정확히 한 줄 */
        .hit-line {
            display: flex;
            align-items: baseline;
            gap: 0.5rem;
            padding: 0.3rem 0;
            border-top: 1px solid #EAECF0;
            font-size: 0.82rem;
        }
        .hit-line:first-of-type { border-top: none; }

        .hit-found {
            font-size: 0.72rem;
            color: #3B5BDB;
            background: rgba(76, 110, 245, 0.12);
            border-radius: 4px;
            padding: 0.02rem 0.32rem;
            white-space: nowrap;
        }
        /* 넘치는 만큼만 '...' 로 잘린다. 글자 수로 자르지 않으므로 창 너비가
           달라져도 항상 한 줄이다. min-width:0 이 없으면 flex 항목이 안 줄어든다. */
        .hit-oneline {
            flex: 1;
            min-width: 0;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            color: #475467;
        }

        /* 4. 리랭킹 표 */
        .rr-up   { color: #2F9E44; font-weight: 700; }
        .rr-down { color: #E03131; font-weight: 700; }
        .rr-same { color: #C1C7D0; }
        .rr-llm  { font-weight: 700; color: #344054; }
        .rr-reason {
            color: #667085;
            white-space: normal;      /* 근거 문장만 줄바꿈 허용 */
            min-width: 18rem;
        }
        .qtable tr.rr-picked td { background: rgba(76, 110, 245, 0.09); }

        /* 6. LLM 답변 */
        .answer-card {
            background: #FFFFFF;
            border-color: #B9C6FF;
        }
        .answer-text {
            font-size: 1.05rem;
            line-height: 1.65;
            font-weight: 500;
        }
        .cite-row {
            margin-top: 0.7rem;
            font-size: 0.8rem;
            color: #98A2B3;
        }
        .cite {
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            font-size: 0.76rem;
            background: rgba(76, 110, 245, 0.12);
            color: #3B5BDB;
            border-radius: 4px;
            padding: 0.08rem 0.4rem;
            margin-left: 0.3rem;
        }
        .cite-none { margin-left: 0.3rem; }

        /* 7. 정답 비교 */
        .grade-card.grade-ok   { border-color: #8CE99A; background: #F4FCF5; }
        .grade-card.grade-no   { border-color: #FFA8A8; background: #FFF5F5; }
        .grade-card.grade-none { border-color: #D0D5DD; }

        .tag-verdict { font-weight: 700; }
        .tag-verdict.grade-ok   { background: rgba(47,158,68,.16);  color: #2B8A3E; }
        .tag-verdict.grade-no   { background: rgba(224,49,49,.14);  color: #C92A2A; }
        .tag-verdict.grade-none { background: #EAECF0; color: #667085; }

        .grade-row {
            display: flex;
            gap: 0.75rem;
            padding: 0.4rem 0;
            border-top: 1px solid rgba(0, 0, 0, 0.06);
            font-size: 0.95rem;
            line-height: 1.5;
        }
        .grade-row:first-of-type { border-top: none; }
        .grade-label {
            flex: 0 0 5.5rem;
            color: #667085;
            font-size: 0.85rem;
            font-weight: 600;
            padding-top: 0.1rem;
        }
        .grade-value { flex: 1; min-width: 0; }
        .grade-gold { font-weight: 700; }

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
    '<p class="subtitle">Multi-Query, harrier-oss-v1, Reranking, LLM</p>',
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
    lang_code = DOCUMENT_OPTIONS[selected_document]

    # 7단계용 실제 정답. data/answer.json 이 QUESTION_OPTIONS 순번(1부터)을
    # key 로 쓴다. 정답이 없는 질문이면 빈 문자열이 와서 7단계를 건너뛴다.
    gold = gold_for(QUESTION_OPTIONS.index(selected_question) + 1)

    # 진행 상태를 한 줄로 보여줄 자리. 단계가 끝날 때마다 문구를 갈아 끼우고
    # 마지막에 지운다.
    progress = st.empty()
    progress.info("Gemini 로 질의를 재작성하고 있습니다.")

    # 3-1 표의 열 이름을 붙이려면 재작성 결과가 필요하다. 콜백끼리 넘기기 위해
    # 바깥 dict 에 담아 둔다.
    state: dict = {}

    def on_stage(stage: str, payload) -> None:
        """
        파이프라인이 한 단계 끝낼 때마다 불린다. 끝난 단계부터 바로 그린다.

        전체가 20초 넘게 걸리는데 다 끝나야 첫 카드가 뜨면 멈춘 것처럼 보인다.
        Streamlit 은 st.* 호출을 그때그때 프런트로 보내므로 여기서 그리면 된다.
        """
        if stage == "rewrite":
            state["rewrite"] = payload
            render_rewrite(payload)          # 1번
            render_expansion(payload)        # 2번
            progress.info(
                f"질의를 임베딩해 {selected_document} 색인에서 청크를 찾고 있습니다. "
                "(예상 시간: 20초)"
            )

        elif stage == "comparison":
            state["comparison"] = payload
            render_query_table(payload, state["rewrite"])   # 3-1
            render_pooled(payload)                          # 3-2
            progress.info(
                f"후보 {len(payload.pooled)}개를 Gemini 로 리랭킹하고 있습니다."
            )

        elif stage == "rerank":
            state["rerank"] = payload
            render_rerank(payload)           # 4번
            render_selected(payload)         # 5번
            progress.info("근거를 읽고 답변을 만들고 있습니다.")

        elif stage == "answer":
            render_answer(payload)           # 6번
            if gold:
                progress.info("실제 정답과 견주고 있습니다.")
            else:
                progress.empty()

        elif stage == "grade":
            progress.empty()
            render_grade(payload)            # 7번

    result = run_pipeline(selected_question, lang=lang_code,
                          gold=gold, on_stage=on_stage)

    for stage_name, message in result.errors().items():
        st.warning(f"{stage_name} 단계가 실패했습니다. {message}")

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

    with st.expander("개발용 데이터 보기"):
        st.json(dev_payload(result))
