#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
파이프라인 오케스트레이터

app.py 의 검색 버튼이 부르는 곳이다. 단계별 모듈을 순서대로 돌리고,
각 단계의 결과를 하나에 담아 돌려준다. 화면 그리는 일은 하지 않는다.

    질문
      |
      +-- 1,2  multi_query.py    재작성 + 확장          -> RewriteResult
      |
      +-- 3    comparison.py     질의별 등수 비교        -> MultiSearch
      |
      +-- 4,5  rerank.py         리랭킹 + 최종 청크 선정 -> RerankResult
      |
      +-- 6    answer.py         LLM 답변 생성          -> AnswerResult
      |
    PipelineResult

단계마다 실패해도 파이프라인이 멈추지 않는다. 각 결과 객체가 error 를 들고
있으므로 화면에서 어디가 어떻게 실패했는지 보여주면서 나머지는 계속 굴린다.
Gemini 가 죽어도 검색은 되고, 리랭킹은 RRF 로 떨어진다.

단계별로 따로 돌려보고 싶으면 각 모듈에 CLI 가 있다.
    python src/multi_query.py "..."
    python src/comparison.py --lang ko "..."
    python src/rerank.py --lang ko --rewrite "..."
    python src/answer.py --lang ko "..."

전체를 한 번에:
    python src/main.py --lang ko "두 페이지만 있는 MDA는?"
    python src/main.py --lang ko --no-llm "..."      # Gemini 없이 (검색만)
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from answer import AnswerResult, generate_answer  # noqa: E402
from comparison import MultiSearch, search_per_query  # noqa: E402
from grade import GradeResult, grade_answer  # noqa: E402
from multi_query import RewriteResult, all_queries, rewrite_query  # noqa: E402
from rerank import FINAL_TOP_N, RerankResult, rerank  # noqa: E402
from search import DEFAULT_TOP_K, LANGUAGES  # noqa: E402


@dataclass
class PipelineResult:
    """6단계 전체의 결과. 화면은 이것만 보고 그린다."""

    question: str                 # 번호 접두사를 뗀 실제 질문
    raw_question: str             # 사용자가 고른 원래 문자열
    lang: str                     # 'ko'
    language_name: str            # '한국어'

    rewrite: RewriteResult        # 1, 2 단계
    queries: list[str] = field(default_factory=list)
    comparison: MultiSearch | None = None   # 3 단계
    rerank: RerankResult | None = None      # 4, 5 단계
    answer: AnswerResult | None = None      # 6 단계
    grade: GradeResult | None = None        # 7 단계 (정답표가 있을 때만)
    elapsed: float = 0.0

    @property
    def selected(self):
        """최종 선정된 청크. 리랭킹을 건너뛰었으면 검색 상위로 대체한다."""
        if self.rerank and self.rerank.selected:
            return self.rerank.selected
        if self.comparison:
            return self.comparison.pooled[:FINAL_TOP_N]
        return []

    def errors(self) -> dict[str, str]:
        """단계 이름 -> 실패 메시지. 화면에서 배너로 띄우기 위한 것."""
        out: dict[str, str] = {}
        if self.rewrite and self.rewrite.error:
            out["질의 재작성"] = self.rewrite.error
        if self.rerank and self.rerank.error:
            out["리랭킹"] = self.rerank.error
        if self.answer and self.answer.error:
            out["답변 생성"] = self.answer.error
        if self.grade and self.grade.error:
            out["정답 비교"] = self.grade.error
        return out


def strip_number(question: str) -> str:
    """
    선택 항목 앞의 "1. " 은 화면 표시용 번호다. 질문 내용이 아니므로 떼고 넘긴다.

    붙인 채로 임베딩하면 점수가 조금 깎인다(측정: 0.6442 -> 0.6351).
    """
    return re.sub(r"^\d+\.\s*", "", question or "").strip()


def run_pipeline(question: str, lang: str = "ko",
                 top_k: int = DEFAULT_TOP_K,
                 final_n: int = FINAL_TOP_N,
                 rerank_method: str = "llm",
                 use_llm: bool = True,
                 gold: list[str] | None = None,
                 llm_backend: str = "gemini",
                 answer_language: str | None = None,
                 on_stage=None) -> PipelineResult:
    """
    질문 하나를 6단계에 통과시킨다.

    use_llm=False 면 Gemini 를 부르지 않는다. 원 질문 하나로 검색하고 리랭킹은
    RRF 로, 답변 생성은 건너뛴다. 키 없이 검색 품질만 확인할 때 쓴다.

    gold 를 주면 7단계(정답 비교)까지 돈다. data/answer.json 에 적어 둔 정답
    후보 목록이며, 화면에서 순번으로 찾아 넘긴다. 비어 있으면 건너뛴다.
    판정은 문자열 포함이라 호출이 없고, use_llm 과 무관하게 돈다.

    llm_backend 는 1·2 단계와 6 단계를 무엇으로 돌릴지 정한다.
        "gemini"  API 호출. 키가 필요하다. (기본)
        "qwen"    로컬 Qwen3-8B. 키가 필요 없고 GPU 가 필요하다.
    4 단계 리랭킹은 rerank_method 로 따로 정한다.

    lang 은 검색할 색인의 언어다. 그 언어 이름을 1·2 단계에는 확장 대상 언어로,
    6 단계에는 근거 문서의 언어로 넘긴다. 답변도 기본적으로 그 언어로 나온다
    (러시아어 색인을 한국어로 물으면 답은 러시아어). 7 단계 정답 후보도 같은
    언어 것으로 넘겨야 한다. 질문 언어로 받고 싶으면 answer_language 를 준다.

    on_stage(단계이름, 결과) 를 주면 단계가 끝날 때마다 부른다. 단계이름은
    "rewrite" / "comparison" / "rerank" / "answer" / "grade" 다. 화면이 결과를
    기다리지 않고 끝난 단계부터 그릴 수 있게 하려는 것이다. 전체가 20초 넘게
    걸리는데 다 끝나야 첫 카드가 뜨면 멈춘 것처럼 보인다.
    """
    started = time.time()
    language_name = LANGUAGES.get(lang, lang)
    clean = strip_number(question)

    def emit(stage: str, payload) -> None:
        if on_stage is not None:
            on_stage(stage, payload)

    # --- 1, 2 단계 : 재작성 + 확장 -----------------------------------------
    # llm_backend="qwen" 이면 Gemini 대신 로컬 Qwen3-8B 를 쓴다. 16GB 모델을
    # 올리므로 그 backend 를 고를 때만 import 한다.
    if not use_llm:
        rw = RewriteResult(question=clean, rewritten=clean)
    elif llm_backend == "qwen":
        from local_llm import rewrite_query_local
        rw = rewrite_query_local(clean, language=language_name)
    else:
        rw = rewrite_query(clean, language=language_name)
    queries = all_queries(rw)
    emit("rewrite", rw)

    # --- 3 단계 : 질의별 등수 비교 -----------------------------------------
    ms = search_per_query(queries, lang=lang, top_k=top_k)
    emit("comparison", ms)

    # --- 4, 5 단계 : 리랭킹 + 최종 선정 -------------------------------------
    # "cross" 는 GPU 크로스인코더(rerank_gpu.py)다. 모델을 하나 더 올리므로
    # 그 방식을 고를 때만 import 한다. 결과 자료구조는 rerank.py 것과 같아서
    # 화면은 어느 쪽으로 돌렸는지 몰라도 된다.
    if rerank_method == "cross":
        from rerank_gpu import rerank_cross
        rr = rerank_cross(clean, ms, top_n=final_n)
    else:
        rr = rerank(clean, ms, top_n=final_n,
                    method=rerank_method if use_llm else "rrf")
    emit("rerank", rr)

    # --- 6 단계 : 답변 생성 -------------------------------------------------
    if not use_llm:
        ans = None
    elif llm_backend == "qwen":
        from local_llm import generate_answer_local
        ans = generate_answer_local(clean, rr.selected,
                                    doc_language=language_name,
                                    answer_language=answer_language)
    else:
        ans = generate_answer(clean, rr.selected,
                              doc_language=language_name,
                              answer_language=answer_language)
    emit("answer", ans)

    # --- 7 단계 : 정답 비교 -------------------------------------------------
    # 정답표에 이 질문의 답이 있을 때만 돈다. 시연용 채점이라 없으면 건너뛴다.
    gr = None
    if gold:
        gr = grade_answer(clean, ans.answer if (ans and ans.ok) else "", gold)
        emit("grade", gr)

    return PipelineResult(
        question=clean,
        raw_question=question,
        lang=lang,
        language_name=language_name,
        rewrite=rw,
        queries=queries,
        comparison=ms,
        rerank=rr,
        answer=ans,
        grade=gr,
        elapsed=time.time() - started,
    )


# --------------------------------------------------------------------------
# 단독 실행
# --------------------------------------------------------------------------

def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    parser = argparse.ArgumentParser(
        description="질문 하나를 RAG 파이프라인 6단계에 통과시킨다.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("question", nargs="*")
    parser.add_argument("--lang", default="ko", choices=list(LANGUAGES),
                        help="검색할 색인(근거 문서)의 언어 (기본: ko)")
    parser.add_argument("--answer-lang", default=None,
                        help="6단계 답변을 쓸 언어 이름. 예: --answer-lang 한국어\n"
                             "(기본: --lang 과 같은 언어 = 검색 문서 언어)")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help=f"질의당 검색할 청크 수 (기본: {DEFAULT_TOP_K})")
    parser.add_argument("--final-n", type=int, default=FINAL_TOP_N,
                        help=f"최종 선정 청크 수 (기본: {FINAL_TOP_N})")
    parser.add_argument("--method", default="llm", choices=["llm", "rrf", "cross"],
                        help="리랭킹 방식 (기본: llm)\n"
                             "  llm    Gemini 가 0~10 점을 매긴다. 판단 근거 문장이 나온다\n"
                             "  cross  bge-reranker-v2-m3 로 직접 계산한다 (GPU 권장)\n"
                             "  rrf    질의별 등수만 합산한다 (호출 없음)")
    parser.add_argument("--backend", default="gemini", choices=["gemini", "qwen"],
                        help="1·2·6 단계를 무엇으로 돌릴지 (기본: gemini)\n"
                             "  gemini  API 호출. 키 필요\n"
                             "  qwen    로컬 Qwen3-8B. 키 불필요, GPU 필요")
    parser.add_argument("--no-llm", action="store_true",
                        help="Gemini 를 전혀 부르지 않는다 (검색만)")
    parser.add_argument("--gold", default=None, nargs="*",
                        help="정답 후보. 주면 7단계(정답 비교)까지 돈다\n"
                             "예: --gold 사법부 사법 Judiciary\n"
                             "질문 순번으로 채점하려면 src/grade.py --run N 을 쓴다")
    args = parser.parse_args()

    question = " ".join(args.question) or "두 페이지만 있는 MDA는?"
    result = run_pipeline(question, lang=args.lang, top_k=args.top_k,
                          final_n=args.final_n, rerank_method=args.method,
                          use_llm=not args.no_llm, gold=args.gold,
                          llm_backend=args.backend,
                          answer_language=args.answer_lang)

    rw, ms, rr, ans = result.rewrite, result.comparison, result.rerank, result.answer

    print(f"\n질문   : {result.question}")
    print(f"색인   : {result.lang} ({result.language_name})")

    print(f"\n[1] 질의 재작성  {'바뀜' if rw.changed else '변경 없음'}")
    print(f"    {rw.rewritten}")
    print(f"[2] 질의 확장    {len(rw.expansions)}개")
    for i, q in enumerate(rw.expansions, 1):
        print(f"    {i}. {q}")

    print(f"\n[3] 질의별 검색  질의 {len(ms.queries)}개 x 상위 {ms.top_k}개 "
          f"= {ms.n_total}개 -> 중복 제거 {len(ms.pooled)}개  ({ms.elapsed:.1f}초)")

    print(f"\n[4] 리랭킹       {rr.method}  ({rr.elapsed:.1f}초)")
    for item in rr.ranked[:args.final_n + 3]:
        mark = " <= 선정" if item.rank_after <= args.final_n else ""
        llm = f"{item.llm_score:>2}" if item.llm_score is not None else " -"
        moved = f"{item.moved:+d}" if item.moved else "  "
        print(f"    {item.rank_after:>2}위 (전 {item.rank_before:>2}위 {moved:>3}) "
              f"LLM {llm}  {item.hit.key}{mark}")

    print(f"\n[5] 최종 선정    {len(rr.selected)}개  "
          f"{', '.join(h.key for h in rr.selected)}")

    if ans is None:
        print("\n[6] 답변 생성    건너뜀 (--no-llm)")
    elif not ans.ok:
        print(f"\n[6] 답변 생성    실패: {ans.error}")
    else:
        print(f"\n[6] 답변 생성    ({ans.elapsed:.1f}초, 근거 충분: "
              f"{'예' if ans.enough else '아니오'})")
        print(f"    {ans.answer}")
        if ans.citations:
            print(f"    인용: {', '.join(ans.citations)}")
        if ans.note:
            print(f"    참고: {ans.note}")

    gr = result.grade
    if gr is not None:
        mark = "O" if gr.correct else ("?" if gr.verdict == "판정 불가" else "X")
        print(f"\n[7] 정답 비교    [{mark}] {gr.verdict}")
        print(f"    LLM 정답  : {gr.llm_answer[:110]}")
        print(f"    실제 정답 : {gr.gold_display}")

    for stage, message in result.errors().items():
        print(f"\n[!] {stage} 실패: {message}")

    print(f"\n전체 {result.elapsed:.1f}초")


if __name__ == "__main__":
    main()
