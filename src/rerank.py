#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
4. 리랭킹 + 5. 최종 청크 선정

comparison.py 가 추린 후보 20개 안팎을 다시 줄 세우고, 그중 몇 개만 골라
answer.py 로 넘긴다.

왜 리랭킹이 필요한가:
  임베딩 검색은 질의와 청크를 각각 따로 벡터로 만들어 비교한다(bi-encoder).
  빠르지만 둘을 같이 읽고 판단하지는 못한다. 그래서 "워크숍 얘기를 많이 하는
  청크"와 "워크숍 일정이 실제로 적힌 청크"를 잘 못 가른다. 후보가 20개로
  줄어든 다음에는 하나씩 질문과 나란히 놓고 따져볼 여유가 있다.

리랭커로 무엇을 쓰나:
  Gemini 를 쓴다. 크로스인코더(bge-reranker-v2-m3 등)가 정석이지만 이 프로젝트에는
  안 맞는다. 568M 짜리 모델을 하나 더 올려야 하고(2.3GB), CPU 에서 후보 20개 x
  500토큰을 재점수하면 10~30초가 걸린다. Gemini 는 후보 전체가 1만 토큰 남짓이라
  한 번 호출로 2~3초에 끝나고, 6개 언어를 그대로 처리하며, 이미 붙어 있다.

  API 없이 돌려야 하면 method="rrf" 로 바꾼다. 질의별 등수만 가지고 계산하므로
  호출이 없다. LLM 호출이 실패해도 자동으로 이쪽으로 떨어진다.

RRF (Reciprocal Rank Fusion):
  질의별 등수의 역수를 더한다.  score = sum over q of 1 / (K + rank_q)
  여러 질의가 공통으로 높게 뽑은 청크가 위로 온다. K=60 은 관례값으로, 1~2등과
  9~10등의 격차를 지나치게 벌리지 않으려고 넣는 완충값이다.
  점수 크기가 아니라 등수만 보므로 질의별 점수 분포가 달라도 안전하다.

최종 몇 개를 고르나 (FINAL_TOP_N = 5):
  이 데이터에서 실제로 재본 값들이다.
    - 정답 청크는 보통 1~2위에 온다 (Q2 1위, Q3 1위, Q1 1·2위)
    - 그런데 정답이 한 청크에 다 들어있지 않다. Q1 은 s0004#1 에 "사법 MDA는
      두 페이지", s0004#2 에 "0개의 자본 프로젝트" 가 나뉘어 있고, 1위 청크에는
      오역된 "법조법인" 만 있어서 2위까지 봐야 "사법부" 라는 답이 나온다.
    - 정답 대화의 인접 청크가 3~4개까지 상위에 붙어 온다 (s0004#0~#3)
  그래서 3개는 위험하고 5개면 정답 대화의 앞뒤가 대체로 들어온다. 500토큰 x 5 =
  2500토큰이라 LLM 입력으로도 가볍다. 10개까지 늘리면 다른 대화의 무관한 청크가
  섞이기 시작해서 오히려 답이 흐려진다.

단독 실행:
    python src/rerank.py --lang ko "두 페이지만 있는 MDA는?"
    python src/rerank.py --lang ko --method rrf "..."     # API 호출 없이
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from comparison import MultiSearch, search_per_query  # noqa: E402
from search import Hit, LANGUAGES  # noqa: E402

# multi_query.py 와 같은 모델을 쓴다. 리랭킹도 짧은 판단을 여러 번 하는 일이라
# 상위 모델을 쓸 이유가 적다.
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

# 최종적으로 answer.py 에 넘길 청크 수. 근거는 위 docstring 참고.
FINAL_TOP_N = 5

# RRF 완충 상수. 정보검색에서 관례적으로 쓰는 값.
RRF_K = 60

# 후보 하나를 프롬프트에 넣을 때의 최대 길이(문자). 500토큰 청크가 한글 기준
# 1000자 안팎이라 여유를 조금 준 값이다.
MAX_CHARS_PER_CANDIDATE = 1400

SYSTEM_PROMPT = """당신은 RAG 파이프라인의 리랭커입니다.

WhatsApp 그룹 채팅 로그에서 뽑은 후보 대목들을 받습니다. 나이지리아 연방정부
예산 문서를 Power BI 로 시각화하는 프로젝트 참여자들의 대화이며, 기계 번역을
거쳐서 같은 단어가 다르게 번역되거나 문장이 반복되는 곳이 있습니다.

각 후보가 **질문에 대한 답을 실제로 담고 있는지** 0~10 으로 매기세요.

  9~10  질문에 대한 답이 그 안에 직접 적혀 있다
  6~8   답의 일부가 있거나, 답을 특정할 결정적 단서가 있다
  3~5   주제는 같지만 답은 없다
  0~2   무관하다

중요한 판단 기준:

  - **주제가 겹치는 것과 답이 있는 것은 다릅니다.** 질문의 키워드를 여러 번
    말하기만 하고 정작 묻는 값(날짜·장소·이름·수치)이 없으면 낮게 주세요.
    반대로 키워드가 하나도 없어도 묻는 값이 적혀 있으면 높게 주세요.
  - 번역 오류로 단어가 이상해도 내용이 맞으면 인정하세요.
  - 같은 문장이 반복되는 것은 번역 결함입니다. 감점 사유가 아닙니다.

모든 후보에 대해 빠짐없이 점수를 매기세요. reason 은 한 문장으로 짧게."""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "rankings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "후보 id (예: s0004#2)"},
                    "score": {"type": "integer", "description": "0~10"},
                    "reason": {"type": "string", "description": "한 문장"},
                },
                "required": ["id", "score", "reason"],
            },
        },
    },
    "required": ["rankings"],
}


@dataclass
class RankedHit:
    """리랭킹을 거친 후보 하나."""

    hit: Hit
    rank_before: int             # 리랭킹 전 등수 (dense 점수순)
    rank_after: int = 0          # 리랭킹 후 등수
    llm_score: int | None = None  # 0~10, rrf 방식이면 None
    reason: str = ""
    rrf_score: float = 0.0

    @property
    def moved(self) -> int:
        """등수가 몇 칸 올라갔는지. 양수면 상승."""
        return self.rank_before - self.rank_after


@dataclass
class RerankResult:
    """4단계 + 5단계 결과."""

    question: str
    method: str                              # "llm" | "rrf"
    ranked: list[RankedHit] = field(default_factory=list)   # 후보 전체, 재정렬됨
    selected: list[Hit] = field(default_factory=list)       # 최종 선정 (top_n)
    model: str = ""
    elapsed: float = 0.0
    error: str | None = None                 # LLM 실패 시 사람이 읽을 메시지

    @property
    def ok(self) -> bool:
        return self.error is None


# --------------------------------------------------------------------------
# RRF (호출 없음)
# --------------------------------------------------------------------------

def rrf_scores(result: MultiSearch) -> dict[str, float]:
    """
    청크별 RRF 점수. 질의별 등수의 역수를 더한다.

    여러 질의가 공통으로 높게 뽑은 청크가 위로 온다. 점수 크기가 아니라 등수만
    보므로 질의마다 점수 분포가 달라도 안전하다.
    """
    scores: dict[str, float] = {}
    for qi in range(len(result.queries)):
        for rank, hit in enumerate(result.per_query[qi], 1):
            scores[hit.key] = scores.get(hit.key, 0.0) + 1.0 / (RRF_K + rank)
    return scores


# --------------------------------------------------------------------------
# Gemini 리랭킹
# --------------------------------------------------------------------------

def _client():
    """multi_query.py 와 같은 방식으로 클라이언트를 만든다."""
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY 가 비어 있습니다. 프로젝트 루트의 .env 에 넣어 주세요."
        )
    from google import genai
    return genai.Client(api_key=key)


def _build_prompt(question: str, candidates: list[Hit]) -> str:
    blocks = []
    for hit in candidates:
        text = " ".join(hit.text.split())[:MAX_CHARS_PER_CANDIDATE]
        blocks.append(f"[{hit.key}] ({hit.started_at[:10]})\n{text}")
    return f"질문: {question}\n\n후보 {len(candidates)}개:\n\n" + "\n\n".join(blocks)


def _llm_scores(question: str, candidates: list[Hit],
                model: str) -> dict[str, tuple[int, str]]:
    """{청크 id: (점수, 근거)}. 실패하면 예외를 올린다."""
    from google.genai import types

    client = _client()
    response = client.models.generate_content(
        model=model,
        contents=_build_prompt(question, candidates),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
            # 후보 20개 x (점수 + 한 문장) 이면 이 정도면 넉넉하다.
            max_output_tokens=4096,
            # 순위 매기기는 매번 같은 답이 나오는 편이 낫다.
            temperature=0.0,
        ),
    )
    data = json.loads(response.text)

    out: dict[str, tuple[int, str]] = {}
    for row in data.get("rankings") or []:
        key = str(row.get("id", "")).strip()
        if not key:
            continue
        score = max(0, min(10, int(row.get("score", 0))))
        out[key] = (score, str(row.get("reason", "")).strip())
    return out


# --------------------------------------------------------------------------
# 4단계 + 5단계
# --------------------------------------------------------------------------

def rerank(question: str, result: MultiSearch, top_n: int = FINAL_TOP_N,
           method: str = "llm", model: str | None = None) -> RerankResult:
    """
    후보를 다시 줄 세우고 상위 top_n 을 고른다.

    method="llm"  Gemini 가 후보마다 0~10 을 매긴다. 동점이면 RRF, 그다음 dense
                  점수 순으로 가른다. 호출이 실패하면 rrf 로 자동 강등된다.
    method="rrf"  호출 없이 질의별 등수만으로 정렬한다.
    """
    started = time.time()
    model = model or DEFAULT_MODEL

    candidates = result.pooled
    if not candidates:
        return RerankResult(question=question, method=method,
                            elapsed=time.time() - started)

    rrf = rrf_scores(result)
    ranked = [
        RankedHit(hit=hit, rank_before=i, rrf_score=rrf.get(hit.key, 0.0))
        for i, hit in enumerate(candidates, 1)
    ]

    error = None
    used = method

    if method == "llm":
        try:
            scores = _llm_scores(question, candidates, model)
            if not scores:
                raise ValueError("리랭킹 응답이 비어 있습니다.")
            for item in ranked:
                score, reason = scores.get(item.hit.key, (0, ""))
                item.llm_score = score
                item.reason = reason
        except Exception as exc:  # noqa: BLE001 - 키/네트워크/스키마 모두 여기로
            error = f"{type(exc).__name__}: {exc}"
            used = "rrf"
            for item in ranked:
                item.llm_score = None

    # 정렬 기준: LLM 점수 -> RRF -> dense 점수.
    # rrf 방식이면 첫 항목이 모두 0 이라 자연스럽게 RRF 기준이 된다.
    ranked.sort(key=lambda x: (x.llm_score or 0, x.rrf_score, x.hit.score),
                reverse=True)
    for rank, item in enumerate(ranked, 1):
        item.rank_after = rank

    return RerankResult(
        question=question,
        method=used,
        ranked=ranked,
        selected=[item.hit for item in ranked[:top_n]],
        model=model if used == "llm" else "",
        elapsed=time.time() - started,
        error=error,
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
        description="검색 후보를 리랭킹하고 최종 청크를 고른다.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("question", nargs="*")
    parser.add_argument("--lang", default="ko", choices=list(LANGUAGES))
    parser.add_argument("--method", default="llm", choices=["llm", "rrf"])
    parser.add_argument("--top-n", type=int, default=FINAL_TOP_N,
                        help=f"최종 선정 청크 수 (기본: {FINAL_TOP_N})")
    parser.add_argument("--rewrite", action="store_true",
                        help="Gemini 재작성/확장 질의까지 함께 검색")
    args = parser.parse_args()

    question = " ".join(args.question) or "두 페이지만 있는 MDA는?"

    queries = [question]
    if args.rewrite:
        from multi_query import all_queries, rewrite_query
        rw = rewrite_query(question, language=LANGUAGES[args.lang])
        if rw.ok:
            queries = all_queries(rw)

    ms = search_per_query(queries, lang=args.lang)
    rr = rerank(question, ms, top_n=args.top_n, method=args.method)

    print(f"\n후보 {len(ms.pooled)}개 -> 리랭킹({rr.method}) {rr.elapsed:.1f}초")
    if rr.error:
        print(f"[!] LLM 리랭킹 실패, RRF 로 대체: {rr.error}")

    print(f"\n{'후':>3} {'전':>3} {'이동':>4} {'LLM':>4} {'RRF':>6} {'dense':>6}  청크")
    for item in rr.ranked:
        mark = "  <= 선정" if item.rank_after <= args.top_n else ""
        moved = f"{item.moved:+d}" if item.moved else "-"
        llm = f"{item.llm_score}" if item.llm_score is not None else "-"
        print(f"{item.rank_after:>3} {item.rank_before:>3} {moved:>4} {llm:>4} "
              f"{item.rrf_score:.4f} {item.hit.score:.4f}  {item.hit.key}{mark}")
        if item.reason:
            print(f"                                    {item.reason[:90]}")


if __name__ == "__main__":
    main()
