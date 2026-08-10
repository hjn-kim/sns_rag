#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
1. 질의 재작성 + 2. 질의 확장  (Gemini Flash-Lite)

질문 하나를 넣으면 한 번의 호출로 둘을 받는다.

    재작성  검색에 맞게 다듬은 질의 하나. 항상 평서문으로 만든다.
    확장    다른 각도로 바꾼 질의 3개.

왜 재작성이 필요한가 (이 데이터 기준):

  1. 질문으로 검색하면 답이 아니라 "같은 질문을 한 메시지"가 걸린다.
     색인 대상이 채팅이라 참여자들이 이미 같은 걸 물어봤기 때문이다.
     그래서 질문형을 서술형(답변처럼 생긴 문장)으로 바꾼다. = HyDE
  2. 어휘가 다르다. 질문의 "부처"는 대화 속 "MDA" 와 임베딩 공간에서 멀다.
     약어를 풀어 쓰거나 대화에서 실제로 쓰는 표현으로 바꾼다.
  3. 색인이 언어별로 나뉘어 있다. 한국어 질문으로 영어 색인을 뒤질 때는
     대상 언어로 옮긴 질의를 함께 던지는 편이 대체로 정확하다.

Streamlit 에 의존하지 않는다. 캐싱은 호출하는 쪽에서 한다.

    from multi_query import rewrite_query
    result = rewrite_query("다음 워크숍이 언제야?", language="한국어")
    result.rewritten      # str
    result.expansions     # list[str] (3개)
    result.error          # 실패했으면 사람이 읽을 메시지, 아니면 None

단독 실행하면 바로 확인할 수 있다:

    python src/multi_query.py "다음 워크숍이 언제야?"
    python src/multi_query.py --list-models
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# .env 의 GEMINI_API_KEY 를 읽는다. python-dotenv 가 없어도 죽지 않는다
# (환경변수로 직접 넣는 경우도 있으므로).
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:  # pragma: no cover
    pass

# Flash-Lite 가 이 작업에 가장 싸다. 재작성은 짧은 문장 몇 개를 만드는 일이라
# 상위 모델을 쓸 이유가 없다. 단가(1M 토큰당, 입력/출력):
#
#   gemini-2.0-flash-lite   $0.075 / $0.30   신규 사용자 사용 불가 (404)
#   gemini-2.5-flash-lite   $0.10  / $0.40   신규 사용자 사용 불가 (404)
#   gemini-3.1-flash-lite   $0.25  / $1.50   사용 가능   <- 기본값
#   gemini-3.5-flash-lite   $0.30  / $2.50   사용 가능
#   gemini-2.5-flash        $0.30  / $2.50
#
# 2.0/2.5-lite 가 더 싸지만 "no longer available to new users" 로 막혀 있어
# 실제로 쓸 수 있는 것 중에서는 3.1-flash-lite 가 가장 싸다. 재작성 한 번이
# 입력 200 · 출력 150토큰쯤이니 질의 1건에 0.03센트, 100건에 3센트 수준이다.
#
# gemini-flash-lite-latest 는 별칭이라 가리키는 모델이 바뀌면 단가도 바뀐다.
# 비용을 고정하려면 지금처럼 버전을 명시한 이름을 쓴다.
#
# 다른 모델을 쓰려면 .env 에 GEMINI_MODEL 을 넣는다.
# 쓸 수 있는 이름은 `python src/multi_query.py --list-models` 로 확인한다.
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

N_EXPANSIONS = 3

SYSTEM_PROMPT = f"""당신은 RAG 검색을 위한 질의->평서문 변환기기 및 확장 모듈입니다.
사용자의 질문을 검색에 적합한 형태의 평서문으로 재작성하고, 동일한 검색 의도를 서로 다른 표현과 관점으로 나타낸 확장 평서문을을 생성하세요.

두 가지를 생성하세요.

[재작성] 검색에 사용할 평서문문 1개

원 질문의 핵심 검색 의도를 유지하면서, 검색 문서의 답변 구간과 의미적으로 가까운 형태로 재작성하세요.

**재작성 평서문은 예외 없이 의문 표현이 아닌 . 로 끝나는 평서문이여야 한다**

사실조회, 정의, 비교, 원인, 절차, 조건, 일정, 인물, 수치 등 질문 유형에 맞게 자연스럽게 변환하세요.
약어, 대명사, 지나치게 일반적인 표현은 원 질문에서 확인 가능한 범위 내에서 더 명확한 표현으로 바꾸세요.
검색 대상 문서의 언어나 용어 체계가 주어진 경우 해당 표현을 우선 사용하세요.
원 질문에 없는 사실, 인물, 기관, 날짜, 수치, 제품명, 사건명을 임의로 추가하지 마세요.

[확장] 서로 다른 검색 관점의 질의 {N_EXPANSIONS}개

1번: 대상 언어({{language}})로 자연스럽게 변환한 질의. 색인 언어와 어휘를 맞추되 원래 의미를 유지하세요.
2번: 검색에 핵심적인 개념과 키워드만 남긴 짧은 질의.
3번: 질문자가 찾고자 하는 동일한 정보를 다른 표현이나 다른 검색 초점으로 나타낸 질의.
추가 확장이 필요한 경우 동의어, 공식 명칭, 관련 표현, 조건 표현 등 검색 Recall을 높일 수 있는 서로 다른 관점을 사용하세요.
각 확장 질의는 의미와 표현이 지나치게 겹치지 않도록 하세요.
원 질문의 범위를 벗어나는 새로운 주제로 확장하지 마세요.
원 질문에 없는 사실을 지어내지 마세요. 날짜, 이름, 수치 등을 임의로 채우지 마세요."""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "rewritten": {
            "type": "string",
            "description": "검색에 쓸 재작성 질의. 반드시 평서문(서술문)이어야 하며 물음표를 쓰지 않는다.",
        },
        "changed": {
            "type": "boolean",
            "description": "원 질문에서 실제로 바뀌었으면 true.",
        },
        "reason": {
            "type": "string",
            "description": "무엇을 왜 바꿨는지 한 문장. 안 바꿨으면 그 이유.",
        },
        "expansions": {
            "type": "array",
            "items": {"type": "string"},
            "description": f"서로 다른 각도의 확장 질의 {N_EXPANSIONS}개.",
        },
    },
    "required": ["rewritten", "changed", "reason", "expansions"],
}


@dataclass
class RewriteResult:
    """재작성 결과. 실패해도 예외를 던지지 않고 error 에 담아 돌려준다."""

    question: str
    rewritten: str
    expansions: list[str] = field(default_factory=list)
    changed: bool = False
    reason: str = ""
    model: str = ""
    elapsed: float = 0.0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def _fallback(question: str, error: str) -> RewriteResult:
    """호출이 실패하면 원 질문만으로 파이프라인이 굴러가게 한다."""
    return RewriteResult(
        question=question,
        rewritten=question,
        expansions=[],
        changed=False,
        reason="",
        error=error,
    )


def _client():
    """genai 클라이언트를 만든다. 키가 없으면 사람이 읽을 메시지로 알린다."""
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY 가 비어 있습니다. 프로젝트 루트의 .env 에 넣어 주세요.\n"
            "  GEMINI_API_KEY=your-key-here"
        )
    try:
        from google import genai
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "google-genai 가 설치되어 있지 않습니다.  pip install google-genai"
        ) from exc
    return genai.Client(api_key=key)


def rewrite_query(question: str, language: str = "영어",
                  model: str | None = None) -> RewriteResult:
    """
    질문 하나를 재작성 + 확장한다. 호출 한 번으로 둘 다 받는다.

    language 는 검색할 색인의 언어("한국어", "영어" ...). 확장 1번을 그 언어로 만든다.
    """
    question = (question or "").strip()
    if not question:
        return _fallback("", "질문이 비어 있습니다.")

    model = model or DEFAULT_MODEL
    started = time.time()

    try:
        from google.genai import types

        client = _client()
        response = client.models.generate_content(
            model=model,
            contents=f"질문: {question}",
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT.replace("{language}", language),
                response_mime_type="application/json",
                response_schema=RESPONSE_SCHEMA,
                # 재작성은 짧다. 길게 쓸 이유가 없다.
                max_output_tokens=1024,
                temperature=0.7,
            ),
        )
        data = json.loads(response.text)
    except Exception as exc:  # noqa: BLE001 - 키/네트워크/스키마 등 모든 실패를 담아 돌려준다
        return _fallback(question, f"{type(exc).__name__}: {exc}")

    rewritten = (data.get("rewritten") or "").strip() or question
    # 재작성은 무조건 평서문이다. 모델이 물음표를 남기면 여기서 떼어낸다.
    rewritten = rewritten.rstrip("?？").strip() or question
    expansions = [str(x).strip() for x in (data.get("expansions") or []) if str(x).strip()]
    # 모델이 개수를 안 맞출 수 있다. 넘치면 자르고, 모자라면 그대로 둔다
    # (없는 질의를 지어내 채우면 검색만 흐려진다).
    expansions = expansions[:N_EXPANSIONS]

    return RewriteResult(
        question=question,
        rewritten=rewritten,
        expansions=expansions,
        changed=bool(data.get("changed")) and rewritten != question,
        reason=(data.get("reason") or "").strip(),
        model=model,
        elapsed=time.time() - started,
    )


def all_queries(result: RewriteResult) -> list[str]:
    """검색에 실제로 던질 질의 목록. 중복은 뺀다."""
    out: list[str] = []
    for q in [result.rewritten, *result.expansions, result.question]:
        q = (q or "").strip()
        if q and q not in out:
            out.append(q)
    return out


def list_models() -> list[str]:
    """쓸 수 있는 모델 이름을 확인할 때 (모델명이 바뀌었을 때 유용)."""
    client = _client()
    return [m.name for m in client.models.list()]


# --------------------------------------------------------------------------
# 단독 실행
# --------------------------------------------------------------------------

def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    args = sys.argv[1:]
    if args and args[0] == "--list-models":
        try:
            for name in list_models():
                print(" ", name)
        except Exception as exc:  # noqa: BLE001
            sys.exit(f"모델 목록을 가져오지 못했습니다: {exc}")
        return

    question = " ".join(args) or "다음 워크숍이 언제야?"
    result = rewrite_query(question, language="영어")

    print(f"모델   : {result.model or DEFAULT_MODEL}")
    print(f"원 질문 : {result.question}")
    if not result.ok:
        sys.exit(f"\n[실패] {result.error}")

    print(f"\n[재작성] {'바뀜' if result.changed else '변경 없음'} "
          f"({result.elapsed:.1f}초)")
    print(f"  {result.rewritten}")
    if result.reason:
        print(f"  근거: {result.reason}")

    print(f"\n[확장] {len(result.expansions)}개")
    for i, q in enumerate(result.expansions, 1):
        print(f"  {i}. {q}")

    print(f"\n검색에 던질 질의 {len(all_queries(result))}개")


if __name__ == "__main__":
    main()
