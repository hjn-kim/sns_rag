#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
6. LLM 답변 생성

rerank.py 가 고른 청크 5개를 근거로 붙여 Gemini 에게 답을 만들게 한다.

이 데이터에 맞춰 프롬프트에서 신경 쓴 것 세 가지:

  1. 근거 밖으로 나가지 말 것
     채팅 로그에 없는 날짜나 이름을 지어내면 데모가 무너진다. 근거가 모자라면
     모자라다고 답하게 하고, 그 사실을 enough 필드로 따로 받는다. 화면에서
     "근거 부족" 배지를 띄우려면 답변 본문을 파싱하는 것보다 이쪽이 안전하다.

  2. 번역 오류를 감안할 것
     원문이 기계 번역이라 같은 단어가 청크마다 다르게 옮겨져 있다. 예를 들어
     Ministry of Power 가 한 청크에서는 "전력부", 다른 청크에서는 "에너지부"로,
     Judiciary 가 "사법 MDA" 와 "법조법인"(law firm 오역)으로 나온다. 여러 청크에
     같은 대상이 다른 이름으로 나올 수 있다고 미리 알려주고, 한 청크만 믿지 말고
     교차 확인하게 한다.

  3. 인용을 청크 id 로 달 것
     s0004#2 같은 id 를 그대로 인용하게 해서 화면에서 근거 청크로 되짚을 수 있게
     한다. citations 를 따로 받아 두면 답변 본문과 별개로 표시할 수 있다.

  4. 답은 검색 문서의 언어로 쓸 것
     색인이 언어별로 나뉘어 있어 근거 청크의 언어가 매번 다르다. 그 언어를
     doc_language 로 받아 프롬프트에 끼우고, 답변도 기본적으로 그 언어로 쓴다.
     러시아어 색인을 한국어로 물으면 답도 러시아어로 나온다. 근거에 적힌
     표기를 그대로 쓰는 편이 번역을 한 겹 더 거치는 것보다 정확하고,
     7단계 정답 비교도 같은 언어의 정답 후보와 맞대면 된다.
     질문 언어로 받고 싶으면 answer_language 를 따로 넘긴다.

단독 실행:
    python src/answer.py --lang ru "두 페이지만 있는 MDA는?"                  # 답도 러시아어
    python src/answer.py --lang ru --answer-lang 한국어 "두 페이지만 있는 MDA는?"
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

from search import Hit, LANGUAGES  # noqa: E402

# 답변 생성은 근거를 읽고 종합하는 일이라 재작성/리랭킹보다 부담이 크지만,
# 청크 5개(2500토큰)를 읽고 몇 문장을 쓰는 정도는 flash-lite 로 충분하다.
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

# 근거 청크가 어느 언어 색인에서 나왔는지. 호출하는 쪽에서 LANGUAGES[lang] 로 넘긴다.
DEFAULT_DOC_LANGUAGE = LANGUAGES["ko"]

# 답을 쓸 언어. 안 주면 검색 문서(doc_language)와 같은 언어로 쓴다.
# "질문과 같은 언어" 처럼 이름 대신 조건을 써 넣어도 그대로 프롬프트에 들어간다.
DEFAULT_ANSWER_LANGUAGE = None

SYSTEM_PROMPT = """당신은 WhatsApp 그룹 채팅 로그를 근거로 질문에 답하는 도우미입니다.
근거 청크는 {doc_language}로 되어 있고, 답변은 {answer_language}로 씁니다.

이 방에서 쓰는 용어:
- MDA = Ministries, Departments and Agencies (나이지리아 정부 부처/기관)
- PBI = Power BI
- CP = capital project (자본 사업)
- Resagratia = 이 프로젝트를 운영하는 데이터 분석 커뮤니티

지켜야 할 것:

1. **주어진 근거 안에서만 답하세요.** 근거에 없는 날짜·이름·수치를 채워 넣지
   마세요. 답을 특정할 수 없으면 enough 를 false 로 두고, "관련 내용의 부재로 답변할 수 없음"을 출력하세요. 절대로 지어내거나 모르는것을 답변하지 마세요.

2. **근거는 {doc_language}로 기계 번역된 로그입니다.** 같은 대상이 청크마다
   다른 이름으로 나올 수 있습니다(예: 같은 부처가 "전력부"와 "에너지부"로).
   한 청크만 믿지 말고 여러 청크를 교차 확인해서 판단하세요. 같은 문장이
   여러 번 반복되는 것도 번역 결함이니 무시하세요.

3. **인용을 다세요.** 답의 근거가 된 청크 id(예: s0004#2)를 citations 에
   빠짐없이 넣으세요. 쓰지 않은 청크는 넣지 마세요.

4. **답변은 전부 {answer_language}로 쓰세요.** 질문이 다른 언어로 들어와도
   답변 언어는 바꾸지 마세요. 부처·기관·인물·직책 같은 고유명사는 근거 청크에
   적힌 {answer_language} 표기를 그대로 쓰고, 다른 언어의 표기를 섞지 마세요.
   근거에 없는 표기로 옮겨 적지 마세요.

5. 답변은 두세 문장으로 짧게 쓰세요. 대화 속 발언을 인용할 때는 발언자
   이름을 함께 적으면 좋습니다."""


def build_system_prompt(doc_language: str | None = None,
                        answer_language: str | None = None) -> str:
    """
    근거 언어와 답변 언어를 끼운 시스템 프롬프트.

    doc_language     근거 청크가 나온 색인의 언어. LANGUAGES[lang] 을 넘긴다.
    answer_language  답을 쓸 언어. 안 주면 doc_language 를 그대로 쓴다.
                     "한국어" 처럼 이름을 주면 검색 문서가 무슨 언어든 그 언어로 쓴다.

    local_llm.py 도 이 함수를 쓴다. 프롬프트가 한 곳에만 있어야 두 backend 의
    답변이 갈리지 않는다.
    """
    doc = doc_language or DEFAULT_DOC_LANGUAGE
    return (SYSTEM_PROMPT
            .replace("{doc_language}", doc)
            .replace("{answer_language}",
                     answer_language or DEFAULT_ANSWER_LANGUAGE or doc))


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "질문에 대한 답. 고유명사까지 전부 근거 문서와 같은 언어로 쓴 두세 문장.",
        },
        "enough": {
            "type": "boolean",
            "description": "주어진 근거만으로 답을 특정할 수 있으면 true.",
        },
        "citations": {
            "type": "array",
            "items": {"type": "string"},
            "description": "답의 근거가 된 청크 id 목록 (예: s0004#2).",
        },
        "note": {
            "type": "string",
            "description": "근거가 부족하거나 번역이 흔들려 판단이 갈린 지점. 없으면 빈 문자열.",
        },
    },
    "required": ["answer", "enough", "citations", "note"],
}


@dataclass
class AnswerResult:
    """6단계 결과. 실패해도 예외를 던지지 않고 error 에 담아 돌려준다."""

    question: str
    answer: str = ""
    enough: bool = False
    citations: list[str] = field(default_factory=list)
    note: str = ""
    model: str = ""
    elapsed: float = 0.0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def _client():
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY 가 비어 있습니다. 프로젝트 루트의 .env 에 넣어 주세요."
        )
    from google import genai
    return genai.Client(api_key=key)


def build_context(chunks: list[Hit]) -> str:
    """
    청크들을 근거 블록으로 만든다.

    id 와 날짜를 머리에 달아준다. "언제" 를 묻는 질문에서 본문에 날짜가 안 적혀
    있어도 대화 날짜로 답할 수 있는 경우가 있어서다.
    """
    blocks = []
    for hit in chunks:
        who = ", ".join(hit.participants[:5])
        blocks.append(
            f"[{hit.key}] {hit.started_at[:10]} · 참여자 {who}\n"
            f"{' '.join(hit.text.split())}"
        )
    return "\n\n---\n\n".join(blocks)


def generate_answer(question: str, chunks: list[Hit],
                    model: str | None = None,
                    doc_language: str | None = None,
                    answer_language: str | None = None) -> AnswerResult:
    """
    청크를 근거로 답을 만든다.

    doc_language     근거 청크가 나온 색인의 언어 (기본: 한국어).
    answer_language  답을 쓸 언어 (기본: doc_language 와 같은 언어).
    """
    question = (question or "").strip()
    if not question:
        return AnswerResult(question="", error="질문이 비어 있습니다.")
    if not chunks:
        return AnswerResult(question=question,
                            error="근거로 쓸 청크가 없습니다.")

    model = model or DEFAULT_MODEL
    started = time.time()

    try:
        from google.genai import types

        client = _client()
        response = client.models.generate_content(
            model=model,
            contents=(f"질문: {question}\n\n"
                      f"근거 청크 {len(chunks)}개:\n\n{build_context(chunks)}"),
            config=types.GenerateContentConfig(
                system_instruction=build_system_prompt(doc_language,
                                                       answer_language),
                response_mime_type="application/json",
                response_schema=RESPONSE_SCHEMA,
                max_output_tokens=2048,
                # 근거에서 답을 뽑는 일이라 매번 흔들릴 이유가 없다.
                temperature=0.2,
            ),
        )
        data = json.loads(response.text)
    except Exception as exc:  # noqa: BLE001
        return AnswerResult(question=question, model=model,
                            elapsed=time.time() - started,
                            error=f"{type(exc).__name__}: {exc}")

    # 모델이 근거에 없는 id 를 지어낼 수 있다. 실제로 넘긴 청크만 남긴다.
    given = {hit.key for hit in chunks}
    citations = [c for c in (data.get("citations") or []) if c in given]

    return AnswerResult(
        question=question,
        answer=(data.get("answer") or "").strip(),
        enough=bool(data.get("enough")),
        citations=citations,
        note=(data.get("note") or "").strip(),
        model=model,
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

    parser = argparse.ArgumentParser(description="검색 결과를 근거로 답을 만든다.")
    parser.add_argument("question", nargs="*")
    parser.add_argument("--lang", default="ko", choices=list(LANGUAGES),
                        help="검색할 색인(근거 문서)의 언어 (기본: ko)")
    parser.add_argument("--answer-lang", default=None,
                        help="답변을 쓸 언어 이름. 예: 한국어 / 영어 "
                             "(기본: --lang 과 같은 언어)")
    args = parser.parse_args()

    from comparison import search_per_query
    from multi_query import all_queries, rewrite_query
    from rerank import rerank

    question = " ".join(args.question) or "두 페이지만 있는 MDA는?"

    rw = rewrite_query(question, language=LANGUAGES[args.lang])
    queries = all_queries(rw) if rw.ok else [question]
    ms = search_per_query(queries, lang=args.lang)
    rr = rerank(question, ms)
    result = generate_answer(question, rr.selected,
                             doc_language=LANGUAGES[args.lang],
                             answer_language=args.answer_lang)

    print(f"\n근거 {len(rr.selected)}개: "
          f"{', '.join(h.key for h in rr.selected)}")
    if not result.ok:
        sys.exit(f"\n[실패] {result.error}")

    print(f"\n답변 ({result.elapsed:.1f}초, 근거 존재: "
          f"{'예' if result.enough else '아니오'})")
    print(f"  {result.answer}")
    if result.citations:
        print(f"\n인용: {', '.join(result.citations)}")
    if result.note:
        print(f"참고: {result.note}")


if __name__ == "__main__":
    main()
