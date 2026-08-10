#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen3-8B 로 Gemini 를 대신한다 (1·2 재작성/확장, 6 답변 생성)

API 키 없이 돌리기 위한 것이다. 결과 자료구조는 multi_query.py / answer.py 것을
그대로 쓰므로 화면과 파이프라인은 어느 쪽으로 돌렸는지 몰라도 된다.

    multi_query.rewrite_query()  <->  rewrite_query_local()
    answer.generate_answer()     <->  generate_answer_local()

왜 Qwen3-8B 인가:

  - 임베딩 모델(Harrier)이 이미 Qwen3 아키텍처다. 토크나이저와 다국어 거동이
    같은 계열이라 6개 언어(한국어·영어·중국어·베트남어·필리핀어·러시아어)에서
    일관되게 움직인다.
  - bf16 으로 약 16GB. Harrier 1.2GB + bge-reranker 1.1GB 를 더해도 24GB GPU 에
    들어간다.
  - 재작성은 짧은 문장 몇 개를 만드는 쉬운 일이고, 답변 생성은 청크 5개
    (2500토큰)를 읽는 일이다. 8B 면 둘 다 감당한다.

JSON 을 어떻게 보장하나 (여기가 제일 까다롭다):

  Gemini 는 response_schema 로 스키마에 맞는 JSON 을 **보장**한다. 로컬 모델은
  그 보장이 없다. 대개 맞는 JSON 이 나오지만 가끔 앞뒤에 설명이 붙거나 코드펜스로
  감싸져 나오고, 그러면 json.loads 가 통째로 실패한다.

  여기서는 세 겹으로 막는다.
    1. 시스템 프롬프트 끝에 스키마와 "JSON 만 출력" 지시를 붙인다
    2. 코드펜스/앞뒤 군더더기를 벗기고 첫 { 부터 짝이 맞는 } 까지 잘라낸다
    3. 그래도 실패하면 온도를 0 으로 낮춰 한 번 더 시도한다

  더 확실한 방법은 vLLM 이나 outlines 의 guided decoding 으로 디코딩 단계에서
  문법을 강제하는 것이다. 그러려면 별도 서버(vLLM)나 추가 패키지가 필요해서
  여기서는 넣지 않았다. 파싱 실패가 잦으면 그쪽으로 옮기는 게 맞다.

Qwen3 의 thinking 모드는 끈다:
  기본값이 켜짐이라 <think>...</think> 를 먼저 뱉는다. JSON 파싱이 깨지고
  토큰도 몇 배로 늘어난다. apply_chat_template(enable_thinking=False) 로 끈다.

단독 실행:
    python src/local_llm.py "다음 워크숍이 언제야?"          # 재작성만
    python src/local_llm.py --lang ko --answer "질문"        # 검색까지 붙여 답변
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from answer import (  # noqa: E402
    RESPONSE_SCHEMA as ANSWER_SCHEMA,
    SYSTEM_PROMPT as ANSWER_SYSTEM,
    AnswerResult,
    build_context,
)
from multi_query import (  # noqa: E402
    N_EXPANSIONS,
    RESPONSE_SCHEMA as REWRITE_SCHEMA,
    SYSTEM_PROMPT as REWRITE_SYSTEM,
    RewriteResult,
)
from search import Hit, LANGUAGES  # noqa: E402

DEFAULT_MODEL = os.getenv("LOCAL_LLM_MODEL", "Qwen/Qwen3-8B")

# 재작성은 짧은 문장 몇 개, 답변은 두세 문장 + 인용이다. 넉넉히 잡아도 이 정도면
# 남는다. 크게 잡을수록 생성이 느려지기만 한다.
MAX_NEW_TOKENS_REWRITE = 512
MAX_NEW_TOKENS_ANSWER = 768


# --------------------------------------------------------------------------
# 모델
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_llm(name: str = DEFAULT_MODEL, device: str | None = None):
    """
    Qwen3-8B 를 올린다. 프로세스당 한 번만.

    16GB 짜리라 Streamlit 이 스크립트를 다시 돌 때마다 올리면 곧바로 OOM 이다.
    search.py / rerank_gpu.py 와 같은 이유로 lru_cache 를 건다.

    device_map="auto" 로 두면 GPU 에 안 들어갈 때 CPU 로 흘려보내며 버틴다.
    그 상태로도 돌긴 하지만 아주 느리므로, 실제로 어디 올라갔는지 확인할 것.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # transformers 4.56 부터 torch_dtype 이 dtype 으로 바뀌었다.
    import transformers
    try:
        tf_version = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    except ValueError:
        tf_version = (99, 99)
    dtype_key = "dtype" if tf_version >= (4, 56) else "torch_dtype"

    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        device_map="auto" if device.startswith("cuda") else None,
        **{dtype_key: "bfloat16" if device.startswith("cuda") else "float32"},
    )
    model.eval()
    return tokenizer, model, device


# --------------------------------------------------------------------------
# JSON 생성
# --------------------------------------------------------------------------

def _schema_hint(schema: dict) -> str:
    """스키마를 프롬프트에 붙일 짧은 안내로 바꾼다."""
    lines = []
    for key, spec in (schema.get("properties") or {}).items():
        kind = spec.get("type", "")
        desc = spec.get("description", "")
        if kind == "array":
            kind = f"array<{(spec.get('items') or {}).get('type', 'string')}>"
        lines.append(f'  "{key}": {kind}   // {desc}')
    return "{\n" + "\n".join(lines) + "\n}"


def _extract_json(text: str) -> dict:
    """
    모델 출력에서 JSON 객체만 꺼낸다.

    코드펜스로 감싸거나 앞뒤에 설명을 붙이는 경우가 있어서, 첫 '{' 부터 괄호
    짝이 맞는 '}' 까지를 잘라 쓴다. 문자열 안의 중괄호는 세지 않는다.
    """
    text = re.sub(r"^\s*```(?:json)?|```\s*$", "", (text or "").strip(),
                  flags=re.MULTILINE)
    # Qwen3 가 thinking 을 못 끈 채 돌면 <think> 가 앞에 붙는다.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()

    start = text.find("{")
    if start < 0:
        raise ValueError(f"JSON 을 찾지 못했습니다: {text[:200]}")

    depth, in_str, escaped = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError(f"JSON 괄호가 닫히지 않았습니다: {text[:200]}")


def generate_json(system: str, user: str, schema: dict,
                  max_new_tokens: int = 512, temperature: float = 0.7,
                  model_name: str = DEFAULT_MODEL,
                  device: str | None = None) -> dict:
    """
    JSON 하나를 받는다. 파싱에 실패하면 온도 0 으로 한 번 더 시도한다.

    재시도를 온도 0 으로 하는 이유: 형식이 깨지는 건 대개 생성이 흔들렸다는
    뜻이라, 같은 온도로 다시 굴리면 또 깨질 확률이 높다.
    """
    import torch

    tokenizer, model, _ = load_llm(model_name, device)

    system = (f"{system}\n\n"
              f"반드시 아래 형태의 JSON 하나만 출력하세요. 설명, 인사말, 코드펜스를\n"
              f"붙이지 마세요.\n\n{_schema_hint(schema)}")

    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]

    # Qwen3 는 thinking 모드가 기본이다. 켜두면 <think> 를 먼저 뱉어 JSON 이
    # 깨지고 토큰도 몇 배가 된다.
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except TypeError:       # enable_thinking 을 모르는 토크나이저
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    last_error: Exception | None = None
    for attempt, temp in enumerate((temperature, 0.0)):
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temp > 0,
                temperature=temp if temp > 0 else None,
                top_p=0.9 if temp > 0 else None,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True)
        try:
            return _extract_json(text)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == 0:
                continue
    raise ValueError(f"JSON 파싱 실패: {last_error}")


# --------------------------------------------------------------------------
# 1, 2 단계 : 재작성 + 확장
# --------------------------------------------------------------------------

def rewrite_query_local(question: str, language: str = "영어",
                        model: str | None = None,
                        device: str | None = None) -> RewriteResult:
    """
    multi_query.rewrite_query() 의 로컬 판. 프롬프트와 스키마를 그대로 쓴다.

    실패해도 예외를 던지지 않고 원 질문만 담아 돌려준다. 그래야 뒤 단계가
    계속 굴러간다.
    """
    question = (question or "").strip()
    if not question:
        return RewriteResult(question="", rewritten="", error="질문이 비어 있습니다.")

    model = model or DEFAULT_MODEL
    started = time.time()

    try:
        data = generate_json(
            REWRITE_SYSTEM.replace("{language}", language),
            f"질문: {question}",
            REWRITE_SCHEMA,
            max_new_tokens=MAX_NEW_TOKENS_REWRITE,
            temperature=0.7,
            model_name=model,
            device=device,
        )
    except Exception as exc:  # noqa: BLE001
        return RewriteResult(question=question, rewritten=question,
                             model=model, elapsed=time.time() - started,
                             error=f"{type(exc).__name__}: {exc}")

    rewritten = (data.get("rewritten") or "").strip() or question
    expansions = [str(x).strip() for x in (data.get("expansions") or [])
                  if str(x).strip()][:N_EXPANSIONS]

    return RewriteResult(
        question=question,
        rewritten=rewritten,
        expansions=expansions,
        changed=bool(data.get("changed")) and rewritten != question,
        reason=(data.get("reason") or "").strip(),
        model=model,
        elapsed=time.time() - started,
    )


# --------------------------------------------------------------------------
# 6 단계 : 답변 생성
# --------------------------------------------------------------------------

def generate_answer_local(question: str, chunks: list[Hit],
                          model: str | None = None,
                          device: str | None = None) -> AnswerResult:
    """
    answer.generate_answer() 의 로컬 판. 프롬프트와 스키마를 그대로 쓴다.

    인용 검증도 원본과 같다. 모델이 지어낸 청크 id 는 버린다. 로컬 모델은
    Gemini 보다 이런 실수가 잦아서 이 방어막이 더 중요하다.
    """
    question = (question or "").strip()
    if not question:
        return AnswerResult(question="", error="질문이 비어 있습니다.")
    if not chunks:
        return AnswerResult(question=question, error="근거로 쓸 청크가 없습니다.")

    model = model or DEFAULT_MODEL
    started = time.time()

    try:
        data = generate_json(
            ANSWER_SYSTEM,
            (f"질문: {question}\n\n"
             f"근거 청크 {len(chunks)}개:\n\n{build_context(chunks)}"),
            ANSWER_SCHEMA,
            max_new_tokens=MAX_NEW_TOKENS_ANSWER,
            # 근거에서 답을 뽑는 일이라 매번 흔들릴 이유가 없다.
            temperature=0.2,
            model_name=model,
            device=device,
        )
    except Exception as exc:  # noqa: BLE001
        return AnswerResult(question=question, model=model,
                            elapsed=time.time() - started,
                            error=f"{type(exc).__name__}: {exc}")

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

    parser = argparse.ArgumentParser(
        description="Qwen3-8B 로 재작성/확장·답변을 만든다 (Gemini 대체).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("question", nargs="*")
    parser.add_argument("--lang", default="ko", choices=list(LANGUAGES))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None, help="cpu / cuda (기본: 자동)")
    parser.add_argument("--answer", action="store_true",
                        help="검색·리랭킹까지 붙여 답변 생성까지 해본다")
    args = parser.parse_args()

    question = " ".join(args.question) or "두 페이지만 있는 MDA는?"

    print(f"모델 로드 중... ({args.model}, 최초 1회는 다운로드에 오래 걸립니다)")
    _, model_obj, device = load_llm(args.model, args.device)
    print(f"장치: {getattr(model_obj, 'device', device)}")

    rw = rewrite_query_local(question, language=LANGUAGES[args.lang],
                             model=args.model, device=args.device)
    print(f"\n[1] 질의 재작성  {'바뀜' if rw.changed else '변경 없음'} "
          f"({rw.elapsed:.1f}초)")
    if not rw.ok:
        print(f"    [!] 실패: {rw.error}")
    print(f"    {rw.rewritten}")
    print(f"[2] 질의 확장    {len(rw.expansions)}개")
    for i, q in enumerate(rw.expansions, 1):
        print(f"    {i}. {q}")

    if not args.answer:
        return

    from comparison import search_per_query
    from multi_query import all_queries
    from rerank_gpu import rerank_cross

    ms = search_per_query(all_queries(rw), lang=args.lang)
    rr = rerank_cross(question, ms)
    print(f"\n[3-5] 후보 {len(ms.pooled)}개 -> 선정 "
          f"{', '.join(h.key for h in rr.selected)}")

    ans = generate_answer_local(question, rr.selected,
                                model=args.model, device=args.device)
    if not ans.ok:
        sys.exit(f"\n[6] 답변 생성 실패: {ans.error}")
    print(f"\n[6] 답변 생성    ({ans.elapsed:.1f}초, 근거 충분: "
          f"{'예' if ans.enough else '아니오'})")
    print(f"    {ans.answer}")
    if ans.citations:
        print(f"    인용: {', '.join(ans.citations)}")


if __name__ == "__main__":
    main()
