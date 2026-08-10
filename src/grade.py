#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
7. 정답 비교

data/answer.json 에 적어 둔 정답 후보들과 6단계 답변을 견줘 맞았는지 본다.
호출은 하지 않는다. 문자열 포함 여부만 본다.

    data/answer.json
        {
          "1": ["사법부", "사법", "Judiciary"],
          "2": ["전력부", "에너지부", "Ministry of Power"],
          ...
        }

    key    app.py 의 QUESTION_OPTIONS 순번(1부터)
    value  정답 후보 목록. 낱말 하나면 문자열로 써도 되고 목록으로 써도 된다.

판정 규칙 (any-include):

    후보 중 **하나라도** 답변 안에 들어 있으면 정답.
    비교 전에 양쪽을 정규화한다 - 유니코드 NFKC, 소문자, 공백/문장부호 제거.

    정규화 덕분에 이런 것들이 같은 것으로 취급된다.
        "Power BI" = "PowerBI" = "power bi"
        "Судебная власть" = "судебная власть"

왜 후보를 여러 개 두나:

    원문이 기계 번역이라 같은 대상이 답변마다 다른 이름으로 나온다. 실제로
    나온 것들이다.

        정답 "전력부"    <-> 답변 "...사용된 데이터는 에너지부의 자료입니다."
        정답 "사법부"    <-> 답변 "...'사법(Judiciary)' MDA입니다."
        정답 "司法机构"   <-> 답변 "司法部 (Ministry of Justice) 被描述为..."

    셋 다 맞은 답인데 정답 낱말 하나만 두면 전부 오답이 된다. 그래서 실제로
    나올 법한 표기를 후보로 함께 적어 둔다.

    반대로 후보를 너무 짧게 잡으면 엉뚱한 것이 통과한다. 예를 들어 "전력" 만
    두면 Power BI 를 "전력 BI" 로 오역한 답변이 정답 처리된다. 그래서 후보는
    "전력부" 처럼 대상을 특정할 수 있는 길이로 적는다.

화면 표시:

    정답일 때  실제 정답: 겹친 후보만 보여준다 (어느 표기로 맞았는지)
    오답일 때  실제 정답: 후보 전체를 보여준다 (무엇을 기대했는지)

단독 실행:
    python src/grade.py                 # 정답표를 QUESTION_OPTIONS 와 대조만
    python src/grade.py --run 1         # 1번 질문을 파이프라인에 태우고 채점
    python src/grade.py --run all       # 18개 전부 + 정답률

    # 앱과 같은 설정으로 채점하려면 백엔드를 맞춰야 한다.
    #   app.py                  --backend gemini --method llm     (기본값)
    #   app_gpu_tabs.py         --backend gemini --method cross
    #   app_gpu_tabs_Qwen.py    --backend qwen   --method cross
    python src/grade.py --run all --backend qwen --method cross
    python src/grade.py --run all --backend qwen --method cross --lang ko
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
ANSWER_PATH = ROOT / "data" / "answer.json"


@dataclass
class GradeResult:
    """7단계 결과."""

    question: str
    llm_answer: str = ""
    candidates: list[str] = field(default_factory=list)   # 정답 후보 전체
    matched: list[str] = field(default_factory=list)      # 답변에 실제로 들어 있던 후보
    verdict: str = ""            # "정답" | "오답" | "판정 불가"
    reason: str = ""
    elapsed: float = 0.0
    error: str | None = None

    @property
    def correct(self) -> bool:
        return self.verdict == "정답"

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def gold_display(self) -> str:
        """
        화면의 '실제 정답' 에 넣을 문자열.

        맞았으면 겹친 후보만, 틀렸으면 후보 전체를 보여준다. 맞았을 때 후보를
        전부 늘어놓으면 어느 표기로 맞았는지가 묻힌다.
        """
        words = self.matched if self.correct else self.candidates
        return ", ".join(words)


# --------------------------------------------------------------------------
# 정답표
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_gold() -> dict[int, tuple[str, ...]]:
    """
    data/answer.json 을 {순번: (후보, ...)} 로 읽는다.

    값이 문자열 하나여도 되고 목록이어도 된다. 파일이 없거나 깨졌으면 빈 dict 를
    돌려준다. 정답표가 없어도 1~6 단계는 굴러가야 한다.
    """
    try:
        with ANSWER_PATH.open(encoding="utf-8") as fp:
            data = json.load(fp)
    except (OSError, json.JSONDecodeError):
        return {}

    out: dict[int, tuple[str, ...]] = {}
    for key, value in (data or {}).items():
        try:
            index = int(key)
        except (TypeError, ValueError):
            continue
        items = value if isinstance(value, (list, tuple)) else [value]
        words = tuple(str(v).strip() for v in items if str(v).strip())
        if words:
            out[index] = words
    return out


def gold_for(index: int) -> list[str]:
    """순번(1부터)의 정답 후보 목록. 없으면 빈 목록."""
    return list(load_gold().get(index, ()))


# --------------------------------------------------------------------------
# 판정
# --------------------------------------------------------------------------

def normalize(text: str) -> str:
    """
    비교용으로 다듬는다.

    NFKC 로 전각/반각을 통일하고, 소문자로 내리고, 공백과 문장부호를 지운다.
    "Power BI" 와 "PowerBI" 와 "power  bi" 가 모두 "powerbi" 가 된다.
    \\W 는 유니코드 모드에서 한글/한자/키릴을 글자로 보므로 지워지지 않는다.
    """
    text = unicodedata.normalize("NFKC", text or "").lower()
    return re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE)


def grade_answer(question: str, llm_answer: str,
                 candidates: list[str] | str) -> GradeResult:
    """
    후보 중 하나라도 답변에 들어 있으면 정답으로 본다.

    겹친 후보는 candidates 에 적힌 순서대로 matched 에 담는다. 앞에 적은 것이
    대표 표기가 되도록 정답표를 쓰면 화면에 그 순서로 나온다.
    """
    started = time.time()
    if isinstance(candidates, str):
        candidates = [candidates]
    candidates = [c for c in (candidates or []) if c and c.strip()]

    result = GradeResult(question=question, llm_answer=llm_answer or "",
                         candidates=candidates)

    if not candidates:
        result.verdict = "판정 불가"
        result.reason = "data/answer.json 에 이 질문의 정답 후보가 없습니다."
        result.elapsed = time.time() - started
        return result

    if not (llm_answer or "").strip():
        result.verdict = "오답"
        result.reason = "답변이 비어 있습니다."
        result.elapsed = time.time() - started
        return result

    haystack = normalize(llm_answer)
    result.matched = [c for c in candidates if normalize(c) and normalize(c) in haystack]

    if result.matched:
        result.verdict = "정답"
        result.reason = (f"후보 {len(candidates)}개 중 "
                         f"{len(result.matched)}개가 답변에 들어 있습니다.")
    else:
        result.verdict = "오답"
        result.reason = f"후보 {len(candidates)}개 중 답변에 들어 있는 것이 없습니다."

    result.elapsed = time.time() - started
    return result


# --------------------------------------------------------------------------
# 단독 실행
# --------------------------------------------------------------------------

def question_options() -> list[str]:
    """
    app.py 의 QUESTION_OPTIONS 를 읽는다.

    app.py 를 import 하면 streamlit 이 딸려 올라오므로 소스에서 리터럴만 꺼낸다.
    """
    import ast
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    match = re.search(r"QUESTION_OPTIONS\s*=\s*(\[.*?\n\])", src, re.S)
    return ast.literal_eval(match.group(1)) if match else []


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    parser = argparse.ArgumentParser(
        description="정답 후보와 파이프라인 답변을 견준다 (문자열 포함 판정).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--run", default=None,
                        help="채점할 질문 순번(1~18) 또는 all.\n없으면 정답표만 훑는다")
    parser.add_argument("--backend", default="gemini", choices=["gemini", "qwen"],
                        help="1·2·6 단계를 무엇으로 돌릴지 (기본: gemini)")
    parser.add_argument("--method", default="llm", choices=["llm", "rrf", "cross"],
                        help="4 단계 리랭킹 방식 (기본: llm)")
    parser.add_argument("--lang", default=None, choices=["ko", "en", "zh", "vi", "fil", "ru"],
                        help="이 언어의 3문항만 채점한다 (기본: 전부)")
    args = parser.parse_args()

    gold = load_gold()
    options = question_options()
    if not gold:
        sys.exit(f"정답표를 읽지 못했습니다: {ANSWER_PATH}")

    # --- 정답표만 훑기 ------------------------------------------------------
    if not args.run:
        print(f"정답표 {len(gold)}개  ({ANSWER_PATH})")
        print(f"화면 목록 {len(options)}개  (app.py QUESTION_OPTIONS)\n")
        for i in range(1, max(len(options), len(gold)) + 1):
            q = options[i - 1] if i <= len(options) else "(질문 없음)"
            words = " / ".join(gold.get(i, ("(정답 없음)",)))
            print(f"{i:>2}  {words:<46s} <- {q[:46]}")
        missing = [i for i in range(1, len(options) + 1) if i not in gold]
        if missing:
            print(f"\n[!] 정답이 없는 순번: {missing}")
        return

    # --- 파이프라인에 태워 채점 ---------------------------------------------
    from main import run_pipeline

    DOC_LANGS = ["ko", "en", "zh", "vi", "fil", "ru"]   # 3문항씩 한 언어

    targets = (list(range(1, len(options) + 1)) if args.run == "all"
               else [int(args.run)])
    if args.lang:
        targets = [i for i in targets if DOC_LANGS[(i - 1) // 3] == args.lang]

    print(f"채점 {len(targets)}문항 · 1·2·6단계 {args.backend} · "
          f"4단계 {args.method}")

    rows = []
    for i in targets:
        question = options[i - 1]
        lang = DOC_LANGS[(i - 1) // 3]
        result = run_pipeline(question, lang=lang, gold=gold_for(i),
                              llm_backend=args.backend,
                              rerank_method=args.method)
        gr = result.grade
        rows.append((i, lang, gr))

        mark = "O" if gr.correct else ("?" if gr.verdict == "판정 불가" else "X")
        print(f"\n[{mark}] {i:>2}번 ({lang})  {gr.verdict}   {result.elapsed:.1f}초")
        print(f"     LLM 정답  : {gr.llm_answer[:110]}")
        print(f"     실제 정답 : {gr.gold_display}")
        # 단계별 실패는 조용히 넘어가면 정답률만 보고 원인을 못 찾는다.
        for stage, message in result.errors().items():
            print(f"     [!] {stage} 실패: {message[:90]}")

    if len(rows) > 1:
        n_ok = sum(1 for _, _, g in rows if g.correct)
        print(f"\n정답 {n_ok}/{len(rows)}  "
              f"({args.backend} · {args.method})")
        wrong = [f"{i}({lang})" for i, lang, g in rows if not g.correct]
        if wrong:
            print(f"틀린 문항: {', '.join(wrong)}")

        # 언어별로 나눠 보면 특정 언어의 번역 품질 문제인지 구분된다.
        by_lang: dict[str, list[bool]] = {}
        for _, lang, g in rows:
            by_lang.setdefault(lang, []).append(g.correct)
        print("언어별: " + "  ".join(
            f"{lang} {sum(v)}/{len(v)}" for lang, v in by_lang.items()))


if __name__ == "__main__":
    main()
