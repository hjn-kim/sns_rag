#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
4. 리랭킹 (GPU / 크로스인코더)  - rerank.py 의 Gemini 판정을 대신한다

comparison.py 가 추린 후보 20개 안팎을 BAAI/bge-reranker-v2-m3 로 다시 줄 세운다.
결과 자료구조(RankedHit / RerankResult)는 rerank.py 것을 그대로 쓰므로 화면 표는
Gemini 방식과 똑같이 그려진다.

왜 bge-reranker-v2-m3 인가:

  - sentence-transformers 의 CrossEncoder 로 바로 올라간다. requirements 에 이미
    있으므로 새로 깔 것이 없다.
  - XLM-RoBERTa-large(568M) 기반이라 이 코퍼스의 6개 언어를 전부 커버한다.
    fp16 으로 1.1GB 남짓이고, 임베딩 모델(0.6B)과 같이 올려도 24GB GPU 에 여유롭다.
  - 인코더 전용이라 쌍 하나당 forward 한 번이다. 같은 파라미터 수의 디코더
    모델(Qwen3-Reranker 등)보다 리랭킹 용도로는 가볍다.

bi-encoder(검색) 와 뭐가 다른가:

  검색 단계는 질의와 청크를 각각 따로 벡터로 만들어 비교한다. 빠르지만 둘을
  같이 읽지는 못한다. 크로스인코더는 (질의, 청크) 를 한 입력으로 붙여 넣고
  통째로 읽어 관련성 점수 하나를 낸다. 느린 대신 정확해서, 후보가 20개로
  줄어든 뒤에 쓴다.

Gemini 방식과 견줘 잃는 것:

  판단 근거 문장이 없다. 크로스인코더는 숫자 하나만 낸다. 그래서 화면 표의
  '판단 근거' 칸에는 설명 대신 확률과 logit 을 적는다. 지어낸 설명을 채우는
  것보다 계산된 값을 그대로 보이는 편이 정직하다.

  이 코퍼스 전용 지시(번역 오류 감안, 반복 문장 무시 등)도 넣을 자리가 없다.
  다만 크로스인코더는 "주제가 겹치는 것과 답이 있는 것을 가르는" 일 자체를
  학습한 모델이라, 그 지시 없이도 되는지는 재보면 된다.

CPU 에서는 쓰지 말 것:
  후보 20개 x 500토큰을 CPU 로 재점수하면 10~30초가 걸린다. GPU 가 없으면
  rerank.py 의 method="llm"(Gemini) 이나 "rrf" 를 쓴다.

단독 실행:
    python src/rerank_gpu.py --lang ko "두 페이지만 있는 MDA는?"
    python src/rerank_gpu.py --lang ko --rewrite "..."      # Gemini 질의 확장까지
    python src/rerank_gpu.py --lang ko --compare "..."      # Gemini 방식과 순위 비교
"""

from __future__ import annotations

import argparse
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from comparison import MultiSearch, search_per_query  # noqa: E402
from rerank import FINAL_TOP_N, RankedHit, RerankResult, rrf_scores  # noqa: E402
from search import LANGUAGES  # noqa: E402

DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"

# 청크가 Harrier 토큰 500개인데 XLM-R 토크나이저로는 그보다 늘어난다(언어마다
# 다르지만 한국어 기준 1.3~1.5배). 질의까지 붙으므로 1024 면 잘리지 않는다.
# 모델 자체는 8192 까지 받지만 길게 잡을수록 느려지기만 한다.
MAX_LENGTH = 1024

# 24GB GPU 기준. 후보가 20개 남짓이라 사실상 한 배치로 끝난다.
BATCH_SIZE = 16


@lru_cache(maxsize=1)
def load_reranker(name: str = DEFAULT_MODEL, device: str | None = None):
    """
    크로스인코더를 올린다. 프로세스당 한 번만.

    search.py 의 load_model 과 같은 이유로 lru_cache 를 건다. Streamlit 은 위젯을
    건드릴 때마다 스크립트를 다시 도는데, 캐시가 없으면 클릭마다 1.1GB 를 새로
    올리게 된다.

    dtype 은 GPU 면 float16, CPU 면 float32 로 둔다. CPU 에서 half 는 느리거나
    아예 지원되지 않는 연산이 있다.
    """
    from sentence_transformers import CrossEncoder
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # transformers 4.56 부터 torch_dtype 이 dtype 으로 바뀌었다. 옛 버전에 dtype 을
    # 넘기면 예외 없이 무시되므로 버전을 보고 인자 이름을 고른다. (search.py 와 동일)
    import transformers
    try:
        tf_version = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    except ValueError:
        tf_version = (99, 99)
    dtype_key = "dtype" if tf_version >= (4, 56) else "torch_dtype"
    dtype = "float16" if device.startswith("cuda") else "float32"

    model = CrossEncoder(
        name,
        max_length=MAX_LENGTH,
        device=device,
        model_kwargs={dtype_key: dtype},
    )
    return model, device


def cross_scores(question: str, texts: list[str],
                 model_name: str = DEFAULT_MODEL,
                 device: str | None = None,
                 batch_size: int = BATCH_SIZE) -> np.ndarray:
    """
    (질의, 청크) 쌍마다 raw logit 을 돌려준다. (N,) float32

    activation_fn 에 Identity 를 넘겨 활성함수를 끈다. bge 계열은 num_labels=1
    이라 CrossEncoder 가 기본으로 시그모이드를 씌우는데, 그러면 0~1 로 눌려서
    상위권끼리의 차이가 안 보인다. 표시할 확률은 아래에서 직접 계산한다.
    """
    import torch

    model, _ = load_reranker(model_name, device)
    pairs = [(question, " ".join(t.split())) for t in texts]

    raw = model.predict(
        pairs,
        batch_size=batch_size,
        activation_fn=torch.nn.Identity(),
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.asarray(raw, dtype=np.float32).reshape(-1)


def rerank_cross(question: str, result: MultiSearch, top_n: int = FINAL_TOP_N,
                 model_name: str = DEFAULT_MODEL, device: str | None = None,
                 batch_size: int = BATCH_SIZE) -> RerankResult:
    """
    후보를 크로스인코더로 다시 줄 세우고 상위 top_n 을 고른다.

    rerank.py 의 rerank() 와 같은 모양의 결과를 돌려준다. 화면은 어느 쪽으로
    돌렸는지 몰라도 되게 하려는 것이다.

      RankedHit.llm_score  0~10 정수. 확률 x 10 을 반올림한 값이라 Gemini 방식의
                           같은 열과 눈금이 맞는다.
      RankedHit.reason     설명 대신 "관련성 0.985 (logit +4.21)".
      RerankResult.method  "cross"

    실패하면 rerank.py 와 같은 방식으로 RRF 로 떨어진다. 모델이 없거나 VRAM 이
    모자란 경우가 여기 해당한다.
    """
    started = time.time()

    candidates = result.pooled
    if not candidates:
        return RerankResult(question=question, method="cross",
                            elapsed=time.time() - started)

    rrf = rrf_scores(result)
    ranked = [
        RankedHit(hit=hit, rank_before=i, rrf_score=rrf.get(hit.key, 0.0))
        for i, hit in enumerate(candidates, 1)
    ]

    error = None
    used = "cross"
    model_used = model_name

    try:
        logits = cross_scores(question, [h.hit.text for h in ranked],
                              model_name, device, batch_size)
        # 시그모이드로 0~1 확률을 만든다. 표에는 x10 한 정수를 쓰고, 원래 값은
        # 근거 칸에 남긴다. logit 을 함께 보여야 상위권끼리의 격차가 드러난다.
        probs = 1.0 / (1.0 + np.exp(-logits))
        for item, logit, prob in zip(ranked, logits, probs):
            item.llm_score = int(round(float(prob) * 10))
            item.reason = f"관련성 {prob:.3f} (logit {logit:+.2f})"
    except Exception as exc:  # noqa: BLE001 - 모델 없음/VRAM 부족 등 모두 여기로
        error = f"{type(exc).__name__}: {exc}"
        used = "rrf"
        model_used = ""
        for item in ranked:
            item.llm_score = None
            item.reason = ""

    # 정렬 기준은 rerank.py 와 같다. 점수 -> RRF -> dense.
    # 크로스인코더 점수는 연속값이라 동점이 거의 없어 뒤 두 개는 사실상 예비다.
    ranked.sort(key=lambda x: (x.llm_score or 0, x.rrf_score, x.hit.score),
                reverse=True)
    for rank, item in enumerate(ranked, 1):
        item.rank_after = rank

    return RerankResult(
        question=question,
        method=used,
        ranked=ranked,
        selected=[item.hit for item in ranked[:top_n]],
        model=model_used,
        elapsed=time.time() - started,
        error=error,
    )


# --------------------------------------------------------------------------
# 단독 실행
# --------------------------------------------------------------------------

def _print_table(rr: RerankResult, top_n: int) -> None:
    """rerank.py 의 CLI 와 같은 표를 찍는다."""
    print(f"\n{'후':>3} {'전':>3} {'이동':>4} {'점수':>4} {'RRF':>6} {'dense':>6}  청크")
    for item in rr.ranked:
        mark = "  <= 선정" if item.rank_after <= top_n else ""
        moved = f"{item.moved:+d}" if item.moved else "-"
        score = f"{item.llm_score}" if item.llm_score is not None else "-"
        print(f"{item.rank_after:>3} {item.rank_before:>3} {moved:>4} {score:>4} "
              f"{item.rrf_score:.4f} {item.hit.score:.4f}  {item.hit.key}{mark}")
        if item.reason:
            print(f"                                    {item.reason}")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    parser = argparse.ArgumentParser(
        description="검색 후보를 크로스인코더로 리랭킹한다 (GPU 권장).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("question", nargs="*")
    parser.add_argument("--lang", default="ko", choices=list(LANGUAGES))
    parser.add_argument("--top-n", type=int, default=FINAL_TOP_N,
                        help=f"최종 선정 청크 수 (기본: {FINAL_TOP_N})")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"리랭커 (기본: {DEFAULT_MODEL})")
    parser.add_argument("--device", default=None, help="cpu / cuda (기본: 자동 판별)")
    parser.add_argument("--rewrite", action="store_true",
                        help="Gemini 재작성/확장 질의까지 함께 검색")
    parser.add_argument("--compare", action="store_true",
                        help="Gemini 리랭킹과 순위를 나란히 비교한다")
    args = parser.parse_args()

    question = " ".join(args.question) or "두 페이지만 있는 MDA는?"

    queries = [question]
    if args.rewrite:
        from multi_query import all_queries, rewrite_query
        rw = rewrite_query(question, language=LANGUAGES[args.lang])
        if rw.ok:
            queries = all_queries(rw)
        else:
            print(f"[!] 재작성 실패, 원 질문만: {rw.error}")

    ms = search_per_query(queries, lang=args.lang)
    print(f"\n후보 {len(ms.pooled)}개  (질의 {len(ms.queries)}개 x 상위 {ms.top_k}개 "
          f"= {ms.n_total}개에서 중복 제거)")

    print("\n리랭커 로드 중... (최초 1회만 느립니다)")
    rr = rerank_cross(question, ms, top_n=args.top_n,
                      model_name=args.model, device=args.device)
    _, device = load_reranker(args.model, args.device)
    print(f"장치 {device} · {args.model} · {rr.elapsed:.1f}초")
    if rr.error:
        print(f"[!] 크로스인코더 실패, RRF 로 대체: {rr.error}")

    _print_table(rr, args.top_n)

    if not args.compare:
        return

    # --- Gemini 방식과 나란히 --------------------------------------------
    from rerank import rerank as rerank_llm

    gr = rerank_llm(question, ms, top_n=args.top_n)
    print(f"\n\nGemini 리랭킹 ({gr.method}, {gr.elapsed:.1f}초) 와 비교")

    cross_rank = {x.hit.key: x.rank_after for x in rr.ranked}
    llm_rank = {x.hit.key: x.rank_after for x in gr.ranked}
    keys = sorted(cross_rank, key=lambda k: cross_rank[k])

    print(f"\n{'cross':>6} {'gemini':>7} {'차이':>5}  청크")
    for key in keys:
        c, g = cross_rank[key], llm_rank.get(key, 0)
        diff = g - c
        mark = ""
        if c <= args.top_n and g > args.top_n:
            mark = "  <- cross 만 선정"
        elif g <= args.top_n and c > args.top_n:
            mark = "  <- gemini 만 선정"
        print(f"{c:>6} {g:>7} {diff:>+5}  {key}{mark}")

    same = [k for k in keys[:args.top_n] if llm_rank.get(k, 0) <= args.top_n]
    print(f"\n최종 {args.top_n}개 중 겹친 것 {len(same)}개: {', '.join(same)}")


if __name__ == "__main__":
    main()
