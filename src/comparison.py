#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
3. 질의별 등수 비교  (다중 질의 병렬 검색)

multi_query.py 가 만든 질의들을 각각 따로 검색해 질의마다 top_k 를 뽑고,
그 전부를 합쳐 같은 청크를 하나로 정리한다.

    질의 5개 x 청크 217개 = 1085개 점수를 계산
        -> 질의마다 자기 top-10               = 50개
        -> 같은 청크를 하나로 합침             = 18개 안팎
        -> 이 묶음이 rerank.py 로 넘어갈 후보

왜 질의마다 따로 뽑나 (max-pooling 과의 차이):

  max-pooling 은 청크마다 최고점 하나만 남겨 전체를 한 줄로 세운다. 그러면
  잘 맞는 질의 하나가 상위권을 독식할 때 다른 질의가 건져 올린 청크가 아예
  안 보인다. 실제로 이 데이터에서 재작성 질의가 top-10 중 8칸을 먹고 원 질문이
  한 칸도 못 들어간 적이 있다.

  질의마다 자기 몫 top_k 를 보장하면 각 질의가 무엇을 찾았는지 드러나고,
  화면에 질의별 등수를 나란히 놓고 비교할 수 있다. 여러 질의가 공통으로 뽑은
  청크는 그만큼 확신이 높다는 뜻이라 리랭킹 단계의 신호로도 쓴다.

저수준 검색(색인 로드, 질의 인코딩)은 search.py 에 있다. 이 파일은 그것을
파이프라인 3단계 모양으로 조립하는 일만 한다.

단독 실행:
    python src/comparison.py --lang ko "두 페이지만 있는 MDA는?"
    python src/comparison.py --lang ko --rewrite "..."   # Gemini 질의 확장까지
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from search import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_TOP_K,
    Hit,
    LANGUAGES,
    Index,
    encode_queries,
    load_index,
    load_metadata,
)


@dataclass
class MultiSearch:
    """질의 여러 개를 각각 검색한 결과."""

    queries: list[str]
    lang: str
    top_k: int
    per_query: list[list[Hit]]   # 질의마다 top_k 개씩 (queries 와 같은 순서)
    pooled: list[Hit]            # 위 전부를 합쳐 중복 제거하고 점수순 정렬한 것
    elapsed: float = 0.0

    @property
    def n_total(self) -> int:
        """중복 제거 전 개수. 질의 5개 x top_k 10 이면 50."""
        return sum(len(hits) for hits in self.per_query)

    def rank_of(self, key: str, query_index: int) -> int | None:
        """어떤 청크가 그 질의의 top_k 에서 몇 등이었는지. 없으면 None (RRF 용)."""
        for rank, hit in enumerate(self.per_query[query_index], 1):
            if hit.key == key:
                return rank
        return None


def make_hit(index: Index, meta: dict, row: int, score: float,
             queries: list[str], qi: int, found_by: list[int]) -> Hit:
    """행 번호 하나를 Hit 으로 만든다."""
    info = meta.get(index.dialogue_ids[row], {})
    return Hit(
        score=score,
        text=str(index.texts[row]),
        lang=index.lang,
        dialogue_id=index.dialogue_ids[row],
        chunk_index=int(index.chunk_indices[row]),
        token_start=int(index.token_starts[row]),
        token_end=int(index.token_ends[row]),
        matched_query=queries[qi] if qi < len(queries) else "",
        matched_query_index=qi,
        found_by=found_by,
        started_at=str(info.get("startedAt", "")),
        ended_at=str(info.get("endedAt", "")),
        n_utterances=int(info.get("numberOfUtterances", 0) or 0),
        participants=list(info.get("participants", []) or []),
    )


def search_per_query(queries: list[str], lang: str = "ko",
                     top_k: int = DEFAULT_TOP_K,
                     model_name: str = DEFAULT_MODEL,
                     device: str | None = None) -> MultiSearch:
    """
    질의마다 따로 top_k 를 뽑고, 그 전부를 합쳐 중복을 제거한다.

    중복 제거 규칙:
      같은 청크는 한 번만 남기되 점수는 여러 질의 중 최고점을 쓴다. 어느 질의들이
      뽑았는지는 Hit.found_by 에 전부 담는다.
    """
    started = time.time()

    index = load_index(lang)
    clean = [q.strip() for q in queries if q and q.strip()]
    query_vectors = encode_queries(clean, model_name, device)
    meta = load_metadata(lang)

    # (Q, N) 전부 정규화돼 있으므로 내적이 곧 코사인 유사도다.
    scores = query_vectors @ index.vectors.T
    k = min(top_k, index.size)

    # --- 질의별 top_k ------------------------------------------------------
    per_rows: list[list[int]] = [
        [int(r) for r in np.argsort(-scores[qi])[:k]]
        for qi in range(len(clean))
    ]
    per_query = [
        [make_hit(index, meta, r, float(scores[qi][r]), clean, qi, [qi])
         for r in rows]
        for qi, rows in enumerate(per_rows)
    ]

    # --- 합집합에서 중복 제거 ----------------------------------------------
    best: dict[int, tuple[float, int]] = {}   # row -> (최고점, 그 점수를 낸 질의)
    found_by: dict[int, list[int]] = {}
    for qi, rows in enumerate(per_rows):
        for r in rows:
            found_by.setdefault(r, []).append(qi)
            if r not in best or scores[qi][r] > best[r][0]:
                best[r] = (float(scores[qi][r]), qi)

    pooled = [
        make_hit(index, meta, r, best[r][0], clean, best[r][1], found_by[r])
        for r in sorted(best, key=lambda r: -best[r][0])
    ]

    return MultiSearch(queries=clean, lang=lang, top_k=k,
                       per_query=per_query, pooled=pooled,
                       elapsed=time.time() - started)


# --------------------------------------------------------------------------
# 단독 실행
# --------------------------------------------------------------------------

def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    parser = argparse.ArgumentParser(
        description="질의별로 따로 검색해 등수를 비교한다.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("question", nargs="*", help="검색할 질문")
    parser.add_argument("--lang", default="ko", choices=list(LANGUAGES))
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help=f"질의당 뽑을 청크 수 (기본: {DEFAULT_TOP_K})")
    parser.add_argument("--rewrite", action="store_true",
                        help="Gemini 재작성/확장 질의까지 함께 검색")
    args = parser.parse_args()

    question = " ".join(args.question) or "두 페이지만 있는 MDA는?"
    queries = [question]
    if args.rewrite:
        from multi_query import all_queries, rewrite_query

        result = rewrite_query(question, language=LANGUAGES[args.lang])
        if result.ok:
            queries = all_queries(result)
        else:
            print(f"[!] 재작성 실패, 원 질문만: {result.error}")

    ms = search_per_query(queries, lang=args.lang, top_k=args.top_k)

    print(f"\n색인 {args.lang} · 질의 {len(ms.queries)}개 x 상위 {ms.top_k}개 "
          f"= {ms.n_total}개 -> 중복 제거 {len(ms.pooled)}개 ({ms.elapsed:.1f}초)\n")

    width = 16
    print("순위 | " + " | ".join(f"질의{i + 1}".ljust(width)
                                 for i in range(len(ms.queries))))
    for rank in range(ms.top_k):
        cells = []
        for hits in ms.per_query:
            cells.append(f"{hits[rank].score:.3f} {hits[rank].key}".ljust(width)
                         if rank < len(hits) else " " * width)
        print(f"{rank + 1:>3}  | " + " | ".join(cells))

    print(f"\n중복 제거 후 {len(ms.pooled)}개 (괄호 = 뽑은 질의 번호)")
    for rank, hit in enumerate(ms.pooled, 1):
        print(f"  {rank:2d} {hit.score:.4f} {hit.key:<10s} "
              f"질의{[q + 1 for q in hit.found_by]}  {hit.preview(70)}")


if __name__ == "__main__":
    main()
