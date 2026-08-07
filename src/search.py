#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
질의 임베딩 + 청크 검색 (Harrier / dense / .npz)

embedding.py 가 만들어 둔 data/emb/{lang}/*.npz 를 읽어 하나의 인덱스로 합치고,
질의를 같은 모델로 인코딩해 코사인 유사도로 상위 청크를 찾는다.

    from search import search
    hits = search(["다음 워크숍 일정", "next workshop date"], lang="ko", top_k=5)
    hits[0].text, hits[0].score, hits[0].matched_query

문서 임베딩과 반드시 맞춰야 하는 것 세 가지 (틀리면 조용히 정확도만 떨어진다):

  1. 같은 모델    microsoft/harrier-oss-v1-0.6b, 1024차원.
                  270m 은 벡터 공간이 달라 유사도가 무의미해진다.
  2. 프리픽스     Harrier 는 비대칭 모델이다. 문서는 프리픽스 없이 넣었으므로
                  (embedding.py 참고) 질의에만 prompt_name="web_search_query" 를 붙인다.
  3. float32 재정규화
                  embedding.py 가 bf16 오차를 float32 에서 다시 맞췄다.
                  질의도 같은 처리를 해야 내적을 그대로 코사인으로 쓸 수 있다.

이 파일은 저수준 검색 엔진이다. 파이프라인 3단계(질의별 등수 비교)는
comparison.py 에 있다.

다중 질의(재작성 1 + 확장 3 + 원 질문)는 max-pooling 으로 합친다.
청크마다 "가장 잘 맞은 질의 하나"의 점수를 그 청크의 점수로 쓰고, 어느 질의가
끌어올렸는지 Hit.matched_query 에 남긴다. 화면에서 "확장질의 2번이 찾음" 을
보여줄 수 있고, 질의별 점수 스케일이 같은 모델이라 RRF 를 쓸 이유가 적다.

같은 대화의 인접 청크를 걸러내지 않는다. 100토큰만 겹치고 400토큰은 서로 다른
내용이라 지우면 손해가 더 크다. 자세한 이유는 search() 의 docstring 참고.

인덱스와 모델은 lru_cache 로 프로세스당 한 번만 올린다. Streamlit 은 위젯을
건드릴 때마다 스크립트를 다시 도는데, 캐시가 없으면 클릭마다 2.4GB 모델을
새로 올리게 된다. 앱에서는 여기에 @st.cache_resource 를 한 겹 더 씌워도 좋다.

사용 예:
    python src/search.py "다음 워크숍이 언제야?"
    python src/search.py --lang en "when is the next workshop"
    python src/search.py --lang ko --top-k 10 "예산 PDF 데이터 추출"
    python src/search.py --lang ko --rewrite "다음 워크숍이 언제야?"   # Gemini 재작성까지
    python src/search.py --lang ko --full "..."                        # 청크 본문 전체 출력
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# embedding.py 와 같은 값이어야 한다.
DEFAULT_MODEL = "microsoft/harrier-oss-v1-0.6b"

# config_sentence_transformers.json 에 정의된 질의용 프롬프트.
#   "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
QUERY_PROMPT = "web_search_query"

EMB_ROOT = ROOT / "data" / "emb"
SRC_ROOT = ROOT / "data" / "whatsapp_chat_language"

# 언어당 청크가 118~217개뿐이라 넉넉히 뽑아도 부담이 없다. 500토큰 x 10 = 5천 토큰이면
# LLM 컨텍스트로도 여유롭다. 정답이 5~10위에 있는 경우가 있어 5는 너무 짜다.
DEFAULT_TOP_K = 10

# app.py 의 DOCUMENT_OPTIONS 와 같은 순서/코드
LANGUAGES = {
    "ko": "한국어",
    "en": "영어",
    "vi": "베트남어",
    "fil": "필리핀어",
    "ru": "러시아어",
    "zh": "중국어",
}


# --------------------------------------------------------------------------
# 자료구조
# --------------------------------------------------------------------------

@dataclass
class Index:
    """언어 하나의 청크 전체. 891청크 x 1024차원이라 언어별로 통째로 올려도 4MB 미만."""

    lang: str
    vectors: np.ndarray          # (N, dim) float32, L2 정규화됨
    texts: np.ndarray            # (N,) 청크 원문
    dialogue_ids: list[str]      # (N,) 's0000' 같은 대화 ID
    chunk_indices: np.ndarray    # (N,) int32  대화 안에서 몇 번째 청크인지
    token_starts: np.ndarray     # (N,) int32
    token_ends: np.ndarray       # (N,) int32
    model: str = ""

    @property
    def size(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])


@dataclass
class Hit:
    """검색 결과 한 건."""

    score: float
    text: str
    lang: str
    dialogue_id: str
    chunk_index: int
    token_start: int
    token_end: int

    # 다중 질의 중 이 청크를 끌어올린 질의
    matched_query: str = ""
    matched_query_index: int = 0

    # 이 청크를 자기 top-k 안에 넣은 질의 번호들. 여러 질의가 함께 뽑았다는 것은
    # 그만큼 확신이 높다는 뜻이라, 나중에 리랭킹할 때 쓸 신호가 된다.
    found_by: list[int] = field(default_factory=list)

    # 원본 JSON 에서 가져온 대화 메타 (없으면 빈 값)
    started_at: str = ""
    ended_at: str = ""
    n_utterances: int = 0
    participants: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        """청크를 가리키는 고유 이름. 중복 제거의 기준."""
        return f"{self.dialogue_id}#{self.chunk_index}"

    def preview(self, n: int = 200) -> str:
        one_line = " ".join(self.text.split())
        return one_line[:n] + ("..." if len(one_line) > n else "")



# --------------------------------------------------------------------------
# 인덱스 로드
# --------------------------------------------------------------------------

@lru_cache(maxsize=len(LANGUAGES))
def load_index(lang: str) -> Index:
    """
    data/emb/{lang}/*.npz 를 이름순으로 읽어 하나로 합친다.

    파일명 s0000_embeddings.npz 의 앞부분이 원본 대화 ID(s0000)다.
    lru_cache 라 같은 언어를 다시 부르면 즉시 돌아온다.
    """
    emb_dir = EMB_ROOT / lang
    if not emb_dir.is_dir():
        raise FileNotFoundError(
            f"임베딩 폴더가 없습니다: {emb_dir}\n"
            f"쓸 수 있는 언어: {', '.join(sorted(p.name for p in EMB_ROOT.iterdir() if p.is_dir()))}"
        )

    files = sorted(emb_dir.glob("*_embeddings.npz"))
    if not files:
        raise FileNotFoundError(f"*.npz 가 없습니다: {emb_dir}")

    vec_parts, text_parts = [], []
    dialogue_ids: list[str] = []
    idx_parts, start_parts, end_parts = [], [], []
    model_name = ""

    for path in files:
        data = np.load(path)
        vectors = data["embeddings"]
        n = vectors.shape[0]

        vec_parts.append(vectors)
        text_parts.append(data["texts"])
        idx_parts.append(data["chunk_index"])
        start_parts.append(data["token_start"])
        end_parts.append(data["token_end"])

        # s0000_embeddings.npz -> s0000
        dialogue_ids.extend([path.name[: -len("_embeddings.npz")]] * n)

        if not model_name:
            try:
                model_name = json.loads(str(data["info"])).get("model", "")
            except (ValueError, KeyError):
                pass

    return Index(
        lang=lang,
        vectors=np.vstack(vec_parts).astype(np.float32),
        texts=np.concatenate(text_parts),
        dialogue_ids=dialogue_ids,
        chunk_indices=np.concatenate(idx_parts),
        token_starts=np.concatenate(start_parts),
        token_ends=np.concatenate(end_parts),
        model=model_name,
    )


@lru_cache(maxsize=len(LANGUAGES))
def load_metadata(lang: str) -> dict[str, dict]:
    """
    대화 ID -> {startedAt, endedAt, numberOfUtterances, participants}

    .npz 에는 청크 본문만 있고 날짜/참여자는 원본 JSON 에만 있다.
    검색 결과 카드에 "2019-11-15 · AcidiQ 외 7명" 을 띄우려면 여기서 이어 붙인다.
    원본이 없어도 검색 자체는 되므로 조용히 빈 dict 를 돌려준다.
    """
    src_dir = SRC_ROOT / f"Whatsapp_chat_{lang}"
    if not src_dir.is_dir():
        return {}

    meta: dict[str, dict] = {}
    for path in sorted(src_dir.glob("*.json")):
        try:
            with path.open(encoding="utf-8") as fp:
                data = json.load(fp)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            meta[path.stem] = data
    return meta


# --------------------------------------------------------------------------
# 질의 임베딩
# --------------------------------------------------------------------------

@lru_cache(maxsize=2)
def load_model(name: str = DEFAULT_MODEL, device: str | None = None):
    """
    임베딩 모델을 올린다. 프로세스당 한 번만.

    질의는 한 번에 5개 남짓이라 CPU 로 충분하다(0.5~2초). CUDA 가 있으면 쓴다.
    dtype 은 float32 로 고정한다. CPU 에서 bf16 은 AVX512-BF16 이 없으면
    에뮬레이션으로 떨어져 오히려 느리다.
    """
    from sentence_transformers import SentenceTransformer
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # transformers 4.56 부터 torch_dtype 이 dtype 으로 바뀌었다. 옛 버전에 dtype 을
    # 넘기면 예외 없이 무시되어 버리므로 버전을 보고 인자 이름을 고른다.
    import transformers
    try:
        tf_version = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    except ValueError:
        tf_version = (99, 99)
    dtype_key = "dtype" if tf_version >= (4, 56) else "torch_dtype"

    return SentenceTransformer(name, device=device,
                               model_kwargs={dtype_key: "float32"})


def encode_queries(queries: list[str], model_name: str = DEFAULT_MODEL,
                   device: str | None = None) -> np.ndarray:
    """
    질의를 (Q, dim) float32 정규화 벡터로 만든다.

    prompt_name 을 반드시 붙인다. Harrier 는 질의 쪽에만 instruction 을 붙이는
    비대칭 모델이고, 문서는 프리픽스 없이 색인돼 있다. 안 붙이면 점수가 떨어진다.
    """
    queries = [q.strip() for q in queries if q and q.strip()]
    if not queries:
        raise ValueError("질의가 비어 있습니다.")

    model = load_model(model_name, device)
    vectors = model.encode(
        queries,
        prompt_name=QUERY_PROMPT,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    # embedding.py 와 같은 처리. 내적을 그대로 코사인으로 쓰기 위해 float32 에서 한 번 더.
    vectors = vectors.astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors


# --------------------------------------------------------------------------
# 검색
# --------------------------------------------------------------------------

def search(queries: list[str], lang: str = "ko", top_k: int = DEFAULT_TOP_K,
           model_name: str = DEFAULT_MODEL,
           device: str | None = None) -> list[Hit]:
    """
    질의 여러 개를 max-pooling 으로 합쳐 상위 top_k 를 돌려준다.

    청크마다 가장 잘 맞은 질의의 점수를 그 청크의 점수로 쓴다. 결과가 한 줄로
    나오면 되는 곳(CLI 등)에서 쓴다. 질의별 순위를 따로 보려면 파이프라인
    3단계인 comparison.search_per_query() 를 쓴다.

    인접 청크를 걸러내지 않는 이유:
      청킹이 500토큰 / 중복 100토큰이라 인접 청크는 100토큰만 공유하고 400토큰은
      서로 다른 내용이다. 중복이라고 지우면 20%를 아끼려다 80%를 버리게 된다.
      게다가 인접 청크가 나란히 높은 점수를 받았다는 건 그 구간이 실제로 관련
      있다는 신호다. 전체가 언어당 200청크 안팎이라 컨텍스트를 아낄 이유도 없다.
      겹치는 텍스트 정리는 LLM 에 넣기 직전 조립 단계에서, 버리는 대신 토큰 범위를
      이어붙이는 방식으로 해야 한다.
    """
    index = load_index(lang)
    clean = [q.strip() for q in queries if q and q.strip()]
    query_vectors = encode_queries(clean, model_name, device)
    meta = load_metadata(lang)

    scores = query_vectors @ index.vectors.T
    best_score = scores.max(axis=0)          # (N,) 청크별 최고점
    best_query = scores.argmax(axis=0)       # (N,) 그 점수를 낸 질의 번호

    order = np.argsort(-best_score)[:min(top_k, index.size)]
    return [
        _make_hit(index, meta, int(r), float(best_score[r]),
                  clean, int(best_query[r]), [int(best_query[r])])
        for r in order
    ]


# --------------------------------------------------------------------------
# 단독 실행
# --------------------------------------------------------------------------

def main() -> None:
    # Windows 콘솔(cp949)에서 한글이 깨지지 않도록
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    parser = argparse.ArgumentParser(
        description="질의를 임베딩해 채팅 청크를 검색한다.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("question", nargs="*", help="검색할 질문")
    parser.add_argument("--lang", default="ko", choices=list(LANGUAGES),
                        help="검색할 색인 언어 (기본: ko)")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help=f"보여줄 청크 수 (기본: {DEFAULT_TOP_K})")
    parser.add_argument("--rewrite", action="store_true",
                        help="Gemini 로 재작성/확장한 질의까지 함께 검색한다\n"
                             "(.env 의 GEMINI_API_KEY 필요)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"임베딩 모델 (기본: {DEFAULT_MODEL})")
    parser.add_argument("--device", default=None, help="cpu / cuda (기본: 자동 판별)")
    parser.add_argument("--full", action="store_true", help="청크 본문을 자르지 않고 전부 출력")
    args = parser.parse_args()

    question = " ".join(args.question) or "다음 워크숍이 언제야?"

    queries = [question]
    if args.rewrite:
        # google-genai 가 없어도 검색만 쓸 수 있도록 여기서만 임포트한다.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from query_rewrite import all_queries, rewrite_query

        print("Gemini 로 질의를 재작성하는 중...")
        result = rewrite_query(question, language=LANGUAGES[args.lang])
        if not result.ok:
            print(f"  [!] 재작성 실패, 원 질문만 씁니다: {result.error}")
        else:
            queries = all_queries(result)
            print(f"  재작성 : {result.rewritten}")
            for i, q in enumerate(result.expansions, 1):
                print(f"  확장 {i} : {q}")

    index = load_index(args.lang)
    print(f"\n색인    : {args.lang} ({LANGUAGES[args.lang]}) "
          f"- 청크 {index.size}개 x {index.dim}차원")
    print(f"모델    : {index.model or args.model}")
    print(f"질의    : {len(queries)}개")

    print("\n모델 로드 중... (최초 1회만 느립니다)")
    started = time.time()
    hits = search(queries, lang=args.lang, top_k=args.top_k,
                  model_name=args.model, device=args.device)
    elapsed = time.time() - started

    print(f"검색 완료 - {elapsed:.1f}초\n")
    for rank, hit in enumerate(hits, 1):
        print(f"[{rank}] {hit.score:.4f}  {hit.dialogue_id}#{hit.chunk_index}"
              f"  (토큰 {hit.token_start}~{hit.token_end})")
        if hit.started_at:
            who = ", ".join(hit.participants[:3])
            more = f" 외 {len(hit.participants) - 3}명" if len(hit.participants) > 3 else ""
            print(f"    {hit.started_at[:10]} · {hit.n_utterances}발화 · {who}{more}")
        if len(queries) > 1:
            print(f"    맞은 질의 {hit.matched_query_index + 1}: {hit.matched_query}")
        print(f"    {hit.text if args.full else hit.preview()}\n")


if __name__ == "__main__":
    main()
