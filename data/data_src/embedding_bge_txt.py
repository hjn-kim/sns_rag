#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TXT 문서 임베딩 (BAAI/bge-m3 / dense / .npz)

data/ 안의 *.txt 를 하나씩 순차로 처리한다. 본문을 토큰 수 기준으로 청킹한 뒤
bge-m3 로 dense 벡터를 만들어 파일마다 .npz 하나를 남긴다.

    data/ko마약류관리에관한법률.txt
      -> data/emb_bge_m3/ko마약류관리에관한법률_embeddings.npz
    data/vn부패및경제범죄...txt
      -> data/emb_bge_m3/vn부패및경제범죄..._embeddings.npz

기존 embedding.py(Harrier/JSON) 는 그대로 두고, 이 스크립트만 따로 쓴다.
저장 형식은 embedding.py 와 같아서 src/search.py 의 로더가 그대로 읽는다.
다만 bge-m3 는 1024차원이라 Harrier .npz 와 같은 폴더에 섞으면 안 된다.
그래서 출력 기본값을 data/emb_bge_m3 로 분리해 두었다.

.txt 라도 내용이 JSON 이면(en국가마약위협평가.txt 처럼) text/content/body 키를
찾아 본문만 뽑는다. 평문이면 파일 전체를 본문으로 본다.

청킹은 모델 토크나이저 기준이다. 문자 수가 아니라 실제로 모델이 보는 토큰 수로
자르므로 --chunk-size 가 모델 입력 길이와 정확히 일치한다.

사용 예:
    python data/data_src/embedding_bge_txt.py --dry-run     # 청킹 결과만 확인
    python data/data_src/embedding_bge_txt.py               # 전체 (512/124)
    python data/data_src/embedding_bge_txt.py --limit 5     # 파일당 앞 5청크만
    python data/data_src/embedding_bge_txt.py --data data/ko마약류관리에관한법률.txt
    python data/data_src/embedding_bge_txt.py --batch-size 32 --device cuda

저장 형식 (.npz):
    embeddings   float32 (N, 1024)  L2 정규화된 dense 벡터
    texts        <U      (N,)       청크 원문
    chunk_index  int32   (N,)       청크 순번
    token_start  int32   (N,)       원문 토큰 기준 시작 위치
    token_end    int32   (N,)       원문 토큰 기준 끝 위치
    token_count  int32   (N,)       청크 토큰 수
    info         str                 설정/출처를 담은 JSON 문자열
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# 이 파일은 data/data_src/ 에 있다. 두 단계 위가 프로젝트 루트.
REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATA = REPO_ROOT / "data"
DEFAULT_OUT = REPO_ROOT / "data" / "emb_bge_m3"
DEFAULT_MODEL = "BAAI/bge-m3"

DEFAULT_CHUNK_SIZE = 512
DEFAULT_OVERLAP = 124

# .txt 안이 JSON 일 때 본문으로 볼 key 후보 (순서대로 찾는다)
TEXT_KEY_CANDIDATES = ("text", "content", "body")


# --------------------------------------------------------------------------
# 본문 추출
# --------------------------------------------------------------------------

def read_text(path: Path) -> str:
    """UTF-8 로 읽고, 안 되면 cp949 로 한 번 더 시도한다."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="cp949", errors="replace")


def extract_text(raw: str, field: str | None) -> tuple[str, str]:
    """
    (본문, 어디서 뽑았는지) 를 돌려준다.

    .txt 지만 내용이 JSON object 인 파일이 섞여 있다. 그런 파일은 source/method
    같은 메타까지 임베딩하지 않도록 본문 key 만 골라낸다.
    JSON 이 아니면 파일 전체가 본문이다.
    """
    stripped = raw.lstrip()
    if not stripped.startswith("{"):
        return raw, "(평문)"

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw, "(평문)"
    if not isinstance(data, dict):
        return raw, "(평문)"

    if field:
        if not isinstance(data.get(field), str):
            sys.exit(f"'{field}' 를 문자열로 찾지 못했습니다. 있는 key: {list(data)}")
        return data[field], field

    for candidate in TEXT_KEY_CANDIDATES:
        if isinstance(data.get(candidate), str) and data[candidate].strip():
            return data[candidate], candidate

    # JSON 이긴 한데 본문 key 가 없다. 통째로 임베딩하기보다 알려 주고 멈춘다.
    sys.exit(f"JSON 인데 text/content/body 가 없습니다. 있는 key: {list(data)}")


# --------------------------------------------------------------------------
# 토큰 기준 청킹
# --------------------------------------------------------------------------

def chunk_by_tokens(text: str, tokenizer, size: int, overlap: int) -> list[dict]:
    """
    토큰 size 개씩, 앞 청크와 overlap 개를 겹치도록 자른다.

    슬라이딩 간격(stride)은 size - overlap 이다. 512/124 -> 388 씩 전진.
    """
    if overlap >= size:
        sys.exit(f"--overlap({overlap}) 은 --chunk-size({size}) 보다 작아야 합니다.")

    ids = tokenizer.encode(text, add_special_tokens=False)
    stride = size - overlap

    chunks: list[dict] = []
    for start in range(0, len(ids), stride):
        window = ids[start:start + size]
        if not window:
            break
        # 마지막 조각이 앞 청크에 완전히 포함되면 버린다.
        if chunks and len(window) <= overlap:
            break
        chunks.append({
            "index": len(chunks),
            "token_start": start,
            "token_end": start + len(window),
            "token_count": len(window),
            "text": tokenizer.decode(window, skip_special_tokens=True),
        })
        if start + size >= len(ids):
            break
    return chunks


# --------------------------------------------------------------------------
# 저장
# --------------------------------------------------------------------------

def save_npz(path: Path, vectors: np.ndarray, chunks: list[dict], info: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        embeddings=vectors.astype(np.float32),
        texts=np.array([c["text"] for c in chunks]),
        chunk_index=np.array([c["index"] for c in chunks], dtype=np.int32),
        token_start=np.array([c["token_start"] for c in chunks], dtype=np.int32),
        token_end=np.array([c["token_end"] for c in chunks], dtype=np.int32),
        token_count=np.array([c["token_count"] for c in chunks], dtype=np.int32),
        info=np.array(json.dumps(info, ensure_ascii=False)),
    )


# --------------------------------------------------------------------------
# 파일 하나 처리
# --------------------------------------------------------------------------

def collect_files(data: Path) -> list[Path]:
    """--data 가 파일이면 그 파일만, 폴더면 바로 아래 *.txt 를 이름순으로."""
    if data.is_file():
        return [data]
    return sorted(data.glob("*.txt"))


def run_file(src: Path, dst: Path, tokenizer, model, device: str, args) -> dict | None:
    """파일 하나를 청킹해 임베딩한다. 건너뛰면 None."""
    if dst.exists() and not args.overwrite:
        print("  건너뜀 (이미 있음). 다시 만들려면 --overwrite")
        return None

    text, origin = extract_text(read_text(src), args.field)

    chunks = chunk_by_tokens(text, tokenizer, args.chunk_size, args.overlap)
    if not chunks:
        print("  [!] 청크가 없습니다. 건너뜁니다.")
        return None

    counts = [c["token_count"] for c in chunks]
    print(f"  본문 {origin} {len(text):,}자 -> 청크 {len(chunks)}개 "
          f"(토큰 최소 {min(counts)} / 평균 {sum(counts) // len(counts)} / 최대 {max(counts)})")

    if args.limit > 0:
        chunks = chunks[:args.limit]
        print(f"  --limit {args.limit} 적용 -> {len(chunks)}개만 임베딩")

    if model is None:      # dry-run
        print("  미리보기: " + chunks[0]["text"][:120].replace("\n", " ⏎ "))
        return None

    started = time.time()
    # bge-m3 는 문서에도 질의에도 instruction 프리픽스를 붙이지 않는다.
    vectors = model.encode(
        [c["text"] for c in chunks],
        batch_size=args.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    elapsed = time.time() - started

    # 모델이 fp16/bf16 으로 돌면 정규화 결과가 노름 1 에서 조금 어긋난다.
    # 내적을 그대로 코사인 유사도로 쓰려면 float32 에서 한 번 더 맞춘다.
    vectors = vectors.astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    info = {
        "source": src.name,
        "source_field": origin,
        "model": args.model,
        "dim": int(vectors.shape[1]),
        "normalized": True,
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "n_chunks": len(chunks),
        "device": device,
        "elapsed_sec": round(elapsed, 1),
    }
    save_npz(dst, vectors, chunks, info)

    print(f"  임베딩 {vectors.shape[0]}개 x {vectors.shape[1]}차원 - "
          f"{elapsed / 60:.1f}분 ({elapsed / max(len(chunks), 1):.2f}초/청크)")
    print(f"  -> {dst.name} ({dst.stat().st_size / 1024 / 1024:.1f}MB)")
    return info


# --------------------------------------------------------------------------
# 실행
# --------------------------------------------------------------------------

def main() -> None:
    # Windows 콘솔(cp949)에서 한글 출력이 깨지지 않도록
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    parser = argparse.ArgumentParser(
        description="data/*.txt 를 토큰 기준으로 청킹해 bge-m3 dense 임베딩을 만든다.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--data", default=DEFAULT_DATA, type=Path,
        help=f"임베딩할 TXT 파일 또는 폴더 (기본: {DEFAULT_DATA})\n"
             "폴더면 바로 아래 *.txt 를 이름순으로 하나씩 처리한다",
    )
    parser.add_argument(
        "--out", default=DEFAULT_OUT, type=Path,
        help=f"결과를 저장할 폴더 (기본: {DEFAULT_OUT})\n"
             "파일마다 {파일명}_embeddings.npz 를 만든다\n"
             "차원이 다르므로 Harrier 용 data/emb 와 섞지 말 것",
    )
    parser.add_argument(
        "--field", default=None,
        help="내용이 JSON 일 때 본문으로 쓸 key\n(기본: text -> content -> body 순으로 자동 탐색)",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"임베딩 모델 (기본: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
        help=f"청크당 토큰 수 (기본: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--overlap", type=int, default=DEFAULT_OVERLAP,
        help=f"앞 청크와 겹치는 토큰 수 (기본: {DEFAULT_OVERLAP})",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="한 번에 인코딩할 청크 수 (기본: 8)\nGPU 면 32 정도까지 올려도 된다",
    )
    parser.add_argument(
        "--device", default=None,
        help="cpu / cuda (기본: 자동 판별)",
    )
    parser.add_argument(
        "--dtype", default="auto",
        help="모델 연산 dtype: auto / float32 / float16 (기본: auto)",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="파일당 앞에서부터 N개 청크만 임베딩. 0 이면 전체 (테스트용)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="이미 만들어진 .npz 도 다시 만든다 (기본: 건너뜀)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="모델을 올리지 않고 청킹 결과만 출력한다",
    )
    args = parser.parse_args()

    data_root: Path = args.data.resolve()
    if not data_root.exists():
        sys.exit(f"경로를 찾을 수 없습니다: {data_root}")

    out_dir: Path = args.out.resolve()

    files = collect_files(data_root)
    if not files:
        sys.exit(f"TXT 파일이 없습니다: {data_root}")

    print(f"원본    : {data_root}")
    print(f"출력    : {out_dir}")
    print(f"대상    : {len(files)}개 파일")
    print(f"모델    : {args.model}")
    print(f"청킹    : {args.chunk_size}토큰 / 중복 {args.overlap}토큰 "
          f"(간격 {args.chunk_size - args.overlap})")

    # 토크나이저는 모델과 동일한 것을 쓴다 (청크 길이가 모델 입력 길이와 일치하도록).
    print("\n토크나이저 로드 중...")
    from transformers import AutoTokenizer, logging as hf_logging
    # 본문 전체를 한 번에 토크나이즈하면 "sequence length is longer than..." 경고가 뜬다.
    # 자른 뒤에 모델로 보내므로 실제 문제가 아니어서 끈다.
    hf_logging.set_verbosity_error()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # ---- 모델은 한 번만 올리고 모든 파일에 재사용한다 ----------------------
    model = None
    device = args.device or "cpu"
    if not args.dry_run:
        print("모델 로드 중... (최초 실행 시 다운로드에 약 2.2GB)")
        from sentence_transformers import SentenceTransformer
        import torch

        if args.device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # transformers 4.56 부터 torch_dtype 이 dtype 으로 바뀌었다. 옛 버전에 dtype 을
        # 넘기면 예외 없이 무시되어 fp32 로 도니, 버전을 보고 인자 이름을 고른다.
        import transformers
        try:
            _tf = tuple(int(x) for x in transformers.__version__.split(".")[:2])
        except ValueError:
            _tf = (99, 99)
        dtype_key = "dtype" if _tf >= (4, 56) else "torch_dtype"

        model = SentenceTransformer(args.model, device=device,
                                    model_kwargs={dtype_key: args.dtype})
        # bge-m3 기본 입력 길이는 8192 다. 우리는 청크가 그보다 훨씬 짧으니
        # 잘림 없이 돌면서 메모리도 아끼도록 청크 길이에 맞춰 줄인다(특수토큰 2개 여유).
        model.max_seq_length = args.chunk_size + 2

        print(f"장치    : {device}")
        if device == "cpu":
            print("          [!] CPU 라 느립니다. --limit 로 먼저 소규모 확인을 권합니다.")

    total_started = time.time()
    done = skipped = 0
    total_chunks = 0

    for idx, src in enumerate(files, 1):
        dst = out_dir / f"{src.stem}_embeddings.npz"
        print(f"\n[{idx}/{len(files)}] {src.name}")
        try:
            info = run_file(src, dst, tokenizer, model, device, args)
        except OSError as exc:
            print(f"  [!] 읽기 실패, 건너뜁니다: {exc}")
            skipped += 1
            continue
        if info is None:
            skipped += 1
        else:
            done += 1
            total_chunks += info["n_chunks"]

    if args.dry_run:
        print("\nDRY-RUN: 모델을 올리지 않고 종료합니다.")
        return

    elapsed = time.time() - total_started
    print(f"\n전체 완료 - {elapsed / 60:.1f}분")
    print(f"  임베딩 {done}개 파일 / 청크 {total_chunks:,}개 / 건너뜀 {skipped}개")


if __name__ == "__main__":
    main()
