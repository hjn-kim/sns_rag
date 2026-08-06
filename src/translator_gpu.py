#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JSON 파일 다국어 번역기 - 로컬 GPU (facebook/nllb-200-1.3B)

translator.py 와 같은 일을 하되, Google 무료 API 대신 NLLB 모델을 직접 돌린다.
호출 제한/차단이 없고 GPU 배치 추론이라 훨씬 빠르다.

    data/2024_마약류_범죄백서.json
      -> data/2024_마약류_범죄백서_영어.json
      -> data/2024_마약류_범죄백서_중국어.json
      -> data/2024_마약류_범죄백서_필리핀어.json
      -> data/2024_마약류_범죄백서_베트남어.json
      -> data/2024_마약류_범죄백서_러시아어.json

NLLB 는 문장 단위로 학습된 번역 모델이라 긴 덩어리를 그대로 넣으면 품질이 무너진다.
그래서 문자열을 조각으로 나눠 번역하고 다시 이어붙인다. 나누는 방식은 --mode 로 고른다.

  line     (기본) 줄 하나가 조각 하나. 원문 줄 구조가 그대로 보존된다.
                 PDF 에서 뽑은 표/목차처럼 짧은 줄이 많은 문서에 맞다.
  sentence       빈 줄로 나뉜 문단 안의 줄들을 이어붙인 뒤 문장 단위로 자른다.
                 줄바꿈으로 끊긴 산문의 번역 품질이 올라가지만
                 문단 안의 줄바꿈은 사라진다.

같은 조각은 한 번만 번역한다(중복 제거). 표 머리글처럼 반복되는 줄이 많으면 크게 줄어든다.

번역 결과 파일이 곧 캐시다. 캐시로 보는 곳은 data/ 바로 아래의

    data/{파일명}_{언어}.json

딱 하나다. 여기 있으면 그 안에서 번역이 끝난 문자열을 그대로 가져다 쓰고 나머지만
새로 번역한다. data/old_data/... 처럼 하위 폴더에 같은 이름이 있어도 캐시로 쓰지 않는다
(다른 설정으로 뽑은 결과가 조용히 섞이면 안 되므로). 폴더는 --cache-dir 로 바꾼다.

번역 도중에도 주기적으로 결과 파일을 갱신하므로 중간에 죽어도 다시 실행하면
이어서 진행한다. 처음부터 다시 번역하려면 --overwrite 를 준다.

사용 예:
    pip install -r requirements.txt

    python translator_gpu.py                              # 5개 언어 전부
    python translator_gpu.py --language 영어               # 한 언어만
    python translator_gpu.py --mode sentence              # 산문 위주 문서
    python translator_gpu.py --batch-size 16              # OOM 이 나면 낮춘다
    python translator_gpu.py --num-beams 1                # 품질 조금 낮추고 3배 빠르게
    python translator_gpu.py --overwrite                  # 기존 결과 무시하고 처음부터
    python translator_gpu.py --dry-run                    # 모델 없이 조각 수만 확인
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

# --------------------------------------------------------------------------
# 설정
# --------------------------------------------------------------------------

# 한국어 표기 -> NLLB(FLORES-200) 언어 코드
LANGUAGES: dict[str, str] = {
    "영어": "eng_Latn",
    "중국어": "zho_Hans",
    "필리핀어": "tgl_Latn",
    "베트남어": "vie_Latn",
    "러시아어": "rus_Cyrl",
}

SOURCE_LANG = "kor_Hang"

DEFAULT_MODEL = "facebook/nllb-200-1.3B"

# 캐시를 찾을 폴더. 이 스크립트가 src/ 에 있으므로 프로젝트 루트의 data/ 를 가리킨다.
# 실행 위치(cwd)에 좌우되면 안 되므로 파일 위치 기준으로 잡는다.
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data"

# 번역하지 않고 원문을 그대로 두는 key
#   source            : URL 또는 원본 파일명
#   document_format   : "txt" 같은 포맷 식별자
#   document_date     : 날짜
#   method / backend  : "PyMuPDF (fitz)" 같은 추출 도구 이름
SKIP_KEYS = {"source", "document_format", "document_date", "method", "backend"}

# 파일명과 동일한 식별자라서 번역 대신 언어 접미사만 붙이는 key
SUFFIX_KEYS = {"document_name"}

URL_RE = re.compile(r"^\s*(https?://|www\.)", re.IGNORECASE)
DATE_RE = re.compile(r"^\s*\d{4}\s*[-./년]\s*\d{1,2}\s*[-./월]?\s*\d{0,2}\s*[일.]?\s*$")
HAS_LETTER_RE = re.compile(r"[가-힣A-Za-z]")
# HWP/PDF 메타에 남는 <B4EBB0CBC2FB...> 같은 헥사 덩어리
HEX_BLOB_RE = re.compile(r"^\s*<[0-9A-Fa-f]{16,}>\s*$")
# PDF 추출기가 넣은 "--- [page 12] ---" 같은 구분선. 번역하지 않고 그대로 둔다.
PAGE_MARKER_RE = re.compile(r"^\s*-{2,}\s*\[[^\]]*\]\s*-{2,}\s*$")

# 문장 경계 (한국어 종결부호 포함). 폭 0 으로 잘라서 구분자를 잃지 않는다.
SENT_SPLIT_RE = re.compile(r"(?<=[.!?。．！？])")

# 조각 하나의 최대 글자 수. NLLB 는 문장 단위 모델이라 크게 잡지 않는다.
DEFAULT_MAX_CHARS = 300

# 새로 번역한 조각이 이만큼 쌓이면 결과 파일을 갱신한다.
# 저장할 때마다 문서 전체를 다시 조립하므로 너무 자주 하면 손해다.
FLUSH_EVERY = 1000


# --------------------------------------------------------------------------
# 번역 대상 판별
# --------------------------------------------------------------------------

def should_translate(key: str | None, value: str) -> bool:
    if key in SKIP_KEYS:
        return False
    if not value.strip():
        return False
    if URL_RE.match(value) or DATE_RE.match(value) or HEX_BLOB_RE.match(value):
        return False
    # "-", "1,000" 처럼 글자가 없는 값은 번역 대상이 아니다.
    if not HAS_LETTER_RE.search(value):
        return False
    return True


def is_translatable_piece(piece: str) -> bool:
    """조각 하나가 번역할 값인지. 숫자/기호만이거나 페이지 구분선이면 그대로 둔다."""
    s = piece.strip()
    if not s or not HAS_LETTER_RE.search(s):
        return False
    if PAGE_MARKER_RE.match(s):
        return False
    return True


# --------------------------------------------------------------------------
# 조각내기
# --------------------------------------------------------------------------

def _pack(pieces, limit: int) -> list[str]:
    """조각들을 상한 이하로 이어붙인다."""
    out: list[str] = []
    buf = ""
    for piece in pieces:
        if not piece:
            continue
        if buf and len(buf) + len(piece) > limit:
            out.append(buf)
            buf = piece
        else:
            buf += piece
    if buf:
        out.append(buf)
    return out


def _split_long(text: str, limit: int) -> list[str]:
    """상한을 넘는 조각을 문장 경계로, 그래도 넘으면 강제로 자른다."""
    if len(text) <= limit:
        return [text]
    out: list[str] = []
    for piece in _pack(SENT_SPLIT_RE.split(text), limit):
        while len(piece) > limit:
            out.append(piece[:limit])
            piece = piece[limit:]
        if piece:
            out.append(piece)
    return out


class Planner:
    """
    문자열을 (번역할지, 조각) 목록으로 쪼갠다.

    조각을 순서대로 이어붙이면 원문이 그대로 복원된다(line 모드).
    번역할 조각만 모아 배치로 돌린 뒤 다시 끼워넣는 방식이다.
    """

    def __init__(self, mode: str = "line", max_chars: int = DEFAULT_MAX_CHARS):
        self.mode = mode
        self.max_chars = max_chars
        self.plans: dict[str, list[tuple[bool, str]]] = {}

    def plan(self, text: str) -> list[tuple[bool, str]]:
        cached = self.plans.get(text)
        if cached is not None:
            return cached
        parts = self._plan_line(text) if self.mode == "line" else self._plan_sentence(text)
        self.plans[text] = parts
        return parts

    # ---- line 모드 : 줄 구조를 그대로 둔다 ------------------------------

    def _plan_line(self, text: str) -> list[tuple[bool, str]]:
        parts: list[tuple[bool, str]] = []
        for line in text.splitlines(keepends=True):
            body = line.rstrip("\r\n")
            tail = line[len(body):]          # 줄바꿈은 그대로 보존
            if is_translatable_piece(body):
                stripped = body.strip()
                head = body[:len(body) - len(body.lstrip())]
                foot = body[len(head) + len(stripped):]
                if head:
                    parts.append((False, head))
                for piece in _split_long(stripped, self.max_chars):
                    parts.append((True, piece))
                if foot:
                    parts.append((False, foot))
            elif body:
                parts.append((False, body))
            if tail:
                parts.append((False, tail))
        return parts

    # ---- sentence 모드 : 문단 안의 줄을 이어붙인 뒤 문장으로 자른다 ------

    def _plan_sentence(self, text: str) -> list[tuple[bool, str]]:
        parts: list[tuple[bool, str]] = []
        for bi, block in enumerate(re.split(r"\n\s*\n", text)):
            if bi:
                parts.append((False, "\n\n"))

            # 페이지 구분선은 따로 떼고, 나머지 줄은 이어붙여 한 덩어리로 만든다.
            units: list[str] = []
            buf: list[str] = []
            for line in block.splitlines():
                s = line.strip()
                if not s:
                    continue
                if PAGE_MARKER_RE.match(s):
                    if buf:
                        units.append(" ".join(buf))
                        buf = []
                    units.append(s)
                else:
                    buf.append(s)
            if buf:
                units.append(" ".join(buf))

            for ui, unit in enumerate(units):
                if ui:
                    parts.append((False, "\n"))
                if is_translatable_piece(unit):
                    for piece in _split_long(unit, self.max_chars):
                        parts.append((True, piece))
                else:
                    parts.append((False, unit))
        return parts


# --------------------------------------------------------------------------
# JSON 순회
# --------------------------------------------------------------------------

def walk_strings(node, key: str | None = None):
    """번역 대상인 문자열 value 를 순서대로 내놓는다."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield from walk_strings(v, k)
    elif isinstance(node, list):
        for item in node:
            yield from walk_strings(item, key)
    elif isinstance(node, str):
        if key not in SUFFIX_KEYS and should_translate(key, node):
            yield node


def is_done(text: str, planner: Planner, table: dict[str, str],
            done: dict[str, str]) -> bool:
    """문자열 하나가 끝까지 번역됐는지. 조각이 하나라도 비면 False."""
    if text in done:
        return True
    return all(piece in table for need, piece in planner.plan(text) if need)


def rebuild(node, planner: Planner, table: dict[str, str], lang_name: str,
            done: dict[str, str], key: str | None = None):
    """번역표(table)와 기존 결과(done)를 끼워넣어 JSON 을 다시 만든다."""
    if isinstance(node, dict):
        return {k: rebuild(v, planner, table, lang_name, done, k)
                for k, v in node.items()}
    if isinstance(node, list):
        return [rebuild(item, planner, table, lang_name, done, key) for item in node]
    if isinstance(node, str):
        if key in SUFFIX_KEYS:
            return f"{node}_{lang_name}"
        if not should_translate(key, node):
            return node

        previous = done.get(node)
        if previous is not None:      # 이전 실행에서 끝낸 문자열
            return previous

        parts = planner.plan(node)
        # 조각이 하나라도 빠지면 통째로 원문을 남긴다.
        # 반쯤 번역된 값을 저장하면 다음 실행의 harvest_done 이 '번역 완료'로
        # 오인해서(원문과 다르므로) 남은 조각이 영영 번역되지 않는다.
        if any(need and piece not in table for need, piece in parts):
            return node
        return "".join(table[piece] if need else piece for need, piece in parts)
    # int / float / bool / None 은 그대로
    return node


def harvest_done(node, previous, lang_name: str, key: str | None = None,
                 out: dict[str, str] | None = None) -> dict[str, str]:
    """
    원본과 기존 번역 결과를 나란히 훑어 이미 번역된 문자열을 (원문 -> 번역) 으로 모은다.

    rebuild 가 문자열 단위로 전부-아니면-전무 로 쓰므로, 결과가 원문과 다르면
    그 문자열은 끝까지 번역된 것이다. 구조가 어긋난 가지는 그냥 버린다
    (다른 문서의 결과를 잘못 집어도 조용히 섞이지 않도록).
    """
    if out is None:
        out = {}

    if isinstance(node, dict):
        if isinstance(previous, dict):
            for k, v in node.items():
                if k in previous:
                    harvest_done(v, previous[k], lang_name, k, out)
    elif isinstance(node, list):
        if isinstance(previous, list) and len(previous) == len(node):
            for item, prev_item in zip(node, previous):
                harvest_done(item, prev_item, lang_name, key, out)
    elif isinstance(node, str):
        if key not in SUFFIX_KEYS and should_translate(key, node):
            if isinstance(previous, str) and previous.strip() and previous != node:
                out[node] = previous
    return out


# --------------------------------------------------------------------------
# 모델
# --------------------------------------------------------------------------

def dtype_kwarg_name() -> str:
    """
    transformers 4.56 부터 from_pretrained 의 torch_dtype 이 dtype 으로 바뀌었다.

    옛 버전에 dtype 을 넘기면 예외 없이 **kwargs 로 흘러가 조용히 무시된다
    (fp16 로 돌린 줄 알았는데 fp32 로 도는 사고). 그래서 버전을 보고 고른다.
    """
    import transformers
    try:
        major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    except ValueError:
        return "dtype"
    return "dtype" if (major, minor) >= (4, 56) else "torch_dtype"


def load_model(model_name: str, device: str, dtype_name: str):
    import torch
    import transformers
    from transformers import AutoModelForSeq2SeqLM

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]

    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, **{dtype_kwarg_name(): dtype}
        )
    except ImportError as exc:
        if "DTensor" not in str(exc):
            raise
        sys.exit(
            "torch 와 transformers 버전이 맞지 않습니다.\n"
            f"  설치됨: torch {torch.__version__} / transformers {transformers.__version__}\n"
            "  torch 2.5 부터 DTensor 가 torch.distributed.tensor 로 옮겨졌는데,\n"
            "  transformers 5.x 는 새 경로만 씁니다.\n\n"
            "  해결 1 (권장) torch 를 올린다. CUDA 버전은 nvidia-smi 로 확인:\n"
            "    pip install -U torch --index-url https://download.pytorch.org/whl/cu124\n"
            "  해결 2 transformers 를 낮춘다 (팟의 torch/CUDA 를 건드리지 않음):\n"
            "    pip install 'transformers<5'"
        )

    model.to(device)
    model.eval()
    return model


def translate_segments(segments: list[str], tokenizer, model, device: str,
                       tgt_code: str, args, cache: dict[str, str], flush) -> None:
    """segments 를 배치로 번역해 cache 에 채운다."""
    import torch
    from tqdm import tqdm

    # NLLB 는 목표 언어 토큰을 디코더 첫 토큰으로 강제해서 번역 방향을 정한다.
    bos = tokenizer.convert_tokens_to_ids(tgt_code)
    if bos is None or bos == tokenizer.unk_token_id:
        sys.exit(f"토크나이저가 모르는 언어 코드입니다: {tgt_code}")
    tokenizer.src_lang = SOURCE_LANG

    # 길이가 비슷한 것끼리 묶어야 패딩 낭비가 줄어든다.
    ordered = sorted(segments, key=len)
    pending = 0

    for start in tqdm(range(0, len(ordered), args.batch_size),
                      desc="  번역", unit="batch"):
        batch = ordered[start:start + args.batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=args.max_tokens).to(device)
        with torch.inference_mode():
            out = model.generate(
                **enc,
                forced_bos_token_id=bos,
                num_beams=args.num_beams,
                max_new_tokens=min(512, int(enc["input_ids"].shape[1] * 2) + 32),
            )
        for src, tgt in zip(batch, tokenizer.batch_decode(out, skip_special_tokens=True)):
            cache[src] = tgt.strip()

        pending += len(batch)
        if pending >= FLUSH_EVERY:
            flush()
            pending = 0

    flush()


# --------------------------------------------------------------------------
# 실행
# --------------------------------------------------------------------------

def build_output_path(src: Path, out_dir: Path, lang_name: str) -> Path:
    return out_dir / f"{src.stem}_{lang_name}{src.suffix}"


def cache_path_for(cache_dir: Path, src: Path, lang_name: str) -> Path:
    """
    캐시로 볼 파일은 딱 한 곳, {cache_dir}/{파일명}_{언어}.json 이다.

    하위 폴더는 뒤지지 않는다. data/old_data/... 처럼 다른 실행 설정으로 만든
    같은 이름의 결과가 조용히 섞여 들어가면 안 되기 때문이다.
    """
    return cache_dir / f"{src.stem}_{lang_name}.json"


def warn_shadowed(cache_dir: Path, name: str) -> None:
    """하위 폴더에만 같은 이름이 있으면 '무시했다'고 알린다 (조용히 넘어가지 않도록)."""
    try:
        others = [p for p in cache_dir.rglob(name) if p.parent != cache_dir]
    except OSError:
        return
    if others:
        print(f"  (하위 폴더의 {len(others)}개 동명 파일은 캐시로 쓰지 않습니다: "
              f"{others[0].relative_to(cache_dir)}{' 외' if len(others) > 1 else ''})")


def load_previous(path: Path, data, lang_name: str, label: str) -> dict[str, str]:
    """
    캐시 파일에서 번역이 끝난 문자열을 회수한다.

    별도 캐시 포맷을 두지 않고 번역 결과 JSON 자체를 캐시로 쓴다. 원본과 구조가
    같으므로 나란히 훑어 (원문 -> 번역) 을 뽑아낼 수 있다.
    """
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as fp:
            previous = json.load(fp)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  [!] {label}를 읽지 못해 무시합니다: {exc}")
        return {}

    done = harvest_done(data, previous, lang_name)
    print(f"  {label}에서 문자열 {len(done):,}개 재사용: {path}")
    return done


def write_output(dst: Path, data, planner: Planner, table: dict[str, str],
                 lang_name: str, done: dict[str, str]) -> None:
    """번역 결과를 조립해 저장한다. 중간 저장도 이 함수로 같은 파일에 덮어쓴다."""
    translated = rebuild(data, planner, table, lang_name, done)
    dst.parent.mkdir(parents=True, exist_ok=True)
    # 이 파일이 곧 캐시라 반쯤 쓰다 죽으면 다음 실행이 통째로 날아간다. 원자적으로 바꾼다.
    tmp = dst.with_name(dst.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fp:
        json.dump(translated, fp, ensure_ascii=False, indent=2)
    tmp.replace(dst)


def run_language(lang_name: str, data, src: Path, out_dir: Path,
                 planner: Planner, tokenizer, model, device: str, args) -> None:
    dst = build_output_path(src, out_dir, lang_name)
    tgt_code = LANGUAGES[lang_name]

    print(f"\n{'=' * 70}")
    print(f"  {lang_name} ({tgt_code})  ->  {dst.name}")
    print(f"{'=' * 70}")

    # 캐시는 data/ 바로 아래의 {파일명}_{언어}.json 하나만 본다.
    cpath = cache_path_for(args.cache_dir, src, lang_name)
    done: dict[str, str] = {}
    if not args.overwrite:
        done = load_previous(cpath, data, lang_name, "캐시")
        if not done:
            warn_shadowed(args.cache_dir, cpath.name)
        # --out 을 따로 준 경우엔 지난 실행이 남긴 결과가 캐시가 아니라 dst 에 있다.
        # 이어하기가 끊기지 않도록 그쪽을 덧씌운다 (기본 설정에서는 dst == cpath).
        if dst != cpath:
            done.update(load_previous(dst, data, lang_name, "이전 출력"))

    # 아직 안 끝난 문자열의 조각만 모은다. done 이 문자열 단위로 세므로 여기도 중복을 뺀다.
    cache: dict[str, str] = {}
    pending: dict[str, None] = {}
    remaining: set[str] = set()
    for value in walk_strings(data):
        if value in done or value in remaining:
            continue
        remaining.add(value)
        for need, piece in planner.plan(value):
            if need:
                pending.setdefault(piece, None)

    todo = list(pending)
    if args.limit > 0:
        todo = todo[:args.limit]
        print(f"  --limit {args.limit} 적용")
    print(f"  남은 문자열 {len(remaining):,}개 -> 번역할 조각 {len(todo):,}개")

    started = time.time()
    if todo:
        def _flush() -> None:
            if not args.no_flush:
                write_output(dst, data, planner, cache, lang_name, done)

        translate_segments(todo, tokenizer, model, device, tgt_code,
                           args, cache, _flush)

    write_output(dst, data, planner, cache, lang_name, done)

    elapsed = time.time() - started
    missing = sum(1 for v in remaining if not is_done(v, planner, cache, done))
    print(f"\n  [{lang_name}] 완료 - {elapsed / 60:.1f}분")
    if missing:
        print(f"  [!] {missing:,}개 문자열은 번역되지 않아 원문(한국어)이 남았습니다. "
              f"다시 실행하면 이어서 진행합니다.")
    print(f"  -> {dst}")


def main() -> None:
    # Windows 콘솔(cp949)에서 한글 출력이 깨지지 않도록
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    parser = argparse.ArgumentParser(
        description="JSON 파일 하나를 여러 언어로 번역한다 (NLLB-200 / 로컬 GPU).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--data", default="data/2024_마약류_범죄백서.json", type=Path,
        help="번역할 원본 JSON 파일 (기본: data/2024_마약류_범죄백서.json)",
    )
    parser.add_argument(
        "--out", default=None, type=Path,
        help="결과를 저장할 폴더 (기본: --data 와 같은 위치)",
    )
    parser.add_argument(
        "--language", "--languages", nargs="+", default=list(LANGUAGES),
        choices=list(LANGUAGES), metavar="언어",
        help="번역할 언어. 선택: " + ", ".join(LANGUAGES) + "\n(기본: 전체)",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"번역 모델 (기본: {DEFAULT_MODEL})\n"
             "더 빠르게(품질 조금 손해): facebook/nllb-200-distilled-600M\n"
             "같은 크기의 증류 모델: facebook/nllb-200-distilled-1.3B",
    )
    parser.add_argument(
        "--mode", default="line", choices=["line", "sentence"],
        help="조각내는 방식 (기본: line)\n"
             "  line     - 줄 하나가 조각 하나. 줄 구조가 그대로 남는다\n"
             "  sentence - 문단 안의 줄을 이어붙여 문장으로 자른다. 산문 품질이 좋아진다",
    )
    parser.add_argument(
        "--max-chars", type=int, default=DEFAULT_MAX_CHARS,
        help=f"조각 하나의 최대 글자 수 (기본: {DEFAULT_MAX_CHARS})",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=256,
        help="입력 토큰 상한. 넘으면 잘린다 (기본: 256)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="한 번에 번역할 조각 수 (기본: 32)\n"
             "빔 서치는 batch x beams 만큼 동시에 도니 VRAM 을 그만큼 쓴다.\n"
             "1.3B 는 600M 보다 층이 두 배라 활성값도 그만큼 늘어난다.\n"
             "24GB(A5000) + beams 4 에서 32 가 적당하다. 64 는 긴 조각에서 OOM 위험",
    )
    parser.add_argument(
        "--num-beams", type=int, default=4,
        help="빔 서치 크기 (기본: 4 = 모델 기본값)\n"
             "1 로 두면 3배쯤 빨라지는 대신 품질이 조금 떨어진다",
    )
    parser.add_argument(
        "--device", default=None,
        help="cuda / cpu (기본: 자동 판별)",
    )
    parser.add_argument(
        "--dtype", default=None, choices=["float16", "bfloat16", "float32"],
        help="모델 dtype (기본: GPU 면 float16, CPU 면 float32)",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="언어당 번역할 조각 수 제한. 0 이면 전체 (테스트용)",
    )
    parser.add_argument(
        "--cache-dir", default=DEFAULT_CACHE_DIR, type=Path,
        help=f"캐시를 찾을 폴더 (기본: {DEFAULT_CACHE_DIR})\n"
             "이 폴더 바로 아래의 {파일명}_{언어}.json 만 캐시로 인정한다.\n"
             "하위 폴더에 같은 이름이 있어도 쓰지 않는다",
    )
    parser.add_argument(
        "--no-flush", action="store_true",
        help="번역 도중 중간 저장을 하지 않고 끝날 때 한 번만 쓴다\n"
             "(기본: 조각 1,000개마다 결과 파일을 갱신해 이어하기가 되게 한다)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="기존 결과 파일을 무시하고 처음부터 다시 번역한다\n"
             "(기본: 이미 번역된 문자열은 그대로 두고 나머지만 이어서 번역)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="모델을 올리지 않고 조각 통계만 출력한다",
    )
    args = parser.parse_args()

    src: Path = args.data.resolve()
    if not src.is_file():
        sys.exit(f"파일을 찾을 수 없습니다: {src}")
    out_dir: Path = (args.out or src.parent).resolve()
    args.cache_dir = args.cache_dir.resolve()

    with src.open(encoding="utf-8") as fp:
        data = json.load(fp)

    # ---- 조각내기 (모델 없이 먼저 끝낸다) --------------------------------
    planner = Planner(mode=args.mode, max_chars=args.max_chars)
    unique: dict[str, None] = {}
    total_pieces = 0
    for value in walk_strings(data):
        for need, piece in planner.plan(value):
            if need:
                total_pieces += 1
                unique.setdefault(piece, None)
    segments = list(unique)

    print(f"원본  : {src}")
    print(f"출력  : {out_dir}")
    print(f"캐시  : {args.cache_dir}{'' if args.cache_dir.is_dir() else '  (없음)'}")
    print(f"언어  : {', '.join(args.language)}")
    print(f"모델  : {args.model}")
    print(f"방식  : {args.mode} / 조각 최대 {args.max_chars}자")
    print(f"조각  : {total_pieces:,}개 -> 중복 제거 {len(segments):,}개 "
          f"({sum(len(s) for s in segments):,}자)")

    if not segments:
        sys.exit("번역할 조각이 없습니다.")

    if args.dry_run:
        print("\n--- 조각 샘플 ---")
        for piece in segments[:10]:
            print("  " + piece[:80])
        print("\nDRY-RUN: 모델을 올리지 않고 종료합니다.")
        return

    # ---- 모델 로드 --------------------------------------------------------
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA 를 쓸 수 없습니다. CUDA 빌드 torch 가 설치됐는지 확인하세요.\n"
                 "  pip install torch --index-url https://download.pytorch.org/whl/cu124")
    dtype_name = args.dtype or ("float16" if device == "cuda" else "float32")

    print(f"장치  : {device} / {dtype_name}")
    if device == "cpu":
        print("        [!] CPU 는 매우 느립니다. --limit 로 먼저 확인하세요.")
    else:
        print(f"        {torch.cuda.get_device_name(0)}")

    print("\n모델 로드 중... (최초 실행 시 다운로드에 시간이 걸립니다)")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, src_lang=SOURCE_LANG)
    model = load_model(args.model, device, dtype_name)

    total_started = time.time()
    for lang_name in args.language:
        run_language(lang_name, data, src, out_dir, planner,
                     tokenizer, model, device, args)

    print(f"\n전체 완료 - {(time.time() - total_started) / 60:.1f}분")


if __name__ == "__main__":
    main()
