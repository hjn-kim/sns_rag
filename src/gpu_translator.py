#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SNS 대화 JSON 다국어 번역기 - 로컬 GPU (facebook/nllb-200-1.3B)

cpu_translator.py 와 같은 일을 하되, Google 무료 API 대신 NLLB 모델을 직접 돌린다.
호출 제한/차단이 없고 GPU 배치 추론이라 훨씬 빠르다.

data/sns_sample.json 에서 아래 항목만 번역해 언어별 파일로 낸다.

    utterance / type / topic / gender / age / residentialProvince

나머지 값(대화 ID, 참가자 ID, 날짜, 시각, 개수 등)은 손대지 않는다.

    data/sns_sample.json
      -> data/sns_sample_en.json    영어
      -> data/sns_sample_vi.json    베트남어
      -> data/sns_sample_fil.json   필리핀어
      -> data/sns_sample_ru.json    러시아어
      -> data/sns_sample_zh.json    중국어

문자열은 #@이름# 같은 마스킹 토큰을 기준으로 조각내고, 토큰은 번역 대상에서 뺀다.
그래서 번역 결과에도 토큰이 원래 자리에 그대로 남는다. 같은 조각은 한 번만 번역한다
(발화문 29만 개 -> 고유 조각 23만 개).

번역 결과 파일이 곧 캐시다. 캐시로 보는 곳은 data/ 바로 아래의

    data/{파일명}_{언어}.json

딱 하나다. 여기 있으면 그 안에서 번역이 끝난 문자열을 그대로 가져다 쓰고 나머지만
새로 번역한다. data/old_data/... 처럼 하위 폴더에 같은 이름이 있어도 캐시로 쓰지 않는다
(다른 설정으로 뽑은 결과가 조용히 섞이면 안 되므로). 폴더는 --cache-dir 로 바꾼다.

번역 도중에도 주기적으로 결과 파일을 갱신하므로 중간에 죽어도 다시 실행하면
이어서 진행한다. 처음부터 다시 번역하려면 --overwrite 를 준다.

사용 예:
    pip install torch transformers tqdm

    python src/gpu_translator.py                           # 한국어 -> 나머지 5개 언어
    python src/gpu_translator.py --language en             # 한 언어만

    # 영어 원문(WhatsApp)을 한국어 포함 5개 언어로
    python src/gpu_translator.py --data data/WhatsappChat.json --source-lang en
    python src/gpu_translator.py --batch-size 16           # OOM 이 나면 낮춘다
    python src/gpu_translator.py --num-beams 1             # 품질 조금 낮추고 3배 빠르게
    python src/gpu_translator.py --limit 200               # 200조각만 번역해 확인
    python src/gpu_translator.py --overwrite               # 기존 결과 무시하고 처음부터
    python src/gpu_translator.py --dry-run                 # 모델 없이 조각 수만 확인
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

# 출력 파일 접미사 -> (표시 이름, NLLB(FLORES-200) 언어 코드)
LANGUAGES: dict[str, tuple[str, str]] = {
    "ko": ("한국어", "kor_Hang"),
    "en": ("영어", "eng_Latn"),
    "vi": ("베트남어", "vie_Latn"),
    "fil": ("필리핀어", "tgl_Latn"),
    "ru": ("러시아어", "rus_Cyrl"),
    "zh": ("중국어", "zho_Hans"),
}

# 원문 언어. --source-lang 으로 바꾼다.
#   한국어 SNS 데이터  : ko  (기본)
#   영어 WhatsApp 데이터: en
DEFAULT_SOURCE = "ko"

DEFAULT_MODEL = "facebook/nllb-200-1.3B"

# 캐시를 찾을 폴더. 이 스크립트가 src/ 에 있으므로 프로젝트 루트의 data/ 를 가리킨다.
# 실행 위치(cwd)에 좌우되면 안 되므로 파일 위치 기준으로 잡는다.
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data"

# 번역할 key. 이 셋에 없는 값은 문자열이라도 건드리지 않는다.
#   utterance           : 발화문
#   type                : 대화 유형 ("일상 대화" 등)
#   topic               : 대화 주제 ("개인 및 관계" 등)
#   gender              : 참가자 성별 ("남성" / "여성")
#   age                 : 참가자 연령대 ("20대" 등)
#   residentialProvince : 참가자 거주 지역 ("대구광역시" 등)
TRANSLATE_KEYS = {"utterance", "type", "topic",
                  "gender", "age", "residentialProvince"}

# 개인정보 마스킹 토큰. 번역하지 않고 그대로 둔다.
#   #@이름#  #@계정#  #@URL#  ...          한 칸짜리
#   #@시스템#사진#  #@이모티콘#하트#        두 칸짜리 (뒤에 세부 라벨이 붙는다)
# 두 칸짜리는 앞의 두 종류뿐이다. 다른 토큰까지 두 칸으로 읽으면
# "#@이름#이랑" 의 "이랑" 까지 토큰으로 삼켜버린다.
PLACEHOLDER_RE = re.compile(r"#@(?:시스템|이모티콘)#[^#@\s]*#|#@[^#@\s]*#")

# 번역할 글자가 있는지. 자모(ㅋㅋㅋ, ㅇㅇ)도 채팅에서는 뜻이 있으므로 대상에 넣는다.
HAS_LETTER_RE = re.compile(r"[가-힣ㄱ-ㅎㅏ-ㅣA-Za-z]")

# 문장 경계 (한국어 종결부호 포함). 폭 0 으로 잘라서 구분자를 잃지 않는다.
SENT_SPLIT_RE = re.compile(r"(?<=[.!?。．！？])")

# 조각 하나의 최대 글자 수. NLLB 는 문장 단위 모델이라 크게 잡지 않는다.
# 발화문은 99%가 41자 이하라 걸릴 일이 거의 없다.
DEFAULT_MAX_CHARS = 300

# 새로 번역한 조각이 이만큼 쌓이면 결과 파일을 갱신한다.
# 저장할 때마다 문서 전체를 다시 조립하므로 너무 자주 하면 손해다.
FLUSH_EVERY = 20000


# --------------------------------------------------------------------------
# 조각내기
# --------------------------------------------------------------------------

def is_translatable_piece(piece: str) -> bool:
    """숫자/기호만 있는 조각은 번역해봐야 얻을 게 없다."""
    return bool(HAS_LETTER_RE.search(piece))


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
    문자열 하나를 (번역할지, 조각) 목록으로 쪼갠다.

    조각을 순서대로 이어붙이면 원문이 그대로 복원된다. 마스킹 토큰과 앞뒤 공백은
    '번역 안 함' 으로 남기므로 번역 결과에서도 자리와 모양이 유지된다.
    (모델에 토큰을 그대로 넣으면 #@이름# 이 번역/변형돼 마스킹이 깨진다)
    """

    def __init__(self, max_chars: int = DEFAULT_MAX_CHARS):
        self.max_chars = max_chars
        self.plans: dict[str, list[tuple[bool, str]]] = {}

    def plan(self, text: str) -> list[tuple[bool, str]]:
        cached = self.plans.get(text)
        if cached is not None:
            return cached
        parts: list[tuple[bool, str]] = []
        pos = 0
        for m in PLACEHOLDER_RE.finditer(text):
            self._add_text(parts, text[pos:m.start()])
            parts.append((False, m.group()))
            pos = m.end()
        self._add_text(parts, text[pos:])
        self.plans[text] = parts
        return parts

    def _add_text(self, parts: list[tuple[bool, str]], text: str) -> None:
        if not text:
            return
        stripped = text.strip()
        if not stripped or not is_translatable_piece(stripped):
            parts.append((False, text))
            return
        head = text[:len(text) - len(text.lstrip())]
        foot = text[len(head) + len(stripped):]
        if head:
            parts.append((False, head))
        for piece in _split_long(stripped, self.max_chars):
            parts.append((True, piece))
        if foot:
            parts.append((False, foot))


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
        if key in TRANSLATE_KEYS and node.strip():
            yield node


def is_done(text: str, planner: Planner, table: dict[str, str],
            done: dict[str, str]) -> bool:
    """문자열 하나가 끝까지 번역됐는지. 조각이 하나라도 비면 False."""
    if text in done:
        return True
    return all(piece in table for need, piece in planner.plan(text) if need)


def rebuild(node, planner: Planner, table: dict[str, str],
            done: dict[str, str], key: str | None = None):
    """번역표(table)와 기존 결과(done)를 끼워넣어 JSON 을 다시 만든다."""
    if isinstance(node, dict):
        return {k: rebuild(v, planner, table, done, k) for k, v in node.items()}
    if isinstance(node, list):
        return [rebuild(item, planner, table, done, key) for item in node]
    if isinstance(node, str):
        if key not in TRANSLATE_KEYS or not node.strip():
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


def harvest_done(node, previous, key: str | None = None,
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
                    harvest_done(v, previous[k], k, out)
    elif isinstance(node, list):
        if isinstance(previous, list) and len(previous) == len(node):
            for item, prev_item in zip(node, previous):
                harvest_done(item, prev_item, key, out)
    elif isinstance(node, str):
        if key in TRANSLATE_KEYS and node.strip():
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

    # 배치마다 이 경고가 찍혀 진행바를 덮는다:
    #   Both `max_new_tokens` (=46) and `max_length`(=200) seem to have been set.
    # NLLB 설정의 max_length=200 과 우리가 넘기는 max_new_tokens 가 겹쳐서 나는 것이고,
    # 이기는 쪽은 의도대로 max_new_tokens 다. 번역에는 영향이 없으므로 끈다.
    transformers.logging.set_verbosity_error()

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
    tokenizer.src_lang = args.source_code

    # 길이가 비슷한 것끼리 묶어야 패딩 낭비가 줄어든다.
    # 발화문은 대부분 10자 안팎이라 이 정렬 효과가 특히 크다.
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

def build_output_path(src: Path, out_dir: Path, code: str) -> Path:
    """data/sns_5k.json -> {out_dir}/sns_sample_en.json"""
    return out_dir / f"{src.stem}_{code}{src.suffix}"


def cache_path_for(cache_dir: Path, src: Path, code: str) -> Path:
    """
    캐시로 볼 파일은 딱 한 곳, {cache_dir}/{파일명}_{언어}.json 이다.

    하위 폴더는 뒤지지 않는다. data/old_data/... 처럼 다른 실행 설정으로 만든
    같은 이름의 결과가 조용히 섞여 들어가면 안 되기 때문이다.
    """
    return cache_dir / f"{src.stem}_{code}.json"


def warn_shadowed(cache_dir: Path, name: str) -> None:
    """하위 폴더에만 같은 이름이 있으면 '무시했다'고 알린다 (조용히 넘어가지 않도록)."""
    try:
        others = [p for p in cache_dir.rglob(name) if p.parent != cache_dir]
    except OSError:
        return
    if others:
        print(f"  (하위 폴더의 {len(others)}개 동명 파일은 캐시로 쓰지 않습니다: "
              f"{others[0].relative_to(cache_dir)}{' 외' if len(others) > 1 else ''})")


def load_previous(path: Path, data, label: str) -> dict[str, str]:
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

    done = harvest_done(data, previous)
    print(f"  {label}에서 문자열 {len(done):,}개 재사용: {path}")
    return done


def write_output(dst: Path, data, planner: Planner, table: dict[str, str],
                 done: dict[str, str]) -> None:
    """번역 결과를 조립해 저장한다. 중간 저장도 이 함수로 같은 파일에 덮어쓴다."""
    translated = rebuild(data, planner, table, done)
    dst.parent.mkdir(parents=True, exist_ok=True)
    # 이 파일이 곧 캐시라 반쯤 쓰다 죽으면 다음 실행이 통째로 날아간다. 원자적으로 바꾼다.
    tmp = dst.with_name(dst.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fp:
        json.dump(translated, fp, ensure_ascii=False, indent=2)
    tmp.replace(dst)


def run_language(code: str, data, src: Path, out_dir: Path,
                 planner: Planner, tokenizer, model, device: str, args) -> None:
    label, tgt_code = LANGUAGES[code]
    dst = build_output_path(src, out_dir, code)

    print(f"\n{'=' * 70}")
    print(f"  {label} ({tgt_code})  ->  {dst.name}")
    print(f"{'=' * 70}")

    # 캐시는 data/ 바로 아래의 {파일명}_{언어}.json 하나만 본다.
    cpath = cache_path_for(args.cache_dir, src, code)
    done: dict[str, str] = {}
    if not args.overwrite:
        done = load_previous(cpath, data, "캐시")
        if not done:
            warn_shadowed(args.cache_dir, cpath.name)
        # --out 을 따로 준 경우엔 지난 실행이 남긴 결과가 캐시가 아니라 dst 에 있다.
        # 이어하기가 끊기지 않도록 그쪽을 덧씌운다 (기본 설정에서는 dst == cpath).
        if dst != cpath:
            done.update(load_previous(dst, data, "이전 출력"))

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
                write_output(dst, data, planner, cache, done)

        translate_segments(todo, tokenizer, model, device, tgt_code,
                           args, cache, _flush)

    write_output(dst, data, planner, cache, done)

    elapsed = time.time() - started
    missing = sum(1 for v in remaining if not is_done(v, planner, cache, done))
    print(f"\n  [{label}] 완료 - {elapsed / 60:.1f}분")
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
        description="SNS 대화 JSON 의 지정 항목을 여러 언어로 번역한다 "
                    "(NLLB-200 / 로컬 GPU).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--data", default="data/sns_sample.json", type=Path,
        help="번역할 원본 JSON 파일 (기본: data/sns_sample.json)",
    )
    parser.add_argument(
        "--out", default=None, type=Path,
        help="결과를 저장할 폴더 (기본: --data 와 같은 위치)",
    )
    parser.add_argument(
        "--source-lang", default=DEFAULT_SOURCE, metavar="언어",
        help=f"원문 언어 (기본: {DEFAULT_SOURCE})\n"
             "LANGUAGES 의 접미사(ko, en ...) 또는 NLLB 코드(kor_Hang)를 준다",
    )
    parser.add_argument(
        "--language", "--languages", nargs="+", default=None,
        choices=list(LANGUAGES), metavar="언어",
        help="번역할 언어. 선택: "
             + ", ".join(f"{c}({n})" for c, (n, _) in LANGUAGES.items())
             + "\n(기본: 원문 언어를 뺀 전체)",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"번역 모델 (기본: {DEFAULT_MODEL})\n"
             "더 빠르게(품질 조금 손해): facebook/nllb-200-distilled-600M\n"
             "같은 크기의 증류 모델: facebook/nllb-200-distilled-1.3B",
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
        "--batch-size", type=int, default=64,
        help="한 번에 번역할 조각 수 (기본: 64)\n"
             "빔 서치는 batch x beams 만큼 동시에 도니 VRAM 을 그만큼 쓴다.\n"
             "발화문은 대부분 짧아 백서류보다 배치를 크게 잡을 수 있다.\n"
             "OOM 이 나면 32, 16 으로 낮춘다",
    )
    parser.add_argument(
        "--num-beams", type=int, default=4,
        help="빔 서치 크기 (기본: 4 = 모델 기본값)\n"
             "1 로 두면 3배쯤 빨라지는 대신 품질이 조금 떨어진다.\n"
             "조각이 23만 개라 전체를 돌릴 때는 1 도 고려할 만하다",
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
        help=f"번역 도중 중간 저장을 하지 않고 끝날 때 한 번만 쓴다\n"
             f"(기본: 조각 {FLUSH_EVERY:,}개마다 결과 파일을 갱신해 이어하기가 되게 한다)",
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

    # 원문 언어는 접미사(en)로도 NLLB 코드(eng_Latn)로도 줄 수 있다.
    source_key = args.source_lang if args.source_lang in LANGUAGES else None
    args.source_code = LANGUAGES[source_key][1] if source_key else args.source_lang
    source_label = LANGUAGES[source_key][0] if source_key else args.source_code

    # 기본은 원문 언어를 뺀 나머지 전부. 자기 자신으로 번역할 이유가 없다.
    if args.language is None:
        args.language = [c for c in LANGUAGES if c != source_key]
    elif source_key in args.language:
        print(f"  (원문과 같은 {source_key} 는 건너뜁니다)")
        args.language = [c for c in args.language if c != source_key]
    if not args.language:
        sys.exit("번역할 언어가 없습니다.")

    print("원본 읽는 중...")
    with src.open(encoding="utf-8") as fp:
        data = json.load(fp)

    # ---- 조각내기 (모델 없이 먼저 끝낸다) --------------------------------
    planner = Planner(max_chars=args.max_chars)
    unique: dict[str, None] = {}
    values = 0
    total_pieces = 0
    for value in walk_strings(data):
        values += 1
        for need, piece in planner.plan(value):
            if need:
                total_pieces += 1
                unique.setdefault(piece, None)
    segments = list(unique)

    print(f"원본  : {src}")
    print(f"출력  : {out_dir}")
    print(f"캐시  : {args.cache_dir}{'' if args.cache_dir.is_dir() else '  (없음)'}")
    print(f"원문  : {source_label} ({args.source_code})")
    print(f"언어  : {', '.join(args.language)}")
    print(f"항목  : {', '.join(sorted(TRANSLATE_KEYS))}")
    print(f"모델  : {args.model}")
    print(f"대상  : 문자열 {values:,}개 -> 조각 {total_pieces:,}개 "
          f"-> 중복 제거 {len(segments):,}개 "
          f"({sum(len(s) for s in segments):,}자)")

    if not segments:
        sys.exit("번역할 조각이 없습니다.")

    if args.dry_run:
        print("\n--- 조각 샘플 ---")
        for piece in segments[:15]:
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
        print("        [!] CPU 는 매우 느립니다. --limit 로 먼저 확인하거나 "
              "cpu_translator.py 를 쓰세요.")
    else:
        print(f"        {torch.cuda.get_device_name(0)}")

    print("\n모델 로드 중... (최초 실행 시 다운로드에 시간이 걸립니다)")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, src_lang=args.source_code)
    model = load_model(args.model, device, dtype_name)

    total_started = time.time()
    for code in args.language:
        run_language(code, data, src, out_dir, planner,
                     tokenizer, model, device, args)

    print(f"\n전체 완료 - {(time.time() - total_started) / 60:.1f}분")


if __name__ == "__main__":
    main()
