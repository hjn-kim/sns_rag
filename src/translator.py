#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JSON 파일 다국어 번역기 (deep-translator / Google)

--data 로 지정한 JSON 파일 하나를 언어별 파일로 번역한다.
JSON 의 key 는 그대로 두고 value(문자열)만 번역한다.

    data/2024_마약류_범죄백서.json
      -> data/2024_마약류_범죄백서_영어.json
      -> data/2024_마약류_범죄백서_중국어.json
      -> data/2024_마약류_범죄백서_필리핀어.json
      -> data/2024_마약류_범죄백서_베트남어.json
      -> data/2024_마약류_범죄백서_러시아어.json

Google 무료 엔드포인트는 요청이 몰리면 차단한다. 그래서

  * 번역 결과를 청크 단위로 디스크에 캐시한다.
    중간에 죽어도 다시 실행하면 남은 청크부터 이어서 진행한다.
  * 실패하면 대기 시간을 늘리고 세션을 새로 만든다. 성공하면 서서히 되돌린다.
  * 연속으로 실패하면 한국어가 섞인 결과를 쓰는 대신 그 언어를 중단한다.
    (캐시는 남으므로 나중에 이어서 하면 된다)

사용 예:
    pip install -r requirements.txt

    python translator.py                                # 5개 언어 전부
    python translator.py --language 영어                 # 한 언어만
    python translator.py --sleep 2 --max-chars 1500      # 차단이 잦을 때
    python translator.py --dry-run                      # 호출 없이 계획만 출력
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

try:
    from deep_translator import GoogleTranslator
except ImportError:
    sys.exit("deep-translator 가 설치되어 있지 않습니다.  pip install -r requirements.txt")


# --------------------------------------------------------------------------
# 설정
# --------------------------------------------------------------------------

# 한국어 표기 -> Google 번역 언어 코드
LANGUAGES: dict[str, str] = {
    "영어": "en",
    "중국어": "zh-CN",
    "필리핀어": "tl",
    "베트남어": "vi",
    "러시아어": "ru",
}

SOURCE_LANG = "ko"

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

# Google 은 요청당 5000자까지 받지만, 크게 보낼수록 빈 응답/차단이 잦다.
# 작게 끊는 편이 전체적으로 훨씬 안정적이다.
DEFAULT_MAX_CHARS = 2000

# 문장 경계 (한국어 종결부호 포함). 폭 0 으로 잘라서 구분자를 잃지 않는다.
SENT_SPLIT_RE = re.compile(r"(?<=[.!?。．！？])")

# 캐시에 새 항목이 이만큼 쌓이면 디스크에 저장한다.
CACHE_FLUSH_EVERY = 5


class Aborted(Exception):
    """연속 실패로 해당 언어를 중단할 때 던진다."""


# --------------------------------------------------------------------------
# 문자열 분할
# --------------------------------------------------------------------------

def _pack(pieces, limit: int) -> list[str]:
    """조각들을 상한 이하로 이어붙인다."""
    chunks: list[str] = []
    buf = ""
    for piece in pieces:
        if not piece:
            continue
        if buf and len(buf) + len(piece) > limit:
            chunks.append(buf)
            buf = piece
        else:
            buf += piece
    if buf:
        chunks.append(buf)
    return chunks


def split_text(text: str, limit: int = DEFAULT_MAX_CHARS) -> list[str]:
    """
    상한을 넘는 문자열을 줄바꿈 -> 문장 경계 순으로 나눈다.

    구분자를 버리지 않으므로 "".join(split_text(t)) == t 가 성립한다.
    (백서처럼 줄 구조가 의미를 갖는 문서에서 줄바꿈이 사라지지 않도록)
    """
    if len(text) <= limit:
        return [text]

    result: list[str] = []
    for chunk in _pack(text.splitlines(keepends=True), limit):
        # 한 줄이 통째로 상한을 넘으면 문장 경계로 한 번 더 나눈다.
        if len(chunk) <= limit:
            result.append(chunk)
        else:
            result.extend(_pack(SENT_SPLIT_RE.split(chunk), limit))

    # 문장 하나가 그래도 상한을 넘으면 강제로 자른다.
    final: list[str] = []
    for chunk in result:
        while len(chunk) > limit:
            final.append(chunk[:limit])
            chunk = chunk[limit:]
        if chunk:
            final.append(chunk)
    return final


# --------------------------------------------------------------------------
# 번역기
# --------------------------------------------------------------------------

class Translator:
    """
    언어 하나를 담당한다.

    번역 결과는 청크 단위로 캐시하고 디스크에도 남긴다.
    차단당해 중간에 멈춰도 다시 실행하면 캐시된 청크는 건너뛴다.
    """

    def __init__(self, lang_name: str, sleep: float, retries: int,
                 max_chars: int = DEFAULT_MAX_CHARS, cooldown: float = 60.0,
                 abort_after: int = 3, cache_path: Path | None = None,
                 jitter: float = 0.3):
        self.lang_name = lang_name
        self.code = LANGUAGES[lang_name]
        self.sleep = sleep
        self.retries = retries
        self.max_chars = max_chars
        self.cooldown = cooldown
        self.abort_after = abort_after
        self.jitter = jitter

        self.engine = GoogleTranslator(source=SOURCE_LANG, target=self.code)
        # 실패하면 늘리고 성공하면 줄이는 현재 대기 시간
        self.delay = sleep
        self.max_delay = max(sleep * 20, 20.0)

        self.cache_path = cache_path
        self.cache: dict[str, str] = {}
        self.pending = 0
        self._load_cache()

        self.api_calls = 0
        self.cache_hits = 0
        self.failures = 0
        self.consecutive_failures = 0

    # ---- 캐시 -----------------------------------------------------------

    def _load_cache(self) -> None:
        if not self.cache_path or not self.cache_path.exists():
            return
        try:
            with self.cache_path.open(encoding="utf-8") as fp:
                self.cache = json.load(fp)
            print(f"  캐시 {len(self.cache)}청크 불러옴: {self.cache_path.name}")
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  [!] 캐시를 읽지 못해 새로 시작합니다: {exc}")
            self.cache = {}

    def save_cache(self) -> None:
        if not self.cache_path or not self.pending:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fp:
            json.dump(self.cache, fp, ensure_ascii=False)
        tmp.replace(self.cache_path)
        self.pending = 0

    # ---- 번역 -----------------------------------------------------------

    def translate(self, text: str) -> str:
        key = text.strip()
        if not key:
            return text

        chunks = split_text(key, self.max_chars)
        pieces: list[str] = []
        for i, chunk in enumerate(chunks, 1):
            pieces.append(self._translate_chunk(chunk))
            if len(chunks) > 1:
                print(f"      [{i}/{len(chunks)}] {len(key):,}자 중 "
                      f"{i * 100 // len(chunks)}%", flush=True)
        # 구분자를 보존해 잘랐으므로 그대로 이어붙인다.
        return "".join(pieces)

    def _translate_chunk(self, chunk: str) -> str:
        if chunk in self.cache:
            self.cache_hits += 1
            return self.cache[chunk]

        result = self._request(chunk)
        if result is None:
            # 실패한 청크는 캐시하지 않는다.
            # 캐시하면 다시 실행해도 한국어 원문이 그대로 굳어버린다.
            return chunk
        self.cache[chunk] = result
        self.pending += 1
        if self.pending >= CACHE_FLUSH_EVERY:
            self.save_cache()
        return result

    def _reset_engine(self) -> None:
        """세션이 막히면 새 요청도 계속 실패한다. 엔진을 새로 만든다."""
        self.engine = GoogleTranslator(source=SOURCE_LANG, target=self.code)

    def _request(self, chunk: str) -> str | None:
        """번역문을 돌려준다. 끝내 실패하면 None (호출부가 원문을 유지한다)."""
        for attempt in range(1, self.retries + 1):
            try:
                result = self.engine.translate(chunk)
                if not result:
                    raise ValueError("빈 응답")
                self.api_calls += 1
                self.consecutive_failures = 0
                # 성공이 이어지면 대기 시간을 서서히 되돌린다.
                self.delay = max(self.sleep, self.delay * 0.8)
                time.sleep(self.delay + random.uniform(0, self.jitter))
                return result
            except Exception as exc:  # noqa: BLE001 - 네트워크/차단 등 모든 예외 대응
                # 차단으로 보고 간격을 늘리고 세션을 새로 판다.
                self.delay = min(self.max_delay, max(self.delay * 2, 2.0))
                self._reset_engine()

                if attempt < self.retries:
                    wait = min(self.delay * (2 ** attempt), self.cooldown)
                    print(f"      [~] 재시도 {attempt}/{self.retries - 1} "
                          f"({wait:.0f}s 대기): {exc}")
                    time.sleep(wait + random.uniform(0, 1.0))
                    continue

                # 재시도를 다 썼다.
                self.failures += 1
                self.consecutive_failures += 1
                if self.consecutive_failures >= self.abort_after:
                    self.save_cache()
                    raise Aborted(
                        f"{self.consecutive_failures}개 청크 연속 실패 - "
                        f"차단된 것으로 보고 중단합니다"
                    ) from exc

                print(f"      [!] 청크 실패 ({self.retries}회 시도) - "
                      f"{self.cooldown:.0f}초 쉬고 계속: {exc}")
                self.save_cache()
                time.sleep(self.cooldown)
                self._reset_engine()
                return None
        return None


# --------------------------------------------------------------------------
# JSON 순회
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


def translate_node(node, tr: Translator, key: str | None = None):
    """dict/list/str 를 재귀적으로 순회하며 문자열 value 만 번역한다."""
    if isinstance(node, dict):
        return {k: translate_node(v, tr, k) for k, v in node.items()}
    if isinstance(node, list):
        return [translate_node(item, tr, key) for item in node]
    if isinstance(node, str):
        if key in SUFFIX_KEYS:
            return f"{node}_{tr.lang_name}"
        if not should_translate(key, node):
            return node
        return tr.translate(node)
    # int / float / bool / None 은 그대로
    return node


# --------------------------------------------------------------------------
# 실행
# --------------------------------------------------------------------------

def build_output_path(src: Path, out_dir: Path, lang_name: str) -> Path:
    """
    data/2024_마약류_범죄백서.json
      -> {out_dir}/2024_마약류_범죄백서_영어.json
    """
    return out_dir / f"{src.stem}_{lang_name}{src.suffix}"


def cache_path_for(src: Path, out_dir: Path, lang_name: str) -> Path:
    return out_dir / ".translate_cache" / f"{src.stem}_{lang_name}.json"


def run_language(lang_name: str, src: Path, out_dir: Path, args) -> bool:
    """번역해서 저장한다. 끝까지 마쳤으면 True."""
    dst = build_output_path(src, out_dir, lang_name)

    print(f"\n{'=' * 70}")
    print(f"  {lang_name} ({LANGUAGES[lang_name]})  ->  {dst.name}")
    print(f"{'=' * 70}")

    if dst.exists() and not args.overwrite:
        print("  건너뜀 (이미 있음). 다시 만들려면 --overwrite")
        return True

    if args.dry_run:
        print(f"  {src}\n    -> {dst}")
        return True

    with src.open(encoding="utf-8") as fp:
        data = json.load(fp)

    tr = Translator(
        lang_name,
        sleep=args.sleep,
        retries=args.retries,
        max_chars=args.max_chars,
        cooldown=args.cooldown,
        abort_after=args.abort_after,
        cache_path=None if args.no_cache else cache_path_for(src, out_dir, lang_name),
    )

    started = time.time()
    try:
        translated = translate_node(data, tr)
    except Aborted as exc:
        elapsed = time.time() - started
        print(f"\n  [!] {lang_name} 중단: {exc}")
        print(f"      API 호출 {tr.api_calls}회 / 캐시 적중 {tr.cache_hits}회 "
              f"/ {elapsed / 60:.1f}분")
        print("      번역한 청크는 캐시에 남아 있습니다. 잠시 뒤 같은 명령을 "
              "다시 실행하면 이어서 진행합니다.")
        print(f"      계속 막히면: --sleep {args.sleep * 4:.0f} "
              f"--max-chars {max(args.max_chars // 2, 500)}")
        return False
    finally:
        # Ctrl+C 로 끊어도 여기까지 번역한 청크는 남긴다.
        tr.save_cache()

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as fp:
        json.dump(translated, fp, ensure_ascii=False, indent=2)

    elapsed = time.time() - started
    print(f"\n  [{lang_name}] 완료 - {elapsed / 60:.1f}분")
    print(f"  API 호출 {tr.api_calls}회, 캐시 적중 {tr.cache_hits}회, "
          f"실패 {tr.failures}회")
    if tr.failures:
        print(f"  [!] {tr.failures}개 청크는 번역하지 못해 원문(한국어)이 남았습니다.")
    print(f"  -> {dst}")
    return True


def main() -> None:
    # Windows 콘솔(cp949)에서 한글 출력이 깨지지 않도록
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    parser = argparse.ArgumentParser(
        description="JSON 파일 하나를 여러 언어로 번역한다 (deep-translator / Google).",
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
        "--sleep", type=float, default=1.0,
        help="번역 호출 사이 기본 대기 시간(초).\n"
             "실패하면 자동으로 늘어나고 성공이 이어지면 이 값으로 돌아온다 (기본: 1.0)",
    )
    parser.add_argument(
        "--max-chars", type=int, default=DEFAULT_MAX_CHARS,
        help=f"요청 하나에 보낼 최대 글자 수 (기본: {DEFAULT_MAX_CHARS})\n"
             "차단이 잦으면 줄일 것. Google 상한은 5000",
    )
    parser.add_argument(
        "--retries", type=int, default=4,
        help="청크 하나당 재시도 횟수 (기본: 4)",
    )
    parser.add_argument(
        "--cooldown", type=float, default=60.0,
        help="청크가 완전히 실패했을 때 쉬는 시간(초) (기본: 60)",
    )
    parser.add_argument(
        "--abort-after", type=int, default=3,
        help="청크가 이만큼 연속 실패하면 해당 언어를 중단한다 (기본: 3)\n"
             "한국어가 섞인 결과 파일을 만드는 것보다 낫다",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="청크 캐시를 쓰지 않는다 (기본: .translate_cache 에 저장하고 이어하기)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="이미 만들어진 결과 파일도 다시 번역한다 (기본: 건너뜀)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="번역하지 않고 입/출력 경로만 출력한다",
    )
    args = parser.parse_args()

    src: Path = args.data.resolve()
    if not src.is_file():
        sys.exit(f"파일을 찾을 수 없습니다: {src}")

    out_dir: Path = (args.out or src.parent).resolve()

    print(f"원본  : {src}")
    print(f"출력  : {out_dir}")
    print(f"언어  : {', '.join(args.language)}")
    print(f"대기  : {args.sleep}초 / 청크 {args.max_chars}자")
    if args.no_cache:
        print("캐시  : 사용 안 함")
    if args.dry_run:
        print("모드  : DRY-RUN (번역 호출 없음)")

    total_started = time.time()
    finished: list[str] = []
    stopped: list[str] = []
    for lang_name in args.language:
        try:
            ok = run_language(lang_name, src, out_dir, args)
        except KeyboardInterrupt:
            print("\n중단했습니다. 캐시는 남아 있으니 다시 실행하면 이어집니다.")
            raise
        (finished if ok else stopped).append(lang_name)

    print(f"\n전체 완료 - {(time.time() - total_started) / 60:.1f}분")
    print(f"  성공 {len(finished)}: {', '.join(finished) or '-'}")
    if stopped:
        print(f"  중단 {len(stopped)}: {', '.join(stopped)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
