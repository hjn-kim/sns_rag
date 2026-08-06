#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SNS 대화 JSON 다국어 번역기 (deep-translator / Google)

data/sns_sample.json 에서 아래 항목만 번역해 언어별 파일로 낸다.

    utterance / type / topic / gender / age / residentialProvince

나머지 값(대화 ID, 참가자 ID, 날짜, 시각, 개수 등)은 손대지 않는다.

    data/sns_sample.json
      -> data/sns_sample_en.json    영어
      -> data/sns_sample_vi.json    베트남어
      -> data/sns_sample_fil.json   필리핀어
      -> data/sns_sample_ru.json    러시아어
      -> data/sns_sample_zh.json    중국어

채팅 말뭉치라 문자열이 아주 짧고(중간값 9자) 개수가 아주 많다(고유 23만 개).
하나씩 요청하면 하루가 걸리므로 이렇게 줄인다.

  * #@이름# 같은 마스킹 토큰을 기준으로 문자열을 조각내고, 같은 조각은 한 번만 번역한다.
    토큰 자체는 번역 대상에서 빼므로 원래 자리에 그대로 남는다.
  * 여러 조각을 구분선(\\n@@\\n)으로 이어 한 번에 보낸다. 1800자면 100조각쯤 들어간다.
    돌아온 응답을 같은 구분선으로 나눈다. 조각 수가 안 맞으면 그 묶음만 하나씩 다시 번역한다.
  * 번역한 조각은 디스크에 캐시한다. 중간에 죽어도 다시 실행하면 남은 조각부터 이어서 한다.
  * 실패하면 대기 시간을 늘리고 세션을 새로 만든다. 연속 실패가 이어지면 그 언어를 중단한다.

사용 예:
    pip install deep-translator

    python src/cpu_translator.py                          # 5개 언어 전부
    python src/cpu_translator.py --language en            # 한 언어만
    python src/cpu_translator.py --limit 200              # 200조각만 번역해 확인
    python src/cpu_translator.py --sleep 2 --group-chars 1200   # 차단이 잦을 때
    python src/cpu_translator.py --dry-run                # 호출 없이 조각 통계만
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
    sys.exit("deep-translator 가 설치되어 있지 않습니다.  pip install deep-translator")


# --------------------------------------------------------------------------
# 설정
# --------------------------------------------------------------------------

# 출력 파일 접미사 -> (표시 이름, Google 번역 언어 코드)
LANGUAGES: dict[str, tuple[str, str]] = {
    "en": ("영어", "en"),
    "vi": ("베트남어", "vi"),
    "fil": ("필리핀어", "tl"),
    "ru": ("러시아어", "ru"),
    "zh": ("중국어", "zh-CN"),
}

SOURCE_LANG = "ko"

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

# 조각 하나의 최대 글자 수. 발화문은 99%가 41자 이하라 걸릴 일이 거의 없다.
DEFAULT_MAX_CHARS = 1000

# 한 요청에 담을 글자 수. Google 상한은 5000 이지만 크게 보낼수록 차단이 잦다.
DEFAULT_GROUP_CHARS = 1800

# 여러 조각을 한 요청에 담을 때 쓰는 구분선.
# 5개 언어 모두 번역되지 않고 그대로 돌아오는 것을 확인했다.
SEP = "\n@@\n"
# 응답에서는 공백이 조금 달라질 수 있어 느슨하게 나눈다.
SEP_SPLIT_RE = re.compile(r"\n\s*@\s*@\s*\n")

# 캐시에 새 조각이 이만큼 쌓이면 디스크에 저장한다.
CACHE_FLUSH_EVERY = 500


class Aborted(Exception):
    """연속 실패로 해당 언어를 중단할 때 던진다."""


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


def rebuild(node, planner: Planner, table: dict[str, str], key: str | None = None):
    """번역표(table)를 끼워넣어 JSON 을 다시 만든다. 표에 없는 조각은 원문으로 둔다."""
    if isinstance(node, dict):
        return {k: rebuild(v, planner, table, k) for k, v in node.items()}
    if isinstance(node, list):
        return [rebuild(item, planner, table, key) for item in node]
    if isinstance(node, str):
        if key not in TRANSLATE_KEYS or not node.strip():
            return node
        return "".join(table.get(piece, piece) if need else piece
                       for need, piece in planner.plan(node))
    # int / float / bool / None 은 그대로
    return node


# --------------------------------------------------------------------------
# 번역기
# --------------------------------------------------------------------------

class Translator:
    """
    언어 하나를 담당한다.

    조각 여러 개를 구분선으로 이어 한 번에 보내고, 돌아온 응답을 다시 나눈다.
    조각 수가 맞지 않으면 그 묶음만 하나씩 다시 번역해서 어긋난 채로 저장되는 일을 막는다.
    """

    def __init__(self, code: str, label: str, sleep: float, retries: int,
                 group_chars: int = DEFAULT_GROUP_CHARS, cooldown: float = 60.0,
                 abort_after: int = 3, cache_path: Path | None = None,
                 jitter: float = 0.3):
        self.code = code
        self.label = label
        self.sleep = sleep
        self.retries = retries
        self.group_chars = group_chars
        self.cooldown = cooldown
        self.abort_after = abort_after
        self.jitter = jitter

        self.engine = GoogleTranslator(source=SOURCE_LANG, target=code)
        # 실패하면 늘리고 성공하면 줄이는 현재 대기 시간
        self.delay = sleep
        self.max_delay = max(sleep * 20, 20.0)

        self.cache_path = cache_path
        self.cache: dict[str, str] = {}
        self.pending = 0
        self._load_cache()

        self.api_calls = 0
        self.failures = 0
        self.consecutive_failures = 0
        self.regrouped = 0      # 조각 수가 안 맞아 하나씩 다시 보낸 묶음
        self.skipped = 0        # 끝내 번역하지 못한 조각

    # ---- 캐시 -----------------------------------------------------------

    def _load_cache(self) -> None:
        if not self.cache_path or not self.cache_path.exists():
            return
        try:
            with self.cache_path.open(encoding="utf-8") as fp:
                self.cache = json.load(fp)
            print(f"  캐시 {len(self.cache):,}조각 불러옴: {self.cache_path.name}")
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  [!] 캐시를 읽지 못해 새로 시작합니다: {exc}")
            self.cache = {}

    def save_cache(self) -> None:
        if not self.cache_path or not self.pending:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_name(self.cache_path.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as fp:
            json.dump(self.cache, fp, ensure_ascii=False)
        tmp.replace(self.cache_path)
        self.pending = 0

    def _store(self, piece: str, result: str) -> None:
        self.cache[piece] = result
        self.pending += 1
        if self.pending >= CACHE_FLUSH_EVERY:
            self.save_cache()

    # ---- 묶기 -----------------------------------------------------------

    def group(self, pieces: list[str]) -> list[list[str]]:
        """조각들을 요청 하나 분량씩 나눈다."""
        groups: list[list[str]] = []
        buf: list[str] = []
        size = 0
        for piece in pieces:
            # 구분선처럼 보이는 글자가 든 조각은 나눌 때 헷갈리므로 혼자 보낸다.
            if SEP_SPLIT_RE.search(piece):
                if buf:
                    groups.append(buf)
                    buf, size = [], 0
                groups.append([piece])
                continue
            cost = len(piece) + len(SEP)
            if buf and size + cost > self.group_chars:
                groups.append(buf)
                buf, size = [], 0
            buf.append(piece)
            size += cost
        if buf:
            groups.append(buf)
        return groups

    # ---- 번역 -----------------------------------------------------------

    def translate_group(self, group: list[str]) -> None:
        """묶음 하나를 번역해 캐시에 채운다."""
        todo = [p for p in group if p not in self.cache]
        if not todo:
            return

        if len(todo) == 1:
            result = self._request(todo[0])
            if result:
                self._store(todo[0], result)
            else:
                self.skipped += 1
            return

        merged = self._request(SEP.join(todo))
        if merged is not None:
            parts = SEP_SPLIT_RE.split(merged)
            if len(parts) == len(todo):
                for piece, part in zip(todo, parts):
                    part = part.strip()
                    # 빈 응답은 캐시하지 않는다. 캐시하면 다시 실행해도 굳어버린다.
                    if part:
                        self._store(piece, part)
                    else:
                        self.skipped += 1
                return
            # 구분선이 뭉개졌다. 어긋난 채로 저장하느니 하나씩 다시 보낸다.
            self.regrouped += 1

        for piece in todo:
            result = self._request(piece)
            if result:
                self._store(piece, result)
            else:
                self.skipped += 1

    def _reset_engine(self) -> None:
        """세션이 막히면 새 요청도 계속 실패한다. 엔진을 새로 만든다."""
        self.engine = GoogleTranslator(source=SOURCE_LANG, target=self.code)

    def _request(self, text: str) -> str | None:
        """번역문을 돌려준다. 끝내 실패하면 None (호출부가 원문을 유지한다)."""
        for attempt in range(1, self.retries + 1):
            try:
                result = self.engine.translate(text)
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
                        f"{self.consecutive_failures}회 연속 실패 - "
                        f"차단된 것으로 보고 중단합니다"
                    ) from exc

                print(f"      [!] 요청 실패 ({self.retries}회 시도) - "
                      f"{self.cooldown:.0f}초 쉬고 계속: {exc}")
                self.save_cache()
                time.sleep(self.cooldown)
                self._reset_engine()
                return None
        return None


# --------------------------------------------------------------------------
# 실행
# --------------------------------------------------------------------------

def build_output_path(src: Path, out_dir: Path, code: str) -> Path:
    """data/sns_sample.json -> {out_dir}/sns_sample_en.json"""
    return out_dir / f"{src.stem}_{code}{src.suffix}"


def cache_path_for(src: Path, out_dir: Path, code: str) -> Path:
    return out_dir / ".translate_cache" / f"{src.stem}_{code}.json"


def run_language(code: str, data, src: Path, out_dir: Path,
                 planner: Planner, segments: list[str], args) -> bool:
    """번역해서 저장한다. 끝까지 마쳤으면 True."""
    label, google_code = LANGUAGES[code]
    dst = build_output_path(src, out_dir, code)

    print(f"\n{'=' * 70}")
    print(f"  {label} ({google_code})  ->  {dst.name}")
    print(f"{'=' * 70}")

    if dst.exists() and not args.overwrite:
        print("  건너뜀 (이미 있음). 다시 만들려면 --overwrite")
        return True

    tr = Translator(
        google_code, label,
        sleep=args.sleep,
        retries=args.retries,
        group_chars=args.group_chars,
        cooldown=args.cooldown,
        abort_after=args.abort_after,
        cache_path=None if args.no_cache else cache_path_for(src, out_dir, code),
    )

    todo = [p for p in segments if p not in tr.cache]
    if args.limit > 0:
        todo = todo[:args.limit]
        print(f"  --limit {args.limit} 적용")
    groups = tr.group(todo)
    print(f"  남은 조각 {len(todo):,}개 -> 요청 {len(groups):,}회")

    started = time.time()
    try:
        for i, grp in enumerate(groups, 1):
            tr.translate_group(grp)
            if i % 10 == 0 or i == len(groups):
                done_ratio = i * 100 // len(groups)
                elapsed = time.time() - started
                eta = elapsed / i * (len(groups) - i) / 60
                print(f"    [{i:,}/{len(groups):,}] {done_ratio}% "
                      f"| 캐시 {len(tr.cache):,} | 실패 {tr.failures} "
                      f"| 재분할 {tr.regrouped} | 남은 시간 {eta:.0f}분", flush=True)
    except Aborted as exc:
        print(f"\n  [!] {label} 중단: {exc}")
        print(f"      API 호출 {tr.api_calls:,}회 / "
              f"{(time.time() - started) / 60:.1f}분")
        print("      번역한 조각은 캐시에 남아 있습니다. 잠시 뒤 같은 명령을 "
              "다시 실행하면 이어서 진행합니다.")
        print(f"      계속 막히면: --sleep {args.sleep * 4:.0f} "
              f"--group-chars {max(args.group_chars // 2, 400)}")
        return False
    finally:
        # Ctrl+C 로 끊어도 여기까지 번역한 조각은 남긴다.
        tr.save_cache()

    print("  결과 조립 중...")
    translated = rebuild(data, planner, tr.cache)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fp:
        json.dump(translated, fp, ensure_ascii=False, indent=2)
    tmp.replace(dst)

    missing = sum(1 for p in segments if p not in tr.cache)
    elapsed = time.time() - started
    print(f"\n  [{label}] 완료 - {elapsed / 60:.1f}분")
    print(f"  API 호출 {tr.api_calls:,}회, 재분할 {tr.regrouped}회, 실패 {tr.failures}회")
    if missing:
        print(f"  [!] {missing:,}개 조각은 번역하지 못해 원문(한국어)이 남았습니다. "
              f"다시 실행하면 이어서 진행합니다.")
    print(f"  -> {dst}")
    return True


def main() -> None:
    # Windows 콘솔(cp949)에서 한글 출력이 깨지지 않도록
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    parser = argparse.ArgumentParser(
        description="SNS 대화 JSON 의 지정 항목을 여러 언어로 번역한다 "
                    "(deep-translator / Google).",
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
        "--language", "--languages", nargs="+", default=list(LANGUAGES),
        choices=list(LANGUAGES), metavar="언어",
        help="번역할 언어. 선택: "
             + ", ".join(f"{c}({n})" for c, (n, _) in LANGUAGES.items())
             + "\n(기본: 전체)",
    )
    parser.add_argument(
        "--sleep", type=float, default=1.0,
        help="요청 사이 기본 대기 시간(초).\n"
             "실패하면 자동으로 늘어나고 성공이 이어지면 이 값으로 돌아온다 (기본: 1.0)",
    )
    parser.add_argument(
        "--max-chars", type=int, default=DEFAULT_MAX_CHARS,
        help=f"조각 하나의 최대 글자 수 (기본: {DEFAULT_MAX_CHARS})",
    )
    parser.add_argument(
        "--group-chars", type=int, default=DEFAULT_GROUP_CHARS,
        help=f"요청 하나에 담을 최대 글자 수 (기본: {DEFAULT_GROUP_CHARS})\n"
             "차단이 잦으면 줄일 것. Google 상한은 5000",
    )
    parser.add_argument(
        "--retries", type=int, default=4,
        help="요청 하나당 재시도 횟수 (기본: 4)",
    )
    parser.add_argument(
        "--cooldown", type=float, default=60.0,
        help="요청이 완전히 실패했을 때 쉬는 시간(초) (기본: 60)",
    )
    parser.add_argument(
        "--abort-after", type=int, default=3,
        help="요청이 이만큼 연속 실패하면 해당 언어를 중단한다 (기본: 3)",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="언어당 번역할 조각 수 제한. 0 이면 전체 (테스트용)",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="조각 캐시를 쓰지 않는다 (기본: .translate_cache 에 저장하고 이어하기)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="이미 만들어진 결과 파일도 다시 만든다 (기본: 건너뜀)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="번역하지 않고 조각 통계만 출력한다",
    )
    args = parser.parse_args()

    src: Path = args.data.resolve()
    if not src.is_file():
        sys.exit(f"파일을 찾을 수 없습니다: {src}")
    out_dir: Path = (args.out or src.parent).resolve()

    print(f"원본  : {src}")
    print(f"출력  : {out_dir}")
    print(f"언어  : {', '.join(args.language)}")
    print(f"항목  : {', '.join(sorted(TRANSLATE_KEYS))}")
    print(f"대기  : {args.sleep}초 / 요청 {args.group_chars}자")
    if args.no_cache:
        print("캐시  : 사용 안 함")

    print("\n원본 읽는 중...")
    with src.open(encoding="utf-8") as fp:
        data = json.load(fp)

    # ---- 조각내기 (요청 전에 먼저 끝낸다) --------------------------------
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

    print(f"대상  : 문자열 {values:,}개 -> 조각 {total_pieces:,}개 "
          f"-> 중복 제거 {len(segments):,}개 "
          f"({sum(len(s) for s in segments):,}자)")

    if not segments:
        sys.exit("번역할 조각이 없습니다.")

    if args.dry_run:
        print("\n--- 조각 샘플 ---")
        for piece in segments[:15]:
            print("  " + piece[:80])
        print("\nDRY-RUN: 번역 호출 없이 종료합니다.")
        return

    total_started = time.time()
    finished: list[str] = []
    stopped: list[str] = []
    for code in args.language:
        try:
            ok = run_language(code, data, src, out_dir, planner, segments, args)
        except KeyboardInterrupt:
            print("\n중단했습니다. 캐시는 남아 있으니 다시 실행하면 이어집니다.")
            raise
        (finished if ok else stopped).append(code)

    print(f"\n전체 완료 - {(time.time() - total_started) / 60:.1f}분")
    print(f"  성공 {len(finished)}: {', '.join(finished) or '-'}")
    if stopped:
        print(f"  중단 {len(stopped)}: {', '.join(stopped)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
