#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
translate_jabber.py
-------------------
jabber_en.jsonl 의 `text` 필드 중 '실제 채팅 발화 부분'만 한국어로 번역한다.

번역 대상 (O)
    "10:46 tilar: doxodit?"            -> "10:46 tilar: 도착해?"
    "stern: Hello, How are the projects?"  -> "stern: 안녕, 프로젝트는 어떻게 돼가?"

번역 제외 (X)
    - 헤더 라인:  "[대화] 8383 | boby | 2020-10-07 18:57~19:02", "[공지] stern -> 8명 | ..."
    - 수신자 라인: "수신: bullet, ghost, gus, ..."
    - 타임스탬프 "HH:MM", 닉네임, 그 외 모든 JSON 메타 필드

사용법
    export ANTHROPIC_API_KEY=sk-ant-...
    python3 translate_jabber.py --input jabber_en.jsonl --output jabber_ko.jsonl

    # 먼저 규모/비용만 확인
    python3 translate_jabber.py --input jabber_en.jsonl --dry-run

    # 앞의 200줄만 시험 번역
    python3 translate_jabber.py --input jabber_en.jsonl --output sample_ko.jsonl --limit 200
"""

import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"

# "10:46 tilar: 본문"  또는  "stern: 본문"(broadcast)
MSG_RE = re.compile(r"^(?:(?P<ts>\d{2}:\d{2}) )?(?P<who>[^:]{1,40}): (?P<body>.*)$")
SKIP_PREFIXES = ("수신: ", "[대화] ", "[공지] ")

SYSTEM_PROMPT = """당신은 채팅 로그 번역 엔진이다. 입력은 XMPP(Jabber) 채팅 메시지 본문 목록이며, 대부분 러시아어에서 영어로 기계번역된 비격식 대화체다.

규칙:
1. 각 항목을 자연스러운 한국어 구어체로 번역한다. 반말/구어 톤을 유지한다.
2. 다음은 절대 번역·변형하지 않고 원문 그대로 둔다: 닉네임/핸들, URL, 도메인, IP, 파일명, 확장자, 해시, 지갑주소, 코드·명령어·경로, 소프트웨어 제품명, 숫자, 통화 표기.
3. 원문이 의미 불명(오타, 러시아어 음차, 깨진 문자열)이면 억지로 지어내지 말고 원문을 그대로 반환한다.
4. 이모티콘, ")))" 같은 웃음 표기, +/- 같은 단답 기호는 그대로 둔다.
5. 설명·주석·따옴표를 추가하지 않는다. 내용을 요약하거나 생략하지 않는다.
6. 출력은 오직 JSON 배열. 형식: [{"i": 1, "t": "번역문"}, ...]. 입력 항목 수와 반드시 동일한 개수를, 동일한 i 값으로 반환한다."""


# ---------------------------------------------------------------- API 호출
def call_api(api_key, model, user_text, max_tokens=8000, timeout=180, retries=5):
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_text}],
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    last_err = None
    for attempt in range(retries):
        req = urllib.request.Request(
            API_URL,
            data=data,
            headers={
                "content-type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": API_VERSION,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return "".join(b.get("text", "") for b in body.get("content", []) if b.get("type") == "text")
        except urllib.error.HTTPError as e:
            code = e.code
            last_err = f"HTTP {code}: {e.read().decode('utf-8', 'ignore')[:300]}"
            if code in (429, 500, 502, 503, 504, 529):
                time.sleep(min(60, 2 ** attempt * 2))
                continue
            break
        except Exception as e:  # 네트워크/타임아웃
            last_err = repr(e)
            time.sleep(min(60, 2 ** attempt * 2))
    raise RuntimeError(f"API 호출 실패: {last_err}")


def parse_json_array(raw):
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.S)
    start, end = s.find("["), s.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("JSON 배열을 찾지 못함")
    return json.loads(s[start : end + 1])


def translate_batch(api_key, model, items):
    """items: [(key, text), ...] -> {key: 번역문}"""
    lines = [f'{i + 1}. {json.dumps(t, ensure_ascii=False)}' for i, (_, t) in enumerate(items)]
    user_text = (
        f"다음 {len(items)}개 채팅 메시지를 규칙에 따라 한국어로 번역하라.\n"
        "각 줄은 `번호. JSON문자열` 형식이다.\n\n" + "\n".join(lines)
    )
    raw = call_api(api_key, model, user_text)
    arr = parse_json_array(raw)
    out = {}
    for obj in arr:
        idx = int(obj["i"]) - 1
        if 0 <= idx < len(items):
            out[items[idx][0]] = str(obj["t"])
    if len(out) != len(items):  # 개수 불일치 -> 누락분만 개별 재시도
        for key, text in items:
            if key not in out:
                try:
                    one = parse_json_array(
                        call_api(api_key, model, f'다음 1개 메시지를 번역하라.\n\n1. {json.dumps(text, ensure_ascii=False)}', max_tokens=4000)
                    )
                    out[key] = str(one[0]["t"])
                except Exception:
                    out[key] = text  # 최종 실패 시 원문 유지
    return out


# ---------------------------------------------------------------- 수집/재조립
def iter_records(path, limit=None):
    with open(path, "r", encoding="utf-8") as f:
        for n, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if limit is not None and n >= limit:
                break
            yield json.loads(line)


def split_lines(text):
    """text -> [(kind, ts, who, body) ...]  kind: 'msg' | 'raw'"""
    parsed = []
    for L in text.split("\n"):
        if L.startswith(SKIP_PREFIXES):
            parsed.append(("raw", None, None, L))
            continue
        m = MSG_RE.match(L)
        if m:
            parsed.append(("msg", m.group("ts"), m.group("who"), m.group("body")))
        else:
            parsed.append(("raw", None, None, L))
    return parsed


def rebuild(parsed, table):
    out = []
    for kind, ts, who, body in parsed:
        if kind == "raw":
            out.append(body)
        else:
            ko = table.get(body, body)
            out.append(f"{ts} {who}: {ko}" if ts else f"{who}: {ko}")
    return "\n".join(out)


# ---------------------------------------------------------------- 캐시
class Cache:
    def __init__(self, path):
        self.path = path
        self.data = {}
        self.lock = threading.Lock()
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        o = json.loads(line)
                        self.data[o["en"]] = o["ko"]
                    except Exception:
                        pass
            print(f"[cache] 기존 번역 {len(self.data):,}건 로드", file=sys.stderr)
        self.fh = open(path, "a", encoding="utf-8") if path else None

    def update(self, mapping):
        with self.lock:
            self.data.update(mapping)
            if self.fh:
                for en, ko in mapping.items():
                    self.fh.write(json.dumps({"en": en, "ko": ko}, ensure_ascii=False) + "\n")
                self.fh.flush()

    def close(self):
        if self.fh:
            self.fh.close()


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="jabber_en.jsonl")
    ap.add_argument("--output", default="jabber_ko.jsonl")
    ap.add_argument("--cache", default="translation_cache.jsonl")
    ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--batch-size", type=int, default=40, help="1회 요청 최대 항목 수")
    ap.add_argument("--batch-chars", type=int, default=4000, help="1회 요청 최대 문자 수")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="앞 N줄만 처리(테스트용)")
    ap.add_argument("--keep-original", action="store_true", help="원문을 text_en 필드로 보존")
    ap.add_argument("--dry-run", action="store_true", help="번역 없이 규모만 출력")
    args = ap.parse_args()

    # 1) 전체 레코드 파싱 + 번역 대상 본문 수집(중복 제거)
    records, parsed_all, bodies = [], [], set()
    for rec in iter_records(args.input, args.limit):
        p = split_lines(rec["text"])
        records.append(rec)
        parsed_all.append(p)
        for kind, _, _, body in p:
            if kind == "msg" and body.strip():
                bodies.add(body)

    n_msg = sum(1 for p in parsed_all for k, *_ in p if k == "msg")
    print(f"[stat] 레코드 {len(records):,} / 발화 라인 {n_msg:,} / 고유 본문 {len(bodies):,} "
          f"({sum(len(b) for b in bodies):,} chars)", file=sys.stderr)
    if args.dry_run:
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("ANTHROPIC_API_KEY 환경변수가 필요합니다.")

    cache = Cache(args.cache)
    todo = [b for b in bodies if b not in cache.data]
    print(f"[stat] 신규 번역 대상 {len(todo):,}건", file=sys.stderr)

    # 2) 길이순 정렬 후 배치 구성(길이 유사한 것끼리 묶어 토큰 낭비 감소)
    todo.sort(key=len)
    batches, cur, cur_chars = [], [], 0
    for b in todo:
        if cur and (len(cur) >= args.batch_size or cur_chars + len(b) > args.batch_chars):
            batches.append(cur)
            cur, cur_chars = [], 0
        cur.append(b)
        cur_chars += len(b)
    if cur:
        batches.append(cur)
    print(f"[stat] 배치 {len(batches):,}개, 동시 요청 {args.workers}", file=sys.stderr)

    # 3) 병렬 번역
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(translate_batch, api_key, args.model, [(b, b) for b in bt]): bt for bt in batches}
        for fut in as_completed(futs):
            bt = futs[fut]
            try:
                cache.update(fut.result())
            except Exception as e:
                print(f"[warn] 배치 실패({len(bt)}건, 원문 유지): {e}", file=sys.stderr)
                cache.update({b: b for b in bt})
            done += 1
            if done % 20 == 0 or done == len(batches):
                print(f"  진행 {done:,}/{len(batches):,} 배치", file=sys.stderr)

    # 4) 재조립 및 저장
    with open(args.output, "w", encoding="utf-8") as f:
        for rec, p in zip(records, parsed_all):
            new = dict(rec)
            if args.keep_original:
                new["text_en"] = rec["text"]
            new["text"] = rebuild(p, cache.data)
            f.write(json.dumps(new, ensure_ascii=False) + "\n")
    cache.close()
    print(f"[done] {args.output} 저장 완료", file=sys.stderr)


if __name__ == "__main__":
    main()
