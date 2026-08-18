#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
translate_jabber_local.py
-------------------------
translate_jabber.py 와 같은 일을 하되, Anthropic API 대신 로컬 GPU 의 instruct 모델을
돌린다. API 키도 비용도 없다.

핵심은 모델 선택이다. NLLB 같은 문장 단위 MT 모델(gpu_translator.py)은 이 데이터에
맞지 않는다. 전체 텍스트의 22% 가 IP/도메인/계정이 나열된 덩어리라 원문 보존 규칙을
지킬 수 있어야 하는데, MT 모델에는 그런 지시를 넣을 방법이 없다. 그래서 여기서는
translate_jabber.py 의 시스템 프롬프트를 그대로 따를 수 있는 instruct 모델을 쓴다.

바뀐 것은 번역 계층 하나뿐이다.

    translate_jabber.py        translate_jabber_local.py
    ------------------------   ---------------------------------
    HTTP 요청 + 스레드 8개  ->  로컬 배치 추론 (한 번에 batch_size 개 프롬프트)
    1회 요청에 40개 묶기    ->  1회 프롬프트에 pack 개 묶기 (기본 8)
    JSON 배열로 응답 요구   ->  번호 붙은 줄로 응답 요구 (작은 모델이 훨씬 안정적)

라인 분리, 중복 제거, 캐시, 재조립은 translate_jabber.py 와 동일하다. 캐시 파일
포맷도 같아서 두 스크립트가 서로의 결과를 이어받을 수 있다.

사용법
    pip install torch transformers accelerate

    python3 translate_jabber_local.py --input jabber_en.jsonl --output jabber_ko.jsonl

    # 규모/필터 결과만 확인 (모델 로드 안 함)
    python3 translate_jabber_local.py --input jabber_en.jsonl --dry-run

    # 앞의 200줄만 시험 번역해 품질/속도 확인 (먼저 이걸 권함)
    python3 translate_jabber_local.py --input jabber_en.jsonl --output sample_ko.jsonl --limit 200

    # VRAM 부족하면
    python3 translate_jabber_local.py --batch-size 8
    python3 translate_jabber_local.py --load-4bit          # pip install bitsandbytes

모델 (A5000 24GB 기준, 전부 bf16 으로 들어감)
    Qwen/Qwen2.5-7B-Instruct              기본값. Apache-2.0, 한국어 무난
    LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct  한국어 특화, 구어체가 더 자연스러움
                                          (라이선스가 연구/비상업 용도로 제한됨)
    Qwen/Qwen3-8B                         더 최신. thinking 모드는 자동으로 끈다
"""

import argparse
import json
import os
import re
import sys
import time

# ---------------------------------------------------------------- 설정

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# "10:46 tilar: 본문"  또는  "stern: 본문"(broadcast)
MSG_RE = re.compile(r"^(?:(?P<ts>\d{2}:\d{2}) )?(?P<who>[^:]{1,40}): (?P<body>.*)$")
SKIP_PREFIXES = ("수신: ", "[대화] ", "[공지] ")

# 번역할 글자가 하나라도 있는지. 라틴/키릴/한글.
LETTER_RE = re.compile(r"[A-Za-zЀ-ӿ가-힣]")

# 자격증명 덩어리 판별용. IP / URL / 이메일.
CREDENTIAL_RE = re.compile(
    r"https?://|\b\d{1,3}(?:\.\d{1,3}){3}\b|\b[\w.+-]+@[\w-]+\.[\w.]+\b"
)

# 이 길이를 넘으면 다른 항목과 묶지 않고 혼자 한 프롬프트를 쓴다.
SOLO_CHARS = 300

SYSTEM_PROMPT = """당신은 채팅 로그 번역 엔진이다. 입력은 XMPP(Jabber) 채팅 메시지 본문 목록이며, 대부분 러시아어에서 영어로 기계번역된 비격식 대화체다.

규칙:
1. 각 항목을 자연스러운 한국어 구어체로 번역한다. 반말/구어 톤을 유지한다.
2. 다음은 절대 번역·변형하지 않고 원문 그대로 둔다: 닉네임/핸들, URL, 도메인, IP, 파일명, 확장자, 해시, 지갑주소, 코드·명령어·경로, 소프트웨어 제품명, 숫자, 통화 표기.
3. 원문이 의미 불명(오타, 러시아어 음차, 깨진 문자열)이면 억지로 지어내지 말고 원문을 그대로 반환한다.
4. 이모티콘, ")))" 같은 웃음 표기, +/- 같은 단답 기호는 그대로 둔다.
5. 설명·주석·따옴표를 추가하지 않는다. 내용을 요약하거나 생략하지 않는다.
6. 출력은 `번호. 번역문` 형식의 줄만 나열한다. 한 항목당 정확히 한 줄이며, 줄바꿈을 넣지 않는다. 입력 항목 수와 같은 개수를, 같은 번호로 반환한다. 다른 말은 일절 쓰지 않는다."""

# 번호 붙은 출력 줄
NUM_LINE_RE = re.compile(r"^\s*(\d+)\s*[.)]\s?(.*)$")
# Qwen3 등이 남기는 thinking 블록
THINK_RE = re.compile(r"<think>.*?</think>\s*", re.S)


# ---------------------------------------------------------------- 노이즈 판별
def is_noise(body):
    """
    번역해봐야 얻을 게 없고, 모델에 넣으면 오히려 망가지는 본문.

    원문을 그대로 통과시킨다. 전체 고유 본문 72,517개 중 1,300개 남짓이지만
    글자 수로는 20% 가 넘어서, 걸러내면 속도가 눈에 띄게 붙는다.
    """
    s = body.strip()
    if not s:
        return True
    if not LETTER_RE.search(s):          # 숫자/기호만 ("3300", ")", "0-3=?")
        return True
    # 긴 자격증명 덤프 (root <ip> <pw> <host> <email> ... 가 반복되는 형태)
    if len(s) > 400 and len(CREDENTIAL_RE.findall(s)) >= 3:
        return True
    return False


# ---------------------------------------------------------------- 수집/재조립
def iter_records(path, limit=None):
    with open(path, "r", encoding="utf-8") as f:
        n = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            if limit is not None and n >= limit:
                break
            n += 1
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
    """translate_jabber.py 와 같은 JSONL 포맷. 두 스크립트가 캐시를 공유한다."""

    def __init__(self, path):
        self.path = path
        self.data = {}
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        o = json.loads(line)
                        self.data[o["en"]] = o["ko"]
                    except Exception:
                        pass
            print(f"[cache] 기존 번역 {len(self.data):,}건 로드", file=sys.stderr)
        # 실제로 쓸 일이 생길 때 연다. --dry-run 이 빈 캐시 파일을 만들지 않도록.
        self.fh = None

    def update(self, mapping):
        self.data.update(mapping)
        if not mapping or not self.path:
            return
        if self.fh is None:
            self.fh = open(self.path, "a", encoding="utf-8")
        for en, ko in mapping.items():
            self.fh.write(json.dumps({"en": en, "ko": ko}, ensure_ascii=False) + "\n")
        self.fh.flush()

    def close(self):
        if self.fh:
            self.fh.close()


# ---------------------------------------------------------------- 로컬 모델
class LocalTranslator:
    def __init__(self, model_name, device, load_4bit=False, max_new_tokens=1024):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.max_new_tokens = max_new_tokens

        print(f"[model] 로드 중: {model_name}", file=sys.stderr)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # decoder-only 모델을 배치로 돌리려면 왼쪽 패딩이어야 한다.
        # 오른쪽으로 채우면 패딩 토큰 뒤에서 생성이 시작돼 출력이 깨진다.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs = {"trust_remote_code": True}
        if load_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
            kwargs["device_map"] = "auto"
        else:
            kwargs["dtype"] = torch.bfloat16 if device == "cuda" else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        if not load_4bit:
            self.model.to(device)
        self.model.eval()
        self.device = device

        # Qwen3 계열은 chat template 에 thinking 모드가 있다. 번역에는 방해만 된다.
        self.template_kwargs = {}
        try:
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": "x"}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            self.template_kwargs["enable_thinking"] = False
        except TypeError:
            pass

    def _prompt(self, items):
        """items: [본문, ...] -> chat template 적용된 문자열"""
        lines = [f"{i + 1}. {t}" for i, t in enumerate(items)]
        user = (
            f"다음 {len(items)}개 채팅 메시지를 규칙에 따라 한국어로 번역하라.\n"
            f"`번호. 번역문` 형식으로 정확히 {len(items)}줄만 출력하라.\n\n"
            + "\n".join(lines)
        )
        return self.tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True, **self.template_kwargs,
        )

    def _generate(self, prompts):
        """프롬프트 여러 개를 한 번에 돌려 생성분만 디코드한다."""
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        with self.torch.inference_mode():
            out = self.model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,                       # 번역은 결정적으로
                pad_token_id=self.tokenizer.pad_token_id,
            )
        gen = out[:, enc["input_ids"].shape[1]:]       # 입력 부분은 잘라낸다
        return self.tokenizer.batch_decode(gen, skip_special_tokens=True)

    def translate_packs(self, packs):
        """
        packs: [[본문, ...], ...]  ->  {본문: 번역문}

        묶음 단위로 번역하고, 줄 수가 안 맞는 묶음은 항목별로 다시 돌린다.
        그래도 실패하면 원문을 유지한다 (번역 누락이 조용히 섞이지 않도록).
        """
        out = {}
        retry = []

        for pack, raw in zip(packs, self._generate([self._prompt(p) for p in packs])):
            got = self._parse(raw, len(pack))
            for i, body in enumerate(pack):
                if i in got:
                    out[body] = got[i]
                else:
                    retry.append(body)

        if retry:
            for body, raw in zip(retry, self._generate([self._prompt([b]) for b in retry])):
                got = self._parse(raw, 1)
                out[body] = got.get(0, body)

        return out

    @staticmethod
    def _parse(raw, n):
        """`번호. 번역문` 줄들을 {0-based index: 번역문} 으로."""
        raw = THINK_RE.sub("", raw).strip()
        got = {}
        last = None
        for line in raw.split("\n"):
            m = NUM_LINE_RE.match(line)
            if m:
                idx = int(m.group(1)) - 1
                if 0 <= idx < n:
                    got[idx] = m.group(2).strip()
                    last = idx
                else:
                    last = None
            elif last is not None and line.strip():
                # 모델이 줄바꿈을 넣은 경우 직전 항목에 이어붙인다
                got[last] += " " + line.strip()
        return got


# ---------------------------------------------------------------- 묶기
def build_packs(todo, pack_size, pack_chars):
    """
    길이순으로 정렬된 todo 를 묶음으로 자른다.

    길이가 비슷한 것끼리 묶여야 패딩 낭비가 줄고, 긴 본문은 혼자 돌아야
    묶음 전체의 출력 길이를 잡아먹지 않는다.
    """
    packs, cur, cur_chars = [], [], 0
    for b in todo:
        if len(b) > SOLO_CHARS:
            if cur:
                packs.append(cur)
                cur, cur_chars = [], 0
            packs.append([b])
            continue
        if cur and (len(cur) >= pack_size or cur_chars + len(b) > pack_chars):
            packs.append(cur)
            cur, cur_chars = [], 0
        cur.append(b)
        cur_chars += len(b)
    if cur:
        packs.append(cur)
    return packs


# ---------------------------------------------------------------- main
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(
        description="jabber_en.jsonl 의 채팅 발화만 로컬 GPU 로 한국어 번역한다.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--input", default="jabber_en.jsonl")
    ap.add_argument("--output", default="jabber_ko.jsonl")
    ap.add_argument("--cache", default="translation_cache.jsonl",
                    help="translate_jabber.py 와 같은 포맷. 이어하기에 쓰인다")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default=None, help="cuda / cpu (기본: 자동)")
    ap.add_argument("--load-4bit", action="store_true",
                    help="4bit 로 로드해 VRAM 을 아낀다 (bitsandbytes 필요)")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="한 번에 GPU 에 올릴 프롬프트 수 (기본: 16)\nOOM 이면 8, 4 로 낮춘다")
    ap.add_argument("--pack", type=int, default=8,
                    help="한 프롬프트에 묶을 메시지 수 (기본: 8)\n1 로 두면 가장 안정적이지만 느리다")
    ap.add_argument("--pack-chars", type=int, default=1200,
                    help="한 프롬프트의 최대 문자 수 (기본: 1200)")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None, help="앞 N줄만 처리 (테스트용)")
    ap.add_argument("--keep-original", action="store_true", help="원문을 text_en 필드로 보존")
    ap.add_argument("--translate-noise", action="store_true",
                    help="숫자/기호만 있는 본문과 자격증명 덤프도 모델에 넣는다\n"
                         "(기본: 원문 그대로 통과시킨다)")
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

    # 2) 노이즈 분리 — 모델에 넣지 않고 원문 그대로 둔다
    noise = set()
    if not args.translate_noise:
        noise = {b for b in bodies if is_noise(b)}
        if noise:
            print(f"[stat] 노이즈 통과 {len(noise):,}건 "
                  f"({sum(len(b) for b in noise):,} chars, "
                  f"{sum(len(b) for b in noise) * 100 // max(1, sum(len(b) for b in bodies))}% of chars) "
                  f"- 숫자/기호 전용, 자격증명 덤프", file=sys.stderr)

    cache = Cache(args.cache)
    todo = [b for b in bodies if b not in cache.data and b not in noise]
    todo.sort(key=len)
    packs = build_packs(todo, args.pack, args.pack_chars)
    print(f"[stat] 신규 번역 대상 {len(todo):,}건 -> 프롬프트 {len(packs):,}개, "
          f"배치 {args.batch_size}", file=sys.stderr)

    if args.dry_run:
        print("[dry-run] 모델을 로드하지 않고 종료합니다.", file=sys.stderr)
        cache.close()
        return

    # 3) 로컬 배치 추론
    if todo:
        import torch
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if device == "cpu":
            print("[warn] CPU 로는 매우 느립니다. --limit 로 먼저 확인하세요.", file=sys.stderr)
        else:
            print(f"[model] {torch.cuda.get_device_name(0)}", file=sys.stderr)

        tr = LocalTranslator(args.model, device, args.load_4bit, args.max_new_tokens)

        started = time.time()
        done_items = 0
        for i in range(0, len(packs), args.batch_size):
            chunk = packs[i:i + args.batch_size]
            try:
                cache.update(tr.translate_packs(chunk))
            except Exception as e:
                flat = [b for p in chunk for b in p]
                print(f"[warn] 배치 실패({len(flat)}건, 원문 유지): {e!r}", file=sys.stderr)
                cache.update({b: b for b in flat})

            done_items += sum(len(p) for p in chunk)
            elapsed = time.time() - started
            rate = done_items / elapsed if elapsed else 0
            eta = (len(todo) - done_items) / rate / 60 if rate else 0
            print(f"  진행 {min(i + args.batch_size, len(packs)):,}/{len(packs):,} 프롬프트 "
                  f"({done_items:,}/{len(todo):,}건, {rate:.1f}건/초, ETA {eta:.0f}분)",
                  file=sys.stderr)

    # 4) 재조립 및 저장 (노이즈는 table 에 없으므로 rebuild 가 원문을 그대로 쓴다)
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
